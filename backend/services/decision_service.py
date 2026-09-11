"""NaviFreight Decision Orchestration Service.

Combines:
    1. Commodity Procurement Valuation (procurement_service.py)
    2. Vessel Suitability & Optimization (vessel_service.py)
    3. Model V3 Freight Forecasting Economics (forecast_service.py)

Synthesizes procurement valuation and shipping market dynamics into ONE
unified, actionable strategic recommendation:
    - BUY CARGO — CHARTER NOW
    - BUY CARGO — WAIT TO CHARTER
    - BUY CARGO — MONITOR FREIGHT
    - MONITOR CARGO — CHARTER NOW
    - MONITOR CARGO — WAIT TO CHARTER
    - MONITOR CARGO — MONITOR FREIGHT
    - WAIT FOR CARGO — WAIT TO CHARTER
    - WAIT FOR CARGO — MONITOR FREIGHT
    - NO SUITABLE VESSEL — RECONFIGURE SHIPMENT
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_ROOT = _THIS_DIR.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from services import landed_cost_service, procurement_service, vessel_service

# -------------------------------------------------------------------------
# STRATEGY DECISION MATRIX
# (Procurement Signal, Charter Decision) -> Overall Strategy Directive
# -------------------------------------------------------------------------
STRATEGY_MAP = {
    ("BUY", "CHARTER NOW"): "BUY CARGO — CHARTER NOW",
    ("BUY", "WAIT TO CHARTER"): "BUY CARGO — WAIT TO CHARTER",
    ("BUY", "MONITOR FREIGHT"): "BUY CARGO — MONITOR FREIGHT",
    ("MONITOR", "CHARTER NOW"): "MONITOR CARGO — CHARTER NOW",
    ("MONITOR", "WAIT TO CHARTER"): "MONITOR CARGO — WAIT TO CHARTER",
    ("MONITOR", "MONITOR FREIGHT"): "MONITOR CARGO — MONITOR FREIGHT",
    ("WAIT", "CHARTER NOW"): "WAIT FOR CARGO — CHARTER NOW",
    ("WAIT", "WAIT TO CHARTER"): "WAIT FOR CARGO — WAIT TO CHARTER",
    ("WAIT", "MONITOR FREIGHT"): "WAIT FOR CARGO — MONITOR FREIGHT",
}


def analyze_decision(
    origin: str,
    destination: str,
    commodity: str,
    cargo_tonnes: float,
    current_freight_usd_per_tonne: Optional[float] = None,
    commodity_df_override: Optional[pd.DataFrame] = None,
    market_overrides: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Execute end-to-end decision orchestration combining commodity procurement and vessel chartering.
    
    Args:
        origin: Loading port or trade corridor (e.g. 'Hay Point', 'Taboneo')
        destination: Discharge port or region (e.g. 'Dhamra', 'Paradip')
        commodity: Cargo commodity ('Coal', 'Thermal Coal', 'Iron Ore')
        cargo_tonnes: Shipment cargo volume in metric tonnes (strictly > 0)
        current_freight_usd_per_tonne: Optional current freight rate override
        commodity_df_override: Optional test DataFrame override for procurement service
        market_overrides: Optional test dictionary of market/weather overrides
        
    Returns:
        Structured decision result containing procurement valuation, vessel optimization,
        charter decision, overall strategy, and transparent synthesis bullets.
    """
    if cargo_tonnes <= 0:
        raise ValueError("cargo_tonnes must be strictly greater than zero.")

    # 1. Commodity Procurement Valuation
    proc_res = procurement_service.calculate_procurement_valuation(
        commodity=commodity,
        df_override=commodity_df_override,
    )
    proc_signal = proc_res["signal"]  # "BUY" | "MONITOR" | "WAIT"

    # 2. Vessel Suitability & Optimization
    vessel_res = vessel_service.optimize_vessel_chartering(
        origin=origin,
        destination=destination,
        commodity=commodity,
        cargo_tonnes=cargo_tonnes,
        current_freight_usd_per_tonne=current_freight_usd_per_tonne,
        market_overrides=market_overrides,
    )

    vessel_status = vessel_res["status"]  # "OPTIMIZED" | "NO_SUITABLE_VESSEL"
    recommended_vessel_name = vessel_res.get("recommended_vessel")

    winning_vessel: Optional[dict[str, Any]] = None
    if vessel_status == "OPTIMIZED" and recommended_vessel_name and vessel_res.get("evaluated_vessels"):
        for v in vessel_res["evaluated_vessels"]:
            if v.get("vessel_type") == recommended_vessel_name:
                winning_vessel = v
                break

    # 3. Freight Timing & Overall Strategy Synthesis
    decision_reasons: list[str] = []

    # (A) Procurement Bullet
    decision_reasons.append(
        f"Commodity Valuation ({proc_res['commodity']}): Signal is {proc_signal}. "
        f"Benchmark is ${proc_res['benchmark_price_usd_per_mt']:.2f} {proc_res['unit']} "
        f"as of {proc_res['benchmark_date']} ({proc_res['percentile']:.1f}th historical percentile, "
        f"3m momentum {proc_res['momentum_3m_pct']:+.1f}%)."
    )

    if vessel_status == "NO_SUITABLE_VESSEL" or not winning_vessel:
        charter_decision = "NO SUITABLE VESSEL"
        overall_strategy = "NO SUITABLE VESSEL — RECONFIGURE SHIPMENT"
        weather_override = False
        weather_risk_level = "LOW"
        if vessel_res.get("evaluated_vessels"):
            for ev in vessel_res["evaluated_vessels"]:
                if ev.get("forecast_risk_level"):
                    weather_risk_level = ev["forecast_risk_level"]
                    break

        decision_reasons.append(
            f"Vessel Feasibility Gate: No suitable vessel class could be chartered for "
            f"{cargo_tonnes:,.0f} mt on the {origin} -> {destination} corridor. "
            f"{vessel_res['recommendation_reason']}"
        )
        decision_reasons.append(
            "Freight Timing: Inactive due to physical draft or capacity infeasibility."
        )
        decision_reasons.append(
            f"Strategic Directive: {overall_strategy}. Split parcel into smaller lots "
            f"or redirect shipment to a deeper water port."
        )

        vessel_summary = {
            "recommended_vessel": None,
            "status": vessel_status,
            "predicted_freight_usd_per_tonne": None,
            "estimated_freight_outlay_usd": None,
            "suitability_score": None,
            "reasons": [vessel_res["recommendation_reason"]],
            "evaluated_vessels": vessel_res.get("evaluated_vessels", []),
        }
    else:
        # Map Model v3 freight recommendation
        model_rec = winning_vessel.get("forecast_recommendation", "MONITOR")
        f_change = winning_vessel.get("forecast_change_percent", 0.0)
        pred_rate = winning_vessel.get("predicted_freight_usd_per_tonne")
        outlay = winning_vessel.get("estimated_freight_outlay_usd")
        weather_risk_level = winning_vessel.get("forecast_risk_level") or "LOW"

        # Weather override is True ONLY when elevated weather risk actually changes the charter timing decision
        # Under Model V3:
        # - If f_change >= 5.0%, model_rec is "CHARTER NOW" purely due to rate surge (no weather override).
        # - If f_change < 5.0%, rate forecast alone would be "WAIT" or "MONITOR", but HIGH weather risk forces "CHARTER NOW".
        weather_override = bool(model_rec == "CHARTER NOW" and weather_risk_level == "HIGH" and f_change < 5.0)

        if model_rec == "CHARTER NOW":
            charter_decision = "CHARTER NOW"
            if weather_override:
                freight_reason = (
                    f"Elevated weather risk ({weather_risk_level}) on voyage corridor forces early fixture "
                    f"(forecast: ${pred_rate:.2f}/t, market delta: {f_change:+.1f}%). "
                    f"Lock in tonnage immediately to avoid cyclone delays, port disruptions, and spot rate spikes."
                )
            else:
                freight_reason = (
                    f"Model v3 forecast indicates freight rates will rise by {f_change:+.1f}% "
                    f"(forecast: ${pred_rate:.2f}/t). Lock in tonnage immediately to protect shipping margins."
                )
        elif model_rec == "WAIT":
            charter_decision = "WAIT TO CHARTER"
            freight_reason = (
                f"Model v3 forecast indicates freight rates will soften by {abs(f_change):.1f}% "
                f"(forecast: ${pred_rate:.2f}/t). Deferring charter fixture is projected to secure lower freight costs."
            )
        else:  # "MONITOR"
            charter_decision = "MONITOR FREIGHT"
            freight_reason = (
                f"Model v3 forecast indicates stable freight conditions ({f_change:+.1f}%, forecast: ${pred_rate:.2f}/t). "
                f"Monitor spot market for opportunistic charter fixtures."
            )

        overall_strategy = STRATEGY_MAP.get(
            (proc_signal, charter_decision),
            f"{proc_signal} CARGO — {charter_decision}",
        )

        # (B) Vessel Bullet
        decision_reasons.append(
            f"Vessel Recommendation: {recommended_vessel_name} selected with suitability score "
            f"{winning_vessel.get('suitability_score', 0):.1f}/100 "
            f"({winning_vessel.get('utilization_pct', 0):.1f}% capacity utilization, "
            f"{winning_vessel.get('draft_m', 0):.2f}m draft)."
        )

        # (C) Freight Bullet
        decision_reasons.append(f"Freight Timing: {charter_decision}. {freight_reason}")

        # (C.1) Explicit Weather Risk Override Bullet if triggered
        if weather_override:
            baseline_timing = "WAIT TO CHARTER" if f_change <= -5.0 else "MONITOR FREIGHT"
            decision_reasons.append(
                f"Weather Risk Override: Elevated weather risk ({weather_risk_level}) overrode baseline freight "
                f"timing ({baseline_timing}, {f_change:+.1f}%), prompting immediate fixture to mitigate weather hazards."
            )

        # (D) Overall Directive Bullet
        if overall_strategy == "BUY CARGO — CHARTER NOW":
            directive_summary = (
                f"Capitalize on favorable commodity pricing (${proc_res['benchmark_price_usd_per_mt']:.2f}) "
                f"and lock in {recommended_vessel_name} freight before anticipated rate increases."
            )
        elif overall_strategy == "BUY CARGO — WAIT TO CHARTER":
            directive_summary = (
                f"Secure physical cargo procurement now at attractive benchmark price, "
                f"but delay chartering {recommended_vessel_name} to capture falling freight rates."
            )
        elif overall_strategy == "BUY CARGO — MONITOR FREIGHT":
            directive_summary = (
                f"Execute cargo purchase at favorable levels while monitoring {recommended_vessel_name} "
                f"freight fixtures in a steady rate environment."
            )
        elif overall_strategy == "MONITOR CARGO — CHARTER NOW":
            directive_summary = (
                f"Track commodity price action for optimal entry while securing forward freight "
                f"on {recommended_vessel_name} to protect against rising shipping costs."
            )
        elif overall_strategy == "MONITOR CARGO — WAIT TO CHARTER":
            directive_summary = (
                f"Hold procurement execution while tracking commodity market, and delay chartering "
                f"as freight rates are projected to decrease."
            )
        elif overall_strategy == "MONITOR CARGO — MONITOR FREIGHT":
            directive_summary = (
                f"Maintain flexible monitoring posture across both commodity procurement and freight markets; "
                f"no immediate commitment required."
            )
        elif overall_strategy == "WAIT FOR CARGO — CHARTER NOW":
            if weather_override:
                directive_summary = (
                    f"Hold cargo procurement due to elevated commodity benchmark levels, "
                    f"but pre-book {recommended_vessel_name} charter immediately due to elevated weather risk."
                )
            else:
                directive_summary = (
                    f"Hold cargo procurement due to elevated commodity benchmark levels, "
                    f"but lock in {recommended_vessel_name} charter now to protect against rising freight rates."
                )
        elif overall_strategy == "WAIT FOR CARGO — WAIT TO CHARTER":
            directive_summary = (
                f"Postpone both cargo acquisition and chartering; commodity prices are elevated and "
                f"freight rates are expected to drop."
            )
        elif overall_strategy == "WAIT FOR CARGO — MONITOR FREIGHT":
            directive_summary = (
                f"Hold cargo procurement due to elevated commodity benchmark levels and monitor freight trends closely."
            )
        else:
            directive_summary = f"Align procurement posture ({proc_signal}) with freight fixture timing ({charter_decision})."

        decision_reasons.append(f"Strategic Directive: {overall_strategy}. {directive_summary}")

        vessel_summary = {
            "recommended_vessel": recommended_vessel_name,
            "status": vessel_status,
            "predicted_freight_usd_per_tonne": pred_rate,
            "estimated_freight_outlay_usd": outlay,
            "suitability_score": winning_vessel.get("suitability_score"),
            "reasons": winning_vessel.get("reasons", []),
            "evaluated_vessels": vessel_res.get("evaluated_vessels", []),
        }

    # 4. Landed Cost Economics (Commodity FOB + Ocean Freight)
    pred_freight_rate = vessel_summary.get("predicted_freight_usd_per_tonne")
    landed_cost_res = landed_cost_service.calculate_landed_cost(
        commodity=proc_res["commodity"],
        commodity_fob_usd=proc_res["benchmark_price_usd_per_mt"],
        ocean_freight_usd_per_tonne=pred_freight_rate,
        cargo_tonnes=cargo_tonnes,
        commodity_unit=proc_res.get("unit"),
    )

    # (E) Landed Cost Bullet
    if landed_cost_res.get("estimated_landed_cost_usd") is not None:
        decision_reasons.append(
            f"Delivered Acquisition Cost: Estimated Landed Cost is ${landed_cost_res['estimated_landed_cost_usd']:.2f} {landed_cost_res['landed_cost_unit']} "
            f"(FOB benchmark ${landed_cost_res['commodity_fob_usd']:.2f} {landed_cost_res['commodity_unit']} + "
            f"Ocean Freight ${landed_cost_res['ocean_freight_usd_per_tonne']:.2f}/t). "
            f"Total delivered shipment outlay: ${landed_cost_res['estimated_total_landed_outlay_usd']:,.2f}."
        )
    else:
        decision_reasons.append(
            "Delivered Acquisition Cost: Estimated Landed Cost is unavailable because no suitable vessel could be chartered."
        )

    return {
        "commodity": proc_res["commodity"],
        "origin": vessel_res["origin"],
        "destination": vessel_res["destination"],
        "cargo_tonnes": float(cargo_tonnes),
        "procurement": {
            "signal": proc_res["signal"],
            "benchmark_price_usd_per_mt": proc_res["benchmark_price_usd_per_mt"],
            "unit": proc_res["unit"],
            "benchmark_date": proc_res["benchmark_date"],
            "percentile": proc_res["percentile"],
            "momentum_3m_pct": proc_res["momentum_3m_pct"],
            "data_freshness": proc_res["data_freshness"],
            "reasons": proc_res["reasons"],
        },
        "vessel": vessel_summary,
        "landed_cost": landed_cost_res,
        "charter_decision": charter_decision,
        "overall_strategy": overall_strategy,
        "weather_override": weather_override,
        "weather_risk_level": weather_risk_level,
        "decision_reasons": decision_reasons,
    }
