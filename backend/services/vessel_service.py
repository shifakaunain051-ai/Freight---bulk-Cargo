"""Vessel Suitability and Chartering Evaluation Service for NaviFreight.

Evaluates vessel classes for bulk cargo voyages to East Coast India using:
    - data/vessel_specs.csv (Baltic Exchange standard dimensions)
    - data/port_constraints.csv (East Coast Indian port hydrographic & berth limits)
    - data/baltic_route_specs.csv (Route benchmark definitions for reference)
    - Model v3 Freight Forecasting via existing forecast service

Provides multi-criteria optimization:
    Physical Draft Compatibility + Cargo Volume Fit + Model v3 Freight Outlay Economics
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_ROOT = _THIS_DIR.parent
_REPO_ROOT = _BACKEND_ROOT.parent

if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from services.analytics_service import _load_historical_data
from services.forecast_service import forecast

# File paths
VESSEL_SPECS_PATH = _REPO_ROOT / "data" / "vessel_specs.csv"
PORT_CONSTRAINTS_PATH = _REPO_ROOT / "data" / "port_constraints.csv"
BALTIC_ROUTES_PATH = _REPO_ROOT / "data" / "baltic_route_specs.csv"

# Cached DataFrames
_VESSEL_SPECS_DF: Optional[pd.DataFrame] = None
_PORT_CONSTRAINTS_DF: Optional[pd.DataFrame] = None
_BALTIC_ROUTES_DF: Optional[pd.DataFrame] = None


# -------------------------------------------------------------------------
# SCORING CONFIGURATION & THRESHOLDS
# -------------------------------------------------------------------------
VESSEL_SCORING_CONFIG = {
    # Capacity Utilization Thresholds (cargo_tonnes / standard_dwt)
    "min_economic_utilization": 0.45,   # Below 45% DWT: severe deadfreight, uneconomic
    "max_physical_utilization": 1.05,   # Above 105% DWT: physical overload, impossible
    "ideal_utilization_min": 0.80,      # 80% - 95%: optimal commercial utilization
    "ideal_utilization_max": 0.95,
    
    # Scoring Weights (Scale: 0 to 100)
    "utilization_weight": 40.0,         # Points for capacity fit
    "economics_weight": 40.0,           # Points for total freight outlay efficiency
    "corridor_alignment_weight": 20.0,  # Points for standard corridor fixture preference
}


# Canonical corridors and default benchmark rates (derived from master training data)
CANONICAL_CORRIDORS = {
    "Hay Point": {
        "supported_vessels": {
            "Capesize": {"benchmark_rate": 17.20, "is_primary": True},
            "Panamax": {"benchmark_rate": 20.00, "is_primary": True},
            "Supramax": {"benchmark_rate": 22.50, "is_primary": False},
            "Handysize": {"benchmark_rate": 25.50, "is_primary": False},
        },
        "default_commodity": "Coal",
    },
    "Australia West Coast": {
        "supported_vessels": {
            "Capesize": {"benchmark_rate": 12.90, "is_primary": True},
            "Panamax": {"benchmark_rate": 15.50, "is_primary": False},
            "Supramax": {"benchmark_rate": 18.00, "is_primary": False},
            "Handysize": {"benchmark_rate": 21.00, "is_primary": False},
        },
        "default_commodity": "Iron Ore",
    },
    "Taboneo": {
        "supported_vessels": {
            "Panamax": {"benchmark_rate": 11.80, "is_primary": True},
            "Supramax": {"benchmark_rate": 13.80, "is_primary": True},
            "Handysize": {"benchmark_rate": 16.50, "is_primary": True},
        },
        "default_commodity": "Thermal Coal",
    },
}


def load_vessel_specs() -> pd.DataFrame:
    """Load and cache Baltic Exchange standard vessel specifications."""
    global _VESSEL_SPECS_DF
    if _VESSEL_SPECS_DF is not None:
        return _VESSEL_SPECS_DF

    if not VESSEL_SPECS_PATH.exists():
        raise FileNotFoundError(f"Vessel specs missing at {VESSEL_SPECS_PATH}")

    df = pd.read_csv(VESSEL_SPECS_PATH)
    _VESSEL_SPECS_DF = df
    return _VESSEL_SPECS_DF


def load_port_constraints() -> pd.DataFrame:
    """Load and cache East Coast India port constraints."""
    global _PORT_CONSTRAINTS_DF
    if _PORT_CONSTRAINTS_DF is not None:
        return _PORT_CONSTRAINTS_DF

    if not PORT_CONSTRAINTS_PATH.exists():
        raise FileNotFoundError(f"Port constraints missing at {PORT_CONSTRAINTS_PATH}")

    df = pd.read_csv(PORT_CONSTRAINTS_PATH)
    _PORT_CONSTRAINTS_DF = df
    return _PORT_CONSTRAINTS_DF


def load_baltic_routes() -> pd.DataFrame:
    """Load and cache Baltic route benchmark specifications (reference only)."""
    global _BALTIC_ROUTES_DF
    if _BALTIC_ROUTES_DF is not None:
        return _BALTIC_ROUTES_DF

    if not BALTIC_ROUTES_PATH.exists():
        raise FileNotFoundError(f"Baltic routes missing at {BALTIC_ROUTES_PATH}")

    df = pd.read_csv(BALTIC_ROUTES_PATH)
    _BALTIC_ROUTES_DF = df
    return _BALTIC_ROUTES_DF


def check_port_compatibility(vessel_type: str, draft_m: float, destination: str) -> tuple[bool, str]:
    """Check whether vessel draft and class are compatible with destination port.
    
    Returns:
        (is_compatible, reason_string)
    """
    dest_clean = destination.strip()

    # Generic regional destination: East Coast India deepwater ports accommodate all classes
    if dest_clean.lower() in {"east coast india", "east_coast_india", "eci"}:
        return True, "Destination represents general East Coast India region (deepwater ports accessible)."

    ports_df = load_port_constraints()
    
    # Match specific port (case-insensitive substring or exact match)
    match = ports_df[ports_df["port"].str.lower() == dest_clean.lower()]
    if len(match) == 0:
        # Check partial match
        match = ports_df[ports_df["port"].str.lower().apply(lambda p: p in dest_clean.lower() or dest_clean.lower() in p)]

    if len(match) == 0:
        # Unknown port: fallback to physical draft comparison against general 14.0m baseline
        return True, f"Port '{destination}' unlisted in specific constraint table; assuming regional standards."

    port_row = match.iloc[0]
    port_name = port_row["port"]
    op_draft = float(port_row["operational_draft_m"])
    notes = str(port_row.get("notes", ""))

    # Check explicit compatibility flags and operational draft
    v_lower = vessel_type.lower()
    if v_lower == "capesize":
        cap_ok = bool(port_row["capesize_compatible"])
        if not cap_ok:
            return False, (
                f"Capesize draft ({draft_m:.2f}m) exceeds {port_name} operational draft limit ({op_draft:.2f}m). "
                f"{notes}"
            )
        return True, f"{port_name} accommodates Capesize bulk carriers ({op_draft:.2f}m permissible draft)."

    elif v_lower == "panamax":
        pan_ok = bool(port_row["panamax_compatible"])
        if not pan_ok or draft_m > op_draft:
            return False, (
                f"Panamax draft ({draft_m:.2f}m) exceeds {port_name} operational draft limit ({op_draft:.2f}m). "
                f"{notes}"
            )
        return True, f"{port_name} accommodates fully laden Panamax ({op_draft:.2f}m permissible draft)."

    elif v_lower == "supramax":
        if draft_m > op_draft:
            return False, (
                f"Supramax draft ({draft_m:.2f}m) exceeds {port_name} operational draft limit ({op_draft:.2f}m). "
                f"{notes}"
            )
        return True, f"{port_name} accommodates fully laden Supramax ({op_draft:.2f}m permissible draft)."

    elif v_lower == "handysize":
        if draft_m > op_draft:
            return False, (
                f"Handysize draft ({draft_m:.2f}m) exceeds {port_name} operational draft limit ({op_draft:.2f}m). "
                f"{notes}"
            )
        return True, f"{port_name} accommodates fully laden Handysize ({op_draft:.2f}m permissible draft)."

    return True, f"Port compatibility verified for {vessel_type} at {port_name}."


def optimize_vessel_chartering(
    origin: str,
    destination: str,
    commodity: str,
    cargo_tonnes: float,
    current_freight_usd_per_tonne: Optional[float] = None,
    market_overrides: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Evaluate all vessel classes on cargo volume, specifications, port limits, and economics.
    
    Returns structured optimization decision with one recommended vessel.
    """
    if cargo_tonnes <= 0:
        raise ValueError("cargo_tonnes must be strictly greater than zero.")

    # Match canonical origin
    origin_clean = origin.strip()
    corridor_spec = None
    canonical_origin = None

    for o_name, spec in CANONICAL_CORRIDORS.items():
        if o_name.lower() == origin_clean.lower() or origin_clean.lower() in o_name.lower():
            corridor_spec = spec
            canonical_origin = o_name
            break

    if not corridor_spec:
        supported_origins = list(CANONICAL_CORRIDORS.keys())
        raise ValueError(
            f"Unsupported origin '{origin}'. Supported canonical corridors: {', '.join(supported_origins)}"
        )

    vessel_specs_df = load_vessel_specs()
    supported_vessels_map = corridor_spec["supported_vessels"]

    evaluated_vessels = []
    eligible_candidates = []

    # Evaluate each vessel class defined in vessel_specs.csv
    for _, v_row in vessel_specs_df.iterrows():
        v_type = str(v_row["vessel_type"])
        std_dwt = float(v_row["standard_dwt"])
        draft_m = float(v_row["draft_m"])

        reasons: list[str] = []
        
        # 1. Utilization & Cargo Volume Fit
        utilization_pct = round((cargo_tonnes / std_dwt) * 100.0, 1)

        cargo_fit = True
        cargo_fit_label = "Suitable"
        min_econ_util = 0.28 if v_type == "Handysize" else VESSEL_SCORING_CONFIG["min_economic_utilization"]
        if cargo_tonnes > std_dwt * VESSEL_SCORING_CONFIG["max_physical_utilization"]:
            cargo_fit = False
            cargo_fit_label = "Cargo exceeds practical capacity"
            reasons.append(
                f"Cargo volume ({cargo_tonnes:,.0f} mt) exceeds {v_type} maximum capacity ({std_dwt:,.0f} DWT, {utilization_pct}%)."
            )
        elif cargo_tonnes < std_dwt * min_econ_util:
            cargo_fit = False
            cargo_fit_label = "Excessive unused capacity"
            reasons.append(
                f"Cargo volume ({cargo_tonnes:,.0f} mt) severely underutilizes {v_type} capacity ({std_dwt:,.0f} DWT, {utilization_pct}%); uneconomic deadfreight."
            )
        elif 80.0 <= utilization_pct <= 98.0:
            cargo_fit_label = "Optimal fit"
            reasons.append(
                f"Cargo volume ({cargo_tonnes:,.0f} mt) achieves optimal {utilization_pct}% utilization of {std_dwt:,.0f} DWT capacity."
            )
        else:
            cargo_fit_label = "Suitable"
            reasons.append(
                f"Cargo volume ({cargo_tonnes:,.0f} mt) achieves viable {utilization_pct}% utilization of {std_dwt:,.0f} DWT capacity."
            )

        # 2. Port Hydrographic Compatibility
        port_compatible, port_reason = check_port_compatibility(v_type, draft_m, destination)
        port_fit_label = f"Compatible ({draft_m:.1f}m draft)" if port_compatible else f"Draft exceeds limit ({draft_m:.1f}m)"
        reasons.append(port_reason)

        # 3. Corridor Operational Viability
        corridor_supported = v_type in supported_vessels_map
        if not corridor_supported:
            reasons.append(
                f"{v_type} is not an active canonical vessel class for the {canonical_origin} corridor."
            )
        else:
            reasons.append(f"{v_type} is commercially established on the {canonical_origin} trade lane.")

        # Hard Exclusion Eligibility Gate
        is_eligible = bool(cargo_fit and port_compatible and corridor_supported)

        pred_freight = None
        freight_outlay = None
        forecast_change_pct = None
        forecast_rec = None
        forecast_risk = "LOW"
        suitability_score = 0.0

        if is_eligible:
            # Determine benchmark base rate for Model v3 inference
            base_rate = current_freight_usd_per_tonne
            if base_rate is None or base_rate <= 0:
                base_rate = supported_vessels_map[v_type]["benchmark_rate"]

            # Query Model v3 forecast using existing forecast service
            predict_payload = {
                "origin": canonical_origin,
                "destination": "East Coast India",
                "commodity": commodity,
                "vessel_type": v_type,
                "current_freight_usd_per_tonne": base_rate,
            }
            # Populate authentic market & weather baseline features from genuine historical dataset
            try:
                hist_df = _load_historical_data()
                latest_date = str(hist_df["date"].max())
                match_row = hist_df[
                    (hist_df["date"] == latest_date)
                    & (hist_df["origin"] == canonical_origin)
                    & (hist_df["vessel_type"] == v_type)
                ]
                if len(match_row) == 0:
                    match_row = hist_df[
                        (hist_df["date"] == latest_date)
                        & (hist_df["origin"] == canonical_origin)
                    ]
                if len(match_row) > 0:
                    row = match_row.iloc[0]
                    for field in [
                        "bdi",
                        "vlsfo_usd_per_tonne",
                        "coal_price_usd_per_mt",
                        "iron_ore_price_usd_per_dmt",
                        "wind_kmh",
                        "wave_height_m",
                        "cyclone_risk",
                        "weather_delay_days",
                    ]:
                        if field not in predict_payload or predict_payload[field] is None:
                            predict_payload[field] = float(row[field])
            except Exception:
                pass

            # Apply market/weather overrides if provided (e.g. for scenario testing)
            if market_overrides:
                for k, v in market_overrides.items():
                    if v is not None:
                        predict_payload[k] = float(v)

            try:
                forecast_res = forecast(predict_payload)
                pred_freight = float(forecast_res["predicted_next_month_freight_usd_per_tonne"])
                freight_outlay = round(pred_freight * cargo_tonnes, 2)
                forecast_change_pct = float(forecast_res.get("forecast_change_percent", 0.0))
                forecast_rec = str(forecast_res.get("recommendation", "MONITOR"))
                forecast_risk = str(forecast_res.get("risk_level", "LOW"))
            except Exception as exc:
                is_eligible = False
                reasons.append(f"Model v3 forecast failed: {exc}")

        eval_item = {
            "vessel_type": v_type,
            "standard_dwt": std_dwt,
            "draft_m": draft_m,
            "cargo_tonnes": float(cargo_tonnes),
            "utilization_pct": utilization_pct,
            "predicted_freight_usd_per_tonne": pred_freight,
            "estimated_freight_outlay_usd": freight_outlay,
            "forecast_change_percent": forecast_change_pct,
            "forecast_recommendation": forecast_rec,
            "forecast_risk_level": forecast_risk,
            "port_compatible": port_compatible,
            "port_fit_label": port_fit_label,
            "cargo_fit": cargo_fit,
            "cargo_fit_label": cargo_fit_label,
            "corridor_supported": corridor_supported,
            "eligible": is_eligible,
            "suitability_score": 0.0,
            "reasons": reasons,
        }

        evaluated_vessels.append(eval_item)
        if is_eligible and freight_outlay is not None:
            eligible_candidates.append(eval_item)

    # Calculate Suitability Scores for Eligible Candidates
    if eligible_candidates:
        max_outlay = max(c["estimated_freight_outlay_usd"] for c in eligible_candidates)
        min_outlay = min(c["estimated_freight_outlay_usd"] for c in eligible_candidates)

        for cand in eligible_candidates:
            v_type = cand["vessel_type"]
            util_pct = cand["utilization_pct"] / 100.0

            # (A) Utilization Score (up to 50 pts)
            # Prefer the smallest suitable vessel that carries cargo without excessive unused capacity
            if VESSEL_SCORING_CONFIG["ideal_utilization_min"] <= util_pct <= VESSEL_SCORING_CONFIG["ideal_utilization_max"]:
                util_score = 50.0
            elif (0.65 <= util_pct < 0.80) or (0.95 < util_pct <= 1.00):
                util_score = 38.0
            elif 0.50 <= util_pct < 0.65:
                util_score = 26.0
            else:
                util_score = 5.0

            # (B) Economic Efficiency Score (up to 30 pts)
            if max_outlay > min_outlay:
                savings_ratio = (max_outlay - cand["estimated_freight_outlay_usd"]) / (max_outlay - min_outlay)
                econ_score = 10.0 + (20.0 * savings_ratio)
            else:
                econ_score = 30.0

            # (C) Corridor Alignment (up to 20 pts)
            is_primary = supported_vessels_map.get(v_type, {}).get("is_primary", False)
            align_score = 20.0 if is_primary else 10.0

            total_score = round(util_score + econ_score + align_score, 1)
            cand["suitability_score"] = total_score
            cand["reasons"].append(
                f"Suitability Score: {total_score}/100 (Utilization: {util_score:.0f}pts, Economics: {econ_score:.0f}pts, Corridor: {align_score:.0f}pts)."
            )

        # Sort eligible by suitability score descending
        eligible_candidates.sort(key=lambda c: c["suitability_score"], reverse=True)
        winner = eligible_candidates[0]
        status = "OPTIMIZED"
        recommended_vessel = winner["vessel_type"]
        recommendation_reason = (
            f"{recommended_vessel} is recommended with highest suitability score ({winner['suitability_score']}/100). "
            f"Achieves {winner['utilization_pct']}% cargo utilization for {cargo_tonnes:,.0f} mt with forecasted freight "
            f"rate ${winner['predicted_freight_usd_per_tonne']:.2f}/t (Total outlay: ${winner['estimated_freight_outlay_usd']:,.2f})."
        )
    else:
        status = "NO_SUITABLE_VESSEL"
        recommended_vessel = None
        recommendation_reason = (
            f"No suitable vessel class found for {cargo_tonnes:,.0f} mt on the {canonical_origin} -> {destination} corridor. "
            f"All evaluated vessel options failed either port draft constraints, cargo volume thresholds, or corridor operational support."
        )

    return {
        "origin": canonical_origin,
        "destination": destination,
        "commodity": commodity,
        "cargo_tonnes": float(cargo_tonnes),
        "status": status,
        "recommended_vessel": recommended_vessel,
        "recommendation_reason": recommendation_reason,
        "evaluated_vessels": evaluated_vessels,
        "source_data": {
            "vessel_specs": "data/vessel_specs.csv",
            "port_constraints": "data/port_constraints.csv",
            "route_specs": "data/baltic_route_specs.csv",
        },
    }
