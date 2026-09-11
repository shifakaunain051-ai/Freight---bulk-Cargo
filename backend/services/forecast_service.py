"""Forecast service: merges user-supplied input with stored DB data and logs telemetry.

Strategy:
    User-provided values ALWAYS win. Any field the user omits is filled from
    the database when a recent value is available:
        - weather (wind_kmh, wave_height_m, cyclone_risk, weather_delay_days)
          <- latest weather_data row for the ORIGIN port
        - market (bdi, vlsfo_usd_per_tonne, coal_price_usd_per_mt,
          iron_ore_price_usd_per_dmt) <- latest market_data rows
        - current_freight_usd_per_tonne <- strictly required from the user

If any model-required field is still missing after merging, a structured
ForecastDataError is raised so the API layer can return a clear 422 JSON response.
Zero fabricated values are used.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ensure backend root is importable
_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_ROOT = _THIS_DIR.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from data import database
from predict import FEATURE_METADATA, FEATURES, MODEL_PATH, predict_freight


# Default freshness thresholds
MAX_WEATHER_AGE_HOURS = 24.0
MAX_MARKET_AGE_HOURS = 72.0


class ForecastDataError(ValueError):
    """Structured error for missing or invalid forecast inputs."""

    def __init__(
        self,
        error_code: str,
        message: str,
        missing_fields: list[str] | None = None,
        detail: str | None = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.missing_fields = missing_fields or []
        self.detail = detail or message


# Fields the model needs that the user is allowed to omit (filled from DB).
DB_FILLABLE = {
    "wind_kmh", "wave_height_m", "cyclone_risk", "weather_delay_days",
    "bdi", "vlsfo_usd_per_tonne", "coal_price_usd_per_mt",
    "iron_ore_price_usd_per_dmt",
}

# Fields the user MUST always provide (identity + current freight).
REQUIRED_FROM_USER = {
    "origin", "destination", "commodity", "vessel_type",
    "current_freight_usd_per_tonne",
}


def _parse_iso_utc(ts_str: str) -> Optional[datetime]:
    """Parse ISO timestamp string into timezone-aware UTC datetime."""
    try:
        clean = ts_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _fill_weather(merged: dict, sources: dict, port: str) -> None:
    weather = database.get_latest_weather(port)
    if not weather:
        return
    mapping = {
        "wind_kmh": weather.get("wind_kmh"),
        "wave_height_m": weather.get("wave_height_m"),
        "cyclone_risk": weather.get("cyclone_risk"),
        "weather_delay_days": weather.get("weather_delay_days"),
    }
    fetched_at = weather.get("fetched_at", "")

    # Calculate age and freshness tag
    dt = _parse_iso_utc(fetched_at)
    if dt:
        age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
        if age_hours > MAX_WEATHER_AGE_HOURS:
            tag = f"weather_db[{port}@{fetched_at}:STALE({age_hours:.1f}h_old)]"
        else:
            tag = f"weather_db[{port}@{fetched_at}]"
    else:
        tag = f"weather_db[{port}@{fetched_at}]"

    for field, value in mapping.items():
        if value is None:
            continue
        if merged.get(field) is None:
            merged[field] = value
            sources[field] = tag


def _fill_market(merged: dict, sources: dict) -> None:
    market = database.get_latest_market()
    if not market:
        return
    mapping = {
        "bdi": market.get("bdi"),
        "vlsfo_usd_per_tonne": market.get("vlsfo_usd_per_tonne"),
        "coal_price_usd_per_mt": market.get("coal_price_usd_per_mt"),
        "iron_ore_price_usd_per_dmt": market.get("iron_ore_price_usd_per_dmt"),
    }
    for field, quote in mapping.items():
        if quote is None:
            continue
        value = quote.get("value") if isinstance(quote, dict) else None
        if value is None:
            continue
        fetched_at = quote.get("fetched_at", "")
        dt = _parse_iso_utc(fetched_at)
        if dt:
            age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
            if age_hours > MAX_MARKET_AGE_HOURS:
                tag = f"market_db[{field}@{fetched_at}:STALE({age_hours:.1f}h_old)]"
            else:
                tag = f"market_db[{field}@{fetched_at}]"
        else:
            tag = f"market_db[{field}@{fetched_at}]"

        if merged.get(field) is None:
            merged[field] = value
            sources[field] = tag


def build_forecast_input(user_input: dict) -> tuple[dict, dict]:
    """Merge user input with stored data and enforce zero-fabrication.

    Returns:
        (merged_input, sources) where sources maps field_name -> provenance string.
    Raises:
        ForecastDataError if any required feature is missing.
    """
    merged: dict = {}
    sources: dict = {}
    for field in FEATURES:
        val = user_input.get(field)
        if val is not None:
            merged[field] = val
            sources[field] = "user"

    # Identity fields must come from user
    missing_identity = [f for f in REQUIRED_FROM_USER if f not in merged or merged[f] is None]
    if missing_identity:
        raise ForecastDataError(
            error_code="INVALID_FORECAST_INPUT",
            message=f"Missing required identity fields: {', '.join(missing_identity)}",
            missing_fields=missing_identity,
            detail="Ensure origin, destination, commodity, vessel_type, and current_freight_usd_per_tonne are provided.",
        )

    # Fill weather from origin port
    _fill_weather(merged, sources, merged["origin"])
    # Fill market data
    _fill_market(merged, sources)

    # Check for missing values and classify error
    missing = [f for f in FEATURES if f not in merged or merged[f] is None]
    if missing:
        market_missing = [f for f in missing if f in {"bdi", "vlsfo_usd_per_tonne", "coal_price_usd_per_mt", "iron_ore_price_usd_per_dmt"}]
        weather_missing = [f for f in missing if f in {"wind_kmh", "wave_height_m", "cyclone_risk", "weather_delay_days"}]

        if market_missing and not weather_missing:
            error_code = "MARKET_DATA_MISSING"
        elif weather_missing and not market_missing:
            error_code = "WEATHER_DATA_MISSING"
        else:
            error_code = "DATA_MISSING"

        raise ForecastDataError(
            error_code=error_code,
            message=f"Missing model inputs that could not be filled from the database: {', '.join(missing)}.",
            missing_fields=missing,
            detail="Provide missing fields in the request or populate the database using `python -m data.update_data`.",
        )

    return merged, sources


def forecast(user_input: dict) -> dict:
    """Build complete input dict, run model inference, and record telemetry log."""
    start_time = time.perf_counter()
    merged, sources = build_forecast_input(user_input)
    result = predict_freight(merged)
    result["sources"] = sources
    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    # Insert prediction telemetry log (safely, non-blocking)
    database.insert_prediction_log(
        model_version=MODEL_PATH.stem,
        origin=str(merged.get("origin", "")),
        destination=str(merged.get("destination", "")),
        commodity=str(merged.get("commodity", "")),
        vessel_type=str(merged.get("vessel_type", "")),
        current_freight_usd_per_tonne=float(result.get("current_freight_usd_per_tonne", 0.0)),
        predicted_next_month_freight_usd_per_tonne=float(result.get("predicted_next_month_freight_usd_per_tonne", 0.0)),
        forecast_change_percent=float(result.get("forecast_change_percent", 0.0)),
        risk_level=str(result.get("risk_level", "LOW")),
        recommendation=str(result.get("recommendation", "MONITOR")),
        latency_ms=latency_ms,
        provenance=sources,
    )

    return result


def run_scenario_forecast(user_input: dict) -> dict:
    """Execute a what-if scenario simulation against Model v3.

    Calculates both baseline and scenario forecasts using the same underlying pipeline,
    computes comparative impact metrics, and verifies safety constraints.
    """
    start_time = time.perf_counter()

    # 1. Build and run baseline forecast
    base_merged, base_sources = build_forecast_input(user_input)
    baseline_result = predict_freight(base_merged)
    baseline_result["sources"] = base_sources

    # 2. Extract and validate scenario modifications
    scenario_changes_input = user_input.get("scenario_changes")
    if hasattr(scenario_changes_input, "model_dump"):
        scen_dict = scenario_changes_input.model_dump(exclude_none=True)
    elif isinstance(scenario_changes_input, dict):
        scen_dict = {k: v for k, v in scenario_changes_input.items() if v is not None}
    else:
        scen_dict = {}

    scen_merged = dict(base_merged)
    scen_sources = dict(base_sources)
    changes_list = []

    # Mapping of scenario keys to (target_feature, is_relative, change_type)
    param_map = {
        "bdi": ("bdi", False),
        "bdi_change_percent": ("bdi", True),
        "vlsfo_usd_per_tonne": ("vlsfo_usd_per_tonne", False),
        "vlsfo_change_percent": ("vlsfo_usd_per_tonne", True),
        "coal_price_usd_per_mt": ("coal_price_usd_per_mt", False),
        "coal_price_change_percent": ("coal_price_usd_per_mt", True),
        "iron_ore_price_usd_per_dmt": ("iron_ore_price_usd_per_dmt", False),
        "iron_ore_price_change_percent": ("iron_ore_price_usd_per_dmt", True),
        "wind_kmh": ("wind_kmh", False),
        "wind_change_percent": ("wind_kmh", True),
        "wave_height_m": ("wave_height_m", False),
        "wave_height_change_percent": ("wave_height_m", True),
        "cyclone_risk": ("cyclone_risk", False),
        "cyclone_risk_change": ("cyclone_risk", "delta"),
        "weather_delay_days": ("weather_delay_days", False),
        "weather_delay_change_percent": ("weather_delay_days", True),
        "current_freight_usd_per_tonne": ("current_freight_usd_per_tonne", False),
        "current_freight_change_percent": ("current_freight_usd_per_tonne", True),
    }

    # Disallowed modifications
    for forbidden in ["origin", "destination", "commodity", "vessel_type"]:
        if forbidden in scen_dict:
            raise ForecastDataError(
                error_code="INVALID_SCENARIO_INPUT",
                message=f"Modifying trade lane category '{forbidden}' is not supported in scenario analysis.",
                detail="Scenario analysis tests market and weather shocks on an established trade lane.",
            )

    # Process each provided scenario change
    processed_features = set()

    for key, mod_val in scen_dict.items():
        if key not in param_map:
            raise ForecastDataError(
                error_code="INVALID_SCENARIO_INPUT",
                message=f"Unrecognized scenario parameter: '{key}'.",
                detail="Supported scenario parameters include bdi, vlsfo_usd_per_tonne, cyclone_risk, etc. (absolute and percent changes).",
            )

        target_feat, mode = param_map[key]
        if target_feat in processed_features:
            continue  # Avoid duplicate overrides on same feature

        base_val = float(base_merged[target_feat])
        label, unit = FEATURE_METADATA.get(target_feat, (target_feat.replace("_", " ").title(), ""))

        if mode is False:  # Absolute override
            new_val = float(mod_val)
            pct_change = round((new_val - base_val) / base_val * 100.0, 2) if base_val > 0 else None
            prov_tag = f"scenario[={new_val}]"
        elif mode is True:  # Relative percent shock
            pct_change = float(mod_val)
            new_val = base_val * (1.0 + pct_change / 100.0)
            prov_tag = f"scenario[{pct_change:+.1f}%]"
        elif mode == "delta":  # Absolute score delta (e.g. +2 points on cyclone)
            new_val = base_val + float(mod_val)
            pct_change = round((new_val - base_val) / base_val * 100.0, 2) if base_val > 0 else None
            prov_tag = f"scenario[{float(mod_val):+.1f}pts]"

        # Safety Validation per feature
        if target_feat == "bdi" and new_val <= 0:
            raise ForecastDataError(
                error_code="INVALID_SCENARIO_INPUT",
                message=f"Simulated BDI must be strictly positive (>0), got {new_val:.2f}.",
                detail="BDI values cannot be zero or negative.",
            )
        if target_feat in {"vlsfo_usd_per_tonne", "coal_price_usd_per_mt", "iron_ore_price_usd_per_dmt"} and new_val < 0:
            raise ForecastDataError(
                error_code="INVALID_SCENARIO_INPUT",
                message=f"Simulated price for {target_feat} cannot be negative, got {new_val:.2f}.",
                detail="Commodity and bunker prices must be >= 0.",
            )
        if target_feat in {"wind_kmh", "wave_height_m", "weather_delay_days"} and new_val < 0:
            raise ForecastDataError(
                error_code="INVALID_SCENARIO_INPUT",
                message=f"Simulated weather parameter {target_feat} cannot be negative, got {new_val:.2f}.",
                detail="Weather parameters must be >= 0.",
            )
        if target_feat == "cyclone_risk" and not (0.0 <= new_val <= 5.0):
            raise ForecastDataError(
                error_code="INVALID_SCENARIO_INPUT",
                message=f"Simulated cyclone risk score must be between 0 and 5, got {new_val:.2f}.",
                detail="Cyclone alert index operates on a 0-5 scale.",
            )
        if target_feat == "current_freight_usd_per_tonne" and new_val <= 0:
            raise ForecastDataError(
                error_code="INVALID_SCENARIO_INPUT",
                message=f"Simulated base freight must be strictly positive (>0), got {new_val:.2f}.",
                detail="Current freight rate must be > 0.",
            )

        scen_merged[target_feat] = new_val
        scen_sources[target_feat] = prov_tag
        processed_features.add(target_feat)

        changes_list.append({
            "feature": target_feat,
            "feature_label": label,
            "baseline": round(base_val, 2),
            "scenario": round(new_val, 2),
            "absolute_change": round(new_val - base_val, 2),
            "percentage_change": pct_change,
            "unit": unit,
        })

    # Limit check (maximum 5 simultaneous parameter changes for hackathon safety)
    if len(changes_list) > 5:
        raise ForecastDataError(
            error_code="INVALID_SCENARIO_INPUT",
            message=f"A maximum of 5 scenario modifications is supported per request, received {len(changes_list)}.",
            detail="Reduce the number of modified scenario parameters.",
        )

    # 3. Run scenario forecast
    scenario_result = predict_freight(scen_merged)
    scenario_result["sources"] = scen_sources

    # 4. Compute comparative impact metrics
    base_pred_rate = baseline_result["predicted_next_month_freight_usd_per_tonne"]
    scen_pred_rate = scenario_result["predicted_next_month_freight_usd_per_tonne"]

    diff_usd = round(scen_pred_rate - base_pred_rate, 2)
    diff_pct = round((diff_usd / base_pred_rate) * 100.0, 2) if base_pred_rate > 0 else 0.0

    base_risk = baseline_result["risk_level"]
    scen_risk = scenario_result["risk_level"]
    risk_shift = f"{base_risk} -> {scen_risk}" if base_risk != scen_risk else f"{base_risk} (unchanged)"

    base_rec = baseline_result["recommendation"]
    scen_rec = scenario_result["recommendation"]
    rec_shift = f"{base_rec} -> {scen_rec}" if base_rec != scen_rec else f"{base_rec} (unchanged)"

    impact = {
        "difference_usd_per_tonne": diff_usd,
        "difference_percent": diff_pct,
        "baseline_change_percent": baseline_result["forecast_change_percent"],
        "scenario_change_percent": scenario_result["forecast_change_percent"],
        "risk_level_shift": risk_shift,
        "recommendation_shift": rec_shift,
    }

    # 5. Formulate dynamic natural language executive summary
    if not changes_list:
        summary = (
            f"No scenario modifications applied. Forecast remains identical to baseline at "
            f"${base_pred_rate:.2f}/t ({baseline_result['forecast_change_percent']:+.2f}%)."
        )
    else:
        shift_str = f"+${diff_usd:.2f}/t (+{diff_pct:.2f}%)" if diff_usd >= 0 else f"-${abs(diff_usd):.2f}/t ({diff_pct:.2f}%)"
        rec_comment = f" Recommendation shifts from {base_rec} to {scen_rec}." if base_rec != scen_rec else f" Recommendation remains {base_rec}."
        summary = (
            f"Under this scenario, the next-month freight forecast shifts by {shift_str} "
            f"from ${base_pred_rate:.2f}/t (baseline) to ${scen_pred_rate:.2f}/t.{rec_comment}"
        )

    # 6. Record scenario telemetry (isolated from baseline production stats)
    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    database.insert_prediction_log(
        model_version=MODEL_PATH.stem,
        origin=str(base_merged.get("origin", "")),
        destination=str(base_merged.get("destination", "")),
        commodity=str(base_merged.get("commodity", "")),
        vessel_type=str(base_merged.get("vessel_type", "")),
        current_freight_usd_per_tonne=float(scenario_result.get("current_freight_usd_per_tonne", 0.0)),
        predicted_next_month_freight_usd_per_tonne=float(scen_pred_rate),
        forecast_change_percent=float(scenario_result.get("forecast_change_percent", 0.0)),
        risk_level=str(scen_risk),
        recommendation=str(scen_rec),
        latency_ms=latency_ms,
        provenance={"type": "scenario_simulation", "diff_usd": diff_usd, "changes_count": len(changes_list)},
    )

    return {
        "summary": summary,
        "baseline": baseline_result,
        "scenario": scenario_result,
        "impact": impact,
        "changes": changes_list,
    }


def init_data_layer() -> None:
    """Make sure the database exists before serving requests."""
    database.init_db()
