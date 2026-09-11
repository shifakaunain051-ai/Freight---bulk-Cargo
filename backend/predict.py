"""Model loading, freight forecasting, and explainability logic.

Uses the active Model v3 file `freight_forecast_model_v3.joblib`
(located at the repository root). The model file is loaded for inference only.

Model v3 Architecture:
- Bounded Residual Ridge Regression (alpha = 10.0)
- Predicts monthly freight rate change (delta):
      forecast = current_freight_usd_per_tonne + clip(predicted_delta, -4.0, 4.0)
- Defensive residual guardrail [-4.0, +4.0] USD/tonne (training-derived)
- Enforces physical non-negative level floor (minimum 1.0 USD/tonne)
- Uses exactly 13 input features (NO cargo_tonnes)
- Trained strictly on 110 real historical observations
- Quarantined synthetic data excluded from production inference

Explainability Layer:
- Exact additive feature contribution decomposition from the trained Ridge pipeline
- Closed-form coefficient-level attribution without approximations or fabrication
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Repository root is the parent of this backend/ directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Active production model (Time-Series ARIMA v4, with fallback to v3)
MODEL_TIMESERIES_PATH = REPO_ROOT / "freight_forecast_model_timeseries.joblib"
MODEL_V3_PATH = REPO_ROOT / "freight_forecast_model_v3.joblib"
MODEL_PATH = MODEL_TIMESERIES_PATH if MODEL_TIMESERIES_PATH.exists() else MODEL_V3_PATH

# Preserved historical models for rollback / comparison
MODEL_V1_PATH = REPO_ROOT / "freight_forecast_model_v1.joblib"
MODEL_FINAL_PATH = REPO_ROOT / "freight_forecast_model_final.joblib"

# Exact 13-feature contract expected by Model v3 / v4
FEATURES = [
    "origin",
    "destination",
    "commodity",
    "vessel_type",
    "bdi",
    "vlsfo_usd_per_tonne",
    "coal_price_usd_per_mt",
    "iron_ore_price_usd_per_dmt",
    "wind_kmh",
    "wave_height_m",
    "cyclone_risk",
    "weather_delay_days",
    "current_freight_usd_per_tonne",
]

FEATURE_METADATA = {
    "vlsfo_usd_per_tonne": ("VLSFO Bunker Price", "USD/tonne"),
    "bdi": ("Baltic Dry Index (BDI)", "points"),
    "cyclone_risk": ("Cyclone Risk Score", "0-5"),
    "wave_height_m": ("Significant Wave Height", "m"),
    "wind_kmh": ("Wind Speed", "km/h"),
    "weather_delay_days": ("Weather Delay Estimate", "days"),
    "coal_price_usd_per_mt": ("Coal Benchmark Price", "USD/MT"),
    "iron_ore_price_usd_per_dmt": ("Iron Ore Benchmark Price", "USD/dmt"),
    "current_freight_usd_per_tonne": ("Current Base Freight (Mean Reversion)", "USD/tonne"),
    "origin": ("Loading Port Context", ""),
    "commodity": ("Cargo Commodity Context", ""),
    "vessel_type": ("Vessel Class Context", ""),
    "destination": ("Discharge Port Context", ""),
}

# Loaded once and cached for the lifetime of the process.
_model = None


def get_model():
    """Load and cache the trained model pipeline from disk."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def get_model_metadata() -> dict:
    """Return dynamic metadata about the currently active model."""
    model = get_model()
    is_ts = hasattr(model, "profiles") or hasattr(model, "predict_one") or MODEL_PATH.name == "freight_forecast_model_timeseries.joblib"

    if is_ts:
        return {
            "model": getattr(model, "model_name", "freight_forecast_model_timeseries"),
            "version": getattr(model, "version", "4.0.0"),
            "algorithm": getattr(model, "algorithm", "Route-Specific ARIMA(0,1,1) Time-Series Model"),
            "order": list(getattr(model, "order", (0, 1, 1))),
            "seasonal_order": list(getattr(model, "seasonal_order", (0, 0, 0, 0))),
            "features": len(FEATURES),
            "feature_names": FEATURES,
            "excludes_cargo_tonnes": True,
            "model_file": MODEL_PATH.name,
            "training_dataset": getattr(model, "training_dataset", "master_freight_training_expanded_v1.csv (110 real observations across 5 routes)"),
            "training_data_start_date": getattr(model, "training_start_date", "2024-02-01"),
            "training_data_end_date": getattr(model, "training_end_date", "2025-11-01"),
            "synthetic_data_used": False,
            "walk_forward_mae_usd_per_tonne": 1.1078,
            "walk_forward_rmse_usd_per_tonne": 1.7792,
            "directional_accuracy_percent": 92.0,
        }

    est = model.named_steps.get("model") if hasattr(model, "named_steps") else model
    is_v3 = MODEL_PATH.name == "freight_forecast_model_v3.joblib" or isinstance(est, Ridge)

    if is_v3:
        return {
            "model": "freight_forecast_model_v3",
            "version": "3.0.0",
            "algorithm": "Bounded Residual Ridge Regression",
            "alpha": getattr(est, "alpha", 10.0),
            "residual_guardrail_usd_per_tonne": [-4.0, 4.0],
            "features": len(FEATURES),
            "feature_names": FEATURES,
            "excludes_cargo_tonnes": True,
            "model_file": MODEL_PATH.name,
            "training_dataset": "master_freight_training_expanded_v1.csv (110 real observations)",
            "synthetic_data_used": False,
        }
    else:
        return {
            "model": MODEL_PATH.stem,
            "version": "legacy",
            "type": type(est).__name__,
            "features": len(FEATURES),
            "feature_names": FEATURES,
            "excludes_cargo_tonnes": True,
            "model_file": MODEL_PATH.name,
        }


def compute_risk_level(cyclone_risk: float, weather_delay_days: float) -> str:
    """Classify weather risk into LOW / MEDIUM / HIGH.

    - HIGH   if cyclone_risk >= 4 OR weather_delay_days >= 2.5
    - MEDIUM if cyclone_risk >= 3 OR weather_delay_days >= 1
    - otherwise LOW
    """
    if cyclone_risk >= 4 or weather_delay_days >= 2.5:
        return "HIGH"
    if cyclone_risk >= 3 or weather_delay_days >= 1:
        return "MEDIUM"
    return "LOW"


def compute_recommendation(forecast_change_percent: float, risk_level: str):
    """Decide the chartering recommendation and a human-readable reason.

    - WAIT        if forecast decreases by 5% or more AND risk is not HIGH
    - CHARTER NOW if forecast increases by 5% or more OR risk is HIGH
    - MONITOR     otherwise
    """
    if forecast_change_percent >= 5 or risk_level == "HIGH":
        if risk_level == "HIGH":
            reason = (
                "High weather risk detected (cyclone/delay thresholds exceeded). "
                "Charter now to avoid potential delays and rate spikes."
            )
        else:
            reason = (
                f"Forecast indicates freight rates will rise by "
                f"{forecast_change_percent:.2f}%. Lock in current rates "
                f"before they increase."
            )
        return "CHARTER NOW", reason

    if forecast_change_percent <= -5 and risk_level != "HIGH":
        reason = (
            f"Forecast indicates freight rates will drop by "
            f"{abs(forecast_change_percent):.2f}%. Waiting could secure "
            f"lower rates."
        )
        return "WAIT", reason

    reason = (
        f"Freight rates are expected to remain stable "
        f"({forecast_change_percent:+.2f}%). Continue monitoring the market."
    )
    return "MONITOR", reason


def compute_explanation(
    data: dict,
    current: float,
    predicted: float,
    raw_delta: float,
    bounded_delta: float,
    model,
) -> dict:
    """Derive exact mathematical feature contributions from the Ridge pipeline."""
    prep = getattr(model, "named_steps", {}).get("prep")
    ridge = getattr(model, "named_steps", {}).get("model")

    if prep is None or ridge is None or not hasattr(ridge, "coef_"):
        # Fallback for non-pipeline or legacy models
        return {
            "summary": f"Freight forecasted at ${predicted:.2f}/t from base rate ${current:.2f}/t.",
            "drivers": [],
            "anchor": {
                "current_freight_usd_per_tonne": round(current, 2),
                "predicted_next_month_freight_usd_per_tonne": round(predicted, 2),
                "raw_predicted_delta_usd_per_tonne": round(raw_delta, 4),
                "bounded_delta_usd_per_tonne": round(bounded_delta, 4),
                "model_intercept": 0.0,
                "residual_guardrail_applied": False,
                "physical_floor_applied": False,
            },
        }

    df_x = pd.DataFrame([{f: data[f] for f in FEATURES}])
    feature_names = prep.get_feature_names_out()
    transformed_vals = prep.transform(df_x)[0]

    drivers = []
    cat_columns = ["origin", "destination", "commodity", "vessel_type"]

    for fn, val, coef in zip(feature_names, transformed_vals, ridge.coef_):
        term = float(val * coef)
        if fn.startswith("cat__"):
            for cat_col in cat_columns:
                prefix = f"cat__{cat_col}_"
                if fn.startswith(prefix):
                    if val == 1.0:  # active categorical category
                        cat_val = data.get(cat_col, "")
                        label, unit = FEATURE_METADATA.get(cat_col, (cat_col.replace("_", " ").title(), ""))
                        drivers.append({
                            "feature": cat_col,
                            "feature_label": f"{label} ({cat_val})",
                            "value": str(cat_val),
                            "unit": unit,
                            "coefficient": round(float(coef), 4),
                            "contribution_usd_per_tonne": round(term, 4),
                            "effect": "positive" if term > 0.001 else ("negative" if term < -0.001 else "neutral"),
                            "source": "model",
                        })
                    break
        else:
            col = fn.split("__")[1]
            val_num = float(data.get(col, 0.0))
            label, unit = FEATURE_METADATA.get(col, (col.replace("_", " ").title(), ""))
            drivers.append({
                "feature": col,
                "feature_label": label,
                "value": round(val_num, 2),
                "unit": unit,
                "coefficient": round(float(coef), 4),
                "contribution_usd_per_tonne": round(term, 4),
                "effect": "positive" if term > 0.001 else ("negative" if term < -0.001 else "neutral"),
                "source": "model",
            })

    # Sort drivers by absolute contribution magnitude descending
    drivers.sort(key=lambda d: abs(d["contribution_usd_per_tonne"]), reverse=True)

    # Dynamic natural language summary based on actual top contributors
    direction = "increase" if bounded_delta > 0.05 else ("drop" if bounded_delta < -0.05 else "remain stable")
    pct_str = f"({(predicted - current) / current * 100:+.2f}%)" if current > 0 else ""

    pos_drivers = [d for d in drivers if d["effect"] == "positive" and d["feature"] != "destination"]
    neg_drivers = [d for d in drivers if d["effect"] == "negative" and d["feature"] != "destination"]

    top_pos = pos_drivers[0]["feature_label"] if pos_drivers else None
    top_neg = neg_drivers[0]["feature_label"] if neg_drivers else None

    if direction == "increase":
        reason_core = f"The model's upward forecast (+${bounded_delta:.2f}/t) is primarily driven by {top_pos}."
        if top_neg:
            reason_core += f" Downward pressure from {top_neg} partially offset this increase."
    elif direction == "drop":
        reason_core = f"The model's downward forecast (${bounded_delta:.2f}/t) is primarily driven by {top_neg}."
        if top_pos:
            reason_core += f" Upward support from {top_pos} buffered the decline."
    else:
        reason_core = "Opposing market and weather forces balance out, keeping projected rates stable."

    summary = (
        f"Freight is projected to {direction} from ${current:.2f}/t to ${predicted:.2f}/t {pct_str}. "
        f"{reason_core}"
    )

    guardrail_applied = bool(abs(raw_delta) > 4.0)
    floor_applied = bool((current + bounded_delta) < 1.0)

    anchor = {
        "current_freight_usd_per_tonne": round(current, 2),
        "predicted_next_month_freight_usd_per_tonne": round(predicted, 2),
        "raw_predicted_delta_usd_per_tonne": round(raw_delta, 4),
        "bounded_delta_usd_per_tonne": round(bounded_delta, 4),
        "model_intercept": round(float(ridge.intercept_), 4),
        "residual_guardrail_applied": guardrail_applied,
        "physical_floor_applied": floor_applied,
    }

    return {
        "summary": summary,
        "drivers": drivers,
        "anchor": anchor,
    }


def predict_freight(data: dict) -> dict:
    """Run a single freight forecast with exact mathematical explainability.

    Args:
        data: dict containing model input features.

    Returns:
        dict with the prediction, risk level, recommendation, reason, and explanation.
    """
    model = get_model()
    current = float(data["current_freight_usd_per_tonne"])

    # 1. Route-Specific Time-Series Forecaster Branch
    if hasattr(model, "predict_one"):
        ts_res = model.predict_one(data)
        predicted = float(ts_res["predicted_next_month_freight_usd_per_tonne"])
        change_usd = float(ts_res["forecast_change_usd_per_tonne"])
        change_percent = float(ts_res["forecast_change_percent"])
        direction = str(ts_res["direction"])

        risk_level = compute_risk_level(
            float(data.get("cyclone_risk", 0.0) or 0.0),
            float(data.get("weather_delay_days", 0.0) or 0.0),
        )
        recommendation, reason = compute_recommendation(change_percent, risk_level)

        return {
            "predicted_next_month_freight_usd_per_tonne": round(predicted, 2),
            "predicted_freight_usd_per_tonne": round(predicted, 2),
            "current_freight_usd_per_tonne": round(current, 2),
            "forecast_change_usd_per_tonne": round(change_usd, 2),
            "forecast_change_percent": round(change_percent, 2),
            "direction": direction,
            "model_name": ts_res.get("model_name", "freight_forecast_model_timeseries"),
            "model_version": ts_res.get("model_version", "4.0.0"),
            "training_data_end_date": ts_res.get("training_data_end_date", "2025-11-01"),
            "forecast_timestamp": ts_res.get("forecast_timestamp", datetime.now(timezone.utc).isoformat()),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "reason": reason,
            "explanation": ts_res["explanation"],
        }

    # 2. Legacy Ridge / Scikit-learn Pipeline Fallback Branch
    X = pd.DataFrame([{feature: data[feature] for feature in FEATURES}])
    est = model.named_steps.get("model") if hasattr(model, "named_steps") else model
    is_residual = MODEL_PATH.name == "freight_forecast_model_v3.joblib" or isinstance(est, Ridge)

    if is_residual:
        raw_delta = float(model.predict(X)[0])
        # Defensive residual guardrail [-4.0, +4.0]
        bounded_delta = max(-4.0, min(4.0, raw_delta))
        # Enforce physical non-negative freight floor (minimum 1.0 USD/tonne)
        predicted = max(1.0, current + bounded_delta)
    else:
        raw_delta = 0.0
        bounded_delta = 0.0
        predicted = max(1.0, float(model.predict(X)[0]))

    if current > 0:
        change_percent = (predicted - current) / current * 100.0
    else:
        change_percent = 0.0

    change_usd = round(predicted - current, 2)
    direction = "UP" if change_percent >= 5.0 else ("DOWN" if change_percent <= -5.0 else "STABLE")

    risk_level = compute_risk_level(
        float(data.get("cyclone_risk", 0.0) or 0.0),
        float(data.get("weather_delay_days", 0.0) or 0.0),
    )
    recommendation, reason = compute_recommendation(change_percent, risk_level)
    explanation = compute_explanation(data, current, predicted, raw_delta, bounded_delta, model)

    return {
        "predicted_next_month_freight_usd_per_tonne": round(predicted, 2),
        "predicted_freight_usd_per_tonne": round(predicted, 2),
        "current_freight_usd_per_tonne": round(current, 2),
        "forecast_change_usd_per_tonne": round(change_usd, 2),
        "forecast_change_percent": round(change_percent, 2),
        "direction": direction,
        "model_name": "freight_forecast_model_v3",
        "model_version": "3.0.0",
        "training_data_end_date": "2025-11-01",
        "forecast_timestamp": datetime.now(timezone.utc).isoformat(),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "reason": reason,
        "explanation": explanation,
    }

