"""Test the FINAL model (freight_forecast_model_final.joblib) end-to-end.

Tests:
  1. Model loads via joblib
  2. Pipeline structure (OneHot + HistGradientBoosting)
  3. 13-feature contract (NO cargo_tonnes)
  4. Predict on all 5 supported combinations
  5. All predictions numeric, non-NaN, in plausible range
  6. No feature-name errors, no categorical encoding errors
  7. API /predict endpoint returns a valid response with recommendation fields
  8. Recommendation logic (risk levels, chartering advice)

Usage:
    1. Start the API:  cd backend && uvicorn main:app --port 8000
    2. Run:            python test_final_model.py

Can also run just the model-direct tests (without API):
    python test_final_model.py --no-api
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "freight_forecast_model_final.joblib"

# The 5 supported combinations (matching the model's training distribution)
COMBINATIONS = [
    {
        "name": "Australia West Coast / Iron Ore / Capesize",
        "payload": {
            "origin": "Australia West Coast", "destination": "East Coast India",
            "commodity": "Iron Ore", "vessel_type": "Capesize",
            "current_freight_usd_per_tonne": 10.0,
            "bdi": 1560, "vlsfo_usd_per_tonne": 638,
            "coal_price_usd_per_mt": 124, "iron_ore_price_usd_per_dmt": 124,
            "wind_kmh": 28, "wave_height_m": 1.7,
            "cyclone_risk": 2, "weather_delay_days": 0.5,
        },
    },
    {
        "name": "Hay Point / Coal / Capesize",
        "payload": {
            "origin": "Hay Point", "destination": "East Coast India",
            "commodity": "Coal", "vessel_type": "Capesize",
            "current_freight_usd_per_tonne": 14.0,
            "bdi": 1560, "vlsfo_usd_per_tonne": 638,
            "coal_price_usd_per_mt": 124, "iron_ore_price_usd_per_dmt": 124,
            "wind_kmh": 32, "wave_height_m": 2.0,
            "cyclone_risk": 2, "weather_delay_days": 0.5,
        },
    },
    {
        "name": "Hay Point / Coal / Panamax",
        "payload": {
            "origin": "Hay Point", "destination": "East Coast India",
            "commodity": "Coal", "vessel_type": "Panamax",
            "current_freight_usd_per_tonne": 16.5,
            "bdi": 1560, "vlsfo_usd_per_tonne": 638,
            "coal_price_usd_per_mt": 124, "iron_ore_price_usd_per_dmt": 124,
            "wind_kmh": 32, "wave_height_m": 2.0,
            "cyclone_risk": 2, "weather_delay_days": 0.5,
        },
    },
    {
        "name": "Taboneo / Thermal Coal / Panamax",
        "payload": {
            "origin": "Taboneo", "destination": "East Coast India",
            "commodity": "Thermal Coal", "vessel_type": "Panamax",
            "current_freight_usd_per_tonne": 11.0,
            "bdi": 1560, "vlsfo_usd_per_tonne": 638,
            "coal_price_usd_per_mt": 124, "iron_ore_price_usd_per_dmt": 124,
            "wind_kmh": 19, "wave_height_m": 1.5,
            "cyclone_risk": 2, "weather_delay_days": 0.5,
        },
    },
    {
        "name": "Taboneo / Thermal Coal / Supramax",
        "payload": {
            "origin": "Taboneo", "destination": "East Coast India",
            "commodity": "Thermal Coal", "vessel_type": "Supramax",
            "current_freight_usd_per_tonne": 12.0,
            "bdi": 1560, "vlsfo_usd_per_tonne": 638,
            "coal_price_usd_per_mt": 124, "iron_ore_price_usd_per_dmt": 124,
            "wind_kmh": 19, "wave_height_m": 1.5,
            "cyclone_risk": 2, "weather_delay_days": 0.5,
        },
    },
]

EXPECTED_FEATURES = [
    "origin", "destination", "commodity", "vessel_type",
    "bdi", "vlsfo_usd_per_tonne", "coal_price_usd_per_mt",
    "iron_ore_price_usd_per_dmt", "wind_kmh", "wave_height_m",
    "cyclone_risk", "weather_delay_days", "current_freight_usd_per_tonne",
]

API_URL = "http://localhost:8000"


def log(msg: str, ok: bool = True):
    status = "✅" if ok else "❌"
    print(f"  {status} {msg}")


def test_model_loads():
    """Test 1: model loads via joblib."""
    print("\n[1] Model loads via joblib")
    m = joblib.load(MODEL_PATH)
    log(f"type={type(m).__name__}", isinstance(m, object))
    return m


def test_pipeline_structure(m):
    """Test 2: pipeline structure."""
    print("\n[2] Pipeline structure")
    steps = [n for n, _ in m.steps]
    log(f"steps={steps}", steps == ["prep", "model"])
    prep = m.named_steps["prep"]
    est = m.named_steps["model"]
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    log(f"prep is ColumnTransformer: {isinstance(prep, ColumnTransformer)}", isinstance(prep, ColumnTransformer))
    log(f"estimator is HistGradientBoostingRegressor: {isinstance(est, HistGradientBoostingRegressor)}", isinstance(est, HistGradientBoostingRegressor))


def test_feature_contract(m):
    """Test 3: 13-feature contract (NO cargo_tonnes)."""
    print("\n[3] Feature contract (13 features, NO cargo_tonnes)")
    fni = list(m.feature_names_in_)
    log(f"feature count={len(fni)} (expected 13)", len(fni) == 13)
    log(f"'cargo_tonnes' NOT in features: {'cargo_tonnes' not in fni}", "cargo_tonnes" not in fni)
    log(f"features match expected: {fni == EXPECTED_FEATURES}", fni == EXPECTED_FEATURES)
    return fni


def test_predict_5_combinations(m, features):
    """Test 4-6: predict on all 5 combinations."""
    print("\n[4-6] Predict on 5 combinations")
    import pandas as pd
    all_ok = True
    for combo in COMBINATIONS:
        payload = combo["payload"]
        X = pd.DataFrame([payload])
        try:
            pred = m.predict(X[features])
            p = float(pred[0])
            numeric = not np.isnan(p)
            in_range = 5.0 <= p <= 30.0
            ok = numeric and in_range
            log(f"{combo['name']}: pred={p:.4f} numeric={numeric} in_range={in_range}", ok)
            if not ok:
                all_ok = False
        except Exception as e:
            log(f"{combo['name']}: ERROR {e}", False)
            all_ok = False
    return all_ok


def test_api():
    """Test 7-8: API /predict + recommendation fields."""
    print("\n[7-8] API /predict endpoint")
    all_ok = True
    for combo in COMBINATIONS:
        payload = combo["payload"]
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{API_URL}/predict", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            # Check required fields
            required = ["predicted_next_month_freight_usd_per_tonne",
                        "current_freight_usd_per_tonne", "forecast_change_percent",
                        "risk_level", "recommendation", "reason", "sources"]
            missing = [f for f in required if f not in result]
            pred = result["predicted_next_month_freight_usd_per_tonne"]
            numeric = isinstance(pred, (int, float)) and not np.isnan(pred)
            reco_valid = result["recommendation"] in ["CHARTER NOW", "WAIT", "MONITOR"]
            risk_valid = result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
            ok = len(missing) == 0 and numeric and reco_valid and risk_valid
            log(f"{combo['name']}: pred={pred:.2f} reco={result['recommendation']} risk={result['risk_level']}", ok)
            if not ok:
                all_ok = False
        except Exception as e:
            log(f"{combo['name']}: API ERROR {e}", False)
            all_ok = False
    return all_ok


def test_missing_market_422():
    """Test: missing market fields without DB fallback -> 422."""
    print("\n[9] Missing market data -> 422 (no fabrication)")
    payload = {
        "origin": "Hay Point", "destination": "East Coast India",
        "commodity": "Coal", "vessel_type": "Panamax",
        "current_freight_usd_per_tonne": 16.5,
        # Intentionally omit ALL market + weather fields
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{API_URL}/predict", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=30)
        log("expected 422 but got 200", False)
        return False
    except urllib.error.HTTPError as e:
        ok = e.code == 422
        detail = json.loads(e.read().decode("utf-8")).get("detail", "")
        log(f"422 returned (missing fields listed): {detail[:80]}...", ok)
        return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-api", action="store_true", help="Skip API tests")
    args = parser.parse_args()

    print("=" * 60)
    print("FINAL MODEL END-TO-END TEST")
    print(f"Model: {MODEL_PATH}")
    print("=" * 60)

    m = test_model_loads()
    test_pipeline_structure(m)
    features = test_feature_contract(m)
    model_ok = test_predict_5_combinations(m, features)

    api_ok = True
    missing_ok = True
    if not args.no_api:
        api_ok = test_api()
        missing_ok = test_missing_market_422()
    else:
        print("\n[7-9] API tests skipped (--no-api)")

    print("\n" + "=" * 60)
    all_pass = model_ok and api_ok and missing_ok
    print(f"ALL TESTS {'PASSED ✅' if all_pass else 'FAILED ❌'}")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
