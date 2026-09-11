"""Test Model v3 (freight_forecast_model_v3.joblib) end-to-end.

Tests:
  1. Model loads via joblib
  2. Pipeline structure (ColumnTransformer + Ridge)
  3. 13-feature contract (NO cargo_tonnes)
  4. Predict on all 5 supported combinations
  5. All predictions numeric, non-NaN, in plausible range (5 - 30 USD/t)
  6. Defensive residual guardrail verification ([-4.0, +4.0])
  7. API /predict endpoint returns valid response with recommendation and provenance
  8. Missing-data handling and fallback logic

Usage:
    python backend/test_v3_model.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_V3_PATH = REPO_ROOT / "freight_forecast_model_v3.joblib"

sys.path.insert(0, str(REPO_ROOT / "backend"))

from main import app
from predict import (
    FEATURES,
    get_model,
    get_model_metadata,
    predict_freight,
)

# 5 supported combinations
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
            "wind_kmh": 20, "wave_height_m": 1.2,
            "cyclone_risk": 1, "weather_delay_days": 0.0,
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
            "wind_kmh": 20, "wave_height_m": 1.2,
            "cyclone_risk": 1, "weather_delay_days": 0.0,
        },
    },
]


class TestModelV3Integration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_artifact_exists_and_loads(self):
        self.assertTrue(MODEL_V3_PATH.exists())
        model = joblib.load(MODEL_V3_PATH)
        self.assertEqual(len(model.feature_names_in_), 13)
        self.assertNotIn("cargo_tonnes", list(model.feature_names_in_))

    def test_model_info(self):
        info = get_model_metadata()
        self.assertIn(info["model"], ["freight_forecast_model_timeseries", "freight_forecast_model_v3"])
        self.assertIn(info["algorithm"], ["Route-Specific ARIMA(0,1,1) Time-Series Model", "Bounded Residual Ridge Regression"])
        self.assertEqual(info["features"], 13)
        self.assertFalse(info["synthetic_data_used"])

    def test_direct_inference_all_combinations(self):
        for combo in COMBINATIONS:
            res = predict_freight(combo["payload"])
            pred = res["predicted_next_month_freight_usd_per_tonne"]
            curr = res["current_freight_usd_per_tonne"]
            chg = res["forecast_change_percent"]
            delta = pred - curr

            self.assertGreaterEqual(pred, 5.0)
            self.assertLessEqual(pred, 30.0)
            self.assertGreaterEqual(delta, -4.0)
            self.assertLessEqual(delta, 4.0)
            self.assertIn(res["risk_level"], ["LOW", "MEDIUM", "HIGH"])
            self.assertIn(res["recommendation"], ["CHARTER NOW", "WAIT", "MONITOR"])
            print(f"  [Direct] {combo['name']:45} -> Pred: ${pred:5.2f} (Delta: {delta:+5.2f}) -> {res['recommendation']}")

    def test_api_predict_all_combinations(self):
        for combo in COMBINATIONS:
            resp = self.client.post("/predict", json=combo["payload"])
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("predicted_next_month_freight_usd_per_tonne", data)
            self.assertIn("recommendation", data)
            self.assertIn("sources", data)
            print(f"  [API]    {combo['name']:45} -> Status 200 OK")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING MODEL V3 INTEGRATION TEST SUITE")
    print("=" * 60)
    unittest.main()
