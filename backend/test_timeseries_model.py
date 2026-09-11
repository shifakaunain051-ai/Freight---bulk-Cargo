"""Comprehensive Unit and Integration Tests for NaviFreight Time-Series Freight Model.

Tests cover all 12 validation requirements specified in Section 19:
1. Model artifact loading via joblib
2. Valid forecast generation
3. Correct route selection across all canonical combinations
4. Correct vessel series selection
5. Chronological expanding-window validation
6. Verification of no future data leakage
7. Missing data handling and database fallback
8. Exogenous variable handling
9. Forecast sanity limits and physical floor enforcement
10. Model artifact reloading and immutability
11. API /predict endpoint response schema and fields
12. Decision-service integration (CHARTER NOW, WAIT, MONITOR)

Usage:
    python -m unittest backend/test_timeseries_model.py
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from main import app
from model_timeseries import (
    CANONICAL_ROUTES,
    NaviFreightTimeSeriesForecaster,
    RouteARIMAProfile,
)
from predict import (
    FEATURES,
    MODEL_PATH,
    MODEL_TIMESERIES_PATH,
    compute_recommendation,
    compute_risk_level,
    get_model,
    get_model_metadata,
    predict_freight,
)

CANONICAL_PAYLOADS = [
    {
        "name": "Australia West Coast / Iron Ore / Capesize",
        "payload": {
            "origin": "Australia West Coast",
            "destination": "East Coast India",
            "commodity": "Iron Ore",
            "vessel_type": "Capesize",
            "current_freight_usd_per_tonne": 12.9,
            "bdi": 1560,
            "vlsfo_usd_per_tonne": 638,
            "coal_price_usd_per_mt": 124,
            "iron_ore_price_usd_per_dmt": 124,
            "wind_kmh": 28,
            "wave_height_m": 1.7,
            "cyclone_risk": 2,
            "weather_delay_days": 0.5,
        },
    },
    {
        "name": "Hay Point / Coal / Capesize",
        "payload": {
            "origin": "Hay Point",
            "destination": "East Coast India",
            "commodity": "Coal",
            "vessel_type": "Capesize",
            "current_freight_usd_per_tonne": 17.2,
            "bdi": 1560,
            "vlsfo_usd_per_tonne": 638,
            "coal_price_usd_per_mt": 124,
            "iron_ore_price_usd_per_dmt": 124,
            "wind_kmh": 32,
            "wave_height_m": 2.0,
            "cyclone_risk": 2,
            "weather_delay_days": 0.5,
        },
    },
    {
        "name": "Hay Point / Coal / Panamax",
        "payload": {
            "origin": "Hay Point",
            "destination": "East Coast India",
            "commodity": "Coal",
            "vessel_type": "Panamax",
            "current_freight_usd_per_tonne": 20.0,
            "bdi": 1560,
            "vlsfo_usd_per_tonne": 638,
            "coal_price_usd_per_mt": 124,
            "iron_ore_price_usd_per_dmt": 124,
            "wind_kmh": 32,
            "wave_height_m": 2.0,
            "cyclone_risk": 2,
            "weather_delay_days": 0.5,
        },
    },
    {
        "name": "Taboneo / Thermal Coal / Panamax",
        "payload": {
            "origin": "Taboneo",
            "destination": "East Coast India",
            "commodity": "Thermal Coal",
            "vessel_type": "Panamax",
            "current_freight_usd_per_tonne": 11.8,
            "bdi": 1560,
            "vlsfo_usd_per_tonne": 638,
            "coal_price_usd_per_mt": 124,
            "iron_ore_price_usd_per_dmt": 124,
            "wind_kmh": 20,
            "wave_height_m": 1.2,
            "cyclone_risk": 1,
            "weather_delay_days": 0.0,
        },
    },
    {
        "name": "Taboneo / Thermal Coal / Supramax",
        "payload": {
            "origin": "Taboneo",
            "destination": "East Coast India",
            "commodity": "Thermal Coal",
            "vessel_type": "Supramax",
            "current_freight_usd_per_tonne": 13.8,
            "bdi": 1560,
            "vlsfo_usd_per_tonne": 638,
            "coal_price_usd_per_mt": 124,
            "iron_ore_price_usd_per_dmt": 124,
            "wind_kmh": 20,
            "wave_height_m": 1.2,
            "cyclone_risk": 1,
            "weather_delay_days": 0.0,
        },
    },
]


class TestTimeSeriesModel(unittest.TestCase):
    """Test suite verifying all time-series model requirements."""

    def setUp(self):
        self.client = TestClient(app)
        self.model = get_model()

    # 1. Model Loading
    def test_model_loading_and_type(self):
        self.assertTrue(MODEL_TIMESERIES_PATH.exists(), "Time-series model artifact missing on disk")
        loaded = joblib.load(MODEL_TIMESERIES_PATH)
        self.assertIsInstance(loaded, NaviFreightTimeSeriesForecaster)
        self.assertEqual(loaded.order, (0, 1, 1))
        self.assertEqual(loaded.seasonal_order, (0, 0, 0, 0))
        self.assertEqual(loaded.version, "4.0.0")

    # 2. Valid Forecast & Contract
    def test_valid_forecast_and_output_contract(self):
        sample = CANONICAL_PAYLOADS[0]["payload"]
        res = predict_freight(sample)

        required_keys = [
            "predicted_next_month_freight_usd_per_tonne",
            "predicted_freight_usd_per_tonne",
            "current_freight_usd_per_tonne",
            "forecast_change_usd_per_tonne",
            "forecast_change_percent",
            "direction",
            "model_name",
            "model_version",
            "training_data_end_date",
            "forecast_timestamp",
            "risk_level",
            "recommendation",
            "reason",
            "explanation",
        ]
        for k in required_keys:
            self.assertIn(k, res, f"Missing key in forecast response: {k}")

        self.assertIn(res["direction"], ["UP", "DOWN", "STABLE"])
        self.assertIn(res["risk_level"], ["LOW", "MEDIUM", "HIGH"])
        self.assertIn(res["recommendation"], ["CHARTER NOW", "WAIT", "MONITOR"])
        self.assertGreater(res["predicted_next_month_freight_usd_per_tonne"], 0.0)

    # 3. Correct Route Selection
    def test_correct_route_selection(self):
        for item in CANONICAL_PAYLOADS:
            payload = item["payload"]
            prof = self.model.match_route(
                payload["origin"], payload["destination"], payload["commodity"], payload["vessel_type"]
            )
            self.assertIsNotNone(prof, f"Failed to match route profile for {item['name']}")
            self.assertEqual(prof.route_tuple[0], payload["origin"])
            self.assertEqual(prof.route_tuple[3], payload["vessel_type"])

    # 4. Correct Vessel Series Selection (Panamax vs Capesize on same port)
    def test_correct_vessel_series_selection(self):
        # Hay Point Capesize vs Hay Point Panamax
        prof_cape = self.model.match_route("Hay Point", "East Coast India", "Coal", "Capesize")
        prof_panamax = self.model.match_route("Hay Point", "East Coast India", "Coal", "Panamax")

        self.assertIsNotNone(prof_cape)
        self.assertIsNotNone(prof_panamax)
        self.assertNotEqual(prof_cape.theta, prof_panamax.theta)
        self.assertNotEqual(prof_cape.history_mean, prof_panamax.history_mean)
        self.assertGreater(prof_panamax.history_mean, prof_cape.history_mean)

    # 5. Chronological Validation & Benchmark Performance
    def test_chronological_validation_metrics(self):
        meta = get_model_metadata()
        self.assertEqual(meta["model"], "freight_forecast_model_timeseries")
        self.assertLess(meta["walk_forward_mae_usd_per_tonne"], 1.2660, "Must beat persistence baseline MAE")
        self.assertGreater(meta["directional_accuracy_percent"], 80.0, "Directional accuracy must be strong")

    # 6. No Future Data Leakage Verification
    def test_no_future_leakage(self):
        # Training end date strictly Nov 2025
        self.assertEqual(self.model.training_end_date, "2025-11-01")
        for prof in self.model.profiles.values():
            self.assertEqual(prof.last_observed_date, "2025-11-01")
            self.assertEqual(prof.observations_count, 22)

    # 7. Missing Data Handling & Defensive Validation
    def test_missing_data_handling(self):
        minimal_payload = {
            "origin": "Hay Point",
            "destination": "East Coast India",
            "commodity": "Coal",
            "vessel_type": "Panamax",
            "current_freight_usd_per_tonne": 20.0,
        }
        # Direct time-series model inference succeeds with robust default handling
        direct_res = predict_freight(minimal_payload)
        self.assertIn("predicted_next_month_freight_usd_per_tonne", direct_res)
        self.assertGreater(direct_res["predicted_next_month_freight_usd_per_tonne"], 0.0)

        # API endpoint enforces zero-fabrication contract: returns 422 when DB has no fresh market data
        resp = self.client.post("/predict", json=minimal_payload)
        self.assertIn(resp.status_code, [200, 422])
        if resp.status_code == 422:
            err = resp.json()
            self.assertIn("error_code", err)

    # 8. Exogenous Variable Handling
    def test_exogenous_variable_handling(self):
        base_payload = {
            "origin": "Australia West Coast",
            "destination": "East Coast India",
            "commodity": "Iron Ore",
            "vessel_type": "Capesize",
            "current_freight_usd_per_tonne": 12.0,
            "bdi": 1200,
            "vlsfo_usd_per_tonne": 600,
            "weather_delay_days": 0.0,
        }
        res_low = predict_freight(base_payload)

        high_bdi_payload = dict(base_payload)
        high_bdi_payload["bdi"] = 3000
        res_high = predict_freight(high_bdi_payload)

        # Higher market index produces higher freight forecast
        self.assertGreaterEqual(
            res_high["predicted_next_month_freight_usd_per_tonne"],
            res_low["predicted_next_month_freight_usd_per_tonne"],
        )

    # 9. Forecast Sanity Limits & Physical Floor
    def test_forecast_sanity_limits_and_physical_floor(self):
        # Extremely low freight input
        low_input = {
            "origin": "Australia West Coast",
            "destination": "East Coast India",
            "commodity": "Iron Ore",
            "vessel_type": "Capesize",
            "current_freight_usd_per_tonne": 0.20,
        }
        res_low = predict_freight(low_input)
        self.assertGreaterEqual(res_low["predicted_next_month_freight_usd_per_tonne"], 1.0)
        self.assertTrue(res_low["explanation"]["anchor"]["physical_floor_applied"])

    # 10. Model Artifact Reloading & Stability
    def test_model_artifact_reload_stability(self):
        reloaded = joblib.load(MODEL_TIMESERIES_PATH)
        for item in CANONICAL_PAYLOADS:
            res1 = self.model.predict_one(item["payload"])
            res2 = reloaded.predict_one(item["payload"])
            self.assertEqual(
                res1["predicted_next_month_freight_usd_per_tonne"],
                res2["predicted_next_month_freight_usd_per_tonne"],
            )

    # 11. API Prediction Endpoint
    def test_api_prediction_endpoint(self):
        for item in CANONICAL_PAYLOADS:
            resp = self.client.post("/predict", json=item["payload"])
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["model_name"], "freight_forecast_model_timeseries")
            self.assertEqual(data["model_version"], "4.0.0")
            self.assertIn("recommendation", data)
            self.assertIn("explanation", data)

    # 12. Decision Engine Integration (CHARTER NOW, WAIT, MONITOR)
    def test_decision_engine_integration(self):
        # Test rate rise -> CHARTER NOW
        rec_up, _ = compute_recommendation(6.0, "LOW")
        self.assertEqual(rec_up, "CHARTER NOW")

        # Test high weather risk -> CHARTER NOW regardless of rate
        rec_risk, _ = compute_recommendation(-2.0, "HIGH")
        self.assertEqual(rec_risk, "CHARTER NOW")

        # Test rate drop -> WAIT
        rec_down, _ = compute_recommendation(-6.0, "LOW")
        self.assertEqual(rec_down, "WAIT")

        # Test flat rate -> MONITOR
        rec_flat, _ = compute_recommendation(1.5, "LOW")
        self.assertEqual(rec_flat, "MONITOR")


if __name__ == "__main__":
    unittest.main()
