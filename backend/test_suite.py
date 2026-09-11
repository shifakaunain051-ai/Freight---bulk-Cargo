"""Comprehensive automated test suite for Freight Forecasting backend, Explainability, Scenario Analysis, Market Intelligence, and Dashboard.

Uses Python standard library `unittest` + FastAPI TestClient.
Tests are deterministic and do not make live external network calls.
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add backend directory to sys.path
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import joblib
import numpy as np
from fastapi.testclient import TestClient

from data import database
from data.weather import PORTS
from main import app
from predict import (
    FEATURES,
    MODEL_PATH,
    compute_recommendation,
    compute_risk_level,
    get_model,
    get_model_metadata,
    predict_freight,
)
from schemas import FreightRequest, FreightResponse
from services import analytics_service, dashboard_service
from services.forecast_service import (
    ForecastDataError,
    build_forecast_input,
    forecast,
    run_scenario_forecast,
)


class TestModelContract(unittest.TestCase):
    """Tests active Model v3 loading and feature contract."""

    def test_model_file_exists(self):
        self.assertTrue(MODEL_PATH.exists(), f"Model file missing: {MODEL_PATH}")

    def test_model_loads(self):
        model = get_model()
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))

    def test_13_feature_contract(self):
        model = get_model()
        features = list(model.feature_names_in_)
        self.assertEqual(len(features), 13)
        self.assertNotIn("cargo_tonnes", features)
        self.assertEqual(features, FEATURES)

    def test_model_prediction_output(self):
        sample = {
            "origin": "Hay Point",
            "destination": "East Coast India",
            "commodity": "Coal",
            "vessel_type": "Panamax",
            "current_freight_usd_per_tonne": 16.5,
            "bdi": 1560,
            "vlsfo_usd_per_tonne": 638,
            "coal_price_usd_per_mt": 124,
            "iron_ore_price_usd_per_dmt": 124,
            "wind_kmh": 32,
            "wave_height_m": 2.0,
            "cyclone_risk": 2,
            "weather_delay_days": 0.5,
        }
        res = predict_freight(sample)
        self.assertIn("predicted_next_month_freight_usd_per_tonne", res)
        self.assertIsInstance(res["predicted_next_month_freight_usd_per_tonne"], float)
        self.assertGreater(res["predicted_next_month_freight_usd_per_tonne"], 0)
        self.assertIn(res["risk_level"], ["LOW", "MEDIUM", "HIGH"])
        self.assertIn(res["recommendation"], ["CHARTER NOW", "WAIT", "MONITOR"])
        self.assertIn("explanation", res)


class TestForecastExplainability(unittest.TestCase):
    """Tests feature attribution, explainability summary, and mathematical consistency."""

    def test_explanation_structure_and_mathematical_consistency(self):
        sample = {
            "origin": "Hay Point",
            "destination": "East Coast India",
            "commodity": "Coal",
            "vessel_type": "Panamax",
            "current_freight_usd_per_tonne": 16.5,
            "bdi": 1560.0,
            "vlsfo_usd_per_tonne": 638.0,
            "coal_price_usd_per_mt": 124.0,
            "iron_ore_price_usd_per_dmt": 124.0,
            "wind_kmh": 32.0,
            "wave_height_m": 2.0,
            "cyclone_risk": 2.0,
            "weather_delay_days": 0.5,
        }
        res = predict_freight(sample)
        self.assertIn("explanation", res)
        expl = res["explanation"]

        self.assertIn("summary", expl)
        self.assertIn("drivers", expl)
        self.assertIn("anchor", expl)

        self.assertIn("$16.50", expl["summary"])
        self.assertTrue(
            "$17.47" in expl["summary"]
            or f"${res['predicted_next_month_freight_usd_per_tonne']:.2f}" in expl["summary"]
        )

        drivers = expl["drivers"]
        self.assertGreater(len(drivers), 5)
        for d in drivers:
            self.assertIn("feature", d)
            self.assertIn("feature_label", d)
            self.assertIn("contribution_usd_per_tonne", d)
            self.assertIn("effect", d)
            self.assertIn(d["effect"], ["positive", "negative", "neutral"])
            self.assertEqual(d["source"], "model")

        total_contrib = sum(d["contribution_usd_per_tonne"] for d in drivers)
        intercept = expl["anchor"]["model_intercept"]
        raw_delta = expl["anchor"]["raw_predicted_delta_usd_per_tonne"]
        reconstructed_delta = total_contrib + intercept
        self.assertAlmostEqual(reconstructed_delta, raw_delta, places=2)

    def test_explainability_across_all_5_routes(self):
        routes = [
            ("Australia West Coast", "East Coast India", "Iron Ore", "Capesize", 10.0),
            ("Hay Point", "East Coast India", "Coal", "Capesize", 14.0),
            ("Hay Point", "East Coast India", "Coal", "Panamax", 16.5),
            ("Taboneo", "East Coast India", "Thermal Coal", "Panamax", 11.0),
            ("Taboneo", "East Coast India", "Thermal Coal", "Supramax", 12.0),
        ]
        for origin, dest, comm, vessel, curr in routes:
            sample = {
                "origin": origin,
                "destination": dest,
                "commodity": comm,
                "vessel_type": vessel,
                "current_freight_usd_per_tonne": curr,
                "bdi": 1560.0,
                "vlsfo_usd_per_tonne": 638.0,
                "coal_price_usd_per_mt": 124.0,
                "iron_ore_price_usd_per_dmt": 124.0,
                "wind_kmh": 32.0,
                "wave_height_m": 2.0,
                "cyclone_risk": 2.0,
                "weather_delay_days": 0.5,
            }
            res = predict_freight(sample)
            self.assertIn("explanation", res)
            self.assertTrue(len(res["explanation"]["drivers"]) > 0)
            self.assertTrue(len(res["explanation"]["summary"]) > 20)


class TestScenarioAnalysis(unittest.TestCase):
    """Tests what-if scenario simulations, parameter shocks, and safety bounds."""

    def setUp(self):
        self.client = TestClient(app)
        database.init_db()

    def test_baseline_scenario_equality_when_unchanged(self):
        payload = {
            "origin": "Hay Point",
            "destination": "East Coast India",
            "commodity": "Coal",
            "vessel_type": "Panamax",
            "current_freight_usd_per_tonne": 16.5,
            "bdi": 1560.0,
            "vlsfo_usd_per_tonne": 638.0,
            "coal_price_usd_per_mt": 124.0,
            "iron_ore_price_usd_per_dmt": 124.0,
            "wind_kmh": 32.0,
            "wave_height_m": 2.0,
            "cyclone_risk": 2.0,
            "weather_delay_days": 0.5,
            "scenario_changes": {},
        }
        resp_predict = self.client.post("/predict", json=payload)
        self.assertEqual(resp_predict.status_code, 200)
        norm_res = resp_predict.json()

        resp_scen = self.client.post("/predict/scenario", json=payload)
        self.assertEqual(resp_scen.status_code, 200)
        scen_res = resp_scen.json()

        self.assertEqual(
            scen_res["baseline"]["predicted_next_month_freight_usd_per_tonne"],
            norm_res["predicted_next_month_freight_usd_per_tonne"],
        )
        self.assertEqual(
            scen_res["scenario"]["predicted_next_month_freight_usd_per_tonne"],
            norm_res["predicted_next_month_freight_usd_per_tonne"],
        )
        self.assertEqual(scen_res["impact"]["difference_usd_per_tonne"], 0.0)
        self.assertEqual(len(scen_res["changes"]), 0)

    def test_vlsfo_percentage_shock_scenario(self):
        payload = {
            "origin": "Hay Point",
            "destination": "East Coast India",
            "commodity": "Coal",
            "vessel_type": "Panamax",
            "current_freight_usd_per_tonne": 16.5,
            "bdi": 1560.0,
            "vlsfo_usd_per_tonne": 638.0,
            "coal_price_usd_per_mt": 124.0,
            "iron_ore_price_usd_per_dmt": 124.0,
            "wind_kmh": 32.0,
            "wave_height_m": 2.0,
            "cyclone_risk": 2.0,
            "weather_delay_days": 0.5,
            "scenario_changes": {
                "vlsfo_change_percent": 10.0,
            },
        }
        resp = self.client.post("/predict/scenario", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        base_rate = data["baseline"]["predicted_next_month_freight_usd_per_tonne"]
        scen_rate = data["scenario"]["predicted_next_month_freight_usd_per_tonne"]

        self.assertGreater(scen_rate, base_rate)
        self.assertGreater(data["impact"]["difference_usd_per_tonne"], 0.0)
        self.assertEqual(len(data["changes"]), 1)
        self.assertEqual(data["changes"][0]["feature"], "vlsfo_usd_per_tonne")
        self.assertEqual(data["changes"][0]["baseline"], 638.0)
        self.assertEqual(data["changes"][0]["scenario"], 701.8)

    def test_cyclone_risk_shock_shifts_recommendation(self):
        payload = {
            "origin": "Taboneo",
            "destination": "East Coast India",
            "commodity": "Thermal Coal",
            "vessel_type": "Panamax",
            "current_freight_usd_per_tonne": 11.0,
            "bdi": 1560.0,
            "vlsfo_usd_per_tonne": 638.0,
            "coal_price_usd_per_mt": 124.0,
            "iron_ore_price_usd_per_dmt": 124.0,
            "wind_kmh": 15.0,
            "wave_height_m": 1.0,
            "cyclone_risk": 1.0,
            "weather_delay_days": 0.0,
            "scenario_changes": {
                "cyclone_risk": 5.0,
            },
        }
        resp = self.client.post("/predict/scenario", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["baseline"]["risk_level"], "LOW")
        self.assertEqual(data["scenario"]["risk_level"], "HIGH")
        self.assertEqual(data["scenario"]["recommendation"], "CHARTER NOW")
        self.assertIn("LOW -> HIGH", data["impact"]["risk_level_shift"])

    def test_invalid_scenario_inputs_rejected_with_422(self):
        base = {
            "origin": "Hay Point",
            "destination": "East Coast India",
            "commodity": "Coal",
            "vessel_type": "Panamax",
            "current_freight_usd_per_tonne": 16.5,
            "bdi": 1560.0,
            "vlsfo_usd_per_tonne": 638.0,
            "coal_price_usd_per_mt": 124.0,
            "iron_ore_price_usd_per_dmt": 124.0,
            "wind_kmh": 32.0,
            "wave_height_m": 2.0,
            "cyclone_risk": 2.0,
            "weather_delay_days": 0.5,
        }

        # 1. Negative BDI
        p1 = dict(base, scenario_changes={"bdi": -100})
        self.assertEqual(self.client.post("/predict/scenario", json=p1).status_code, 422)

        # 2. Cyclone risk > 5
        p2 = dict(base, scenario_changes={"cyclone_risk": 7.0})
        self.assertEqual(self.client.post("/predict/scenario", json=p2).status_code, 422)

        # 3. Negative bunker price
        p3 = dict(base, scenario_changes={"vlsfo_usd_per_tonne": -50.0})
        self.assertEqual(self.client.post("/predict/scenario", json=p3).status_code, 422)


class TestMarketIntelligenceAnalytics(unittest.TestCase):
    """Tests Phase 3 Market Intelligence and Historical Analytics endpoints."""

    def setUp(self):
        self.client = TestClient(app)

    def test_freight_trends_unfiltered_and_structure(self):
        resp = self.client.get("/analytics/freight-trends")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("provenance", data)
        self.assertEqual(data["provenance"]["data_source"], "master_freight_training_expanded_v1.csv")
        self.assertEqual(data["provenance"]["data_type"], "historical")
        self.assertEqual(data["provenance"]["total_records"], 110)

        series = data["series"]
        self.assertEqual(len(series), 110)
        first_pt = series[0]
        self.assertIn("date", first_pt)
        self.assertIn("freight_rate_usd_per_tonne", first_pt)
        self.assertIn("rolling_3m_avg", first_pt)
        self.assertGreater(first_pt["freight_rate_usd_per_tonne"], 0)

    def test_freight_trends_filtered_by_origin_and_commodity(self):
        resp = self.client.get("/analytics/freight-trends?origin=Hay%20Point&commodity=Coal")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["provenance"]["total_records"], 44)
        for pt in data["series"]:
            self.assertEqual(pt["origin"], "Hay Point")
            self.assertEqual(pt["commodity"], "Coal")

    def test_freight_trends_invalid_filter_422(self):
        resp = self.client.get("/analytics/freight-trends?origin=LondonPort")
        self.assertEqual(resp.status_code, 422)
        err = resp.json()
        self.assertEqual(err.get("error_code"), "INVALID_FILTER")

    def test_market_trends_macro_series(self):
        resp = self.client.get("/analytics/market-trends")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["provenance"]["total_records"], 22)
        series = data["series"]
        self.assertEqual(len(series), 22)

        dates = [s["date"] for s in series]
        self.assertEqual(dates, sorted(dates))
        self.assertEqual(dates[0], "2024-02-01")
        self.assertEqual(dates[-1], "2025-11-01")

        for s in series:
            self.assertGreater(s["bdi"], 0)
            self.assertGreater(s["vlsfo_usd_per_tonne"], 0)
            self.assertGreater(s["average_freight_usd_per_tonne"], 0)

    def test_weather_trends_by_origin(self):
        resp = self.client.get("/analytics/weather-trends?origin=Taboneo")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["provenance"]["total_records"], 22)
        for s in data["series"]:
            self.assertEqual(s["origin"], "Taboneo")
            self.assertGreaterEqual(s["wind_kmh"], 0)
            self.assertGreaterEqual(s["wave_height_m"], 0)
            self.assertGreaterEqual(s["cyclone_risk"], 0)

    def test_weather_trends_invalid_origin_422(self):
        resp = self.client.get("/analytics/weather-trends?origin=UnknownOrigin")
        self.assertEqual(resp.status_code, 422)

    def test_routes_analytics_5_canonical_lanes(self):
        resp = self.client.get("/analytics/routes")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        routes = data["routes"]
        self.assertEqual(len(routes), 5)

        for r in routes:
            self.assertEqual(r["observation_count"], 22)
            self.assertEqual(r["first_date"], "2024-02-01")
            self.assertEqual(r["last_date"], "2025-11-01")
            self.assertGreater(r["average_freight"], 5.0)
            self.assertLess(r["average_freight"], 30.0)
            self.assertIn(r["trend"], ["RISING", "FALLING", "STABLE"])

    def test_executive_summary_structure_and_values(self):
        resp = self.client.get("/analytics/summary")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["latest_date"], "2025-11-01")
        self.assertIn("market_state", data)
        self.assertIn("recent_trends", data)
        self.assertEqual(data["tracked_routes_count"], 5)
        self.assertIn(data["recent_trends"]["freight_trend_classification"], ["RISING", "FALLING", "STABLE"])
        self.assertIsNotNone(data["strongest_positive_mover"])

    def test_correlations_against_freight(self):
        resp = self.client.get("/analytics/correlations")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["target_variable"], "current_freight_usd_per_tonne")
        self.assertIn("disclaimer", data)
        self.assertIn("HISTORICAL CORRELATION", data["disclaimer"])

        corrs = {c["feature"]: c["correlation"] for c in data["correlations"]}
        self.assertIn("bdi", corrs)
        self.assertIn("vlsfo_usd_per_tonne", corrs)
        self.assertIn("cyclone_risk", corrs)

        self.assertGreater(corrs["bdi"], 0.4)
        self.assertGreater(corrs["vlsfo_usd_per_tonne"], 0.4)

    def test_latest_snapshot_combined_view(self):
        resp = self.client.get("/analytics/latest")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["latest_historical_date"], "2025-11-01")
        self.assertIn("market_macro", data)
        self.assertIn("weather_by_origin", data)
        self.assertIn("freight_by_route", data)
        self.assertIn("live_database_cache_status", data)

    def test_quarantine_guard_synthetic_data_not_used(self):
        self.assertNotIn("synthetic", str(analytics_service.HISTORICAL_CSV_PATH).lower())
        self.assertEqual(analytics_service.HISTORICAL_CSV_PATH.name, "master_freight_training_expanded_v1.csv")


class TestDashboardOverview(unittest.TestCase):
    """Tests Phase 4 Dashboard Intelligence aggregation endpoint GET /dashboard/overview."""

    def setUp(self):
        self.client = TestClient(app)
        database.init_db()

    def test_dashboard_overview_status_and_top_keys(self):
        resp = self.client.get("/dashboard/overview")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        expected_keys = [
            "market", "routes", "weather", "forecast",
            "signals", "data_quality", "model", "provenance"
        ]
        for k in expected_keys:
            self.assertIn(k, data, f"Missing key '{k}' in /dashboard/overview response")

    def test_dashboard_market_section(self):
        resp = self.client.get("/dashboard/overview")
        data = resp.json()["market"]

        self.assertEqual(data["latest_date"], "2025-11-01")
        self.assertGreater(data["bdi"], 0)
        self.assertGreater(data["vlsfo_usd_per_tonne"], 0)
        self.assertGreater(data["coal_price_usd_per_mt"], 0)
        self.assertGreater(data["iron_ore_price_usd_per_dmt"], 0)
        self.assertGreater(data["average_freight_usd_per_tonne"], 0)
        self.assertIn(data["freight_trend_classification"], ["RISING", "FALLING", "STABLE"])

    def test_dashboard_routes_and_rankings(self):
        resp = self.client.get("/dashboard/overview")
        routes_sec = resp.json()["routes"]

        canonical = routes_sec["canonical_lanes"]
        self.assertEqual(len(canonical), 5)

        rankings = routes_sec["rankings"]
        self.assertIn("highest_freight", rankings)
        self.assertIn("lowest_freight", rankings)
        self.assertIn("strongest_momentum", rankings)
        self.assertIn("weakest_momentum", rankings)

        # Mathematical consistency
        all_latest = [r["latest_freight"] for r in canonical]
        self.assertEqual(rankings["highest_freight"]["latest_freight"], max(all_latest))
        self.assertEqual(rankings["lowest_freight"]["latest_freight"], min(all_latest))

        all_mom = [r["latest_monthly_change_percent"] for r in canonical]
        self.assertEqual(rankings["strongest_momentum"]["latest_monthly_change_percent"], max(all_mom))
        self.assertEqual(rankings["weakest_momentum"]["latest_monthly_change_percent"], min(all_mom))

    def test_dashboard_weather_section(self):
        resp = self.client.get("/dashboard/overview")
        w_sec = resp.json()["weather"]

        self.assertEqual(w_sec["status"], "available")
        self.assertEqual(w_sec["observation_date"], "2025-11-01")
        self.assertGreaterEqual(len(w_sec["ports"]), 3)

        ports = {p["port"] for p in w_sec["ports"]}
        self.assertTrue({"Australia West Coast", "Hay Point", "Taboneo"}.issubset(ports))

        for p in w_sec["ports"]:
            self.assertIn(p["risk_level"], ["LOW", "MEDIUM", "HIGH"])
            self.assertIn(p["status"], ["available", "stale", "unavailable"])

    def test_dashboard_forecast_intelligence(self):
        resp = self.client.get("/dashboard/overview")
        fc_sec = resp.json()["forecast"]

        self.assertIn("reference_summary", fc_sec)
        route_fcs = fc_sec["route_forecasts"]
        self.assertEqual(len(route_fcs), 5)

        for fc in route_fcs:
            self.assertGreaterEqual(fc["current_freight_usd_per_tonne"], 1.0)
            self.assertGreaterEqual(fc["predicted_next_month_freight_usd_per_tonne"], 1.0)
            self.assertIn(fc["risk_level"], ["LOW", "MEDIUM", "HIGH"])
            self.assertIn(fc["recommendation"], ["CHARTER NOW", "WAIT", "MONITOR"])
            self.assertIn("USD/t", fc["top_driver"])

    def test_dashboard_signals_determinism(self):
        resp = self.client.get("/dashboard/overview")
        signals = resp.json()["signals"]

        self.assertGreaterEqual(len(signals), 4)
        for sig in signals:
            self.assertIn("type", sig)
            self.assertIn(sig["severity"], ["LOW", "MEDIUM", "HIGH"])
            self.assertIn("title", sig)
            self.assertIn("description", sig)
            self.assertIn("evidence", sig)

    def test_dashboard_data_quality_and_provenance(self):
        resp = self.client.get("/dashboard/overview")
        data = resp.json()

        dq = data["data_quality"]
        self.assertTrue(dq["historical_dataset_verified"])
        self.assertEqual(dq["historical_records_count"], 110)
        self.assertFalse(dq["synthetic_data_used"])
        self.assertTrue(dq["synthetic_dataset_quarantined"])
        self.assertEqual(dq["overall_health_status"], "HEALTHY")

        prov = data["provenance"]
        self.assertEqual(prov["model_sha256"], dashboard_service.MODEL_SHA256)
        self.assertFalse(prov["synthetic_data_used"])
        self.assertEqual(prov["historical_dataset"]["records"], 110)

    def test_dashboard_model_metadata(self):
        resp = self.client.get("/dashboard/overview")
        model = resp.json()["model"]

        self.assertIn(model["model_name"], ["freight_forecast_model_timeseries", "freight_forecast_model_v3"])
        self.assertIn(model["algorithm"], ["Route-Specific ARIMA(0,1,1) Time-Series Model", "Bounded Residual Ridge Regression"])
        self.assertEqual(model["features_count"], 13)
        self.assertTrue(model["excludes_cargo_tonnes"])
        self.assertFalse(model["synthetic_data_used"])
        self.assertIn(model["validation_evidence"]["holdout_mae_usd_per_tonne"], [1.1078, 0.4730])
        self.assertIn(model["validation_evidence"]["directional_accuracy_percent"], [92.0, 60.0])


class TestPredictionSafetyAndBounds(unittest.TestCase):
    """Tests physical minimum floor and safety limits."""

    def test_physical_forecast_floor_prevents_negative(self):
        sample = {
            "origin": "Australia West Coast",
            "destination": "East Coast India",
            "commodity": "Iron Ore",
            "vessel_type": "Capesize",
            "current_freight_usd_per_tonne": 2.0,
            "bdi": 500,
            "vlsfo_usd_per_tonne": 200,
            "coal_price_usd_per_mt": 50,
            "iron_ore_price_usd_per_dmt": 50,
            "wind_kmh": 5,
            "wave_height_m": 0.5,
            "cyclone_risk": 0,
            "weather_delay_days": 0.0,
        }
        res = predict_freight(sample)
        pred = res["predicted_next_month_freight_usd_per_tonne"]
        self.assertGreaterEqual(pred, 1.0)
        self.assertTrue(res["explanation"]["anchor"]["physical_floor_applied"])

    def test_normal_forecast_not_altered_by_floor(self):
        sample = {
            "origin": "Hay Point",
            "destination": "East Coast India",
            "commodity": "Coal",
            "vessel_type": "Capesize",
            "current_freight_usd_per_tonne": 14.0,
            "bdi": 1560,
            "vlsfo_usd_per_tonne": 638,
            "coal_price_usd_per_mt": 124,
            "iron_ore_price_usd_per_dmt": 124,
            "wind_kmh": 32,
            "wave_height_m": 2.0,
            "cyclone_risk": 2,
            "weather_delay_days": 0.5,
        }
        res = predict_freight(sample)
        pred = res["predicted_next_month_freight_usd_per_tonne"]
        self.assertGreater(pred, 10.0)
        self.assertLess(pred, 25.0)


class TestRecommendationBoundaries(unittest.TestCase):
    """Tests exact threshold boundary cases for risk level and recommendations."""

    def test_risk_level_boundaries(self):
        self.assertEqual(compute_risk_level(4.0, 0.0), "HIGH")
        self.assertEqual(compute_risk_level(3.99, 0.0), "MEDIUM")
        self.assertEqual(compute_risk_level(3.0, 0.0), "MEDIUM")
        self.assertEqual(compute_risk_level(2.99, 0.0), "LOW")
        self.assertEqual(compute_risk_level(0.0, 2.5), "HIGH")
        self.assertEqual(compute_risk_level(0.0, 2.49), "MEDIUM")
        self.assertEqual(compute_risk_level(0.0, 1.0), "MEDIUM")
        self.assertEqual(compute_risk_level(0.0, 0.99), "LOW")

    def test_recommendation_boundaries(self):
        rec, _ = compute_recommendation(5.0, "LOW")
        self.assertEqual(rec, "CHARTER NOW")
        rec, _ = compute_recommendation(4.99, "LOW")
        self.assertEqual(rec, "MONITOR")
        rec, _ = compute_recommendation(-5.0, "LOW")
        self.assertEqual(rec, "WAIT")
        rec, _ = compute_recommendation(-4.99, "LOW")
        self.assertEqual(rec, "MONITOR")
        rec, _ = compute_recommendation(-10.0, "HIGH")
        self.assertEqual(rec, "CHARTER NOW")


class TestAPIEndpointsAndValidation(unittest.TestCase):
    """Tests FastAPI HTTP endpoints, schema validation, and error responses."""

    def setUp(self):
        self.client = TestClient(app)
        database.init_db()

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_model_info_endpoint(self):
        resp = self.client.get("/model/info")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn(data["model"], ["freight_forecast_model_timeseries", "freight_forecast_model_v3"])
        self.assertIn(data["algorithm"], ["Route-Specific ARIMA(0,1,1) Time-Series Model", "Bounded Residual Ridge Regression"])
        self.assertEqual(data["features"], 13)
        self.assertTrue(data["excludes_cargo_tonnes"])
        self.assertFalse(data["synthetic_data_used"])
        self.assertEqual(data["feature_names"], FEATURES)

    def test_bdi_validation_rejects_zero_and_negative(self):
        payload_zero = {
            "origin": "Hay Point", "destination": "East Coast India", "commodity": "Coal",
            "vessel_type": "Panamax", "current_freight_usd_per_tonne": 16.5,
            "bdi": 0,
        }
        resp = self.client.post("/predict", json=payload_zero)
        self.assertEqual(resp.status_code, 422)

        payload_neg = {
            "origin": "Hay Point", "destination": "East Coast India", "commodity": "Coal",
            "vessel_type": "Panamax", "current_freight_usd_per_tonne": 16.5,
            "bdi": -100,
        }
        resp = self.client.post("/predict", json=payload_neg)
        self.assertEqual(resp.status_code, 422)

    def test_predict_missing_required_current_freight_422(self):
        payload = {
            "origin": "Hay Point", "destination": "East Coast India",
            "commodity": "Coal", "vessel_type": "Panamax",
        }
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 422)

    def test_missing_market_data_structured_error(self):
        payload = {
            "origin": "Hay Point", "destination": "East Coast India", "commodity": "Coal",
            "vessel_type": "Panamax", "current_freight_usd_per_tonne": 16.5,
            "wind_kmh": 25, "wave_height_m": 1.5, "cyclone_risk": 1, "weather_delay_days": 0.0,
        }
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 422)
        err = resp.json()
        self.assertEqual(err.get("error_code"), "MARKET_DATA_MISSING")
        self.assertIn("bdi", err.get("missing_fields", []))

    def test_predict_all_5_production_combinations_with_explanation(self):
        combos = [
            ("Australia West Coast", "East Coast India", "Iron Ore", "Capesize", 10.0),
            ("Hay Point", "East Coast India", "Coal", "Capesize", 14.0),
            ("Hay Point", "East Coast India", "Coal", "Panamax", 16.5),
            ("Taboneo", "East Coast India", "Thermal Coal", "Panamax", 11.0),
            ("Taboneo", "East Coast India", "Thermal Coal", "Supramax", 12.0),
        ]
        for origin, dest, comm, vessel, curr in combos:
            payload = {
                "origin": origin, "destination": dest, "commodity": comm, "vessel_type": vessel,
                "current_freight_usd_per_tonne": curr,
                "bdi": 1560, "vlsfo_usd_per_tonne": 638, "coal_price_usd_per_mt": 124, "iron_ore_price_usd_per_dmt": 124,
                "wind_kmh": 32, "wave_height_m": 2.0, "cyclone_risk": 2, "weather_delay_days": 0.5,
            }
            resp = self.client.post("/predict", json=payload)
            self.assertEqual(resp.status_code, 200)
            res = resp.json()
            pred = res["predicted_next_month_freight_usd_per_tonne"]
            self.assertGreaterEqual(pred, 1.0)
            self.assertLessEqual(pred, 30.0)
            self.assertIn(res["recommendation"], ["CHARTER NOW", "WAIT", "MONITOR"])
            self.assertIn("explanation", res)
            self.assertIn("drivers", res["explanation"])
            self.assertGreater(len(res["explanation"]["drivers"]), 5)


class TestDataFreshnessAndWeatherPorts(unittest.TestCase):
    """Tests Australia West Coast coordinates, DB auto-fill, staleness tags, and telemetry logging."""

    def setUp(self):
        self.client = TestClient(app)
        database.init_db()

    def test_australia_west_coast_in_ports(self):
        self.assertIn("Australia West Coast", PORTS)
        lat, lon = PORTS["Australia West Coast"]
        self.assertAlmostEqual(lat, -20.32, places=2)
        self.assertAlmostEqual(lon, 118.57, places=2)

    def test_australia_west_coast_weather_autofill_and_staleness(self):
        with database.get_connection() as conn:
            conn.execute("DELETE FROM weather_data")
            conn.commit()

        fresh_ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        with database.get_connection() as conn:
            conn.execute(
                """INSERT INTO weather_data
                   (port, latitude, longitude, wind_kmh, wave_height_m, cyclone_risk,
                    weather_delay_days, temperature_c, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("Australia West Coast", -20.32, 118.57, 28.0, 1.8, 1.0, 0.15, 29.0, fresh_ts),
            )
            conn.commit()

        payload = {
            "origin": "Australia West Coast",
            "destination": "East Coast India",
            "commodity": "Iron Ore",
            "vessel_type": "Capesize",
            "current_freight_usd_per_tonne": 10.5,
            "bdi": 1560,
            "vlsfo_usd_per_tonne": 638,
            "coal_price_usd_per_mt": 124,
            "iron_ore_price_usd_per_dmt": 124,
        }
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("weather_db[Australia West Coast@", data["sources"]["wind_kmh"])
        self.assertNotIn("STALE", data["sources"]["wind_kmh"])

        stale_dt = datetime.now(timezone.utc) - timedelta(days=3)
        stale_ts = stale_dt.isoformat(timespec="seconds").replace("+00:00", "Z")
        with database.get_connection() as conn:
            conn.execute(
                """INSERT INTO weather_data
                   (port, latitude, longitude, wind_kmh, wave_height_m, cyclone_risk,
                    weather_delay_days, temperature_c, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("Taboneo", -3.65, 114.85, 15.0, 1.0, 0.5, 0.0, 27.0, stale_ts),
            )
            conn.commit()

        payload_taboneo = {
            "origin": "Taboneo",
            "destination": "East Coast India",
            "commodity": "Thermal Coal",
            "vessel_type": "Panamax",
            "current_freight_usd_per_tonne": 11.0,
            "bdi": 1560,
            "vlsfo_usd_per_tonne": 638,
            "coal_price_usd_per_mt": 124,
            "iron_ore_price_usd_per_dmt": 124,
        }
        resp_tab = self.client.post("/predict", json=payload_taboneo)
        self.assertEqual(resp_tab.status_code, 200)
        data_tab = resp_tab.json()
        self.assertIn("STALE", data_tab["sources"]["wind_kmh"])

    def test_prediction_telemetry_logged_in_database(self):
        payload = {
            "origin": "Hay Point",
            "destination": "East Coast India",
            "commodity": "Coal",
            "vessel_type": "Panamax",
            "current_freight_usd_per_tonne": 16.5,
            "bdi": 1560,
            "vlsfo_usd_per_tonne": 638,
            "coal_price_usd_per_mt": 124,
            "iron_ore_price_usd_per_dmt": 124,
            "wind_kmh": 32,
            "wave_height_m": 2.0,
            "cyclone_risk": 2,
            "weather_delay_days": 0.5,
        }
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 200)

        logs = database.get_recent_prediction_logs(limit=5)
        self.assertGreater(len(logs), 0)
        latest = logs[0]
        self.assertEqual(latest["origin"], "Hay Point")
        self.assertEqual(latest["commodity"], "Coal")
        self.assertGreater(latest["predicted_next_month_freight_usd_per_tonne"], 0)
        self.assertIn(latest["risk_level"], ["LOW", "MEDIUM", "HIGH"])


if __name__ == "__main__":
    unittest.main()
