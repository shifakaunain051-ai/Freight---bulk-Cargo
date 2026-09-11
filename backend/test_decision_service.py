"""Automated Test Suite for NaviFreight Decision Orchestration Service.

Tests:
- BUY + falling freight -> BUY CARGO — WAIT TO CHARTER
- BUY + rising freight -> BUY CARGO — CHARTER NOW
- WAIT + falling freight -> WAIT FOR CARGO — WAIT TO CHARTER
- MONITOR + uncertain freight -> MONITOR CARGO — MONITOR FREIGHT
- No suitable vessel physical/capacity exclusion
- Invalid commodity rejection (422)
- Invalid trade route rejection (422)
- Invalid cargo volume rejection (422)
- End-to-end API orchestration via POST /decision/analyze
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from main import app
from services.decision_service import analyze_decision, STRATEGY_MAP


class TestDecisionOrchestrationService(unittest.TestCase):
    """Test suite covering multi-layer procurement and chartering decision orchestration."""

    def setUp(self):
        self.client = TestClient(app)
        # Mock DataFrame that deterministically triggers a BUY procurement signal (low percentile + falling)
        self.mock_buy_df = pd.DataFrame({
            "date": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"],
            "date_dt": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"]),
            "coal_price_usd_per_mt": [150.0, 140.0, 130.0, 120.0, 80.0],
            "iron_ore_price_usd_per_dmt": [150.0, 140.0, 130.0, 120.0, 80.0],
            "source": ["WB"] * 5,
            "source_date": ["Test"] * 5,
        })

    # -------------------------------------------------------------------------
    # 1. BUY + Rising Freight -> BUY CARGO — CHARTER NOW
    # -------------------------------------------------------------------------
    def test_buy_commodity_rising_freight(self):
        """When commodity signal is BUY and Model v3 projects rising freight, strategy is BUY CARGO — CHARTER NOW."""
        # Market inputs that trigger Model v3 positive residual delta (+8.02% > +5.0%) -> CHARTER NOW
        market_rising = {
            "bdi": 1560.0, "vlsfo_usd_per_tonne": 638.0,
            "coal_price_usd_per_mt": 124.0, "iron_ore_price_usd_per_dmt": 124.0,
            "wind_kmh": 28.0, "wave_height_m": 1.7,
            "cyclone_risk": 1.0, "weather_delay_days": 0.15,
        }
        res = analyze_decision(
            origin="Australia West Coast",
            destination="Dhamra",
            commodity="Iron Ore",
            cargo_tonnes=150000,
            current_freight_usd_per_tonne=10.00,
            commodity_df_override=self.mock_buy_df,
            market_overrides=market_rising,
        )
        self.assertEqual(res["procurement"]["signal"], "BUY")
        self.assertEqual(res["charter_decision"], "CHARTER NOW")
        self.assertEqual(res["overall_strategy"], "BUY CARGO — CHARTER NOW")
        self.assertEqual(res["vessel"]["recommended_vessel"], "Capesize")
        self.assertGreater(len(res["decision_reasons"]), 3)
        self.assertTrue(any("BUY CARGO — CHARTER NOW" in r for r in res["decision_reasons"]))

    # -------------------------------------------------------------------------
    # 2. BUY + Falling Freight -> BUY CARGO — WAIT TO CHARTER
    # -------------------------------------------------------------------------
    def test_buy_commodity_falling_freight(self):
        """When commodity signal is BUY and Model v3 projects falling freight, strategy is BUY CARGO — WAIT TO CHARTER."""
        # Market inputs with higher base freight ($20.00) produce -20.0% delta (< -5.0%) -> WAIT TO CHARTER
        market_falling = {
            "bdi": 1970.0, "vlsfo_usd_per_tonne": 602.0,
            "coal_price_usd_per_mt": 142.5, "iron_ore_price_usd_per_dmt": 105.0,
            "wind_kmh": 28.0, "wave_height_m": 1.8,
            "cyclone_risk": 1.0, "weather_delay_days": 0.15,
        }
        res = analyze_decision(
            origin="Australia West Coast",
            destination="Dhamra",
            commodity="Iron Ore",
            cargo_tonnes=150000,
            current_freight_usd_per_tonne=20.00,
            commodity_df_override=self.mock_buy_df,
            market_overrides=market_falling,
        )
        self.assertEqual(res["procurement"]["signal"], "BUY")
        self.assertEqual(res["charter_decision"], "WAIT TO CHARTER")
        self.assertEqual(res["overall_strategy"], "BUY CARGO — WAIT TO CHARTER")
        self.assertEqual(res["vessel"]["recommended_vessel"], "Capesize")
        self.assertTrue(any("WAIT TO CHARTER" in r for r in res["decision_reasons"]))

    # -------------------------------------------------------------------------
    # 3. WAIT + Falling Freight -> WAIT FOR CARGO — WAIT TO CHARTER
    # -------------------------------------------------------------------------
    def test_wait_commodity_falling_freight(self):
        """When commodity prices are elevated (WAIT) and freight is softening, strategy is WAIT FOR CARGO — WAIT TO CHARTER."""
        # Authentic Coal dataset naturally produces WAIT signal
        market_falling = {
            "bdi": 1970.0, "vlsfo_usd_per_tonne": 602.0,
            "coal_price_usd_per_mt": 142.5, "iron_ore_price_usd_per_dmt": 105.0,
            "wind_kmh": 28.0, "wave_height_m": 1.8,
            "cyclone_risk": 1.0, "weather_delay_days": 0.15,
        }
        res = analyze_decision(
            origin="Hay Point",
            destination="Dhamra",
            commodity="Coal",
            cargo_tonnes=150000,
            current_freight_usd_per_tonne=20.00,
            market_overrides=market_falling,
        )
        self.assertEqual(res["procurement"]["signal"], "WAIT")
        self.assertEqual(res["charter_decision"], "WAIT TO CHARTER")
        self.assertEqual(res["overall_strategy"], "WAIT FOR CARGO — WAIT TO CHARTER")
        self.assertTrue(any("WAIT FOR CARGO — WAIT TO CHARTER" in r for r in res["decision_reasons"]))

    # -------------------------------------------------------------------------
    # 4. MONITOR + Uncertain/Stable Freight -> MONITOR CARGO — MONITOR FREIGHT
    # -------------------------------------------------------------------------
    def test_monitor_commodity_uncertain_freight(self):
        """When commodity signal is MONITOR and freight change is within +/- 5%, strategy is MONITOR CARGO — MONITOR FREIGHT."""
        # Authentic Iron Ore naturally produces MONITOR signal
        market_neutral = {
            "bdi": 1560.0, "vlsfo_usd_per_tonne": 638.0,
            "coal_price_usd_per_mt": 124.0, "iron_ore_price_usd_per_dmt": 124.0,
            "wind_kmh": 28.0, "wave_height_m": 1.7,
            "cyclone_risk": 1.0, "weather_delay_days": 0.15,
        }
        # Base freight equal to $15.00 produces +4.21% change (within [-5%, +5%]) -> MONITOR FREIGHT
        res = analyze_decision(
            origin="Australia West Coast",
            destination="Dhamra",
            commodity="Iron Ore",
            cargo_tonnes=150000,
            current_freight_usd_per_tonne=15.00,
            market_overrides=market_neutral,
        )
        self.assertEqual(res["procurement"]["signal"], "MONITOR")
        self.assertEqual(res["charter_decision"], "MONITOR FREIGHT")
        self.assertEqual(res["overall_strategy"], "MONITOR CARGO — MONITOR FREIGHT")
        self.assertFalse(res["weather_override"])
        self.assertEqual(res["weather_risk_level"], "LOW")

    # -------------------------------------------------------------------------
    # 5. WAIT + CHARTER NOW (Weather Override = True)
    # -------------------------------------------------------------------------
    def test_wait_commodity_charter_now_weather_override_true(self):
        """When commodity is WAIT but elevated weather risk overrides freight softening, strategy is WAIT FOR CARGO — CHARTER NOW."""
        # Elevated weather risk (cyclone_risk=4.5, delay=3.0) forces CHARTER NOW even when baseline freight delta is negative (-22%)
        market_severe_weather = {
            "bdi": 1970.0, "vlsfo_usd_per_tonne": 602.0,
            "coal_price_usd_per_mt": 142.5, "iron_ore_price_usd_per_dmt": 105.0,
            "wind_kmh": 65.0, "wave_height_m": 4.5,
            "cyclone_risk": 4.5, "weather_delay_days": 3.0,
        }
        res = analyze_decision(
            origin="Hay Point",
            destination="Dhamra",
            commodity="Coal",
            cargo_tonnes=150000,
            current_freight_usd_per_tonne=20.00,
            market_overrides=market_severe_weather,
        )
        # Procurement: Coal benchmark is historically elevated -> WAIT
        self.assertEqual(res["procurement"]["signal"], "WAIT")
        # Charter: High weather risk overrides softening freight -> CHARTER NOW
        self.assertEqual(res["charter_decision"], "CHARTER NOW")
        # Overall Strategy must strictly be WAIT FOR CARGO — CHARTER NOW
        self.assertEqual(res["overall_strategy"], "WAIT FOR CARGO — CHARTER NOW")
        # Weather decision fields
        self.assertTrue(res["weather_override"])
        self.assertEqual(res["weather_risk_level"], "HIGH")
        # Evidence must contain explicit weather override explanation
        self.assertTrue(any("Weather Risk Override" in r for r in res["decision_reasons"]))
        self.assertTrue(any("WAIT FOR CARGO — CHARTER NOW" in r for r in res["decision_reasons"]))

    # -------------------------------------------------------------------------
    # 6. Normal Weather -> Weather Override = False
    # -------------------------------------------------------------------------
    def test_weather_override_false_under_normal_weather(self):
        """Under normal weather conditions (LOW risk), weather_override must remain False."""
        # Normal weather with softening freight (-22%) -> WAIT TO CHARTER
        market_normal = {
            "bdi": 1970.0, "vlsfo_usd_per_tonne": 602.0,
            "coal_price_usd_per_mt": 142.5, "iron_ore_price_usd_per_dmt": 105.0,
            "wind_kmh": 20.0, "wave_height_m": 1.2,
            "cyclone_risk": 1.0, "weather_delay_days": 0.1,
        }
        res = analyze_decision(
            origin="Hay Point",
            destination="Dhamra",
            commodity="Coal",
            cargo_tonnes=150000,
            current_freight_usd_per_tonne=20.00,
            market_overrides=market_normal,
        )
        self.assertEqual(res["charter_decision"], "WAIT TO CHARTER")
        self.assertFalse(res["weather_override"])
        self.assertEqual(res["weather_risk_level"], "LOW")
        self.assertFalse(any("Weather Risk Override" in r for r in res["decision_reasons"]))

    # -------------------------------------------------------------------------
    # 7. Rate Increase Alone Triggers CHARTER NOW -> Weather Override = False
    # -------------------------------------------------------------------------
    def test_weather_override_false_when_rate_alone_triggers_charter_now(self):
        """When forecast rate rise >= 5% alone justifies CHARTER NOW, weather_override must be False even if risk is HIGH."""
        # Current freight $10.00, baseline market produces rate surge (+27.37% >= 5.0%).
        # Even with elevated cyclone risk (4.0 -> HIGH), rate surge alone dictates CHARTER NOW,
        # so weather risk did NOT change the decision.
        market_high_rate_and_weather = {
            "bdi": 1560.0, "vlsfo_usd_per_tonne": 638.0,
            "coal_price_usd_per_mt": 124.0, "iron_ore_price_usd_per_dmt": 124.0,
            "wind_kmh": 28.0, "wave_height_m": 1.7,
            "cyclone_risk": 4.0, "weather_delay_days": 0.15,
        }
        res = analyze_decision(
            origin="Australia West Coast",
            destination="Dhamra",
            commodity="Iron Ore",
            cargo_tonnes=150000,
            current_freight_usd_per_tonne=10.00,
            commodity_df_override=self.mock_buy_df,
            market_overrides=market_high_rate_and_weather,
        )
        self.assertEqual(res["charter_decision"], "CHARTER NOW")
        self.assertEqual(res["weather_risk_level"], "HIGH")
        # Must be FALSE because rate surge >= 5.0% independently required CHARTER NOW
        self.assertFalse(res["weather_override"])
        self.assertFalse(any("Weather Risk Override" in r for r in res["decision_reasons"]))

    # -------------------------------------------------------------------------
    # 8. Semantic Consistency: charter_decision vs overall_strategy
    # -------------------------------------------------------------------------
    def test_semantic_consistency_all_strategy_combinations(self):
        """Verify semantic consistency across all 9 procurement and chartering combinations."""
        for (proc_sig, charter_dec), overall_strat in STRATEGY_MAP.items():
            # Verify procurement signal prefix
            if proc_sig == "BUY":
                self.assertTrue(overall_strat.startswith("BUY CARGO — "), f"Mismatch for {proc_sig}, {charter_dec}")
            elif proc_sig == "MONITOR":
                self.assertTrue(overall_strat.startswith("MONITOR CARGO — "), f"Mismatch for {proc_sig}, {charter_dec}")
            elif proc_sig == "WAIT":
                self.assertTrue(overall_strat.startswith("WAIT FOR CARGO — "), f"Mismatch for {proc_sig}, {charter_dec}")

            # Verify charter decision suffix
            if charter_dec == "CHARTER NOW":
                self.assertTrue(overall_strat.endswith("— CHARTER NOW"), f"Mismatch for {proc_sig}, {charter_dec}")
            elif charter_dec == "WAIT TO CHARTER":
                self.assertTrue(overall_strat.endswith("— WAIT TO CHARTER"), f"Mismatch for {proc_sig}, {charter_dec}")
            elif charter_dec == "MONITOR FREIGHT":
                self.assertTrue(overall_strat.endswith("— MONITOR FREIGHT"), f"Mismatch for {proc_sig}, {charter_dec}")

        # Explicit check for WAIT + CHARTER NOW requirement
        self.assertEqual(STRATEGY_MAP[("WAIT", "CHARTER NOW")], "WAIT FOR CARGO — CHARTER NOW")
        self.assertEqual(STRATEGY_MAP[("WAIT", "WAIT TO CHARTER")], "WAIT FOR CARGO — WAIT TO CHARTER")
        self.assertEqual(STRATEGY_MAP[("WAIT", "MONITOR FREIGHT")], "WAIT FOR CARGO — MONITOR FREIGHT")

    # -------------------------------------------------------------------------
    # 9. No Suitable Vessel Case
    # -------------------------------------------------------------------------
    def test_no_suitable_vessel_reconfigure(self):
        """When destination port draft excludes Capesize and cargo volume overloads Panamax, strategy indicates reconfiguring shipment."""
        res = analyze_decision(
            origin="Hay Point",
            destination="Paradip",
            commodity="Coal",
            cargo_tonnes=150000,  # 150k mt to Paradip (14.5m limit excludes Capesize; 150k overloads Panamax)
        )
        self.assertEqual(res["vessel"]["status"], "NO_SUITABLE_VESSEL")
        self.assertIsNone(res["vessel"]["recommended_vessel"])
        self.assertEqual(res["charter_decision"], "NO SUITABLE VESSEL")
        self.assertEqual(res["overall_strategy"], "NO SUITABLE VESSEL — RECONFIGURE SHIPMENT")
        self.assertTrue(any("Feasibility Gate" in r for r in res["decision_reasons"]))

    # -------------------------------------------------------------------------
    # 6. Input Validation: Invalid Commodity
    # -------------------------------------------------------------------------
    def test_invalid_commodity_raises_error(self):
        """Unsupported commodity must raise ValueError with supported names."""
        with self.assertRaises(ValueError) as ctx:
            analyze_decision(
                origin="Hay Point",
                destination="Dhamra",
                commodity="Uranium",
                cargo_tonnes=150000,
            )
        self.assertIn("Unsupported commodity", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 7. Input Validation: Invalid Trade Corridor
    # -------------------------------------------------------------------------
    def test_invalid_route_raises_error(self):
        """Unsupported origin corridor must raise ValueError with canonical corridor hints."""
        with self.assertRaises(ValueError) as ctx:
            analyze_decision(
                origin="Rotterdam",
                destination="Dhamra",
                commodity="Coal",
                cargo_tonnes=150000,
            )
        self.assertIn("Unsupported origin", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 8. Input Validation: Non-positive Cargo Volume
    # -------------------------------------------------------------------------
    def test_invalid_cargo_volume_raises_error(self):
        """Cargo volume <= 0 must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            analyze_decision(
                origin="Hay Point",
                destination="Dhamra",
                commodity="Coal",
                cargo_tonnes=0,
            )
        self.assertIn("cargo_tonnes must be strictly greater than zero", str(ctx.exception))

        with self.assertRaises(ValueError):
            analyze_decision(
                origin="Hay Point",
                destination="Dhamra",
                commodity="Coal",
                cargo_tonnes=-1000,
            )

    # -------------------------------------------------------------------------
    # 9. API Integration: POST /decision/analyze (End-to-End Orchestration)
    # -------------------------------------------------------------------------
    def test_api_analyze_decision_success(self):
        """Verify POST /decision/analyze returns 200 with full structured response."""
        payload = {
            "origin": "Hay Point",
            "destination": "Dhamra",
            "commodity": "Coal",
            "cargo_tonnes": 150000,
            "current_freight_usd_per_tonne": 17.20,
        }
        resp = self.client.post("/decision/analyze", json=payload)
        self.assertEqual(resp.status_code, 200, f"Error: {resp.text}")
        data = resp.json()

        # Root fields
        self.assertEqual(data["origin"], "Hay Point")
        self.assertEqual(data["destination"], "Dhamra")
        self.assertEqual(data["commodity"], "Coal")
        self.assertEqual(data["cargo_tonnes"], 150000.0)
        self.assertIn("charter_decision", data)
        self.assertIn("overall_strategy", data)
        self.assertIsInstance(data["weather_override"], bool)
        self.assertIn(data["weather_risk_level"], ["LOW", "MEDIUM", "HIGH"])
        self.assertIsInstance(data["decision_reasons"], list)
        self.assertGreaterEqual(len(data["decision_reasons"]), 4)

        # Procurement sub-block
        proc = data["procurement"]
        self.assertIn(proc["signal"], ["BUY", "MONITOR", "WAIT"])
        self.assertIsInstance(proc["benchmark_price_usd_per_mt"], float)
        self.assertIn("percentile", proc)
        self.assertIn("momentum_3m_pct", proc)
        self.assertIn("data_freshness", proc)
        self.assertIsInstance(proc["reasons"], list)

        # Vessel sub-block
        vessel = data["vessel"]
        self.assertEqual(vessel["recommended_vessel"], "Capesize")
        self.assertEqual(vessel["status"], "OPTIMIZED")
        self.assertIsInstance(vessel["predicted_freight_usd_per_tonne"], float)
        self.assertIsInstance(vessel["estimated_freight_outlay_usd"], float)
        self.assertIsInstance(vessel["suitability_score"], float)
        self.assertIsInstance(vessel["reasons"], list)

        # Landed cost sub-block
        landed = data["landed_cost"]
        self.assertIsInstance(landed["commodity_fob_usd"], float)
        self.assertIsInstance(landed["ocean_freight_usd_per_tonne"], float)
        self.assertIsInstance(landed["estimated_landed_cost_usd"], float)
        self.assertIsInstance(landed["estimated_total_landed_outlay_usd"], float)
        self.assertEqual(landed["commodity_unit"], "USD/mt")
        self.assertEqual(landed["freight_unit"], "USD/t")
        self.assertEqual(landed["landed_cost_unit"], "USD/mt")
        self.assertEqual(
            landed["estimated_landed_cost_usd"],
            round(landed["commodity_fob_usd"] + landed["ocean_freight_usd_per_tonne"], 2),
        )
        self.assertEqual(landed["formula"], "Landed Cost = Commodity FOB + Ocean Freight")

    def test_decision_includes_landed_cost_iron_ore(self):
        """Decision analysis for Iron Ore must preserve USD/dmt unit discipline in landed cost."""
        res = analyze_decision(
            origin="Australia West Coast",
            destination="Dhamra",
            commodity="Iron Ore",
            cargo_tonnes=150000,
            current_freight_usd_per_tonne=12.00,
        )
        landed = res["landed_cost"]
        self.assertIsNotNone(landed["estimated_landed_cost_usd"])
        self.assertEqual(landed["commodity_unit"], "USD/dmt")
        self.assertEqual(landed["freight_unit"], "USD/t")
        self.assertEqual(landed["landed_cost_unit"], "USD/dmt")
        self.assertEqual(
            landed["estimated_landed_cost_usd"],
            round(landed["commodity_fob_usd"] + landed["ocean_freight_usd_per_tonne"], 2),
        )
        self.assertTrue(any("Delivered Acquisition Cost" in r for r in res["decision_reasons"]))

    def test_decision_landed_cost_none_when_no_suitable_vessel(self):
        """When vessel optimization yields NO_SUITABLE_VESSEL, landed cost must be None."""
        res = analyze_decision(
            origin="Hay Point",
            destination="Paradip",
            commodity="Coal",
            cargo_tonnes=150000,
        )
        self.assertEqual(res["vessel"]["status"], "NO_SUITABLE_VESSEL")
        landed = res["landed_cost"]
        self.assertIsNone(landed["ocean_freight_usd_per_tonne"])
        self.assertIsNone(landed["estimated_landed_cost_usd"])
        self.assertIsNone(landed["estimated_total_landed_outlay_usd"])
        self.assertTrue(any("unavailable" in r.lower() for r in landed["reasons"]))

    def test_api_analyze_decision_weather_override_success(self):
        """Verify POST /decision/analyze preserves structured weather override when elevated weather risk is injected."""
        # Using service directly with market_overrides is tested, here verify API schema serializes cleanly
        payload = {
            "origin": "Hay Point",
            "destination": "Dhamra",
            "commodity": "Coal",
            "cargo_tonnes": 150000,
            "current_freight_usd_per_tonne": 25.00,
        }
        resp = self.client.post("/decision/analyze", json=payload)
        self.assertEqual(resp.status_code, 200, f"Error: {resp.text}")
        data = resp.json()
        self.assertIn("weather_override", data)
        self.assertIn("weather_risk_level", data)
        self.assertIsInstance(data["weather_override"], bool)
        self.assertIn(data["weather_risk_level"], ["LOW", "MEDIUM", "HIGH"])

    def test_api_analyze_decision_invalid_commodity_422(self):
        """Verify POST /decision/analyze returns 422 for unsupported commodity."""
        payload = {
            "origin": "Hay Point",
            "destination": "Dhamra",
            "commodity": "Lithium",
            "cargo_tonnes": 150000,
        }
        resp = self.client.post("/decision/analyze", json=payload)
        self.assertEqual(resp.status_code, 422)
        err = resp.json()
        self.assertEqual(err["error_code"], "INVALID_DECISION_INPUT")

    def test_api_analyze_decision_invalid_cargo_422(self):
        """Verify POST /decision/analyze returns 422 for zero cargo tonnes."""
        payload = {
            "origin": "Hay Point",
            "destination": "Dhamra",
            "commodity": "Coal",
            "cargo_tonnes": 0,
        }
        resp = self.client.post("/decision/analyze", json=payload)
        self.assertEqual(resp.status_code, 422)
        err = resp.json()
        self.assertEqual(err["error_code"], "INVALID_DECISION_INPUT")

    def test_api_analyze_arbitrary_cargo_volumes_accepted_unrounded(self):
        """Verify POST /decision/analyze accepts all arbitrary unrounded custom whole-tonne volumes."""
        test_volumes = [
            10000, 10001, 12345, 12346, 15000, 15001, 15750, 18000, 18437, 18500,
            22375, 25001, 37250, 46038, 51700, 51729, 75000, 75001, 82741,
            100001, 100123, 125500, 127843, 150000, 150001, 165700, 169999, 170000, 181999
        ]
        for vol in test_volumes:
            with self.subTest(cargo_volume=vol):
                payload = {
                    "origin": "Taboneo" if vol <= 55000 else "Hay Point",
                    "destination": "East Coast India" if vol <= 55000 else "Dhamra",
                    "commodity": "Thermal Coal" if vol <= 55000 else "Coal",
                    "cargo_tonnes": vol,
                }
                resp = self.client.post("/decision/analyze", json=payload)
                self.assertEqual(resp.status_code, 200, f"Failed on volume {vol}: {resp.text}")
                data = resp.json()
                # Verify exact value is preserved without rounding
                self.assertEqual(data["cargo_tonnes"], float(vol))
                self.assertEqual(data["vessel"]["status"], "OPTIMIZED")
                self.assertIsNotNone(data["vessel"]["recommended_vessel"])


if __name__ == "__main__":
    unittest.main()

