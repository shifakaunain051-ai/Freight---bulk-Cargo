"""Automated Test Suite for Vessel Suitability & Chartering Decision Service.

Tests:
- Authentic Baltic vessel specifications loading and dimension verification
- East Coast India port constraint loading and hydrographic limits
- Cargo-volume fit and deadfreight/overload eligibility boundaries
- Physical draft compatibility against port constraints (e.g. Paradip vs Dhamra vs Haldia)
- Trade lane / canonical corridor operational viability
- Total freight outlay calculation (predicted_freight * cargo_tonnes)
- Recommendation selection based on multi-criteria suitability (not merely lowest $/t)
- Structured "NO_SUITABLE_VESSEL" output when no class is feasible
- FastAPI integration for POST /vessel/optimize with schema and error validation
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
from services import vessel_service
from services.vessel_service import (
    VESSEL_SPECS_PATH,
    PORT_CONSTRAINTS_PATH,
    BALTIC_ROUTES_PATH,
    load_vessel_specs,
    load_port_constraints,
    load_baltic_routes,
    check_port_compatibility,
    optimize_vessel_chartering,
)


class TestVesselService(unittest.TestCase):
    """Comprehensive automated tests for NaviFreight vessel suitability service."""

    def setUp(self):
        self.client = TestClient(app)

    # -------------------------------------------------------------------------
    # 1. Vessel Specifications Loading
    # -------------------------------------------------------------------------
    def test_vessel_specs_loading_and_dimensions(self):
        """Verify authentic Baltic Exchange vessel specifications exist and have exact dimensions."""
        self.assertTrue(VESSEL_SPECS_PATH.exists(), f"Vessel specs missing at {VESSEL_SPECS_PATH}")
        df = load_vessel_specs()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df), 3)

        v_map = df.set_index("vessel_type").to_dict(orient="index")
        self.assertIn("Capesize", v_map)
        self.assertIn("Panamax", v_map)
        self.assertIn("Supramax", v_map)
        self.assertIn("Handysize", v_map)

        # Authentic dimensions check
        self.assertEqual(float(v_map["Capesize"]["standard_dwt"]), 182000.0)
        self.assertEqual(float(v_map["Capesize"]["draft_m"]), 18.20)

        self.assertEqual(float(v_map["Panamax"]["standard_dwt"]), 82500.0)
        self.assertEqual(float(v_map["Panamax"]["draft_m"]), 14.43)

        self.assertEqual(float(v_map["Supramax"]["standard_dwt"]), 58328.0)
        self.assertEqual(float(v_map["Supramax"]["draft_m"]), 12.80)

        self.assertEqual(float(v_map["Handysize"]["standard_dwt"]), 35000.0)
        self.assertEqual(float(v_map["Handysize"]["draft_m"]), 10.00)

    # -------------------------------------------------------------------------
    # 2. Port Constraint Loading
    # -------------------------------------------------------------------------
    def test_port_constraints_loading(self):
        """Verify East Coast India port constraints load correctly with draft limits."""
        self.assertTrue(PORT_CONSTRAINTS_PATH.exists(), f"Port constraints missing at {PORT_CONSTRAINTS_PATH}")
        df = load_port_constraints()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df), 7)

        ports = df["port"].tolist()
        self.assertIn("Dhamra", ports)
        self.assertIn("Paradip", ports)
        self.assertIn("Haldia", ports)
        self.assertIn("Visakhapatnam", ports)

        p_map = df.set_index("port").to_dict(orient="index")
        self.assertEqual(float(p_map["Dhamra"]["operational_draft_m"]), 18.00)
        self.assertEqual(bool(p_map["Dhamra"]["capesize_compatible"]), True)

        self.assertEqual(float(p_map["Paradip"]["operational_draft_m"]), 14.50)
        self.assertEqual(bool(p_map["Paradip"]["capesize_compatible"]), False)
        self.assertEqual(bool(p_map["Paradip"]["panamax_compatible"]), True)

        self.assertEqual(float(p_map["Haldia"]["operational_draft_m"]), 8.00)
        self.assertEqual(bool(p_map["Haldia"]["capesize_compatible"]), False)
        self.assertEqual(bool(p_map["Haldia"]["panamax_compatible"]), False)

    # -------------------------------------------------------------------------
    # 3. Baltic Route Specs Loading (Reference Only)
    # -------------------------------------------------------------------------
    def test_baltic_routes_reference_loading(self):
        """Verify Baltic route benchmark specifications load as reference definitions."""
        self.assertTrue(BALTIC_ROUTES_PATH.exists(), f"Baltic routes missing at {BALTIC_ROUTES_PATH}")
        df = load_baltic_routes()
        self.assertIsInstance(df, pd.DataFrame)
        codes = df["route_code"].tolist()
        self.assertIn("C18", codes)
        self.assertIn("P9", codes)

    # -------------------------------------------------------------------------
    # 4. Draft Compatibility Checks
    # -------------------------------------------------------------------------
    def test_draft_compatibility_paradip(self):
        """Paradip has 14.50m draft limit: Capesize (18.2m) must be excluded; Panamax, Supramax, Handysize allowed."""
        # Capesize at Paradip -> Incompatible
        ok_cape, reason_cape = check_port_compatibility("Capesize", 18.20, "Paradip")
        self.assertFalse(ok_cape)
        self.assertIn("exceeds Paradip operational draft limit", reason_cape)

        # Panamax at Paradip -> Compatible
        ok_pan, reason_pan = check_port_compatibility("Panamax", 14.43, "Paradip")
        self.assertTrue(ok_pan)
        self.assertIn("accommodates fully laden Panamax", reason_pan)

        # Supramax at Paradip -> Compatible (12.80m <= 14.50m)
        ok_sup, reason_sup = check_port_compatibility("Supramax", 12.80, "Paradip")
        self.assertTrue(ok_sup)

        # Handysize at Paradip -> Compatible (10.00m <= 14.50m)
        ok_hdy, reason_hdy = check_port_compatibility("Handysize", 10.00, "Paradip")
        self.assertTrue(ok_hdy)
        self.assertIn("accommodates fully laden Handysize", reason_hdy)

    def test_draft_compatibility_dhamra(self):
        """Dhamra is a deepwater port (18.00m permissible draft) accommodating Capesize."""
        ok_cape, reason_cape = check_port_compatibility("Capesize", 18.20, "Dhamra")
        self.assertTrue(ok_cape)
        self.assertIn("accommodates Capesize bulk carriers", reason_cape)

    def test_draft_compatibility_haldia(self):
        """Haldia has 8.00m riverine draft: all standard bulk carriers must be excluded."""
        ok_cape, _ = check_port_compatibility("Capesize", 18.20, "Haldia")
        self.assertFalse(ok_cape)

        ok_pan, _ = check_port_compatibility("Panamax", 14.43, "Haldia")
        self.assertFalse(ok_pan)

        ok_sup, _ = check_port_compatibility("Supramax", 12.80, "Haldia")
        self.assertFalse(ok_sup)

        ok_hdy, _ = check_port_compatibility("Handysize", 10.00, "Haldia")
        self.assertFalse(ok_hdy)

    # -------------------------------------------------------------------------
    # 5. Cargo Volume Recommendation Across All 4 Vessel Classes
    # -------------------------------------------------------------------------
    def test_small_cargo_handysize_recommended(self):
        """Small parcel (30,000 mt) on Taboneo must recommend Handysize (35k DWT, ~85% util)."""
        res = optimize_vessel_chartering(
            origin="Taboneo",
            destination="East Coast India",
            commodity="Thermal Coal",
            cargo_tonnes=30000,
        )
        self.assertEqual(res["status"], "OPTIMIZED")
        self.assertEqual(res["recommended_vessel"], "Handysize")

        hdy_eval = next(v for v in res["evaluated_vessels"] if v["vessel_type"] == "Handysize")
        self.assertTrue(hdy_eval["eligible"])
        self.assertAlmostEqual(hdy_eval["utilization_pct"], 85.7, places=1)

    def test_medium_cargo_supramax_recommended(self):
        """Medium parcel (50,000 mt) on Taboneo must recommend Supramax (58.3k DWT, ~85% util)."""
        res = optimize_vessel_chartering(
            origin="Taboneo",
            destination="East Coast India",
            commodity="Thermal Coal",
            cargo_tonnes=50000,
        )
        self.assertEqual(res["status"], "OPTIMIZED")
        self.assertEqual(res["recommended_vessel"], "Supramax")

        # Handysize must be rejected for practical capacity overload
        hdy_eval = next(v for v in res["evaluated_vessels"] if v["vessel_type"] == "Handysize")
        self.assertFalse(hdy_eval["cargo_fit"])
        self.assertFalse(hdy_eval["eligible"])

        sup_eval = next(v for v in res["evaluated_vessels"] if v["vessel_type"] == "Supramax")
        self.assertTrue(sup_eval["eligible"])

    def test_larger_cargo_panamax_recommended(self):
        """Larger parcel (75,000 mt) on Hay Point must recommend Panamax (82.5k DWT, ~90% util)."""
        res = optimize_vessel_chartering(
            origin="Hay Point",
            destination="East Coast India",
            commodity="Coal",
            cargo_tonnes=75000,
        )
        self.assertEqual(res["status"], "OPTIMIZED")
        self.assertEqual(res["recommended_vessel"], "Panamax")

        # Capesize (182k DWT) must be excluded for deadfreight (<45% util)
        cape_eval = next(v for v in res["evaluated_vessels"] if v["vessel_type"] == "Capesize")
        self.assertFalse(cape_eval["cargo_fit"])
        self.assertFalse(cape_eval["eligible"])

        # Handysize & Supramax excluded for capacity overload
        hdy_eval = next(v for v in res["evaluated_vessels"] if v["vessel_type"] == "Handysize")
        self.assertFalse(hdy_eval["cargo_fit"])

        sup_eval = next(v for v in res["evaluated_vessels"] if v["vessel_type"] == "Supramax")
        self.assertFalse(sup_eval["cargo_fit"])

    def test_very_large_cargo_capesize_recommended(self):
        """Very large parcel (150,000 mt) on Hay Point -> Dhamra must recommend Capesize (182k DWT, ~82% util)."""
        res = optimize_vessel_chartering(
            origin="Hay Point",
            destination="Dhamra",
            commodity="Coal",
            cargo_tonnes=150000,
        )
        self.assertEqual(res["status"], "OPTIMIZED")
        self.assertEqual(res["recommended_vessel"], "Capesize")

        cape_eval = next(v for v in res["evaluated_vessels"] if v["vessel_type"] == "Capesize")
        self.assertTrue(cape_eval["eligible"])
        self.assertAlmostEqual(cape_eval["utilization_pct"], 82.4, places=1)

    def test_cargo_volume_overload_exclusion(self):
        """A 160,000 mt parcel must exclude Panamax, Supramax, and Handysize as physical overloads."""
        res = optimize_vessel_chartering(
            origin="Hay Point",
            destination="East Coast India",
            commodity="Coal",
            cargo_tonnes=160000,
        )
        self.assertEqual(res["status"], "OPTIMIZED")
        self.assertEqual(res["recommended_vessel"], "Capesize")

        for v_name in ["Panamax", "Supramax", "Handysize"]:
            v_eval = next(v for v in res["evaluated_vessels"] if v["vessel_type"] == v_name)
            self.assertFalse(v_eval["cargo_fit"])
            self.assertFalse(v_eval["eligible"])
    def test_arbitrary_cargo_volumes_accepted_and_evaluated(self):
        """Verify arbitrary non-5000 increment cargo volumes (10k, 15k, 18k, 18.5k, 22.5k, 37.25k, 51.7k, 75k, 100k, 150k) evaluate cleanly."""
        test_cases = [
            (10000, "Taboneo", "Thermal Coal", "East Coast India", "Handysize"),
            (15000, "Taboneo", "Thermal Coal", "East Coast India", "Handysize"),
            (18000, "Taboneo", "Thermal Coal", "East Coast India", "Handysize"),
            (18500, "Taboneo", "Thermal Coal", "East Coast India", "Handysize"),
            (22500, "Taboneo", "Thermal Coal", "East Coast India", "Handysize"),
            (37250, "Taboneo", "Thermal Coal", "East Coast India", "Supramax"),
            (51700, "Taboneo", "Thermal Coal", "East Coast India", "Supramax"),
            (75000, "Hay Point", "Coal", "Dhamra", "Panamax"),
            (100000, "Hay Point", "Coal", "Dhamra", "Capesize"),
            (150000, "Hay Point", "Coal", "Dhamra", "Capesize"),
        ]
        for cargo, orig, comm, dest, expected_vessel in test_cases:
            with self.subTest(cargo=cargo, expected=expected_vessel):
                res = optimize_vessel_chartering(
                    origin=orig,
                    destination=dest,
                    commodity=comm,
                    cargo_tonnes=cargo,
                )
                self.assertEqual(res["status"], "OPTIMIZED")
                self.assertEqual(res["recommended_vessel"], expected_vessel)
                winner = next(v for v in res["evaluated_vessels"] if v["vessel_type"] == expected_vessel)
                self.assertTrue(winner["eligible"])
                self.assertTrue(winner["cargo_fit"])

    # -------------------------------------------------------------------------
    # 6. Trade Lane / Corridor Support
    # -------------------------------------------------------------------------
    def test_unsupported_corridor_vessel(self):
        """Capesize is not an active canonical vessel on the Taboneo -> East Coast India lane."""
        res = optimize_vessel_chartering(
            origin="Taboneo",
            destination="East Coast India",
            commodity="Thermal Coal",
            cargo_tonnes=70000,
        )
        cape_eval = next(v for v in res["evaluated_vessels"] if v["vessel_type"] == "Capesize")
        self.assertFalse(cape_eval["corridor_supported"])
        self.assertFalse(cape_eval["eligible"])
        self.assertTrue(any("not an active canonical vessel class" in r for r in cape_eval["reasons"]))

    def test_unsupported_origin_validation(self):
        """Unsupported origin should raise ValueError with supported corridor hints."""
        with self.assertRaises(ValueError) as ctx:
            optimize_vessel_chartering(
                origin="Rotterdam",
                destination="East Coast India",
                commodity="Coal",
                cargo_tonnes=75000,
            )
        self.assertIn("Unsupported origin", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 7. Outlay Arithmetic & Model v3 Integration
    # -------------------------------------------------------------------------
    def test_freight_outlay_calculation(self):
        """Verify estimated_freight_outlay_usd = predicted_freight_usd_per_tonne * cargo_tonnes."""
        cargo = 150000.0
        res = optimize_vessel_chartering(
            origin="Hay Point",
            destination="Dhamra",
            commodity="Coal",
            cargo_tonnes=cargo,
        )
        cape_eval = next(v for v in res["evaluated_vessels"] if v["vessel_type"] == "Capesize")
        self.assertTrue(cape_eval["eligible"])
        pred_rate = cape_eval["predicted_freight_usd_per_tonne"]
        outlay = cape_eval["estimated_freight_outlay_usd"]

        self.assertIsNotNone(pred_rate)
        self.assertGreater(pred_rate, 5.0)
        expected_outlay = round(pred_rate * cargo, 2)
        self.assertEqual(outlay, expected_outlay)

    # -------------------------------------------------------------------------
    # 8. Recommendation Selection Multi-Criteria & Best-Fit
    # -------------------------------------------------------------------------
    def test_recommendation_selection_multi_criteria(self):
        """Recommendation chooses highest suitability score (capacity fit + economics + corridor)."""
        res = optimize_vessel_chartering(
            origin="Hay Point",
            destination="East Coast India",
            commodity="Coal",
            cargo_tonnes=150000,
        )
        self.assertEqual(res["status"], "OPTIMIZED")
        self.assertEqual(res["recommended_vessel"], "Capesize")

        winner_eval = next(v for v in res["evaluated_vessels"] if v["vessel_type"] == res["recommended_vessel"])
        self.assertGreater(winner_eval["suitability_score"], 60.0)

    def test_multiple_feasible_vessels_best_fit_selected(self):
        """When multiple vessels are feasible (e.g. 50k mt on Taboneo: Supramax & Panamax), Supramax is chosen as best fit."""
        res = optimize_vessel_chartering(
            origin="Taboneo",
            destination="East Coast India",
            commodity="Thermal Coal",
            cargo_tonnes=50000,
        )
        self.assertEqual(res["status"], "OPTIMIZED")
        self.assertEqual(res["recommended_vessel"], "Supramax")

        sup = next(v for v in res["evaluated_vessels"] if v["vessel_type"] == "Supramax")
        pan = next(v for v in res["evaluated_vessels"] if v["vessel_type"] == "Panamax")

        self.assertTrue(sup["eligible"])
        self.assertTrue(pan["eligible"])
        self.assertGreater(sup["suitability_score"], pan["suitability_score"])

    # -------------------------------------------------------------------------
    # 9. No Suitable Vessel Structured Case
    # -------------------------------------------------------------------------
    def test_no_suitable_vessel_draft_infeasible(self):
        """Large cargo (150,000 mt) to Paradip must return NO_SUITABLE_VESSEL due to draft and overload limits."""
        res = optimize_vessel_chartering(
            origin="Hay Point",
            destination="Paradip",
            commodity="Coal",
            cargo_tonnes=150000,
        )
        self.assertEqual(res["status"], "NO_SUITABLE_VESSEL")
        self.assertIsNone(res["recommended_vessel"])
        self.assertIn("No suitable vessel class found", res["recommendation_reason"])

        # Capesize was excluded for port draft
        cape = next(v for v in res["evaluated_vessels"] if v["vessel_type"] == "Capesize")
        self.assertFalse(cape["port_compatible"])
        self.assertFalse(cape["eligible"])

        # Panamax, Supramax, Handysize were excluded for capacity overload
        for v_name in ["Panamax", "Supramax", "Handysize"]:
            v_eval = next(v for v in res["evaluated_vessels"] if v["vessel_type"] == v_name)
            self.assertFalse(v_eval["cargo_fit"])
            self.assertFalse(v_eval["eligible"])

    def test_no_suitable_vessel_haldia_shallow(self):
        """Any standard bulk shipment to Haldia must return NO_SUITABLE_VESSEL due to riverine shallow draft."""
        res = optimize_vessel_chartering(
            origin="Hay Point",
            destination="Haldia",
            commodity="Coal",
            cargo_tonnes=75000,
        )
        self.assertEqual(res["status"], "NO_SUITABLE_VESSEL")
        self.assertIsNone(res["recommended_vessel"])

    # -------------------------------------------------------------------------
    # 10. API Integration: POST /vessel/optimize
    # -------------------------------------------------------------------------
    def test_api_optimize_vessel_valid(self):
        """Verify POST /vessel/optimize endpoint returns 200 with all 4 vessel classes evaluated."""
        payload = {
            "origin": "Hay Point",
            "destination": "Dhamra",
            "commodity": "Coal",
            "cargo_tonnes": 150000,
        }
        resp = self.client.post("/vessel/optimize", json=payload)
        self.assertEqual(resp.status_code, 200, f"Error: {resp.text}")
        data = resp.json()

        self.assertEqual(data["origin"], "Hay Point")
        self.assertEqual(data["destination"], "Dhamra")
        self.assertEqual(data["status"], "OPTIMIZED")
        self.assertEqual(data["recommended_vessel"], "Capesize")
        self.assertIsInstance(data["evaluated_vessels"], list)
        self.assertEqual(len(data["evaluated_vessels"]), 4)

        # Check required evaluation item fields
        first = data["evaluated_vessels"][0]
        self.assertIn("vessel_type", first)
        self.assertIn("standard_dwt", first)
        self.assertIn("draft_m", first)
        self.assertIn("cargo_tonnes", first)
        self.assertIn("utilization_pct", first)
        self.assertIn("port_compatible", first)
        self.assertIn("port_fit_label", first)
        self.assertIn("cargo_fit", first)
        self.assertIn("cargo_fit_label", first)
        self.assertIn("eligible", first)
        self.assertIn("suitability_score", first)
        self.assertIn("reasons", first)

        v_names = [v["vessel_type"] for v in data["evaluated_vessels"]]
        self.assertIn("Handysize", v_names)
        self.assertIn("Supramax", v_names)
        self.assertIn("Panamax", v_names)
        self.assertIn("Capesize", v_names)

    def test_api_optimize_vessel_invalid_cargo(self):
        """Verify POST /vessel/optimize returns 422 for non-positive cargo."""
        payload = {
            "origin": "Hay Point",
            "destination": "Dhamra",
            "commodity": "Coal",
            "cargo_tonnes": 0,
        }
        resp = self.client.post("/vessel/optimize", json=payload)
        self.assertEqual(resp.status_code, 422)
        data = resp.json()
        self.assertEqual(data["error_code"], "INVALID_VESSEL_OPTIMIZATION_INPUT")

    def test_api_optimize_vessel_invalid_origin(self):
        """Verify POST /vessel/optimize returns 422 for unsupported origin."""
        payload = {
            "origin": "Antwerp",
            "destination": "Dhamra",
            "commodity": "Coal",
            "cargo_tonnes": 75000,
        }
        resp = self.client.post("/vessel/optimize", json=payload)
        self.assertEqual(resp.status_code, 422)
        data = resp.json()
        self.assertEqual(data["error_code"], "INVALID_VESSEL_OPTIMIZATION_INPUT")


if __name__ == "__main__":
    unittest.main()

