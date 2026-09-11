"""Automated Test Suite for Commodity Procurement Decision Service.

Tests:
- Authentic CSV loading & caching from data/commodity_prices_worldbank.csv
- Latest available observation detection without future calendar assumptions
- Historical percentile calculation
- 3-month momentum calculation
- BUY / MONITOR / WAIT rule-based classification
- Invalid and unsupported commodity rejection (422)
- Missing data handling & mock resilience
- FastAPI endpoint integration for /procurement/valuation and /procurement/overview
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from main import app
from services import procurement_service
from services.procurement_service import (
    COMMODITY_CSV_PATH,
    COMMODITY_REGISTRY,
    PROCUREMENT_CONFIG,
    calculate_procurement_valuation,
    get_procurement_overview,
    load_commodity_data,
    resolve_commodity,
)


class TestProcurementService(unittest.TestCase):
    """Unit tests for the procurement service logic."""

    def setUp(self):
        self.client = TestClient(app)

    def test_csv_file_exists_and_loads(self):
        """Verify the authentic World Bank CSV exists and loads correctly into DataFrame."""
        self.assertTrue(COMMODITY_CSV_PATH.exists(), f"CSV not found at {COMMODITY_CSV_PATH}")
        df = load_commodity_data(force_reload=True)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 700, "Expected at least 700 monthly historical rows.")
        self.assertIn("coal_price_usd_per_mt", df.columns)
        self.assertIn("iron_ore_price_usd_per_dmt", df.columns)
        self.assertIn("date", df.columns)

    def test_commodity_resolution_and_aliases(self):
        """Verify canonical name resolution and alias handling."""
        # Canonical names
        spec_coal = resolve_commodity("Coal")
        self.assertEqual(spec_coal["canonical_name"], "Coal")
        self.assertEqual(spec_coal["column"], "coal_price_usd_per_mt")

        spec_iron = resolve_commodity("Iron Ore")
        self.assertEqual(spec_iron["canonical_name"], "Iron Ore")
        self.assertEqual(spec_iron["column"], "iron_ore_price_usd_per_dmt")

        # Aliases
        self.assertEqual(resolve_commodity("thermal coal")["canonical_name"], "Coal")
        self.assertEqual(resolve_commodity("iron_ore")["canonical_name"], "Iron Ore")
        self.assertEqual(resolve_commodity("COAL")["canonical_name"], "Coal")

        # Unsupported
        with self.assertRaises(ValueError) as ctx:
            resolve_commodity("Crude Oil")
        self.assertIn("Unsupported commodity", str(ctx.exception))

    def test_latest_available_observation_coal(self):
        """Verify latest available observation for Coal matches actual dataset."""
        res = calculate_procurement_valuation("Coal")
        df = load_commodity_data()
        valid_coal = df.dropna(subset=["coal_price_usd_per_mt"]).sort_values("date_dt")
        expected_latest = valid_coal.iloc[-1]

        self.assertEqual(res["commodity"], "Coal")
        self.assertEqual(res["benchmark_date"], str(expected_latest["date"]))
        self.assertEqual(res["benchmark_price_usd_per_mt"], round(float(expected_latest["coal_price_usd_per_mt"]), 2))
        self.assertEqual(res["unit"], "USD/mt")
        self.assertEqual(res["source"], "World Bank Pink Sheet")
        self.assertIn("2026", res["benchmark_date"])

    def test_latest_available_observation_iron_ore(self):
        """Verify latest available observation for Iron Ore matches actual dataset."""
        res = calculate_procurement_valuation("Iron Ore")
        df = load_commodity_data()
        valid_iron = df.dropna(subset=["iron_ore_price_usd_per_dmt"]).sort_values("date_dt")
        expected_latest = valid_iron.iloc[-1]

        self.assertEqual(res["commodity"], "Iron Ore")
        self.assertEqual(res["benchmark_date"], str(expected_latest["date"]))
        self.assertEqual(res["benchmark_price_usd_per_mt"], round(float(expected_latest["iron_ore_price_usd_per_dmt"]), 2))
        self.assertEqual(res["unit"], "USD/dmt")
        self.assertIn("2026", res["benchmark_date"])

    def test_percentile_and_momentum_calculations(self):
        """Verify historical percentile and 3-month momentum formulas."""
        res = calculate_procurement_valuation("Iron Ore")
        df = load_commodity_data()
        valid_iron = df.dropna(subset=["iron_ore_price_usd_per_dmt"]).sort_values("date_dt")

        latest_price = float(valid_iron.iloc[-1]["iron_ore_price_usd_per_dmt"])
        price_3m_ago = float(valid_iron.iloc[-4]["iron_ore_price_usd_per_dmt"])
        expected_momentum = ((latest_price - price_3m_ago) / price_3m_ago) * 100.0

        self.assertAlmostEqual(res["momentum_3m_pct"], round(expected_momentum, 2), places=1)
        self.assertGreaterEqual(res["percentile"], 0.0)
        self.assertLessEqual(res["percentile"], 100.0)
        self.assertGreaterEqual(res["volatility_3m_pct"], 0.0)

    def test_rule_based_signals_deterministic(self):
        """Verify deterministic BUY, MONITOR, WAIT logic on custom data distributions."""
        # 1. Test BUY scenario: low percentile + downward momentum
        mock_buy_df = pd.DataFrame({
            "date": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"],
            "date_dt": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"]),
            "coal_price_usd_per_mt": [150.0, 140.0, 130.0, 120.0, 80.0],  # Latest is lowest (percentile ~20%) & falling
            "iron_ore_price_usd_per_dmt": [100.0, 100.0, 100.0, 100.0, 100.0],
            "source": ["WB"] * 5,
            "source_date": ["Test"] * 5,
        })
        res_buy = calculate_procurement_valuation("Coal", df_override=mock_buy_df)
        self.assertEqual(res_buy["signal"], "BUY")
        self.assertGreaterEqual(res_buy["signal_score"], 20.0)
        self.assertTrue(any("Favorable Entry Valuation" in r for r in res_buy["reasons"]))

        # 2. Test WAIT scenario: high percentile + accelerating upward momentum
        mock_wait_df = pd.DataFrame({
            "date": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"],
            "date_dt": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"]),
            "coal_price_usd_per_mt": [50.0, 60.0, 70.0, 80.0, 200.0],  # Latest is highest (percentile 100%) & rising sharply
            "iron_ore_price_usd_per_dmt": [100.0, 100.0, 100.0, 100.0, 100.0],
            "source": ["WB"] * 5,
            "source_date": ["Test"] * 5,
        })
        res_wait = calculate_procurement_valuation("Coal", df_override=mock_wait_df)
        self.assertEqual(res_wait["signal"], "WAIT")
        self.assertLessEqual(res_wait["signal_score"], -20.0)
        self.assertTrue(any("Elevated Entry Valuation" in r for r in res_wait["reasons"]))

    def test_procurement_overview(self):
        """Verify overview aggregates all supported commodities."""
        overview = get_procurement_overview()
        self.assertIn("commodities", overview)
        self.assertEqual(overview["total_supported"], 2)
        names = [c["commodity"] for c in overview["commodities"]]
        self.assertIn("Coal", names)
        self.assertIn("Iron Ore", names)

    def test_missing_data_handling(self):
        """Verify handling when commodity series has all NaN values."""
        mock_empty_df = pd.DataFrame({
            "date": ["2024-01-01"],
            "date_dt": pd.to_datetime(["2024-01-01"]),
            "coal_price_usd_per_mt": [np.nan],
            "iron_ore_price_usd_per_dmt": [100.0],
        })
        with self.assertRaises(ValueError) as ctx:
            calculate_procurement_valuation("Coal", df_override=mock_empty_df)
        self.assertIn("No valid historical observations", str(ctx.exception))

    def test_api_endpoint_valuation_valid(self):
        """Test GET /procurement/valuation for valid commodities."""
        for comm in ["Coal", "Iron Ore", "thermal coal"]:
            resp = self.client.get(f"/procurement/valuation?commodity={comm}")
            self.assertEqual(resp.status_code, 200, f"Failed for {comm}: {resp.text}")
            data = resp.json()
            self.assertIn(data["commodity"], ["Coal", "Iron Ore"])
            self.assertIn(data["signal"], ["BUY", "MONITOR", "WAIT"])
            self.assertIsInstance(data["benchmark_price_usd_per_mt"], float)
            self.assertIsInstance(data["reasons"], list)
            self.assertGreater(len(data["reasons"]), 0)

    def test_api_endpoint_valuation_invalid_commodity(self):
        """Test GET /procurement/valuation returns 422 for unsupported commodity."""
        resp = self.client.get("/procurement/valuation?commodity=Lithium")
        self.assertEqual(resp.status_code, 422)
        data = resp.json()
        self.assertEqual(data["error_code"], "INVALID_COMMODITY")
        self.assertIn("Unsupported commodity", data["message"])

    def test_api_endpoint_overview(self):
        """Test GET /procurement/overview returns full portfolio."""
        resp = self.client.get("/procurement/overview")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_supported"], 2)
        self.assertEqual(len(data["commodities"]), 2)


if __name__ == "__main__":
    unittest.main()
