"""Automated Unit Tests for NaviFreight Landed Cost Service.

Tests compliance with implementation_specs/02_LANDED_COST.txt:
- Formula correctness: Landed Cost = Commodity FOB + Ocean Freight
- Unit consistency for Coal (USD/mt) and Iron Ore (USD/dmt)
- Missing price handling (returns None)
- Missing freight handling (returns None)
- Provenance preservation (World Bank + Model V3)
- No unmodeled cost components (port charges, demurrage, insurance, taxes)
"""

import sys
import unittest
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from services.landed_cost_service import calculate_landed_cost


class TestLandedCostService(unittest.TestCase):
    """Test suite verifying landed cost calculation, unit discipline, and edge cases."""

    def test_coal_landed_cost_formula_and_units(self):
        """Coal: Landed cost must equal FOB + Freight with USD/mt and USD/t preserved."""
        res = calculate_landed_cost(
            commodity="Coal",
            commodity_fob_usd=135.20,
            ocean_freight_usd_per_tonne=16.00,
            cargo_tonnes=150000,
        )
        # Landed Cost = 135.20 + 16.00 = 151.20
        self.assertEqual(res["commodity_fob_usd"], 135.20)
        self.assertEqual(res["ocean_freight_usd_per_tonne"], 16.00)
        self.assertEqual(res["estimated_landed_cost_usd"], 151.20)
        self.assertEqual(res["estimated_total_landed_outlay_usd"], 22680000.00)
        # Unit discipline
        self.assertEqual(res["commodity_unit"], "USD/mt")
        self.assertEqual(res["freight_unit"], "USD/t")
        self.assertEqual(res["landed_cost_unit"], "USD/mt")
        # Formula & provenance
        self.assertEqual(res["formula"], "Landed Cost = Commodity FOB + Ocean Freight")
        self.assertIn("World Bank", res["provenance"]["commodity_source"])
        self.assertIn("Model V3", res["provenance"]["freight_source"])
        self.assertTrue(any("Estimated Landed Cost: $151.20 USD/mt" in r for r in res["reasons"]))

    def test_iron_ore_landed_cost_formula_and_units(self):
        """Iron Ore: Landed cost must equal FOB + Freight with USD/dmt unit preserved."""
        res = calculate_landed_cost(
            commodity="Iron Ore",
            commodity_fob_usd=105.00,
            ocean_freight_usd_per_tonne=11.41,
            cargo_tonnes=150000,
        )
        # Landed Cost = 105.00 + 11.41 = 116.41
        self.assertEqual(res["commodity_fob_usd"], 105.00)
        self.assertEqual(res["ocean_freight_usd_per_tonne"], 11.41)
        self.assertEqual(res["estimated_landed_cost_usd"], 116.41)
        self.assertEqual(res["estimated_total_landed_outlay_usd"], 17461500.00)
        # Unit discipline
        self.assertEqual(res["commodity_unit"], "USD/dmt")
        self.assertEqual(res["freight_unit"], "USD/t")
        self.assertEqual(res["landed_cost_unit"], "USD/dmt")
        self.assertTrue(any("Estimated Landed Cost: $116.41 USD/dmt" in r for r in res["reasons"]))

    def test_missing_price_handling(self):
        """Missing commodity FOB benchmark must produce None for landed cost."""
        res = calculate_landed_cost(
            commodity="Coal",
            commodity_fob_usd=None,
            ocean_freight_usd_per_tonne=16.00,
            cargo_tonnes=150000,
        )
        self.assertIsNone(res["commodity_fob_usd"])
        self.assertIsNone(res["estimated_landed_cost_usd"])
        self.assertIsNone(res["estimated_total_landed_outlay_usd"])
        self.assertTrue(any("unavailable" in r.lower() for r in res["reasons"]))

    def test_missing_freight_handling(self):
        """Missing ocean freight rate (e.g. no suitable vessel) must produce None for landed cost."""
        res = calculate_landed_cost(
            commodity="Coal",
            commodity_fob_usd=135.20,
            ocean_freight_usd_per_tonne=None,
            cargo_tonnes=150000,
        )
        self.assertEqual(res["commodity_fob_usd"], 135.20)
        self.assertIsNone(res["ocean_freight_usd_per_tonne"])
        self.assertIsNone(res["estimated_landed_cost_usd"])
        self.assertIsNone(res["estimated_total_landed_outlay_usd"])
        self.assertTrue(any("unavailable" in r.lower() for r in res["reasons"]))

    def test_provenance_preserved(self):
        """Source provenance must document World Bank Pink Sheet and Model V3."""
        res = calculate_landed_cost(
            commodity="Coal",
            commodity_fob_usd=135.20,
            ocean_freight_usd_per_tonne=16.00,
        )
        self.assertIn("commodity_source", res["provenance"])
        self.assertIn("freight_source", res["provenance"])
        self.assertIn("formula", res["provenance"])
        self.assertIn("World Bank", res["provenance"]["commodity_source"])
        self.assertIn("Model V3", res["provenance"]["freight_source"])

    def test_zero_extra_costs_disciplined(self):
        """Landed cost service must not add demurrage, insurance, port charges or taxes."""
        res = calculate_landed_cost(
            commodity="Coal",
            commodity_fob_usd=100.00,
            ocean_freight_usd_per_tonne=20.00,
            cargo_tonnes=1000,
        )
        # Strictly FOB + Freight = 120.00
        self.assertEqual(res["estimated_landed_cost_usd"], 120.00)
        self.assertEqual(res["estimated_total_landed_outlay_usd"], 120000.00)
        # Disallowed invented cost keys
        for forbidden in ["demurrage", "port_charges", "insurance", "taxes", "handling"]:
            self.assertNotIn(forbidden, res)


if __name__ == "__main__":
    unittest.main()
