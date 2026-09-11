"""NaviFreight Landed Cost Service.

Calculates estimated delivered commodity acquisition cost:
    Landed Cost = Commodity FOB + Ocean Freight

Adheres strictly to implementation_specs/02_LANDED_COST.txt:
- Commodity FOB: procurement benchmark commodity price (World Bank Pink Sheet)
- Ocean Freight: forecast ocean freight (Model V3 forecast in USD/t)
- Unit discipline:
  * Coal: commodity in USD/mt, freight in USD/t, landed cost in USD/mt
  * Iron Ore: commodity in USD/dmt, freight in USD/t, landed cost in USD/dmt
- Preserves source provenance
- Does not invent unmodeled costs (insurance, demurrage, taxes, port charges, etc.)
"""

from __future__ import annotations

from typing import Any, Optional


def calculate_landed_cost(
    commodity: str,
    commodity_fob_usd: Optional[float],
    ocean_freight_usd_per_tonne: Optional[float],
    cargo_tonnes: Optional[float] = None,
    commodity_unit: Optional[str] = None,
) -> dict[str, Any]:
    """Calculate delivered commodity acquisition cost and outlay.

    Formula:
        Landed Cost = Commodity FOB + Ocean Freight

    Args:
        commodity: Commodity name (e.g. 'Coal', 'Thermal Coal', 'Iron Ore')
        commodity_fob_usd: Procurement benchmark commodity price FOB (or None if unavailable)
        ocean_freight_usd_per_tonne: Forecasted ocean freight rate USD/t (or None if unavailable)
        cargo_tonnes: Optional shipment cargo volume in metric tonnes
        commodity_unit: Optional explicit commodity unit ('USD/mt' or 'USD/dmt')

    Returns:
        Structured dictionary conforming to DecisionLandedCostSummary.
    """
    clean_comm = (commodity or "").strip().lower()

    # Determine standard units adhering to unit discipline
    if commodity_unit:
        c_unit = commodity_unit
    elif "iron" in clean_comm:
        c_unit = "USD/dmt"
    else:
        c_unit = "USD/mt"

    f_unit = "USD/t"
    landed_unit = c_unit

    reasons: list[str] = []
    estimated_landed_cost_usd: Optional[float] = None
    estimated_total_landed_outlay_usd: Optional[float] = None

    if commodity_fob_usd is None:
        reasons.append("Commodity FOB benchmark is unavailable; landed cost cannot be calculated.")
    elif ocean_freight_usd_per_tonne is None:
        reasons.append("Ocean freight rate is unavailable (no suitable vessel); landed cost cannot be calculated.")
    else:
        # Landed Cost = Commodity FOB + Ocean Freight
        estimated_landed_cost_usd = round(float(commodity_fob_usd) + float(ocean_freight_usd_per_tonne), 2)
        if cargo_tonnes is not None and cargo_tonnes > 0:
            estimated_total_landed_outlay_usd = round(estimated_landed_cost_usd * float(cargo_tonnes), 2)

        reasons.append(
            f"Estimated Landed Cost: ${estimated_landed_cost_usd:.2f} {landed_unit} "
            f"(FOB ${float(commodity_fob_usd):.2f} {c_unit} + Ocean Freight ${float(ocean_freight_usd_per_tonne):.2f}/t)."
        )
        if estimated_total_landed_outlay_usd is not None:
            reasons.append(
                f"Total delivered shipment outlay: ${estimated_total_landed_outlay_usd:,.2f} USD "
                f"for {float(cargo_tonnes):,.0f} mt."
            )

    provenance = {
        "commodity_source": "World Bank Pink Sheet (Monthly Commodity Price Data)",
        "freight_source": "Model V3 Bounded Freight Forecast",
        "formula": "Landed Cost = Commodity FOB + Ocean Freight",
    }

    return {
        "commodity_fob_usd": round(float(commodity_fob_usd), 2) if commodity_fob_usd is not None else None,
        "ocean_freight_usd_per_tonne": (
            round(float(ocean_freight_usd_per_tonne), 2) if ocean_freight_usd_per_tonne is not None else None
        ),
        "estimated_landed_cost_usd": estimated_landed_cost_usd,
        "commodity_unit": c_unit,
        "freight_unit": f_unit,
        "landed_cost_unit": landed_unit,
        "estimated_total_landed_outlay_usd": estimated_total_landed_outlay_usd,
        "formula": "Landed Cost = Commodity FOB + Ocean Freight",
        "provenance": provenance,
        "reasons": reasons,
    }
