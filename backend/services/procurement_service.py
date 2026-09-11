"""Commodity Procurement Decision Service for NaviFreight.

Loads and evaluates authentic historical commodity benchmark series from:
    data/commodity_prices_worldbank.csv (World Bank Commodity Markets "Pink Sheet")

Provides deterministic, rule-based procurement valuation and signals:
    BUY | MONITOR | WAIT
Zero machine learning models are used for commodity prices.
All thresholds are isolated in PROCUREMENT_CONFIG for governance and auditing.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_ROOT = _THIS_DIR.parent
_REPO_ROOT = _BACKEND_ROOT.parent

# Canonical World Bank commodity dataset path
COMMODITY_CSV_PATH = _REPO_ROOT / "data" / "commodity_prices_worldbank.csv"

# In-memory dataframe cache for high-performance sub-millisecond querying
_COMMODITY_DF: Optional[pd.DataFrame] = None


# -------------------------------------------------------------------------
# PROCUREMENT DECISION CONFIGURATION & SCORING THRESHOLDS
# -------------------------------------------------------------------------
PROCUREMENT_CONFIG = {
    # Valuation (Historical Percentile) Thresholds
    "percentile_low_threshold": 35.0,     # <= 35th percentile: historically attractive entry (BUY)
    "percentile_high_threshold": 75.0,    # >= 75th percentile: historically elevated / cyclical peak (WAIT)
    
    # Momentum (3-Month Price Shift %) Thresholds
    "momentum_softening_threshold": -5.0, # <= -5% shift: downward softening discount (Favorable entry)
    "momentum_heating_threshold": 5.0,    # >= +5% shift: upward inflationary heating (Unfavorable entry)
    
    # Volatility Risk Threshold
    "volatility_high_threshold": 8.0,     # >= 8% monthly standard deviation indicates elevated price volatility
    
    # Scoring Weights (Total scale: -100 to +100)
    "valuation_weight": 40.0,             # Max points from percentile valuation
    "momentum_weight": 30.0,              # Max points from 3-month momentum
    
    # Signal Decision Boundaries
    "buy_score_threshold": 20.0,          # score >= +20 -> BUY
    "wait_score_threshold": -20.0,        # score <= -20 -> WAIT
}


# Mapping of commodity user queries to canonical dataset columns & metadata
COMMODITY_REGISTRY = {
    "coal": {
        "canonical_name": "Coal",
        "column": "coal_price_usd_per_mt",
        "unit": "USD/mt",
        "description": "Australian Coal Benchmark (World Bank Pink Sheet)",
        "aliases": ["coal", "thermal coal", "coking coal", "metallurgical coal"],
    },
    "iron ore": {
        "canonical_name": "Iron Ore",
        "column": "iron_ore_price_usd_per_dmt",
        "unit": "USD/dmt",
        "description": "Iron Ore 62% Fe CFR China Benchmark (World Bank Pink Sheet)",
        "aliases": ["iron ore", "iron_ore", "ironore", "iron"],
    },
}


def load_commodity_data(force_reload: bool = False) -> pd.DataFrame:
    """Load and prepare authentic World Bank commodity dataset with caching."""
    global _COMMODITY_DF
    if _COMMODITY_DF is not None and not force_reload:
        return _COMMODITY_DF

    if not COMMODITY_CSV_PATH.exists():
        raise FileNotFoundError(f"Commodity dataset missing at {COMMODITY_CSV_PATH}")

    df = pd.read_csv(COMMODITY_CSV_PATH)
    
    # Verify required schema
    expected_cols = {"date", "coal_price_usd_per_mt", "iron_ore_price_usd_per_dmt"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Invalid commodity dataset schema: missing {expected_cols - set(df.columns)}"
        )

    # Convert date to datetime and sort chronologically
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_dt").reset_index(drop=True)

    _COMMODITY_DF = df
    return _COMMODITY_DF


def resolve_commodity(commodity_query: str) -> dict[str, Any]:
    """Resolve user query string into canonical commodity metadata."""
    if not commodity_query or not isinstance(commodity_query, str):
        raise ValueError("Commodity name must be a non-empty string.")

    cleaned = commodity_query.strip().lower()
    for key, spec in COMMODITY_REGISTRY.items():
        if cleaned == key or cleaned in spec["aliases"]:
            return spec

    supported = [spec["canonical_name"] for spec in COMMODITY_REGISTRY.values()]
    raise ValueError(
        f"Unsupported commodity '{commodity_query}'. Supported commodities: {', '.join(supported)}"
    )


def calculate_procurement_valuation(
    commodity: str,
    df_override: Optional[pd.DataFrame] = None
) -> dict[str, Any]:
    """Calculate deterministic valuation, momentum, and procurement signal for a commodity."""
    spec = resolve_commodity(commodity)
    col = spec["column"]
    canonical_name = spec["canonical_name"]
    unit = spec["unit"]

    df = df_override if df_override is not None else load_commodity_data()

    # Filter to non-null historical observations for this specific commodity
    valid = df.dropna(subset=[col]).sort_values("date_dt").reset_index(drop=True)
    if len(valid) == 0:
        raise ValueError(f"No valid historical observations found for commodity '{canonical_name}'.")

    # 1. Latest Available Observation
    latest_row = valid.iloc[-1]
    benchmark_price = float(latest_row[col])
    benchmark_date = str(latest_row["date"])
    source_agency = str(latest_row.get("source", "World Bank Commodity Markets (Pink Sheet)"))
    source_release = str(latest_row.get("source_date", "Latest Monthly Bulletin"))

    # 2. Historical Valuation Percentile (relative to all available records <= latest date)
    all_prices = valid[col].values
    percentile = float(stats.percentileofscore(all_prices, benchmark_price))

    # 3. 3-Month Momentum (% change from 3 months prior)
    if len(valid) >= 4:
        price_3m_ago = float(valid.iloc[-4][col])
        momentum_3m_pct = float(((benchmark_price - price_3m_ago) / price_3m_ago) * 100.0)
    else:
        price_3m_ago = benchmark_price
        momentum_3m_pct = 0.0

    # 4. Trailing 3-Month Volatility (Standard deviation of MoM percentage returns)
    if len(valid) >= 4:
        trailing_returns = valid[col].pct_change().iloc[-3:] * 100.0
        volatility_3m_pct = float(trailing_returns.std()) if not np.isnan(trailing_returns.std()) else 0.0
    else:
        volatility_3m_pct = 0.0

    # 5. Deterministic Rule-Based Scoring Engine
    reasons: list[str] = []
    signal_score = 0.0

    # (A) Valuation Score Contribution (Historical Percentile)
    p_low = PROCUREMENT_CONFIG["percentile_low_threshold"]
    p_high = PROCUREMENT_CONFIG["percentile_high_threshold"]
    w_val = PROCUREMENT_CONFIG["valuation_weight"]

    if percentile <= p_low:
        val_score = w_val
        reasons.append(
            f"Favorable Entry Valuation: Benchmark price (${benchmark_price:.2f} {unit}) is at the "
            f"{percentile:.1f}th historical percentile, indicating cyclically attractive pricing."
        )
    elif percentile >= p_high:
        val_score = -w_val
        reasons.append(
            f"Elevated Entry Valuation: Benchmark price (${benchmark_price:.2f} {unit}) is at the "
            f"{percentile:.1f}th historical percentile, near upper cyclical boundaries."
        )
    else:
        # Linear interpolation between +20 (at p_low) and -20 (at p_high)
        norm_pos = (percentile - p_low) / (p_high - p_low)  # 0.0 to 1.0
        val_score = 20.0 - (norm_pos * 40.0)
        reasons.append(
            f"Neutral Valuation: Benchmark price is at the {percentile:.1f}th historical percentile (mid-range)."
        )

    signal_score += val_score

    # (B) Momentum Score Contribution (3-Month Rate of Change)
    m_soft = PROCUREMENT_CONFIG["momentum_softening_threshold"]
    m_heat = PROCUREMENT_CONFIG["momentum_heating_threshold"]
    w_mom = PROCUREMENT_CONFIG["momentum_weight"]

    if momentum_3m_pct <= m_soft:
        signal_score += w_mom
        reasons.append(
            f"Softening Momentum: 3-month price change of {momentum_3m_pct:.2f}% offers procurement discounts."
        )
    elif momentum_3m_pct >= m_heat:
        signal_score -= w_mom
        reasons.append(
            f"Heating Momentum: 3-month price acceleration of +{momentum_3m_pct:.2f}% poses near-term inflation risk."
        )
    else:
        reasons.append(
            f"Stable Momentum: Trailing 3-month price shift is modest ({momentum_3m_pct:+.2f}%)."
        )

    # (C) Volatility Warning
    v_high = PROCUREMENT_CONFIG["volatility_high_threshold"]
    if volatility_3m_pct >= v_high:
        reasons.append(
            f"Volatility Alert: 3-month price volatility is elevated at {volatility_3m_pct:.2f}%; timing sensitivity is high."
        )
    else:
        reasons.append(
            f"Volatility Stable: Recent monthly volatility is contained ({volatility_3m_pct:.2f}%)."
        )

    # (D) Final Decision Classification
    if signal_score >= PROCUREMENT_CONFIG["buy_score_threshold"]:
        signal = "BUY"
    elif signal_score <= PROCUREMENT_CONFIG["wait_score_threshold"]:
        signal = "WAIT"
    else:
        signal = "MONITOR"

    # Data Freshness context
    data_freshness = f"Benchmark as of {benchmark_date} (World Bank Pink Sheet released {source_release})"

    return {
        "commodity": canonical_name,
        "benchmark_price_usd_per_mt": round(benchmark_price, 2),
        "unit": unit,
        "benchmark_date": benchmark_date,
        "percentile": round(percentile, 1),
        "momentum_3m_pct": round(momentum_3m_pct, 2),
        "volatility_3m_pct": round(volatility_3m_pct, 2),
        "signal": signal,
        "signal_score": round(signal_score, 1),
        "reasons": reasons,
        "source": "World Bank Pink Sheet",
        "data_freshness": data_freshness,
    }


def get_procurement_overview() -> dict[str, Any]:
    """Retrieve procurement valuations for all supported commodities in a unified overview."""
    overview = []
    for spec in COMMODITY_REGISTRY.values():
        val = calculate_procurement_valuation(spec["canonical_name"])
        overview.append(val)

    return {
        "commodities": overview,
        "total_supported": len(overview),
        "source": "World Bank Pink Sheet",
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
