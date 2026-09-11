"""Analytics Service: exposes genuine historical market intelligence, trends, and correlations.

Source of Truth:
    data/master_freight_training_expanded_v1.csv (110 genuine observations, 2024-02-01 to 2025-11-01).

Zero synthetic data is used.
Zero values are fabricated.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_ROOT = _THIS_DIR.parent
_REPO_ROOT = _BACKEND_ROOT.parent

if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from data import database
from services.forecast_service import ForecastDataError

# Canonical historical dataset path
HISTORICAL_CSV_PATH = _REPO_ROOT / "data" / "master_freight_training_expanded_v1.csv"

# In-memory dataframe cache for sub-millisecond query performance
_HISTORICAL_DF: Optional[pd.DataFrame] = None


def _load_historical_data() -> pd.DataFrame:
    """Load and prepare genuine historical dataset with caching."""
    global _HISTORICAL_DF
    if _HISTORICAL_DF is not None:
        return _HISTORICAL_DF

    if not HISTORICAL_CSV_PATH.exists():
        raise FileNotFoundError(f"Historical dataset missing at {HISTORICAL_CSV_PATH}")

    df = pd.read_csv(HISTORICAL_CSV_PATH)
    if "date" not in df.columns or "current_freight_usd_per_tonne" not in df.columns:
        raise ValueError("Invalid historical dataset format: missing required columns.")

    # Guard against synthetic data
    if "synthetic" in str(HISTORICAL_CSV_PATH).lower():
        raise RuntimeError("CRITICAL: Attempted to load quarantined synthetic dataset!")

    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values(
        ["origin", "destination", "commodity", "vessel_type", "date_dt"]
    ).reset_index(drop=True)

    # Compute previous month freight, MoM changes, and 3-month rolling averages
    route_keys = ["origin", "destination", "commodity", "vessel_type"]
    df["prev_freight"] = df.groupby(route_keys)["current_freight_usd_per_tonne"].shift(1)
    df["mom_change_usd"] = df["current_freight_usd_per_tonne"] - df["prev_freight"]
    df["mom_change_percent"] = (df["mom_change_usd"] / df["prev_freight"]) * 100.0
    df["rolling_3m_avg"] = df.groupby(route_keys)[
        "current_freight_usd_per_tonne"
    ].transform(lambda s: s.rolling(3, min_periods=1).mean())

    _HISTORICAL_DF = df
    return _HISTORICAL_DF


def _get_provenance(total_records: int) -> dict[str, Any]:
    """Standard provenance metadata dictionary."""
    df = _load_historical_data()
    return {
        "data_source": HISTORICAL_CSV_PATH.name,
        "data_type": "historical",
        "date_range": {
            "start": str(df["date"].min()),
            "end": str(df["date"].max()),
        },
        "total_records": total_records,
    }


def get_freight_trends(
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    commodity: Optional[str] = None,
    vessel_type: Optional[str] = None,
) -> dict[str, Any]:
    """Retrieve chronological freight rate trends with optional filtering."""
    df = _load_historical_data()
    filtered = df.copy()

    filters_applied = {
        "origin": origin,
        "destination": destination,
        "commodity": commodity,
        "vessel_type": vessel_type,
    }

    # Validation of filter inputs
    valid_origins = set(df["origin"].unique())
    valid_destinations = set(df["destination"].unique())
    valid_commodities = set(df["commodity"].unique())
    valid_vessels = set(df["vessel_type"].unique()) | {"Handysize"}

    if origin and origin not in valid_origins:
        raise ForecastDataError(
            error_code="INVALID_FILTER",
            message=f"Unknown origin '{origin}'.",
            detail=f"Valid origins: {', '.join(sorted(valid_origins))}",
        )
    if destination and destination not in valid_destinations:
        raise ForecastDataError(
            error_code="INVALID_FILTER",
            message=f"Unknown destination '{destination}'.",
            detail=f"Valid destinations: {', '.join(sorted(valid_destinations))}",
        )
    if commodity and commodity not in valid_commodities:
        raise ForecastDataError(
            error_code="INVALID_FILTER",
            message=f"Unknown commodity '{commodity}'.",
            detail=f"Valid commodities: {', '.join(sorted(valid_commodities))}",
        )
    if vessel_type and vessel_type not in valid_vessels:
        raise ForecastDataError(
            error_code="INVALID_FILTER",
            message=f"Unknown vessel type '{vessel_type}'.",
            detail=f"Valid vessel types: {', '.join(sorted(valid_vessels))}",
        )

    if origin:
        filtered = filtered[filtered["origin"] == origin]
    if destination:
        filtered = filtered[filtered["destination"] == destination]
    if commodity:
        filtered = filtered[filtered["commodity"] == commodity]
    if vessel_type:
        filtered = filtered[filtered["vessel_type"] == vessel_type]

    filtered = filtered.sort_values(
        ["date_dt", "origin", "commodity", "vessel_type"]
    ).reset_index(drop=True)

    series = []
    for _, row in filtered.iterrows():
        series.append({
            "date": str(row["date"]),
            "origin": str(row["origin"]),
            "destination": str(row["destination"]),
            "commodity": str(row["commodity"]),
            "vessel_type": str(row["vessel_type"]),
            "freight_rate_usd_per_tonne": round(float(row["current_freight_usd_per_tonne"]), 2),
            "previous_month_freight": round(float(row["prev_freight"]), 2)
            if pd.notna(row["prev_freight"])
            else None,
            "mom_change_usd": round(float(row["mom_change_usd"]), 2)
            if pd.notna(row["mom_change_usd"])
            else None,
            "mom_change_percent": round(float(row["mom_change_percent"]), 2)
            if pd.notna(row["mom_change_percent"])
            else None,
            "rolling_3m_avg": round(float(row["rolling_3m_avg"]), 2)
            if pd.notna(row["rolling_3m_avg"])
            else None,
        })

    return {
        "provenance": _get_provenance(total_records=len(series)),
        "filters_applied": filters_applied,
        "series": series,
    }


def get_market_trends() -> dict[str, Any]:
    """Retrieve chronological macroeconomic and bunker benchmark trends (deduplicated by month)."""
    df = _load_historical_data()

    grouped = (
        df.groupby("date")
        .agg(
            bdi=("bdi", "first"),
            vlsfo_usd_per_tonne=("vlsfo_usd_per_tonne", "first"),
            coal_price_usd_per_mt=("coal_price_usd_per_mt", "first"),
            iron_ore_price_usd_per_dmt=("iron_ore_price_usd_per_dmt", "first"),
            average_freight_usd_per_tonne=("current_freight_usd_per_tonne", "mean"),
            min_freight_usd_per_tonne=("current_freight_usd_per_tonne", "min"),
            max_freight_usd_per_tonne=("current_freight_usd_per_tonne", "max"),
        )
        .reset_index()
        .sort_values("date")
    )

    series = []
    for _, row in grouped.iterrows():
        series.append({
            "date": str(row["date"]),
            "bdi": round(float(row["bdi"]), 1),
            "vlsfo_usd_per_tonne": round(float(row["vlsfo_usd_per_tonne"]), 2),
            "coal_price_usd_per_mt": round(float(row["coal_price_usd_per_mt"]), 2),
            "iron_ore_price_usd_per_dmt": round(float(row["iron_ore_price_usd_per_dmt"]), 2),
            "average_freight_usd_per_tonne": round(
                float(row["average_freight_usd_per_tonne"]), 2
            ),
            "min_freight_usd_per_tonne": round(float(row["min_freight_usd_per_tonne"]), 2),
            "max_freight_usd_per_tonne": round(float(row["max_freight_usd_per_tonne"]), 2),
        })

    return {
        "provenance": _get_provenance(total_records=len(series)),
        "series": series,
    }


def get_weather_trends(origin: Optional[str] = None) -> dict[str, Any]:
    """Retrieve historical weather trends by loading port/region."""
    df = _load_historical_data()
    valid_origins = set(df["origin"].unique())

    if origin and origin not in valid_origins:
        raise ForecastDataError(
            error_code="INVALID_FILTER",
            message=f"Unknown origin port '{origin}'.",
            detail=f"Valid origin ports: {', '.join(sorted(valid_origins))}",
        )

    filtered = df.copy()
    if origin:
        filtered = filtered[filtered["origin"] == origin]

    # Deduplicate by (date, origin) to ensure clean port observations
    grouped = (
        filtered.groupby(["date", "origin"])
        .agg(
            wind_kmh=("wind_kmh", "first"),
            wave_height_m=("wave_height_m", "first"),
            cyclone_risk=("cyclone_risk", "first"),
            weather_delay_days=("weather_delay_days", "first"),
        )
        .reset_index()
        .sort_values(["origin", "date"])
    )

    series = []
    for _, row in grouped.iterrows():
        series.append({
            "date": str(row["date"]),
            "origin": str(row["origin"]),
            "wind_kmh": round(float(row["wind_kmh"]), 2),
            "wave_height_m": round(float(row["wave_height_m"]), 2),
            "cyclone_risk": round(float(row["cyclone_risk"]), 1),
            "weather_delay_days": round(float(row["weather_delay_days"]), 2),
        })

    return {
        "provenance": _get_provenance(total_records=len(series)),
        "origin_filter": origin,
        "series": series,
    }


def _classify_trend(sub_df: pd.DataFrame) -> str:
    """Classify 3-month recent trajectory against prior 3-month window."""
    if len(sub_df) < 3:
        return "STABLE"
    recent_3m = sub_df["current_freight_usd_per_tonne"].iloc[-3:].mean()
    prior_3m = (
        sub_df["current_freight_usd_per_tonne"].iloc[-6:-3].mean()
        if len(sub_df) >= 6
        else sub_df["current_freight_usd_per_tonne"].iloc[:-3].mean()
    )
    diff_pct = (recent_3m - prior_3m) / prior_3m * 100.0
    if diff_pct >= 2.5:
        return "RISING"
    elif diff_pct <= -2.5:
        return "FALLING"
    return "STABLE"


def get_routes_analytics() -> dict[str, Any]:
    """Retrieve comprehensive summary statistics for each canonical trade lane."""
    df = _load_historical_data()
    routes_list = []

    for (o, d, c, v), g in df.groupby(["origin", "destination", "commodity", "vessel_type"]):
        g_sorted = g.sort_values("date_dt")
        latest_row = g_sorted.iloc[-1]
        prev_row = g_sorted.iloc[-2] if len(g_sorted) > 1 else latest_row

        mom_chg = (
            latest_row["current_freight_usd_per_tonne"] - prev_row["current_freight_usd_per_tonne"]
        )
        mom_pct = (
            (mom_chg / prev_row["current_freight_usd_per_tonne"]) * 100.0
            if prev_row["current_freight_usd_per_tonne"] > 0
            else 0.0
        )
        trend = _classify_trend(g_sorted)

        routes_list.append({
            "origin": o,
            "destination": d,
            "commodity": c,
            "vessel_type": v,
            "observation_count": len(g_sorted),
            "first_date": str(g_sorted["date"].min()),
            "last_date": str(g_sorted["date"].max()),
            "average_freight": round(float(g_sorted["current_freight_usd_per_tonne"].mean()), 2),
            "minimum_freight": round(float(g_sorted["current_freight_usd_per_tonne"].min()), 2),
            "maximum_freight": round(float(g_sorted["current_freight_usd_per_tonne"].max()), 2),
            "latest_freight": round(float(latest_row["current_freight_usd_per_tonne"]), 2),
            "latest_monthly_change": round(float(mom_chg), 2),
            "latest_monthly_change_percent": round(float(mom_pct), 2),
            "trend": trend,
        })

    # Sort routes by average freight descending
    routes_list.sort(key=lambda r: r["average_freight"], reverse=True)

    return {
        "provenance": _get_provenance(total_records=len(routes_list)),
        "routes": routes_list,
    }


def get_executive_summary() -> dict[str, Any]:
    """Return high-level executive snapshot of latest market conditions and recent momentum."""
    df = _load_historical_data()
    latest_date = str(df["date"].max())
    latest_rows = df[df["date"] == latest_date]

    # Market Macro state
    first_row = latest_rows.iloc[0]
    market_state = {
        "bdi": round(float(first_row["bdi"]), 1),
        "vlsfo_usd_per_tonne": round(float(first_row["vlsfo_usd_per_tonne"]), 2),
        "coal_price_usd_per_mt": round(float(first_row["coal_price_usd_per_mt"]), 2),
        "iron_ore_price_usd_per_dmt": round(float(first_row["iron_ore_price_usd_per_dmt"]), 2),
        "average_freight_usd_per_tonne": round(
            float(latest_rows["current_freight_usd_per_tonne"].mean()), 2
        ),
    }

    # Macro trend evaluation (3-month rolling average vs prior 3 months)
    monthly_avg = (
        df.groupby("date")["current_freight_usd_per_tonne"]
        .mean()
        .reset_index()
        .sort_values("date")
    )
    recent_3m = monthly_avg["current_freight_usd_per_tonne"].iloc[-3:].mean()
    prior_3m = monthly_avg["current_freight_usd_per_tonne"].iloc[-6:-3].mean()
    shift_pct = (recent_3m - prior_3m) / prior_3m * 100.0

    if shift_pct >= 2.5:
        overall_trend = "RISING"
    elif shift_pct <= -2.5:
        overall_trend = "FALLING"
    else:
        overall_trend = "STABLE"

    trend_info = {
        "freight_trend_classification": overall_trend,
        "methodology": "3-month rolling average comparison vs preceding 3-month window (threshold: +/-2.5%).",
        "recent_3m_avg_freight": round(float(recent_3m), 2),
        "prior_3m_avg_freight": round(float(prior_3m), 2),
        "shift_percent": round(float(shift_pct), 2),
    }

    # Route MoM movers
    routes_res = get_routes_analytics()["routes"]
    sorted_by_mom = sorted(routes_res, key=lambda r: r["latest_monthly_change_percent"], reverse=True)
    strongest_pos = sorted_by_mom[0] if sorted_by_mom else None
    strongest_neg = sorted_by_mom[-1] if sorted_by_mom else None

    return {
        "provenance": _get_provenance(total_records=len(df)),
        "latest_date": latest_date,
        "market_state": market_state,
        "recent_trends": trend_info,
        "strongest_positive_mover": strongest_pos,
        "strongest_negative_mover": strongest_neg,
        "tracked_routes_count": len(routes_res),
    }


def get_correlations() -> dict[str, Any]:
    """Calculate Pearson correlations between freight rate and market/weather drivers."""
    df = _load_historical_data()
    corrs = []

    numeric_drivers = [
        ("bdi", "Baltic Dry Index (BDI)"),
        ("vlsfo_usd_per_tonne", "VLSFO Bunker Price"),
        ("coal_price_usd_per_mt", "Coal Benchmark Price"),
        ("iron_ore_price_usd_per_dmt", "Iron Ore Benchmark Price"),
        ("wind_kmh", "Wind Speed"),
        ("wave_height_m", "Significant Wave Height"),
        ("cyclone_risk", "Cyclone Risk Score"),
        ("weather_delay_days", "Weather Delay Estimate"),
    ]

    for col, label in numeric_drivers:
        r, p = stats.pearsonr(df["current_freight_usd_per_tonne"], df[col])
        if r > 0.05:
            rel = "positive"
            interp = f"{label} and freight showed a positive historical co-movement (r = {r:.3f}, p = {p:.4f})."
        elif r < -0.05:
            rel = "negative"
            interp = f"{label} and freight showed an inverse historical co-movement (r = {r:.3f}, p = {p:.4f})."
        else:
            rel = "neutral"
            interp = f"{label} showed negligible linear correlation with freight (r = {r:.3f}, p = {p:.4f})."

        corrs.append({
            "feature": col,
            "feature_label": label,
            "correlation": round(float(r), 4),
            "p_value": round(float(p), 6),
            "sample_count": len(df),
            "relationship": rel,
            "interpretation": interp,
        })

    # Sort correlations by absolute magnitude descending
    corrs.sort(key=lambda c: abs(c["correlation"]), reverse=True)

    return {
        "provenance": _get_provenance(total_records=len(df)),
        "target_variable": "current_freight_usd_per_tonne",
        "correlations": corrs,
        "disclaimer": (
            "HISTORICAL CORRELATION (non-causal). Pearson correlation indicates linear association "
            "within genuine historical training observations (N=110, 2024-02 to 2025-11). "
            "It does not establish a causal relationship."
        ),
    }


def get_latest_market_snapshot() -> dict[str, Any]:
    """Provide dashboard snapshot combining latest historical state with runtime database status."""
    df = _load_historical_data()
    latest_date = str(df["date"].max())
    summary = get_executive_summary()
    routes = get_routes_analytics()["routes"]
    weather = get_weather_trends()["series"]
    latest_weather = [w for w in weather if w["date"] == latest_date]

    db_status = database.get_data_status()

    return {
        "provenance": _get_provenance(total_records=len(df)),
        "latest_historical_date": latest_date,
        "market_macro": summary["market_state"],
        "weather_by_origin": latest_weather,
        "freight_by_route": routes,
        "live_database_cache_status": db_status.model_dump()
        if hasattr(db_status, "model_dump")
        else db_status,
    }
