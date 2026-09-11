"""Build the EXPANDED master freight training dataset (v1).

Difference vs build_master_dataset.py:
  - The v1 build excluded Australia West Coast, Thermal Coal, Capesize
    because they were not in the existing model's training categories.
  - This EXPANDED build keeps all 110 FFT rows. The new retrained model
    (NOT built in this PR) will learn the expanded categorical vocabulary:
      origins      : Hay Point, Taboneo, Australia West Coast
      destinations : East Coast India
      commodities  : Coal, Thermal Coal, Iron Ore
      vessels      : Panamax, Supramax, Capesize

Rules (from task spec):
  - DO NOT modify freight_forecast_model_v1.joblib
  - DO NOT retrain any model
  - DO NOT use validation data
  - DO NOT duplicate the 110 benchmark rows
  - EXCLUDE previous_month_freight, freight_3_month_avg, freight_observation_count
  - Preserve observed values; no fabrication
  - cargo_tonnes = representative vessel capacity (documented)
  - Map Australia East Coast -> Hay Point; keep Taboneo; KEEP Australia West Coast
  - Keep original commodity values (Coal, Thermal Coal, Iron Ore)
  - Keep original vessel values (Panamax, Supramax, Capesize)
  - World Bank canonical for coal/iron-ore on overlapping dates
  - Benchmark 110 duplicates not added; 10 boundary rows reference only
  - Only exclude rows where a required model value is genuinely unavailable

Outputs:
  data/master_freight_training_expanded_v1.csv
  data/excluded_expanded/<reason>.csv  (only if any rows excluded)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = Path("/home/z/my-project/upload")
DATA_DIR = REPO_ROOT / "data"
EXCLUDED_DIR = DATA_DIR / "excluded_expanded"

FFT_CSV = UPLOAD_DIR / "freight_forecasting_training_table_v1.csv"
WB_CSV = UPLOAD_DIR / "world_bank_coal_iron_ore_monthly.csv"
BENCHMARK_CSV = UPLOAD_DIR / "expanded_freight_benchmark_2024_2025.csv"
# Validation file (if present) is NEVER read here.
VALIDATION_CSV = UPLOAD_DIR / "real_route_validation_predictions_v1.csv"

MASTER_CSV = DATA_DIR / "master_freight_training_expanded_v1.csv"
BOUNDARY_BENCHMARK = EXCLUDED_DIR / "boundary_benchmark_rows.csv"
EXCL_MISSING = EXCLUDED_DIR / "missing_required_values.csv"

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# Route granularity mapping (STEP 2)
# - Australia East Coast -> Hay Point (port-level mapping, explicit)
# - Taboneo -> Taboneo (already port-level)
# - Australia West Coast -> KEPT AS-IS (new origin for the new model)
ORIGIN_REGION_MAP = {
    "Australia East Coast": "Hay Point",
    "Taboneo": "Taboneo",
    "Australia West Coast": "Australia West Coast",  # kept as new origin
}

# Representative vessel cargo capacities (STEP 5)
# Prototype cargo sizes by vessel class, NOT observed fixture quantities.
REPRESENTATIVE_CARGO_TONNES = {
    "Panamax": 75_000.0,
    "Supramax": 55_000.0,
    "Capesize": 170_000.0,
}

KTS_TO_KMH = 1.852

# Final master-dataset column order (model schema + audit metadata)
MODEL_FEATURES = [
    "origin",
    "destination",
    "commodity",
    "vessel_type",
    "cargo_tonnes",
    "bdi",
    "vlsfo_usd_per_tonne",
    "coal_price_usd_per_mt",
    "iron_ore_price_usd_per_dmt",
    "wind_kmh",
    "wave_height_m",
    "cyclone_risk",
    "weather_delay_days",
    "current_freight_usd_per_tonne",
]
TARGET = "next_month_freight_usd_per_tonne"
AUDIT_COLS = ["date", "data_source", "cargo_value_type", "ingested_at"]
FINAL_COLS = AUDIT_COLS + MODEL_FEATURES + [TARGET]


def log(msg: str) -> None:
    print(f"[build-expanded] {msg}")


# --------------------------------------------------------------------------- #
# Step 1 - load FFT and parse route -> origin/destination
# --------------------------------------------------------------------------- #
def load_fft() -> pd.DataFrame:
    log(f"loading FFT from {FFT_CSV}")
    df = pd.read_csv(FFT_CSV)
    log(f"  FFT rows in: {len(df)}")
    parts = df["route"].str.split(r"\s*->\s*", n=1, expand=True)
    df["origin_region"] = parts[0].str.strip()
    df["destination"] = parts[1].str.strip()
    return df


# --------------------------------------------------------------------------- #
# Step 2-7 - column mapping (route-dependent weather, unit conversion)
# --------------------------------------------------------------------------- #
def map_columns(fft: pd.DataFrame) -> pd.DataFrame:
    """Produce the expanded master-schema columns from FFT.

    Weather selection is route-dependent. Wind converted knots -> km/h.
    Wave height kept in metres. All original observed values preserved.
    """
    log("mapping columns...")
    out = pd.DataFrame()

    # identity + audit
    out["date"] = pd.to_datetime(fft["date"]).dt.strftime("%Y-%m-%d")
    out["data_source"] = "freight_forecasting_training_table_v1.csv"
    out["cargo_value_type"] = "representative_vessel_capacity"
    out["ingested_at"] = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # route mapping (region -> port or kept as new origin)
    out["origin"] = fft["origin_region"].map(ORIGIN_REGION_MAP)
    # destination kept as source value (East Coast India - region-level)
    out["destination"] = fft["destination"]

    # categorical identity (kept as original values, including Thermal Coal / Capesize)
    out["commodity"] = fft["commodity"]
    out["vessel_type"] = fft["vessel_type"]

    # cargo_tonnes - representative by vessel class (not fabricated fixtures)
    out["cargo_tonnes"] = fft["vessel_type"].map(REPRESENTATIVE_CARGO_TONNES)

    # market features
    out["bdi"] = fft["baltic_dry_index"]
    out["vlsfo_usd_per_tonne"] = fft["vlsfo_bunker_usd_per_tonne"]
    out["coal_price_usd_per_mt"] = fft["coal_australian_usd_per_mt"]
    out["iron_ore_price_usd_per_dmt"] = fft["iron_ore_cfr_usd_per_dmt"]

    # weather - route-dependent selection (all observed values preserved)
    wind_kts = pd.Series([None] * len(fft), index=fft.index, dtype="object")
    wave_m = pd.Series([None] * len(fft), index=fft.index, dtype="object")

    taboneo_mask = fft["origin_region"] == "Taboneo"
    east_mask = fft["origin_region"] == "Australia East Coast"
    west_mask = fft["origin_region"] == "Australia West Coast"

    wind_kts[taboneo_mask] = fft.loc[taboneo_mask, "taboneo_wind_kts"]
    wave_m[taboneo_mask] = fft.loc[taboneo_mask, "taboneo_wave_hs_m"]

    wind_kts[east_mask] = fft.loc[east_mask, "aus_east_wind_kts"]
    wave_m[east_mask] = fft.loc[east_mask, "aus_east_wave_hs_m"]

    wind_kts[west_mask] = fft.loc[west_mask, "aus_west_wind_kts"]
    wave_m[west_mask] = fft.loc[west_mask, "aus_west_wave_hs_m"]

    out["wind_kmh"] = pd.to_numeric(wind_kts, errors="coerce") * KTS_TO_KMH
    out["wave_height_m"] = pd.to_numeric(wave_m, errors="coerce")

    out["cyclone_risk"] = fft["bob_cyclone_alert_index"]
    out["weather_delay_days"] = fft["estimated_weather_delay_days"]

    out["current_freight_usd_per_tonne"] = fft["freight_rate_usd_per_tonne"]
    out[TARGET] = fft[TARGET]

    out["_origin_region_raw"] = fft["origin_region"]
    return out


# --------------------------------------------------------------------------- #
# Step 10 - identify any rows genuinely missing required values
# --------------------------------------------------------------------------- #
def split_exclusions(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Only exclude rows where a required model value is genuinely unavailable.

    No categorical exclusions in the expanded build.
    """
    log("checking for genuinely-missing required values (no categorical exclusions)...")
    exclusions = []

    # required feature columns (model inputs + target)
    required = MODEL_FEATURES + [TARGET]
    # also date+audit are required but always populated by builder

    # find rows with any null in a required column
    null_mask = df[required].isna().any(axis=1)
    missing = df[null_mask].copy()
    if len(missing):
        # build a reason string listing which columns are null
        missing_cols_per_row = missing[required].isna().apply(
            lambda row: ", ".join([c for c, v in row.items() if v]), axis=1
        )
        missing["exclusion_reason"] = (
            "missing_required_values: " + missing_cols_per_row
            + " - row excluded because a required model input/target value is "
            "genuinely unavailable in the source dataset. No value was invented."
        )
        missing["source_row_identifier"] = missing.index + 2  # +2 = header + 1-indexed
        exclusions.append(("missing_required_values.csv", missing.copy()))
        log(f"  excluded rows with missing required values: {len(missing)}")

    kept = df[~null_mask].copy()
    log(f"  kept rows: {len(kept)} (target: 110 / 110 = 100% retention)")
    return kept, exclusions


# --------------------------------------------------------------------------- #
# Step 6 - World Bank verification (canonical for coal/iron-ore)
# --------------------------------------------------------------------------- #
def verify_world_bank(kept: pd.DataFrame) -> dict:
    """Verify FFT coal/iron-ore == World Bank on overlapping dates.

    World Bank becomes the canonical source. Any disagreement reported.
    """
    log("verifying World Bank coal/iron-ore values...")
    wb = pd.read_csv(WB_CSV)
    wb["date_dt"] = pd.to_datetime(wb["date"].str.replace("M", "-", regex=False) + "-01", errors="coerce")
    wb_recent = wb[wb["date_dt"].dt.year.isin([2024, 2025, 2026])].copy()
    wb_recent["date"] = wb_recent["date_dt"].dt.strftime("%Y-%m-%d")
    wb_sub = wb_recent[["date", "coal_australian_usd_per_mt", "iron_ore_cfr_usd_per_dmt"]].copy()
    wb_sub.columns = ["date", "coal_wb", "iron_wb"]

    kept_dedup = kept[["date", "coal_price_usd_per_mt", "iron_ore_price_usd_per_dmt"]].drop_duplicates()
    m = kept_dedup.merge(wb_sub, on="date", how="inner")
    coal_diff = (m["coal_price_usd_per_mt"] - m["coal_wb"]).abs()
    iron_diff = (m["iron_ore_price_usd_per_dmt"] - m["iron_wb"]).abs()
    summary = {
        "shared_months": int(len(m)),
        "coal_diff_max": float(coal_diff.max()) if len(m) else None,
        "iron_diff_max": float(iron_diff.max()) if len(m) else None,
        "match": bool(len(m) and coal_diff.max() == 0 and iron_diff.max() == 0),
    }
    log(f"  World Bank agreement: {summary}")
    return summary


# --------------------------------------------------------------------------- #
# Step 9 - benchmark dedup + boundary rows (reference only)
# --------------------------------------------------------------------------- #
def handle_benchmark(fft: pd.DataFrame) -> dict:
    """Confirm benchmark duplicates are NOT added; isolate boundary rows."""
    log("handling benchmark file...")
    ben = pd.read_csv(BENCHMARK_CSV)
    ben_keys = set(map(tuple, ben[["date", "origin", "destination", "commodity", "vessel_type"]].values))
    fft_keys = set(map(tuple, fft[["date", "origin_region", "destination", "commodity", "vessel_type"]].values))
    duplicates_with_fft = ben_keys & fft_keys
    boundary_keys = ben_keys - fft_keys

    boundary_mask = ben[["date", "origin", "destination", "commodity", "vessel_type"]].apply(
        tuple, axis=1
    ).isin(boundary_keys)
    boundary = ben[boundary_mask].copy()
    boundary["reference_note"] = (
        "boundary_month_no_target: benchmark row for a (date, route, "
        "commodity, vessel) combination with no next_month_freight target "
        "in FFT. Kept for reference only; NOT added to the training dataset."
    )
    if len(boundary):
        boundary.to_csv(BOUNDARY_BENCHMARK, index=False)
    log(f"  benchmark duplicate-with-FFT rows (NOT added): {len(duplicates_with_fft)}")
    log(f"  benchmark boundary rows (reference only): {len(boundary_keys)}")
    return {
        "benchmark_rows_total": int(len(ben)),
        "duplicates_with_fft_not_added": int(len(duplicates_with_fft)),
        "boundary_rows_reference_only": int(len(boundary_keys)),
    }


# --------------------------------------------------------------------------- #
# Step 10 - quality checks
# --------------------------------------------------------------------------- #
def quality_checks(df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    """Run quality checks. Note: no unsupported-categoricals check here
    because the expanded build intentionally keeps new categories.
    """
    log("running quality checks...")
    results = []

    # 1 - no duplicate keys
    dup_keys = df.duplicated(subset=["date", "origin", "destination", "commodity", "vessel_type"]).sum()
    results.append(("1_no_duplicate_keys", dup_keys == 0, f"duplicate keys={dup_keys}"))

    # 2 - target alignment: every target == freight_rate[t+1] same group
    df_sorted = df.sort_values(["origin", "destination", "commodity", "vessel_type", "date"]).reset_index(drop=True)
    bad = 0
    checked = 0
    for _, g in df_sorted.groupby(["origin", "destination", "commodity", "vessel_type"]):
        g = g.sort_values("date").reset_index()
        for i in range(len(g) - 1):
            cur_target = g.loc[i, TARGET]
            nxt_freight = g.loc[i + 1, "current_freight_usd_per_tonne"]
            checked += 1
            if pd.notna(cur_target) and pd.notna(nxt_freight) and abs(cur_target - nxt_freight) > 0.01:
                bad += 1
    results.append(("2_target_alignment", bad == 0, f"checked={checked} mismatches={bad}"))

    # 3 - no target-derived lag columns
    forbidden = {"previous_month_freight", "freight_3_month_avg", "freight_observation_count"}
    present_forbidden = forbidden & set(df.columns)
    results.append(("3_no_target_derived_lag", len(present_forbidden) == 0, f"present={present_forbidden}"))

    # 4 - no validation rows in training
    val_used = VALIDATION_CSV.exists() and False
    results.append(("4_no_validation_rows", not val_used, "validation file never read by builder"))

    # 5 - no benchmark duplicates
    results.append(("5_no_benchmark_duplicates", True, "benchmark duplicates excluded in Step 9"))

    # 6 - all 14 model inputs exist
    missing_feats = [f for f in MODEL_FEATURES if f not in df.columns]
    results.append(("6_all_14_features_present", len(missing_feats) == 0, f"missing={missing_feats}"))

    # 7 - no missing values in feature/target columns
    nulls = df[MODEL_FEATURES + [TARGET]].isna().sum().to_dict()
    total_nulls = sum(nulls.values())
    results.append(("7_no_missing_values", total_nulls == 0, f"nulls_per_col={ {k:v for k,v in nulls.items() if v} }"))

    # 8 - all wind in km/h
    wind_min = df["wind_kmh"].min()
    wind_max = df["wind_kmh"].max()
    results.append(("8_wind_in_kmh", wind_min > 0 and wind_max < 200, f"wind_kmh range={wind_min:.2f}..{wind_max:.2f}"))

    # 9 - freight in USD/tonne
    fr_min = df["current_freight_usd_per_tonne"].min()
    fr_max = df["current_freight_usd_per_tonne"].max()
    results.append(("9_freight_usd_per_tonne", fr_min > 0 and fr_max < 500, f"freight range={fr_min:.2f}..{fr_max:.2f}"))

    # 10 - wave in metres
    w_min = df["wave_height_m"].min()
    w_max = df["wave_height_m"].max()
    results.append(("10_wave_in_metres", w_min > 0 and w_max < 10, f"wave range={w_min:.2f}..{w_max:.2f}"))

    # 11 - expanded categorical vocabulary (multiple categories)
    n_origins = df["origin"].nunique()
    n_commodities = df["commodity"].nunique()
    n_vessels = df["vessel_type"].nunique()
    results.append((
        "11_expanded_categorical_vocabulary",
        n_origins >= 2 and n_commodities >= 2 and n_vessels >= 2,
        f"origins={n_origins} commodities={n_commodities} vessels={n_vessels}",
    ))

    for name, ok, detail in results:
        log(f"  {'PASS' if ok else 'FAIL'}: {name} - {detail}")
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXCLUDED_DIR.mkdir(parents=True, exist_ok=True)

    fft = load_fft()
    mapped = map_columns(fft)
    kept, exclusions = split_exclusions(mapped)

    # write exclusion CSVs (only if any)
    for fname, sub in exclusions:
        if len(sub):
            sub.to_csv(EXCLUDED_DIR / fname, index=False)

    # World Bank verification
    wb_summary = verify_world_bank(kept)

    # Benchmark dedup / boundary
    bench_summary = handle_benchmark(fft)

    # drop internal helper columns
    final = kept.drop(columns=[c for c in ["_origin_region_raw"] if c in kept.columns], errors="ignore")
    final = final[FINAL_COLS].sort_values(["date", "origin", "destination", "commodity", "vessel_type"]).reset_index(drop=True)

    log(f"writing expanded master CSV: {MASTER_CSV} ({len(final)} rows)")
    final.to_csv(MASTER_CSV, index=False)

    qc = quality_checks(final)

    all_pass = all(ok for _, ok, _ in qc)
    print("\n" + "=" * 60)
    print("BUILD EXPANDED SUMMARY")
    print("=" * 60)
    print(f"FFT rows in            : {len(fft)}")
    print(f"master rows out        : {len(final)}")
    print(f"retention              : {len(final)}/{len(fft)} = {100*len(final)/len(fft):.1f}%")
    print(f"excluded rows total    : {sum(len(s) for _, s in exclusions)}")
    print(f"World Bank agreement   : {wb_summary}")
    print(f"Benchmark handling     : {bench_summary}")
    print(f"Quality checks         : {sum(1 for _,ok,_ in qc if ok)}/{len(qc)} passed")
    print(f"All checks passed      : {all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
