"""Generate IMPROVED synthetic training extension (v2).

Improvement over v1:
  v1 generated freight via block-bootstrap pct-change perturbation that was
  INDEPENDENT of the bootstrapped market values. This weakened the
  freight<->market correlation (max diff 0.41 for BDI<->freight) and let
  the freight range drift to 5.3-28.8 (orig 8.5-21.2).

  v2 uses a REGRESSION + RESIDUAL BOOTSTRAP approach:
    1. Per combination, fit a linear regression on the original 22 rows:
         current_freight ~ bdi + vlsfo + coal + iron_ore + wind + wave
                           + cyclone_risk + weather_delay
    2. Compute empirical residuals: residuals = y - y_pred
    3. For each synthetic month:
       a. Bootstrap market + weather values from the original block
          (preserves market<->market correlations, same as v1).
       b. Predict base freight via the regression using the bootstrapped
          market/weather values -> THIS COUPLES FREIGHT TO MARKET.
       c. Add a residual bootstrap (random draw from empirical residuals).
       d. Blend with previous month's freight (AR(1) term) for temporal
          persistence.
       e. Clip to combination-specific empirical range (orig min/max + small
          margin) to prevent drift.

This preserves:
  - market<->market correlations (block bootstrap, inherited jointly)
  - freight<->market correlations (regression couples them)
  - temporal persistence (AR(1) blend)
  - forward-shifted target (target[t] = freight[t+1] within trajectory)
  - combination-specific freight range (clipping)

Strict rules:
  - DO NOT modify freight_forecast_model_v1.joblib
  - DO NOT modify freight_forecast_model_v2.joblib (if present)
  - DO NOT overwrite synthetic_v1
  - DO NOT discard the original 110 observations
  - DO NOT retrain
  - DO NOT modify FastAPI backend
  - DO NOT use target-derived columns as features
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent
ORIGINAL_CSV = REPO_ROOT / "data" / "master_freight_training_expanded_v1.csv"
V1_SYNTHETIC_CSV = REPO_ROOT / "data" / "master_freight_training_synthetic_v1.csv"
STATS_JSON = REPO_ROOT / "data" / "synthetic_generation_statistics_v2.json"
V2_SYNTHETIC_CSV = REPO_ROOT / "data" / "master_freight_training_synthetic_v2.csv"
VALIDATION_REPORT = REPO_ROOT / "data" / "synthetic_validation_v2.json"

MODEL_V1 = REPO_ROOT / "freight_forecast_model_v1.joblib"
MODEL_V2 = REPO_ROOT / "freight_forecast_model_v2.joblib"

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
N_SYNTHETIC = 1000
RANDOM_SEED = 42
TRAJECTORY_LENGTH = 50
MAX_TRAJ_ATTEMPTS = 20

COMBINATIONS = [
    ("Australia West Coast", "East Coast India", "Iron Ore", "Capesize"),
    ("Hay Point", "East Coast India", "Coal", "Capesize"),
    ("Hay Point", "East Coast India", "Coal", "Panamax"),
    ("Taboneo", "East Coast India", "Thermal Coal", "Panamax"),
    ("Taboneo", "East Coast India", "Thermal Coal", "Supramax"),
]

CARGO_BY_VESSEL = {"Panamax": 75_000.0, "Supramax": 55_000.0, "Capesize": 170_000.0}
SYNTH_START_DATE = pd.Timestamp("2025-12-01")

# Model features (14) + target
MODEL_FEATURES = [
    "origin", "destination", "commodity", "vessel_type",
    "cargo_tonnes", "bdi", "vlsfo_usd_per_tonne",
    "coal_price_usd_per_mt", "iron_ore_price_usd_per_dmt",
    "wind_kmh", "wave_height_m", "cyclone_risk", "weather_delay_days",
    "current_freight_usd_per_tonne",
]
TARGET = "next_month_freight_usd_per_tonne"

# Market + weather features used as REGRESSION PREDICTORS for freight
FREIGHT_PREDICTORS = [
    "bdi", "vlsfo_usd_per_tonne", "coal_price_usd_per_mt",
    "iron_ore_price_usd_per_dmt", "wind_kmh", "wave_height_m",
    "cyclone_risk", "weather_delay_days",
]

# AR(1) blend factor: alpha = weight on regression prediction, (1-alpha) = weight on previous freight
# alpha = 0.7 means freight is mostly driven by market conditions, with some persistence
AR_ALPHA = 0.7

# Conservative freight clipping: combination-specific observed range + 15% margin
# This prevents the v1 drift (5.3-28.8) while allowing modest variation
FREIGHT_RANGE_MARGIN = 0.15

# Hard outlier bounds (same as v1, used for trajectory-level rejection)
BOUNDS = {
    "current_freight_usd_per_tonne": (5.0, 30.0),
    "bdi": (500.0, 3000.0),
    "vlsfo_usd_per_tonne": (300.0, 900.0),
    "coal_price_usd_per_mt": (50.0, 200.0),
    "iron_ore_price_usd_per_dmt": (50.0, 200.0),
    "wind_kmh": (10.0, 80.0),
    "wave_height_m": (0.5, 6.0),
    "cyclone_risk": (0.0, 5.0),
    "weather_delay_days": (0.0, 5.0),
}


def log(msg: str) -> None:
    print(f"[synth-v2] {msg}")


# --------------------------------------------------------------------------- #
# STEP 1 - analyze original relationships (per-combo regression + correlations)
# --------------------------------------------------------------------------- #
def analyze_original(df: pd.DataFrame) -> dict:
    """Fit per-combination regression of freight on market+weather,
    capture residuals, and compute correlations."""
    log("STEP 1: analyzing original relationships (regression + correlations)...")

    combo_models = {}
    global_corr = df[FREIGHT_PREDICTORS + ["current_freight_usd_per_tonne"]].corr().round(4)

    for combo in COMBINATIONS:
        origin, dest, comm, vessel = combo
        sub = df[(df.origin == origin) & (df.destination == dest)
                 & (df.commodity == comm) & (df.vessel_type == vessel)].sort_values("date").reset_index(drop=True)
        key = f"{origin}|{dest}|{comm}|{vessel}"

        X = sub[FREIGHT_PREDICTORS].values
        y = sub["current_freight_usd_per_tonne"].values

        # Ridge regression with high alpha for stability on 22 rows
        # (avoids overfitting coefficients to noise)
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred

        combo_models[key] = {
            "model": model,
            "residuals": residuals,
            "coefficients": dict(zip(FREIGHT_PREDICTORS, model.coef_.round(4))),
            "intercept": float(model.intercept_),
            "freight_observed_min": float(sub["current_freight_usd_per_tonne"].min()),
            "freight_observed_max": float(sub["current_freight_usd_per_tonne"].max()),
            "freight_observed_mean": float(sub["current_freight_usd_per_tonne"].mean()),
            "freight_observed_std": float(sub["current_freight_usd_per_tonne"].std()),
            "n_rows": int(len(sub)),
            # Conservative clip range: observed min/max + 15% margin
            "freight_clip_min": float(sub["current_freight_usd_per_tonne"].min() - FREIGHT_RANGE_MARGIN * (sub["current_freight_usd_per_tonne"].max() - sub["current_freight_usd_per_tonne"].min())),
            "freight_clip_max": float(sub["current_freight_usd_per_tonne"].max() + FREIGHT_RANGE_MARGIN * (sub["current_freight_usd_per_tonne"].max() - sub["current_freight_usd_per_tonne"].min())),
        }
        log(f"  {key}: coef={dict(zip(FREIGHT_PREDICTORS, model.coef_.round(3)))}")
        log(f"    intercept={model.intercept_:.3f}  resid_std={residuals.std():.3f}  clip=[{combo_models[key]['freight_clip_min']:.2f}, {combo_models[key]['freight_clip_max']:.2f}]")

    stats = {
        "per_combination_models": {k: {kk: vv for kk, vv in v.items() if kk != "model" and kk != "residuals"} for k, v in combo_models.items()},
        "global_correlation": global_corr.where(global_corr.notna(), None).to_dict(),
    }
    STATS_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATS_JSON.write_text(json.dumps(stats, indent=2, default=str))
    log(f"  wrote stats to {STATS_JSON}")
    return combo_models


# --------------------------------------------------------------------------- #
# STEP 5-8 - IMPROVED synthetic generation (regression + residual bootstrap)
# --------------------------------------------------------------------------- #
def generate_synthetic_v2(original: pd.DataFrame, combo_models: dict) -> pd.DataFrame:
    """Generate N_SYNTHETIC synthetic rows using regression + residual bootstrap.

    Per trajectory:
      1. Seed month-0 with a bootstrapped original row.
      2. For each subsequent month:
         a. Bootstrap a block of 1-3 original rows (preserves market<->market
            correlations because values are inherited jointly).
         b. Predict base freight via the per-combo regression using the
            bootstrapped market/weather values -> COUPLES freight to market.
         c. Add a residual bootstrap (random draw from empirical residuals).
         d. Blend with previous month's freight (AR(1)) for persistence.
         e. Clip to combination-specific empirical range + margin.
      3. Target = forward-shifted freight within trajectory.
      4. Trajectory-level outlier rejection.
    """
    log(f"STEP 5-8: generating {N_SYNTHETIC} synthetic rows via regression + residual bootstrap...")
    rng = np.random.default_rng(RANDOM_SEED)

    target_rows_per_combo = N_SYNTHETIC // 5  # 200
    all_synth_rows: List[dict] = []

    for combo_idx, combo in enumerate(COMBINATIONS, 1):
        origin, dest, comm, vessel = combo
        sub = original[
            (original.origin == origin) & (original.destination == dest)
            & (original.commodity == comm) & (original.vessel_type == vessel)
        ].sort_values("date").reset_index(drop=True)
        n_orig = len(sub)
        key = f"{origin}|{dest}|{comm}|{vessel}"
        model_info = combo_models[key]
        model = model_info["model"]
        residuals = model_info["residuals"]
        clip_min = model_info["freight_clip_min"]
        clip_max = model_info["freight_clip_max"]
        cargo_t = CARGO_BY_VESSEL[vessel]

        combo_rows = 0
        traj_attempt = 0
        traj_accepted = 0
        while combo_rows < target_rows_per_combo and traj_attempt < MAX_TRAJ_ATTEMPTS:
            traj_attempt += 1
            traj_id = f"SYN{combo_idx:02d}{traj_attempt:02d}"

            # Seed month-0 with a bootstrapped original row
            start = sub.iloc[rng.integers(0, n_orig)]
            prev_freight = float(start["current_freight_usd_per_tonne"])

            trajectory_rows: List[dict] = []
            for m in range(TRAJECTORY_LENGTH):
                # Block bootstrap: draw 1-3 consecutive original rows
                # (preserves joint distribution of market+weather)
                block_len = int(rng.integers(1, 4))
                block_start = int(rng.integers(0, max(1, n_orig - block_len + 1)))
                block = sub.iloc[block_start:block_start + block_len]
                rep = block.iloc[-1]  # representative row (last in block)

                # Market + weather values (inherited from bootstrapped block)
                market_weather = {
                    "bdi": float(rep["bdi"]),
                    "vlsfo_usd_per_tonne": float(rep["vlsfo_usd_per_tonne"]),
                    "coal_price_usd_per_mt": float(rep["coal_price_usd_per_mt"]),
                    "iron_ore_price_usd_per_dmt": float(rep["iron_ore_price_usd_per_dmt"]),
                    "wind_kmh": float(rep["wind_kmh"]),
                    "wave_height_m": float(rep["wave_height_m"]),
                    "cyclone_risk": float(rep["cyclone_risk"]),
                    "weather_delay_days": float(rep["weather_delay_days"]),
                }

                if m == 0:
                    # First month: use the bootstrapped starting row's freight
                    new_freight = prev_freight
                else:
                    # REGRESSION PREDICTION: couple freight to market/weather
                    X_row = np.array([[market_weather[f] for f in FREIGHT_PREDICTORS]])
                    base_freight = float(model.predict(X_row)[0])

                    # RESIDUAL BOOTSTRAP: add a random draw from empirical residuals
                    residual = float(rng.choice(residuals))

                    # AR(1) BLEND: combine regression prediction with previous freight
                    # for temporal persistence
                    reg_pred = base_freight + residual
                    new_freight = AR_ALPHA * reg_pred + (1.0 - AR_ALPHA) * prev_freight

                    # CLIP to combination-specific empirical range + margin
                    new_freight = float(np.clip(new_freight, clip_min, clip_max))

                # Synthetic date: SYNTH_START_DATE + m months (+ traj_attempt offset
                # so different trajectories of the same combo don't collide)
                synth_date = (SYNTH_START_DATE + pd.DateOffset(months=m + (traj_attempt * TRAJECTORY_LENGTH))).strftime("%Y-%m-%d")

                row = {
                    "date": synth_date,
                    "origin": origin,
                    "destination": dest,
                    "commodity": comm,
                    "vessel_type": vessel,
                    "cargo_tonnes": cargo_t,
                    **market_weather,
                    "current_freight_usd_per_tonne": float(new_freight),
                    "trajectory_id": traj_id,
                    "data_origin": "synthetic",
                    "synthetic_generation_method": "regression_residual_bootstrap_v2",
                }
                trajectory_rows.append(row)
                prev_freight = float(new_freight)

            # STEP 9 - target = forward-shifted freight within trajectory
            for i in range(len(trajectory_rows) - 1):
                trajectory_rows[i][TARGET] = float(trajectory_rows[i + 1]["current_freight_usd_per_tonne"])
            # Drop the last row of each trajectory (no successor -> no target)
            trajectory_rows = trajectory_rows[:-1]

            # STEP 12 - trajectory-level outlier rejection
            if trajectory_is_clean(trajectory_rows):
                all_synth_rows.extend(trajectory_rows)
                combo_rows += len(trajectory_rows)
                traj_accepted += 1
            else:
                log(f"    rejected trajectory {traj_id} (pathological value)")
        log(f"  combo {combo_idx} ({origin} -> {dest}, {comm}, {vessel}): accepted {traj_accepted} trajectories, {combo_rows} rows")

    synth_df = pd.DataFrame(all_synth_rows)
    log(f"  generated {len(synth_df)} synthetic rows (target {N_SYNTHETIC})")

    # Trim to exactly N_SYNTHETIC (same logic as v1: drop whole trajectories or
    # tail rows + dependent row, preserving target alignment)
    trim_pass = 0
    while len(synth_df) > N_SYNTHETIC:
        trim_pass += 1
        excess = len(synth_df) - N_SYNTHETIC
        last_traj = synth_df.iloc[-1]["trajectory_id"]
        last_traj_df = synth_df[synth_df["trajectory_id"] == last_traj]
        if len(last_traj_df) <= excess:
            synth_df = synth_df[synth_df["trajectory_id"] != last_traj].reset_index(drop=True)
            continue
        drop_idx = list(synth_df.tail(excess).index)
        first_dropped_freight = synth_df.loc[drop_idx[0], "current_freight_usd_per_tonne"]
        for idx in range(drop_idx[0] - 1, -1, -1):
            if synth_df.loc[idx, "trajectory_id"] == last_traj:
                if abs(synth_df.loc[idx, TARGET] - first_dropped_freight) < 0.001:
                    drop_idx.insert(0, idx)
                    break
            else:
                break
        synth_df = synth_df.drop(index=drop_idx).reset_index(drop=True)
    log(f"  trimmed to exactly {len(synth_df)} rows (after {trim_pass} pass(es))")

    # If we under-shot, regenerate a small batch
    if len(synth_df) < N_SYNTHETIC:
        deficit = N_SYNTHETIC - len(synth_df)
        log(f"  regenerating {deficit} extra rows...")
        extra = generate_extra_trajectories(original, combo_models, deficit, rng, start_combo_idx=50)
        if len(extra) > deficit:
            extra = extra.head(deficit).copy()
        synth_df = pd.concat([synth_df, extra], ignore_index=True)
        log(f"  final rows: {len(synth_df)}")

    return synth_df


def trajectory_is_clean(rows: List[dict]) -> bool:
    """Return True if all rows in the trajectory are within bounds."""
    for row in rows:
        for col, (lo, hi) in BOUNDS.items():
            v = row.get(col)
            if v is None:
                return False
            try:
                if not (lo <= float(v) <= hi):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def generate_extra_trajectories(original, combo_models, n_needed, rng, start_combo_idx=1):
    """Generate extra trajectories to fill a small deficit."""
    extra_rows = []
    trajectories_per_combo = max(1, (n_needed // 5) // (TRAJECTORY_LENGTH - 1) + 1)
    for offset, combo in enumerate(COMBINATIONS):
        combo_idx = start_combo_idx + offset
        origin, dest, comm, vessel = combo
        sub = original[(original.origin == origin) & (original.destination == dest)
                       & (original.commodity == comm) & (original.vessel_type == vessel)].sort_values("date").reset_index(drop=True)
        n_orig = len(sub)
        if n_orig < 4:
            continue
        key = f"{origin}|{dest}|{comm}|{vessel}"
        model_info = combo_models[key]
        model = model_info["model"]
        residuals = model_info["residuals"]
        clip_min = model_info["freight_clip_min"]
        clip_max = model_info["freight_clip_max"]
        cargo_t = CARGO_BY_VESSEL[vessel]

        traj_attempt = 0
        accepted = 0
        while accepted < trajectories_per_combo and traj_attempt < 6:
            traj_attempt += 1
            traj_id = f"SYNX{combo_idx:02d}{traj_attempt:02d}"
            start = sub.iloc[rng.integers(0, n_orig)]
            prev_freight = float(start["current_freight_usd_per_tonne"])
            trajectory = []
            for m in range(TRAJECTORY_LENGTH):
                block_len = int(rng.integers(1, 4))
                block_start = int(rng.integers(0, max(1, n_orig - block_len + 1)))
                block = sub.iloc[block_start:block_start + block_len]
                rep = block.iloc[-1]
                market_weather = {
                    "bdi": float(rep["bdi"]), "vlsfo_usd_per_tonne": float(rep["vlsfo_usd_per_tonne"]),
                    "coal_price_usd_per_mt": float(rep["coal_price_usd_per_mt"]),
                    "iron_ore_price_usd_per_dmt": float(rep["iron_ore_price_usd_per_dmt"]),
                    "wind_kmh": float(rep["wind_kmh"]), "wave_height_m": float(rep["wave_height_m"]),
                    "cyclone_risk": float(rep["cyclone_risk"]),
                    "weather_delay_days": float(rep["weather_delay_days"]),
                }
                if m == 0:
                    new_freight = prev_freight
                else:
                    X_row = np.array([[market_weather[f] for f in FREIGHT_PREDICTORS]])
                    base_freight = float(model.predict(X_row)[0])
                    residual = float(rng.choice(residuals))
                    reg_pred = base_freight + residual
                    new_freight = AR_ALPHA * reg_pred + (1.0 - AR_ALPHA) * prev_freight
                    new_freight = float(np.clip(new_freight, clip_min, clip_max))
                synth_date = (SYNTH_START_DATE + pd.DateOffset(months=m + 500 + traj_attempt * TRAJECTORY_LENGTH)).strftime("%Y-%m-%d")
                row = {
                    "date": synth_date, "origin": origin, "destination": dest,
                    "commodity": comm, "vessel_type": vessel, "cargo_tonnes": cargo_t,
                    **market_weather,
                    "current_freight_usd_per_tonne": float(new_freight),
                    "trajectory_id": traj_id, "data_origin": "synthetic",
                    "synthetic_generation_method": "regression_residual_bootstrap_v2",
                }
                trajectory.append(row)
                prev_freight = float(new_freight)
            for i in range(len(trajectory) - 1):
                trajectory[i][TARGET] = float(trajectory[i + 1]["current_freight_usd_per_tonne"])
            trows = trajectory[:-1]
            if trajectory_is_clean(trows):
                extra_rows.extend(trows)
                accepted += 1
    return pd.DataFrame(extra_rows)


# --------------------------------------------------------------------------- #
# STEP 14 - validation (vs original AND vs v1)
# --------------------------------------------------------------------------- #
def validate(original: pd.DataFrame, synth: pd.DataFrame, v1_synth: pd.DataFrame) -> dict:
    log("STEP 14: validating v2 vs original AND vs v1...")
    numeric_features = FREIGHT_PREDICTORS + ["current_freight_usd_per_tonne", TARGET]

    def describe(s):
        s = s.dropna()
        return {"mean": float(s.mean()), "median": float(s.median()),
                "std": float(s.std()), "min": float(s.min()), "max": float(s.max())}

    comparison = {}
    for f in numeric_features:
        o = original[f].dropna()
        s = synth[f].dropna() if f in synth.columns else pd.Series(dtype=float)
        v1 = v1_synth[f].dropna() if (v1_synth is not None and f in v1_synth.columns) else pd.Series(dtype=float)
        comparison[f] = {
            "original": describe(o),
            "synthetic_v2": describe(s) if len(s) else None,
            "synthetic_v1": describe(v1) if len(v1) else None,
        }

    # Correlation matrices
    orig_corr = original[numeric_features].corr().round(4)
    v2_corr = synth[numeric_features].corr().round(4)
    v1_corr = v1_synth[numeric_features].corr().round(4) if v1_synth is not None else None

    # Specific correlations of interest
    def safe_corr(df, a, b):
        if a in df.columns and b in df.columns:
            c = df[a].corr(df[b])
            return float(c) if pd.notna(c) else None
        return None

    corr_focus = {}
    pairs = [
        ("current_freight_usd_per_tonne", "bdi"),
        ("current_freight_usd_per_tonne", "vlsfo_usd_per_tonne"),
        ("current_freight_usd_per_tonne", "coal_price_usd_per_mt"),
        ("current_freight_usd_per_tonne", "iron_ore_price_usd_per_dmt"),
        ("current_freight_usd_per_tonne", "wind_kmh"),
        ("current_freight_usd_per_tonne", "wave_height_m"),
        ("current_freight_usd_per_tonne", "cyclone_risk"),
        ("current_freight_usd_per_tonne", "weather_delay_days"),
        ("current_freight_usd_per_tonne", TARGET),
    ]
    for a, b in pairs:
        corr_focus[f"{a}__{b}"] = {
            "original": safe_corr(original, a, b),
            "synthetic_v2": safe_corr(synth, a, b),
            "synthetic_v1": safe_corr(v1_synth, a, b) if v1_synth is not None else None,
        }

    # Freight pct change distributions (per combo)
    def pct_dist(df, label):
        out = {}
        for combo in COMBINATIONS:
            origin, dest, comm, vessel = combo
            sub = df[(df.origin == origin) & (df.destination == dest)
                     & (df.commodity == comm) & (df.vessel_type == vessel)]
            if "trajectory_id" in df.columns and label != "original":
                pcts = []
                for tid, tsub in sub.groupby("trajectory_id"):
                    tsub = tsub.sort_values("date")
                    p = tsub["current_freight_usd_per_tonne"].pct_change().dropna() * 100
                    pcts.extend(p.tolist())
                pcts = pd.Series(pcts)
            else:
                sub = sub.sort_values("date")
                pcts = sub["current_freight_usd_per_tonne"].pct_change().dropna() * 100
            out["|".join(combo)] = {
                "mean": float(pcts.mean()) if len(pcts) else 0,
                "std": float(pcts.std()) if len(pcts) else 0,
                "min": float(pcts.min()) if len(pcts) else 0,
                "max": float(pcts.max()) if len(pcts) else 0,
                "count": int(len(pcts)),
            }
        return out

    pct_compare = {
        "original": pct_dist(original, "original"),
        "synthetic_v2": pct_dist(synth, "synthetic_v2"),
        "synthetic_v1": pct_dist(v1_synth, "synthetic_v1") if v1_synth is not None else None,
    }

    # Correlation matrix difference metrics
    common = [c for c in orig_corr.columns if c in v2_corr.columns]
    diff_v2 = (orig_corr.loc[common, common] - v2_corr.loc[common, common]).abs()
    diff_v1 = (orig_corr.loc[common, common] - v1_corr.loc[common, common]).abs() if v1_corr is not None else None

    matrix_diff = {
        "v2_vs_original": {
            "mean_abs_diff": float(np.nanmean(diff_v2.values)),
            "max_abs_diff": float(np.nanmax(diff_v2.values)),
        },
        "v1_vs_original": {
            "mean_abs_diff": float(np.nanmean(diff_v1.values)) if diff_v1 is not None else None,
            "max_abs_diff": float(np.nanmax(diff_v1.values)) if diff_v1 is not None else None,
        } if diff_v1 is not None else None,
    }

    return {
        "feature_distributions": comparison,
        "correlation_focus": corr_focus,
        "freight_pct_change_comparison": pct_compare,
        "matrix_diff": matrix_diff,
        "original_correlation_matrix": orig_corr.where(orig_corr.notna(), None).to_dict(),
        "synthetic_v2_correlation_matrix": v2_corr.where(v2_corr.notna(), None).to_dict(),
        "synthetic_v1_correlation_matrix": v1_corr.where(v1_corr.notna(), None).to_dict() if v1_corr is not None else None,
    }


# --------------------------------------------------------------------------- #
# STEP 15 - final QC
# --------------------------------------------------------------------------- #
def final_qc(original: pd.DataFrame, synth: pd.DataFrame, combined: pd.DataFrame) -> dict:
    log("STEP 15: final QC checks...")
    qc = {}

    qc["original_rows"] = int(len(original))
    qc["synthetic_rows"] = int(len(synth))
    qc["total_rows"] = int(len(combined))
    qc["expected_total"] = 1110
    qc["rows_match"] = qc["total_rows"] == qc["expected_total"]

    combos_present = combined.groupby(["origin", "destination", "commodity", "vessel_type"]).size()
    qc["combinations_present"] = int(len(combos_present))
    qc["all_5_combinations_present"] = len(combos_present) == 5

    cols_to_check = MODEL_FEATURES + [TARGET]
    nulls = combined[cols_to_check].isna().sum().to_dict()
    qc["nulls_per_col"] = {k: int(v) for k, v in nulls.items()}
    qc["no_missing_values"] = all(v == 0 for v in nulls.values())

    check_df = combined.copy()
    check_df["_traj_key"] = check_df["trajectory_id"].fillna("ORIGINAL")
    dup_keys = check_df.duplicated(subset=["date", "origin", "destination", "commodity", "vessel_type", "_traj_key"]).sum()
    qc["duplicate_keys"] = int(dup_keys)
    qc["no_duplicate_keys"] = dup_keys == 0

    qc["model_features_present"] = all(f in combined.columns for f in MODEL_FEATURES)
    qc["target_present"] = TARGET in combined.columns

    # Original 110 rows unchanged
    orig_in_combined = combined[combined["data_origin"] == "original"].copy()
    compare_cols = ["date", "origin", "destination", "commodity", "vessel_type",
                    "cargo_tonnes", "bdi", "vlsfo_usd_per_tonne", "coal_price_usd_per_mt",
                    "iron_ore_price_usd_per_dmt", "wind_kmh", "wave_height_m",
                    "cyclone_risk", "weather_delay_days", "current_freight_usd_per_tonne", TARGET]
    orig_sorted = original[compare_cols].copy()
    orig_sorted["date"] = pd.to_datetime(orig_sorted["date"]).dt.strftime("%Y-%m-%d")
    orig_sorted = orig_sorted.sort_values(compare_cols[:5]).reset_index(drop=True)
    comb_sorted = orig_in_combined[compare_cols].sort_values(compare_cols[:5]).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(orig_sorted, comb_sorted, check_dtype=False, check_exact=False, rtol=1e-9)
        qc["original_rows_unchanged"] = True
    except AssertionError as e:
        qc["original_rows_unchanged"] = False
        qc["original_rows_diff"] = str(e)[:500]

    # Synthetic target alignment
    bad_target = 0
    checked_target = 0
    for tid, tsub in synth.groupby("trajectory_id"):
        tsub = tsub.sort_values("date").reset_index(drop=True)
        for i in range(len(tsub) - 1):
            checked_target += 1
            cur_target = tsub.loc[i, TARGET]
            nxt_freight = tsub.loc[i + 1, "current_freight_usd_per_tonne"]
            if pd.notna(cur_target) and pd.notna(nxt_freight) and abs(cur_target - nxt_freight) > 0.001:
                bad_target += 1
    qc["synthetic_target_alignment_checked"] = checked_target
    qc["synthetic_target_alignment_mismatches"] = bad_target
    qc["synthetic_target_alignment_ok"] = bad_target == 0

    # Outlier checks
    qc["outlier_checks"] = {}
    for col, (lo, hi) in BOUNDS.items():
        if col in combined.columns:
            bad = ((combined[col] < lo) | (combined[col] > hi)).sum()
            qc["outlier_checks"][col] = {"bad_count": int(bad), "bounds": [lo, hi]}

    # Leakage columns
    leakage_cols = {"previous_month_freight", "freight_3_month_avg", "freight_observation_count"}
    present_leakage = leakage_cols & set(combined.columns)
    qc["no_leakage_columns_in_dataset"] = len(present_leakage) == 0
    qc["leakage_columns_present"] = list(present_leakage)

    qc["all_passed"] = (
        qc["rows_match"]
        and qc["all_5_combinations_present"]
        and qc["no_missing_values"]
        and qc["no_duplicate_keys"]
        and qc["model_features_present"]
        and qc["target_present"]
        and qc["original_rows_unchanged"]
        and qc["synthetic_target_alignment_ok"]
        and qc["no_leakage_columns_in_dataset"]
    )

    log(f"  original_rows            : {qc['original_rows']}")
    log(f"  synthetic_rows           : {qc['synthetic_rows']}")
    log(f"  total_rows               : {qc['total_rows']} (expected {qc['expected_total']})")
    log(f"  rows_match               : {qc['rows_match']}")
    log(f"  all_5_combinations       : {qc['all_5_combinations_present']}")
    log(f"  no_missing_values        : {qc['no_missing_values']}")
    log(f"  no_duplicate_keys        : {qc['no_duplicate_keys']}")
    log(f"  model_features_present   : {qc['model_features_present']}")
    log(f"  target_present           : {qc['target_present']}")
    log(f"  original_rows_unchanged   : {qc['original_rows_unchanged']}")
    log(f"  synth target alignment   : {qc['synthetic_target_alignment_ok']} ({qc['synthetic_target_alignment_checked']} checked, {qc['synthetic_target_alignment_mismatches']} mismatches)")
    log(f"  no_leakage_columns       : {qc['no_leakage_columns_in_dataset']}")
    log(f"  ALL QC PASSED            : {qc['all_passed']}")
    return qc


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    log("=" * 60)
    log("SYNTHETIC TRAINING EXTENSION V2 (regression + residual bootstrap)")
    log("=" * 60)

    # Verify v1 model untouched
    if MODEL_V1.exists():
        v1_sha = hashlib.sha256(MODEL_V1.read_bytes()).hexdigest()
        log(f"v1 model sha256 (baseline): {v1_sha}")
        assert v1_sha == "695fafe3f31b560d5a4412124c0839e0e622c9d2bd090191a5e02eaef6c3819a", "v1 model changed!"
    v2_sha_before = None
    if MODEL_V2.exists():
        v2_sha_before = hashlib.sha256(MODEL_V2.read_bytes()).hexdigest()
        log(f"v2 model sha256 (baseline): {v2_sha_before}")

    # Load original
    log(f"loading original master: {ORIGINAL_CSV}")
    original = pd.read_csv(ORIGINAL_CSV)
    original["date"] = pd.to_datetime(original["date"])
    log(f"  original rows: {len(original)}")

    # Load v1 synthetic for comparison (if exists)
    v1_synth = None
    if V1_SYNTHETIC_CSV.exists():
        log(f"loading v1 synthetic for comparison: {V1_SYNTHETIC_CSV}")
        v1_synth = pd.read_csv(V1_SYNTHETIC_CSV)
        v1_synth = v1_synth[v1_synth["data_origin"] == "synthetic"].copy()
        log(f"  v1 synthetic rows: {len(v1_synth)}")
    else:
        log(f"  v1 synthetic not found at {V1_SYNTHETIC_CSV} - will compare against original only")

    # STEP 1 - analyze original
    combo_models = analyze_original(original)

    # STEP 5-8 - generate v2 synthetic
    synth = generate_synthetic_v2(original, combo_models)
    log(f"  final synthetic rows: {len(synth)}")

    # Combine: original + synthetic (drop v1 audit cols from original)
    original_combined = original.copy()
    original_combined["trajectory_id"] = None
    original_combined["data_origin"] = "original"
    original_combined["synthetic_generation_method"] = "original_observation"
    for col in ["data_source", "cargo_value_type", "ingested_at"]:
        if col in original_combined.columns:
            original_combined = original_combined.drop(columns=[col])

    target_cols = list(original_combined.columns)
    synth = synth[target_cols]

    combined = pd.concat([original_combined, synth], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
    combined = combined.sort_values(["data_origin", "trajectory_id", "date", "origin", "destination", "commodity", "vessel_type"]).reset_index(drop=True)
    log(f"  combined rows: {len(combined)} (expected 1110)")

    # STEP 14 - validation
    validation = validate(original, synth, v1_synth)

    # STEP 15 - final QC
    qc = final_qc(original_combined, synth, combined)

    # Write outputs
    log(f"writing {V2_SYNTHETIC_CSV}")
    combined.to_csv(V2_SYNTHETIC_CSV, index=False)

    VALIDATION_REPORT.write_text(json.dumps({
        "validation": validation,
        "qc": qc,
    }, indent=2, default=str))
    log(f"wrote validation report to {VALIDATION_REPORT}")

    # Verify models untouched
    if MODEL_V1.exists():
        v1_sha_after = hashlib.sha256(MODEL_V1.read_bytes()).hexdigest()
        assert v1_sha_after == "695fafe3f31b560d5a4412124c0839e0e622c9d2bd090191a5e02eaef6c3819a", "v1 model changed!"
        log("v1 model UNTOUCHED ✅")
    if MODEL_V2.exists():
        v2_sha_after = hashlib.sha256(MODEL_V2.read_bytes()).hexdigest()
        assert v2_sha_after == v2_sha_before, "v2 model changed!"
        log("v2 model UNTOUCHED ✅")

    print("\n" + "=" * 60)
    print("SYNTHETIC V2 GENERATION SUMMARY")
    print("=" * 60)
    print(f"original rows           : {qc['original_rows']}")
    print(f"synthetic rows          : {qc['synthetic_rows']}")
    print(f"total rows              : {qc['total_rows']} (expected 1110)")
    print(f"all QC passed           : {qc['all_passed']}")
    # Print key correlation improvements
    if "correlation_focus" in validation:
        log("\nKey correlation improvements (v2 vs v1 vs original):")
        for pair, corrs in validation["correlation_focus"].items():
            o = corrs["original"]; v2 = corrs["synthetic_v2"]; v1 = corrs["synthetic_v1"]
            v2_diff = abs(v2 - o) if (o is not None and v2 is not None) else None
            v1_diff = abs(v1 - o) if (o is not None and v1 is not None) else None
            improvement = ""
            if v2_diff is not None and v1_diff is not None:
                improvement = f"  (v2 {'BETTER' if v2_diff < v1_diff else 'WORSE' if v2_diff > v1_diff else 'SAME'} than v1)"
            print(f"  {pair[:60]:<60}")
            print(f"    orig={o:.4f}  v2={v2:.4f}  v1={v1:.4f if v1 else 'N/A'}{improvement}")
    return 0 if qc["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
