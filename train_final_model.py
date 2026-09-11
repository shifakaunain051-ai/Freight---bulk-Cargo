"""Train the FINAL hackathon model on the approved synthetic v2 dataset.

Trains 4 ML candidates + 2 baselines on data/master_freight_training_synthetic_v2.csv
using a trajectory-aware temporal split. Saves the best ML model as
freight_forecast_model_final.joblib (separate from v1/v2).

Strict rules:
  - DO NOT modify freight_forecast_model_v1.joblib
  - DO NOT modify freight_forecast_model_v2.joblib (if present)
  - DO NOT modify FastAPI backend
  - DO NOT use validation data
  - DO NOT expose data_origin/trajectory_id/synthetic_generation_method as features
  - DO NOT include cargo_tonnes (per spec - representative, not observed)
  - Save final model as freight_forecast_model_final.joblib (new file)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent
DATASET_CSV = REPO_ROOT / "data" / "master_freight_training_synthetic_v2.csv"
MODEL_FINAL_PATH = REPO_ROOT / "freight_forecast_model_final.joblib"
MODEL_V1_PATH = REPO_ROOT / "freight_forecast_model_v1.joblib"
MODEL_V2_PATH = REPO_ROOT / "freight_forecast_model_v2.joblib"
PREDICTIONS_CSV = REPO_ROOT / "data" / "final_model_predictions.csv"
METRICS_JSON = REPO_ROOT / "data" / "final_model_metrics.json"

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# Per spec: 13 input features (NO cargo_tonnes)
MODEL_FEATURES = [
    "origin", "destination", "commodity", "vessel_type",
    "bdi", "vlsfo_usd_per_tonne", "coal_price_usd_per_mt",
    "iron_ore_price_usd_per_dmt", "wind_kmh", "wave_height_m",
    "cyclone_risk", "weather_delay_days", "current_freight_usd_per_tonne",
]
CATEGORICAL = ["origin", "destination", "commodity", "vessel_type"]
TARGET = "next_month_freight_usd_per_tonne"
RANDOM_STATE = 42

# Test split: latest 20% of dated observations, trajectory-aware
TEST_FRACTION = 0.20

# 5 combinations (for smoke test)
COMBINATIONS = [
    ("Australia West Coast", "East Coast India", "Iron Ore", "Capesize"),
    ("Hay Point", "East Coast India", "Coal", "Capesize"),
    ("Hay Point", "East Coast India", "Coal", "Panamax"),
    ("Taboneo", "East Coast India", "Thermal Coal", "Panamax"),
    ("Taboneo", "East Coast India", "Thermal Coal", "Supramax"),
]


def log(msg: str) -> None:
    print(f"[train-final] {msg}")


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------- #
# STEP 1 - load + inspect
# --------------------------------------------------------------------------- #
def load_and_inspect() -> pd.DataFrame:
    log(f"loading {DATASET_CSV}")
    df = pd.read_csv(DATASET_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "origin", "destination", "commodity", "vessel_type"]).reset_index(drop=True)

    log(f"  total rows: {len(df)}")
    log(f"  original rows: {(df.data_origin == 'original').sum()}")
    log(f"  synthetic rows: {(df.data_origin == 'synthetic').sum()}")
    log(f"  combinations: {df.groupby(['origin','destination','commodity','vessel_type']).ngroups}")
    log(f"  date range: {df.date.min().date()} -> {df.date.max().date()}")
    log(f"  missing values (in feature+target cols): {df[MODEL_FEATURES + [TARGET]].isna().sum().sum()}")
    check_keys = df.copy()
    check_keys["trajectory_id"] = check_keys["trajectory_id"].fillna("ORIGINAL")
    dup = check_keys.duplicated(subset=["date", "origin", "destination", "commodity", "vessel_type", "data_origin", "trajectory_id"]).sum()
    log(f"  duplicate keys (date+route+vessel+origin+traj): {dup}")
    log(f"  data_origin distribution: {df.data_origin.value_counts().to_dict()}")
    return df


# --------------------------------------------------------------------------- #
# STEP 3 - trajectory-aware temporal split
# --------------------------------------------------------------------------- #
def trajectory_aware_temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the latest 20% of dated observations for testing.

    Trajectory-aware: for synthetic trajectories, the split date is applied
    within each trajectory so we don't split a trajectory such that future
    rows leak into training. We pick a global split date = the 80th
    percentile of all dates, then ensure no trajectory straddles the
    boundary (trajectories that start before but extend past the split
    date are kept ENTIRELY in test if their majority is in test, else
    entirely in train).

    Implementation: simpler and safer approach - for each trajectory (and
    each original combo), find the temporal cut that puts the latest 20%
    of THAT trajectory's rows into test. Aggregate across trajectories.
    """
    log("STEP 3: trajectory-aware temporal split (latest 20% for test)...")

    train_parts = []
    test_parts = []

    # Original observations: group by combo (each combo is its own "trajectory")
    orig = df[df.data_origin == "original"].copy()
    for combo, sub in orig.groupby(["origin", "destination", "commodity", "vessel_type"]):
        sub = sub.sort_values("date").reset_index(drop=True)
        n = len(sub)
        n_test = max(1, int(round(n * TEST_FRACTION)))
        train_parts.append(sub.iloc[:n - n_test])
        test_parts.append(sub.iloc[n - n_test:])

    # Synthetic observations: group by trajectory_id
    synth = df[df.data_origin == "synthetic"].copy()
    for tid, sub in synth.groupby("trajectory_id"):
        sub = sub.sort_values("date").reset_index(drop=True)
        n = len(sub)
        n_test = max(1, int(round(n * TEST_FRACTION)))
        train_parts.append(sub.iloc[:n - n_test])
        test_parts.append(sub.iloc[n - n_test:])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    log(f"  train rows: {len(train)}  (original={(train.data_origin=='original').sum()}, synthetic={(train.data_origin=='synthetic').sum()})")
    log(f"  test  rows: {len(test)}   (original={(test.data_origin=='original').sum()}, synthetic={(test.data_origin=='synthetic').sum()})")
    log(f"  train date range: {train.date.min().date()} -> {train.date.max().date()}")
    log(f"  test  date range: {test.date.min().date()} -> {test.date.max().date()}")
    return train, test


# --------------------------------------------------------------------------- #
# STEP 4 - baselines
# --------------------------------------------------------------------------- #
def persistence_baseline(train, test) -> dict:
    log("STEP 4A: persistence baseline (predict = current_freight)")
    res = {
        "train": metrics(train[TARGET].values, train["current_freight_usd_per_tonne"].values),
        "test": metrics(test[TARGET].values, test["current_freight_usd_per_tonne"].values),
    }
    log(f"  test: MAE={res['test']['mae']:.4f} RMSE={res['test']['rmse']:.4f} R2={res['test']['r2']:.4f}")
    return res


def linear_baseline(train, test) -> dict:
    log("STEP 4B: linear regression (target ~ current_freight + bdi + vlsfo)")
    feats = ["current_freight_usd_per_tonne", "bdi", "vlsfo_usd_per_tonne"]
    model = LinearRegression()
    model.fit(train[feats], train[TARGET])
    res = {
        "train": metrics(train[TARGET].values, model.predict(train[feats])),
        "test": metrics(test[TARGET].values, model.predict(test[feats])),
    }
    log(f"  test: MAE={res['test']['mae']:.4f} RMSE={res['test']['rmse']:.4f} R2={res['test']['r2']:.4f}")
    return res


# --------------------------------------------------------------------------- #
# STEP 5-6 - ML candidates
# --------------------------------------------------------------------------- #
def build_pipeline(estimator, features: list) -> Pipeline:
    cat_cols = [c for c in CATEGORICAL if c in features]
    num_cols = [c for c in features if c not in cat_cols]
    prep = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("prep", prep), ("model", estimator)])


def train_rf(train, test) -> dict:
    log("STEP 5 MODEL 1: RandomForest (n=300, depth=8, leaf=2)")
    rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2,
                               random_state=RANDOM_STATE, n_jobs=-1)
    pipe = build_pipeline(rf, MODEL_FEATURES)
    pipe.fit(train[MODEL_FEATURES], train[TARGET])
    res = {
        "train": metrics(train[TARGET].values, pipe.predict(train[MODEL_FEATURES])),
        "test": metrics(test[TARGET].values, pipe.predict(test[MODEL_FEATURES])),
        "pipeline": pipe,
    }
    log(f"  train: MAE={res['train']['mae']:.4f} R2={res['train']['r2']:.4f}")
    log(f"  test : MAE={res['test']['mae']:.4f} RMSE={res['test']['rmse']:.4f} R2={res['test']['r2']:.4f}")
    return res


def train_gb(train, test) -> dict:
    log("STEP 5 MODEL 2: GradientBoosting (n=150, depth=2, lr=0.05, leaf=5)")
    gb = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                    min_samples_leaf=5, random_state=RANDOM_STATE)
    pipe = build_pipeline(gb, MODEL_FEATURES)
    pipe.fit(train[MODEL_FEATURES], train[TARGET])
    res = {
        "train": metrics(train[TARGET].values, pipe.predict(train[MODEL_FEATURES])),
        "test": metrics(test[TARGET].values, pipe.predict(test[MODEL_FEATURES])),
        "pipeline": pipe,
    }
    log(f"  train: MAE={res['train']['mae']:.4f} R2={res['train']['r2']:.4f}")
    log(f"  test : MAE={res['test']['mae']:.4f} RMSE={res['test']['rmse']:.4f} R2={res['test']['r2']:.4f}")
    return res


def train_histgb(train, test) -> dict:
    log("STEP 5 MODEL 3: HistGradientBoosting (conservative)")
    # HistGB needs the OneHot in a Pipeline (it doesn't natively handle strings)
    hist = HistGradientBoostingRegressor(max_iter=150, max_depth=4, learning_rate=0.05,
                                         min_samples_leaf=10, l2_regularization=1.0,
                                         random_state=RANDOM_STATE)
    pipe = build_pipeline(hist, MODEL_FEATURES)
    pipe.fit(train[MODEL_FEATURES], train[TARGET])
    res = {
        "train": metrics(train[TARGET].values, pipe.predict(train[MODEL_FEATURES])),
        "test": metrics(test[TARGET].values, pipe.predict(test[MODEL_FEATURES])),
        "pipeline": pipe,
    }
    log(f"  train: MAE={res['train']['mae']:.4f} R2={res['train']['r2']:.4f}")
    log(f"  test : MAE={res['test']['mae']:.4f} RMSE={res['test']['rmse']:.4f} R2={res['test']['r2']:.4f}")
    return res


def train_rf_reg(train, test) -> dict:
    log("STEP 5 MODEL 4: RandomForest regularized (n=300, depth=6, leaf=4)")
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=4,
                               random_state=RANDOM_STATE, n_jobs=-1)
    pipe = build_pipeline(rf, MODEL_FEATURES)
    pipe.fit(train[MODEL_FEATURES], train[TARGET])
    res = {
        "train": metrics(train[TARGET].values, pipe.predict(train[MODEL_FEATURES])),
        "test": metrics(test[TARGET].values, pipe.predict(test[MODEL_FEATURES])),
        "pipeline": pipe,
    }
    log(f"  train: MAE={res['train']['mae']:.4f} R2={res['train']['r2']:.4f}")
    log(f"  test : MAE={res['test']['mae']:.4f} RMSE={res['test']['rmse']:.4f} R2={res['test']['r2']:.4f}")
    return res


# --------------------------------------------------------------------------- #
# STEP 8 - original vs synthetic test performance
# --------------------------------------------------------------------------- #
def eval_by_subset(pipe, test) -> dict:
    """Evaluate the pipeline on: all test / original-only / synthetic-only."""
    out = {"all": metrics(test[TARGET].values, pipe.predict(test[MODEL_FEATURES]))}
    for label in ["original", "synthetic"]:
        sub = test[test.data_origin == label]
        if len(sub):
            out[label] = metrics(sub[TARGET].values, pipe.predict(sub[MODEL_FEATURES]))
        else:
            out[label] = None
    return out


# --------------------------------------------------------------------------- #
# STEP 9 - permutation importance
# --------------------------------------------------------------------------- #
def permutation_importance_on_test(pipe, test, n_repeats=10) -> dict:
    log("STEP 9: permutation importance on test set...")
    result = permutation_importance(pipe, test[MODEL_FEATURES], test[TARGET],
                                    n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1)
    prep = pipe.named_steps["prep"]
    try:
        transformed_names = list(prep.get_feature_names_out())
    except Exception:
        transformed_names = [f"f{i}" for i in range(len(result.importances_mean))]

    agg = {f: 0.0 for f in MODEL_FEATURES}
    for name, imp in zip(transformed_names, result.importances_mean):
        matched = None
        if name.startswith("cat__"):
            rest = name[len("cat__"):]
            for c in CATEGORICAL:
                if rest.startswith(c + "_") and c in agg:
                    matched = c
                    break
        elif name.startswith("num__"):
            rest = name[len("num__"):]
            if rest in agg:
                matched = rest
        if matched is None:
            for f in MODEL_FEATURES:
                if f in name:
                    matched = f
                    break
        if matched is not None:
            agg[matched] += float(imp)
    total = sum(agg.values())
    pct = {k: round(100.0 * v / total, 2) for k, v in agg.items()} if total > 0 else {k: 0.0 for k in agg}
    sorted_pct = dict(sorted(pct.items(), key=lambda x: -x[1]))
    log("  top features (perm importance %):")
    for name, imp in list(sorted_pct.items())[:8]:
        log(f"    {imp:6.2f}%  {name}")
    return {"raw": agg, "pct": pct, "sorted": sorted_pct}


def builtin_importance(pipe) -> dict:
    """Built-in (impurity) importance if available."""
    try:
        est = pipe.named_steps["model"]
        if not hasattr(est, "feature_importances_"):
            return {}
        prep = pipe.named_steps["prep"]
        tnames = list(prep.get_feature_names_out())
        raw = list(zip(tnames, est.feature_importances_))
        agg = {f: 0.0 for f in MODEL_FEATURES}
        for name, imp in raw:
            matched = None
            if name.startswith("cat__"):
                rest = name[len("cat__"):]
                for c in CATEGORICAL:
                    if rest.startswith(c + "_") and c in agg:
                        matched = c
                        break
            elif name.startswith("num__"):
                rest = name[len("num__"):]
                if rest in agg:
                    matched = rest
            if matched is not None:
                agg[matched] += float(imp)
        total = sum(agg.values())
        return {k: round(100.0 * v / total, 2) for k, v in agg.items()} if total > 0 else {k: 0.0 for k in agg}
    except Exception as e:
        log(f"  builtin importance unavailable: {e}")
        return {}


# --------------------------------------------------------------------------- #
# STEP 10 - synthetic-data reliance experiment
# --------------------------------------------------------------------------- #
def synthetic_reliance_experiment(train, test) -> dict:
    """Train on (A) all 1110 rows vs (B) only 110 original rows.
    Compare on the ORIGINAL held-out test observations only.
    """
    log("STEP 10: synthetic-reliance experiment (train on all vs train on original-only)...")
    # Original held-out observations (for evaluation)
    orig_test = test[test.data_origin == "original"].copy()
    if len(orig_test) == 0:
        log("  WARNING: no original rows in test set - cannot run this experiment")
        return {"error": "no original rows in test set"}

    # A: train on ALL train rows (original + synthetic)
    rf_a = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2,
                                 random_state=RANDOM_STATE, n_jobs=-1)
    pipe_a = build_pipeline(rf_a, MODEL_FEATURES)
    pipe_a.fit(train[MODEL_FEATURES], train[TARGET])
    perf_a = metrics(orig_test[TARGET].values, pipe_a.predict(orig_test[MODEL_FEATURES]))

    # B: train on ONLY original train rows
    orig_train = train[train.data_origin == "original"].copy()
    rf_b = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2,
                                 random_state=RANDOM_STATE, n_jobs=-1)
    pipe_b = build_pipeline(rf_b, MODEL_FEATURES)
    pipe_b.fit(orig_train[MODEL_FEATURES], orig_train[TARGET])
    perf_b = metrics(orig_test[TARGET].values, pipe_b.predict(orig_test[MODEL_FEATURES]))

    log(f"  A (all 1110 train) on original test: MAE={perf_a['mae']:.4f} R2={perf_a['r2']:.4f}")
    log(f"  B (110 original only) on original test: MAE={perf_b['mae']:.4f} R2={perf_b['r2']:.4f}")
    improved = perf_a["mae"] < perf_b["mae"]
    log(f"  synthetic data {'IMPROVED' if improved else 'HURT'} generalization to original data (MAE diff={perf_a['mae']-perf_b['mae']:+.4f})")
    return {
        "A_all_train_on_original_test": perf_a,
        "B_original_only_train_on_original_test": perf_b,
        "synthetic_improved_generalization": bool(improved),
        "mae_diff_A_minus_B": float(perf_a["mae"] - perf_b["mae"]),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    log("=" * 60)
    log("FINAL HACKATHON MODEL TRAINING")
    log("=" * 60)

    # Verify v1 (and v2 if present) untouched
    if MODEL_V1_PATH.exists():
        v1_sha = hashlib.sha256(MODEL_V1_PATH.read_bytes()).hexdigest()
        log(f"v1 model sha256 (baseline): {v1_sha}")
        assert v1_sha == "695fafe3f31b560d5a4412124c0839e0e622c9d2bd090191a5e02eaef6c3819a", "v1 model changed!"
    v2_sha_before = None
    if MODEL_V2_PATH.exists():
        v2_sha_before = hashlib.sha256(MODEL_V2_PATH.read_bytes()).hexdigest()
        log(f"v2 model sha256 (baseline): {v2_sha_before}")

    # STEP 1 - load + inspect
    df = load_and_inspect()

    # Verify original 110 rows unchanged (compare to expanded master)
    expanded = pd.read_csv(REPO_ROOT / "data" / "master_freight_training_expanded_v1.csv")
    expanded["date"] = pd.to_datetime(expanded["date"])
    orig_in_df = df[df.data_origin == "original"].copy()
    compare_cols = ["date", "origin", "destination", "commodity", "vessel_type",
                    "bdi", "vlsfo_usd_per_tonne", "coal_price_usd_per_mt",
                    "iron_ore_price_usd_per_dmt", "wind_kmh", "wave_height_m",
                    "cyclone_risk", "weather_delay_days", "current_freight_usd_per_tonne", TARGET]
    orig_in_df_s = orig_in_df[compare_cols].sort_values(compare_cols[:5]).reset_index(drop=True)
    expanded_s = expanded[compare_cols].sort_values(compare_cols[:5]).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(expanded_s, orig_in_df_s, check_dtype=False, check_exact=False, rtol=1e-9)
        log("  original 110 rows verified unchanged ✅")
    except AssertionError as e:
        log(f"  WARNING: original rows differ: {e}")

    # STEP 3 - trajectory-aware temporal split
    train, test = trajectory_aware_temporal_split(df)

    # STEP 4 - baselines
    persistence = persistence_baseline(train, test)
    linear = linear_baseline(train, test)

    # STEP 5 - ML candidates
    rf = train_rf(train, test)
    gb = train_gb(train, test)
    histgb = train_histgb(train, test)
    rf_reg = train_rf_reg(train, test)

    # STEP 6/7 - comparison
    log("\n" + "=" * 70)
    log("MODEL COMPARISON (TEMPORAL TEST SET)")
    log("=" * 70)
    log(f"{'Model':<32} {'TestMAE':>9} {'TestRMSE':>9} {'TestR2':>8}")
    log("-" * 70)
    candidates = [
        ("Persistence (current_freight)", persistence, None),
        ("LinearRegression", linear, None),
        ("RandomForest (depth=8)", rf, "rf"),
        ("GradientBoosting", gb, "gb"),
        ("HistGradientBoosting", histgb, "histgb"),
        ("RandomForest regularized (depth=6)", rf_reg, "rf_reg"),
    ]
    for name, m, key in candidates:
        t = m["test"]
        log(f"{name:<32} {t['mae']:>9.4f} {t['rmse']:>9.4f} {t['r2']:>8.4f}")

    # Selection: lowest test MAE among ML candidates
    ml_candidates = [(n, m, k) for n, m, k in candidates if k is not None]
    ml_candidates.sort(key=lambda c: (c[1]["test"]["mae"], c[1]["test"]["rmse"]))
    selected_name, selected_metrics, selected_key = ml_candidates[0]
    log(f"\nSELECTED (lowest test MAE among ML): {selected_name}")

    # STEP 8 - original vs synthetic test performance
    selected_pipe = selected_metrics["pipeline"]
    subset_perf = eval_by_subset(selected_pipe, test)
    log(f"  all test          : MAE={subset_perf['all']['mae']:.4f} R2={subset_perf['all']['r2']:.4f}")
    if subset_perf.get("original"):
        log(f"  original test only: MAE={subset_perf['original']['mae']:.4f} R2={subset_perf['original']['r2']:.4f}")
    if subset_perf.get("synthetic"):
        log(f"  synthetic test only: MAE={subset_perf['synthetic']['mae']:.4f} R2={subset_perf['synthetic']['r2']:.4f}")

    # STEP 9 - permutation importance
    perm_imp = permutation_importance_on_test(selected_pipe, test)
    builtin_imp = builtin_importance(selected_pipe)
    current_freight_pct = perm_imp["pct"].get("current_freight_usd_per_tonne", 0.0)
    current_freight_builtin = builtin_imp.get("current_freight_usd_per_tonne", 0.0)
    log(f"  current_freight perm importance: {current_freight_pct:.2f}%")
    log(f"  current_freight builtin importance: {current_freight_builtin:.2f}%")
    dominant = current_freight_pct > 90 or current_freight_builtin > 90
    log(f"  current_freight dominant (>90%)? {dominant}")

    # STEP 10 - synthetic-reliance experiment
    reliance = synthetic_reliance_experiment(train, test)

    # STEP 11 - selection + does ML beat persistence?
    ml_beats_persistence = selected_metrics["test"]["mae"] < persistence["test"]["mae"]
    log(f"\nML beats persistence? {ml_beats_persistence} (ML MAE={selected_metrics['test']['mae']:.4f} vs persistence MAE={persistence['test']['mae']:.4f})")

    # STEP 12 - save final model
    log(f"STEP 12: saving final model to {MODEL_FINAL_PATH}")
    joblib.dump(selected_pipe, MODEL_FINAL_PATH)
    loaded = joblib.load(MODEL_FINAL_PATH)
    log(f"  reloaded OK, type={type(loaded).__name__}")

    # STEP 13 - smoke test on 5 combinations
    log("STEP 13: smoke test on 5 combinations...")
    sample_inputs = []
    for combo in COMBINATIONS:
        origin, dest, comm, vessel = combo
        mask = ((df.origin == origin) & (df.destination == dest)
                & (df.commodity == comm) & (df.vessel_type == vessel))
        # prefer an original row if available, else first synthetic
        orig_match = df[mask & (df.data_origin == "original")]
        row = orig_match.iloc[0] if len(orig_match) else df[mask].iloc[0]
        sample_inputs.append(row)
    sample_df = pd.DataFrame(sample_inputs)
    preds = loaded.predict(sample_df[MODEL_FEATURES])
    sample_check = []
    for i, row in enumerate(sample_inputs):
        p = float(preds[i])
        ok = (not np.isnan(p)) and isinstance(p, float) and 5.0 <= p <= 30.0
        sample_check.append({
            "origin": row["origin"], "destination": row["destination"],
            "commodity": row["commodity"], "vessel_type": row["vessel_type"],
            "date": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
            "predicted": p, "numeric": bool(isinstance(p, float)),
            "is_nan": bool(np.isnan(p)), "in_range": bool(5.0 <= p <= 30.0),
        })
        log(f"  {row['origin']:22} -> {row['destination']:18} {row['commodity']:13} {row['vessel_type']:9}: pred={p:.4f}  ok={ok}")
    all_numeric_ok = all(s["numeric"] and not s["is_nan"] and s["in_range"] for s in sample_check)
    log(f"  all 5 predictions numeric, non-NaN, in range [5,30]: {all_numeric_ok}")

    # STEP 14 - predictions CSV (test rows only)
    log(f"STEP 14: writing {PREDICTIONS_CSV}")
    test_pred = loaded.predict(test[MODEL_FEATURES])
    pred_df = test[["date", "origin", "destination", "commodity", "vessel_type",
                    "current_freight_usd_per_tonne", TARGET, "data_origin"]].copy()
    pred_df["predicted_next_month_freight"] = test_pred
    pred_df["absolute_error"] = (pred_df["predicted_next_month_freight"] - pred_df[TARGET]).abs()
    pred_df = pred_df.rename(columns={TARGET: "actual_next_month_freight"})
    pred_df["date"] = pred_df["date"].dt.strftime("%Y-%m-%d")
    pred_df = pred_df[[
        "date", "origin", "destination", "commodity", "vessel_type",
        "current_freight_usd_per_tonne", "actual_next_month_freight",
        "predicted_next_month_freight", "absolute_error", "data_origin",
    ]]
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    log(f"  wrote {len(pred_df)} test predictions")

    # Overfitting analysis
    overfit = {}
    for name, m, key in candidates:
        if key is None:
            continue
        overfit[name] = {
            "train_mae": m["train"]["mae"], "test_mae": m["test"]["mae"],
            "mae_gap": m["test"]["mae"] - m["train"]["mae"],
            "train_r2": m["train"]["r2"], "test_r2": m["test"]["r2"],
            "r2_gap": m["train"]["r2"] - m["test"]["r2"],
        }

    # Persist metrics
    metrics_out = {
        "dataset": {
            "rows_total": int(len(df)),
            "rows_original": int((df.data_origin == "original").sum()),
            "rows_synthetic": int((df.data_origin == "synthetic").sum()),
            "combinations": int(df.groupby(["origin","destination","commodity","vessel_type"]).ngroups),
            "date_range": [str(df.date.min().date()), str(df.date.max().date())],
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_original": int((train.data_origin == "original").sum()),
            "train_synthetic": int((train.data_origin == "synthetic").sum()),
            "test_original": int((test.data_origin == "original").sum()),
            "test_synthetic": int((test.data_origin == "synthetic").sum()),
            "test_fraction": TEST_FRACTION,
        },
        "features": MODEL_FEATURES,
        "excluded_features": ["cargo_tonnes", "previous_month_freight", "freight_3_month_avg",
                               "freight_observation_count", "data_origin", "synthetic_generation_method",
                               "trajectory_id", "year", "month_number", "quarter_number",
                               "data_source", "ingested_at"],
        "persistence_baseline": persistence,
        "linear_baseline": linear,
        "rf": {k: rf[k] for k in ["train", "test"]},
        "gb": {k: gb[k] for k in ["train", "test"]},
        "histgb": {k: histgb[k] for k in ["train", "test"]},
        "rf_reg": {k: rf_reg[k] for k in ["train", "test"]},
        "selected_model": selected_name,
        "selected_test_metrics": selected_metrics["test"],
        "subset_performance": subset_perf,
        "permutation_importance_pct": perm_imp["sorted"],
        "builtin_importance_pct": builtin_imp,
        "current_freight_perm_pct": current_freight_pct,
        "current_freight_builtin_pct": current_freight_builtin,
        "current_freight_dominant": dominant,
        "overfit_analysis": overfit,
        "synthetic_reliance_experiment": reliance,
        "ml_beats_persistence": bool(ml_beats_persistence),
        "sample_predictions": sample_check,
        "model_path": str(MODEL_FINAL_PATH),
    }
    METRICS_JSON.write_text(json.dumps(metrics_out, indent=2, default=str))
    log(f"  wrote metrics to {METRICS_JSON}")

    # Verify models untouched
    if MODEL_V1_PATH.exists():
        v1_sha_after = hashlib.sha256(MODEL_V1_PATH.read_bytes()).hexdigest()
        assert v1_sha_after == v1_sha, "v1 model changed!"
        log("v1 model UNTOUCHED ✅")
    if MODEL_V2_PATH.exists():
        v2_sha_after = hashlib.sha256(MODEL_V2_PATH.read_bytes()).hexdigest()
        assert v2_sha_after == v2_sha_before, "v2 model changed!"
        log("v2 model UNTOUCHED ✅")

    print("\n" + "=" * 60)
    print("FINAL MODEL TRAINING SUMMARY")
    print("=" * 60)
    print(f"selected model         : {selected_name}")
    print(f"saved to               : {MODEL_FINAL_PATH}")
    print(f"persistence test MAE    : {persistence['test']['mae']:.4f}")
    print(f"selected ML test MAE    : {selected_metrics['test']['mae']:.4f}")
    print(f"ML beats persistence?  : {ml_beats_persistence}")
    print(f"synthetic improved gen? : {reliance.get('synthetic_improved_generalization')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
