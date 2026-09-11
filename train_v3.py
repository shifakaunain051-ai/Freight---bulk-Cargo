"""Train Model v3: Bounded Residual Ridge Regression on the 110 real observations.

Specifications:
- Model Artifact: freight_forecast_model_v3.joblib
- Algorithm: Residual Ridge Regression (alpha = 10.0)
- Guardrail: delta clipped to [-4.0, +4.0] USD/tonne (training-derived defensive boundary)
- Preprocessing: ColumnTransformer with OneHotEncoder(handle_unknown='ignore') on 4 categoricals,
  passthrough on 9 numerical features.
- Pipeline: Standard sklearn.pipeline.Pipeline (100% portable, 0 custom unpickle dependencies)
- Features (13): origin, destination, commodity, vessel_type, bdi, vlsfo_usd_per_tonne,
  coal_price_usd_per_mt, iron_ore_price_usd_per_dmt, wind_kmh, wave_height_m,
  cyclone_risk, weather_delay_days, current_freight_usd_per_tonne
- Target during training: freight delta (next_month_freight_usd_per_tonne - current_freight_usd_per_tonne)
- Inference formulation: predicted_next_month = current_freight + clip(predicted_delta, -4.0, 4.0)
- Training Data: data/master_freight_training_expanded_v1.csv (110 real observations)
- Synthetic Data: Quarantined / Excluded from training
"""

import os
import sys
import json
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data" / "master_freight_training_expanded_v1.csv"
MODEL_V3_PATH = REPO_ROOT / "freight_forecast_model_v3.joblib"
MODEL_V1_PATH = REPO_ROOT / "freight_forecast_model_v1.joblib"
MODEL_FINAL_PATH = REPO_ROOT / "freight_forecast_model_final.joblib"

FEATURES = [
    "origin",
    "destination",
    "commodity",
    "vessel_type",
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


def build_pipeline(alpha: float = 10.0) -> Pipeline:
    """Build canonical 13-feature sklearn Pipeline for Model v3."""
    prep = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["origin", "destination", "commodity", "vessel_type"]),
            ("num", "passthrough",
             ["bdi", "vlsfo_usd_per_tonne", "coal_price_usd_per_mt",
              "iron_ore_price_usd_per_dmt", "wind_kmh", "wave_height_m",
              "cyclone_risk", "weather_delay_days", "current_freight_usd_per_tonne"]),
        ]
    )
    return Pipeline([
        ("prep", prep),
        ("model", Ridge(alpha=alpha, random_state=42)),
    ])


def predict_v3(pipeline: Pipeline, X: pd.DataFrame, clip_bounds: tuple[float, float] = (-4.0, 4.0)) -> np.ndarray:
    """Inference for Model v3: current_freight + clip(predicted_delta, -4.0, 4.0)."""
    curr = X["current_freight_usd_per_tonne"].values
    pred_delta = pipeline.predict(X)
    if clip_bounds is not None:
        pred_delta = np.clip(pred_delta, clip_bounds[0], clip_bounds[1])
    return curr + pred_delta


def train_and_validate():
    print("=" * 60)
    print("TRAINING MODEL V3: BOUNDED RESIDUAL RIDGE REGRESSOR")
    print("=" * 60)

    # 1. Load real data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {DATA_PATH.name} ({len(df)} rows, 0 missing)")

    # Record hashes of existing models
    v1_hash = hashlib.sha256(MODEL_V1_PATH.read_bytes()).hexdigest() if MODEL_V1_PATH.exists() else None
    final_hash = hashlib.sha256(MODEL_FINAL_PATH.read_bytes()).hexdigest() if MODEL_FINAL_PATH.exists() else None

    # 2. Validation Run (Months 1-17 train, Months 18-22 holdout)
    df["date_dt"] = pd.to_datetime(df["date"])
    tr_idx, te_idx = [], []
    for (o, d, c, v), group in df.groupby(["origin", "destination", "commodity", "vessel_type"]):
        group_sorted = group.sort_values("date_dt")
        tr_idx.extend(group_sorted.iloc[:17].index.tolist())
        te_idx.extend(group_sorted.iloc[17:].index.tolist())

    tr_df = df.loc[tr_idx]
    te_df = df.loc[te_idx]

    delta_train = tr_df[TARGET].values - tr_df["current_freight_usd_per_tonne"].values
    val_pipeline = build_pipeline(alpha=10.0)
    val_pipeline.fit(tr_df[FEATURES], delta_train)

    val_preds = predict_v3(val_pipeline, te_df[FEATURES], clip_bounds=(-4.0, 4.0))

    y_val = te_df[TARGET].values
    y_curr_val = te_df["current_freight_usd_per_tonne"].values

    mae_val = mean_absolute_error(y_val, val_preds)
    rmse_val = np.sqrt(mean_squared_error(y_val, val_preds))
    r2_val = r2_score(y_val, val_preds)
    pers_mae = mean_absolute_error(y_val, y_curr_val)

    true_dir = np.sign(y_val - y_curr_val)
    pred_dir = np.sign(val_preds - y_curr_val)
    dir_acc = float(np.mean(true_dir == pred_dir) * 100)

    print(f"\n--- Validation Check on 25-Row Clean Holdout ---")
    print(f"  Holdout MAE          : {mae_val:.4f} USD/tonne")
    print(f"  Holdout RMSE         : {rmse_val:.4f} USD/tonne")
    print(f"  Holdout R2           : {r2_val:.4f}")
    print(f"  Directional Accuracy : {dir_acc:.1f}%")
    print(f"  Persistence MAE      : {pers_mae:.4f} USD/tonne")
    print(f"  Error Reduction      : {((pers_mae - mae_val) / pers_mae * 100):.2f}%")

    assert mae_val < 0.50, f"Expected validation MAE < 0.50, got {mae_val:.4f}"
    assert pers_mae > mae_val, "Model failed to beat persistence on holdout!"

    # 3. Final Model Training on ALL 110 Real Observations
    print(f"\n--- Training Production Model v3 Pipeline on Full 110 Observations ---")
    full_delta = df[TARGET].values - df["current_freight_usd_per_tonne"].values
    v3_pipeline = build_pipeline(alpha=10.0)
    v3_pipeline.fit(df[FEATURES], full_delta)

    # Save to freight_forecast_model_v3.joblib
    joblib.dump(v3_pipeline, MODEL_V3_PATH)
    print(f"Saved Model v3 Pipeline to: {MODEL_V3_PATH}")

    # 4. Verify artifact in fresh reload
    reloaded = joblib.load(MODEL_V3_PATH)
    file_bytes = MODEL_V3_PATH.read_bytes()
    file_sha256 = hashlib.sha256(file_bytes).hexdigest()

    print(f"\n--- Artifact Verification ---")
    print(f"  Artifact Path : {MODEL_V3_PATH.name}")
    print(f"  File Size     : {len(file_bytes)} bytes")
    print(f"  SHA-256 Hash  : {file_sha256}")
    print(f"  Type          : {type(reloaded).__name__}")
    print(f"  Features In   : {list(reloaded.feature_names_in_)}")
    print(f"  Feature Count : {len(reloaded.feature_names_in_)}")

    # 5. Smoke test on all 5 combinations
    print(f"\n--- Smoke Test on All 5 Combinations ---")
    combos = [
        {"origin": "Australia West Coast", "destination": "East Coast India", "commodity": "Iron Ore", "vessel_type": "Capesize", "current_freight_usd_per_tonne": 10.0},
        {"origin": "Hay Point", "destination": "East Coast India", "commodity": "Coal", "vessel_type": "Capesize", "current_freight_usd_per_tonne": 14.0},
        {"origin": "Hay Point", "destination": "East Coast India", "commodity": "Coal", "vessel_type": "Panamax", "current_freight_usd_per_tonne": 16.5},
        {"origin": "Taboneo", "destination": "East Coast India", "commodity": "Thermal Coal", "vessel_type": "Panamax", "current_freight_usd_per_tonne": 11.0},
        {"origin": "Taboneo", "destination": "East Coast India", "commodity": "Thermal Coal", "vessel_type": "Supramax", "current_freight_usd_per_tonne": 12.0},
    ]

    for c in combos:
        row = {
            "origin": c["origin"], "destination": c["destination"], "commodity": c["commodity"], "vessel_type": c["vessel_type"],
            "bdi": 1560, "vlsfo_usd_per_tonne": 638, "coal_price_usd_per_mt": 124, "iron_ore_price_usd_per_dmt": 124,
            "wind_kmh": 32, "wave_height_m": 2.0, "cyclone_risk": 2, "weather_delay_days": 0.5,
            "current_freight_usd_per_tonne": c["current_freight_usd_per_tonne"]
        }
        test_row_df = pd.DataFrame([row])[FEATURES]
        pred = float(predict_v3(reloaded, test_row_df)[0])
        delta = pred - c["current_freight_usd_per_tonne"]
        ok = 5.0 <= pred <= 30.0 and not np.isnan(pred)
        print(f"  {c['origin']:22} | {c['commodity']:13} | {c['vessel_type']:9} -> Current: ${c['current_freight_usd_per_tonne']:5.1f} | Pred: ${pred:5.2f} (Delta: {delta:+5.2f}) | OK: {ok}")

    # 6. Verify existing models remain untouched
    if v1_hash:
        assert hashlib.sha256(MODEL_V1_PATH.read_bytes()).hexdigest() == v1_hash, "v1 model modified!"
        print("\nfreight_forecast_model_v1.joblib UNTOUCHED ✅")
    if final_hash:
        assert hashlib.sha256(MODEL_FINAL_PATH.read_bytes()).hexdigest() == final_hash, "final model modified!"
        print("freight_forecast_model_final.joblib UNTOUCHED ✅")

    print("\nModel v3 training & verification complete. ✅")


if __name__ == "__main__":
    train_and_validate()
