"""Investigate detailed fold predictions of top models vs Frozen Model v3.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

DATA_PATH = Path("data/master_freight_training_expanded_v1.csv")
MODEL_V3_PATH = Path("freight_forecast_model_v3.joblib")
v3_model = joblib.load(MODEL_V3_PATH)

df = pd.read_csv(DATA_PATH)
df["date_dt"] = pd.to_datetime(df["date"])
df = df.sort_values(["date_dt", "origin", "vessel_type"]).reset_index(drop=True)

# Check last 5 months (holdout: 2025-07 to 2025-11)
holdout_dates = sorted(df["date_dt"].unique())[-5:]
print(f"Holdout months: {[d.strftime('%Y-%m') for d in holdout_dates]}")

for d in holdout_dates:
    sub = df[df["date_dt"] == d]
    print(f"\n--- Month: {d.strftime('%Y-%m')} ---")
    for _, row in sub.iterrows():
        y_curr = row["current_freight_usd_per_tonne"]
        y_true = row["next_month_freight_usd_per_tonne"]
        
        # v3 prediction
        v3_row = pd.DataFrame([row])[list(v3_model.feature_names_in_)]
        delta_v3 = np.clip(v3_model.predict(v3_row)[0], -4.0, 4.0)
        pred_v3 = y_curr + delta_v3
        err_v3 = abs(y_true - pred_v3)
        pers_err = abs(y_true - y_curr)
        
        print(f"  {row['origin'][:12]:12} | {row['vessel_type']:8} | Curr: ${y_curr:5.2f} | True: ${y_true:5.2f} (True_d: {y_true-y_curr:+5.2f}) | v3: ${pred_v3:5.2f} (v3_d: {delta_v3:+5.2f}, Err: {err_v3:.2f}) | PersErr: {pers_err:.2f}")
