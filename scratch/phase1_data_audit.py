"""Phase 1: Existing Data Audit for Model Improvement Research.
Inspects data/master_freight_training_expanded_v1.csv and freight_forecast_model_v3.joblib.
"""
import hashlib
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

DATA_PATH = Path("data/master_freight_training_expanded_v1.csv")
MODEL_PATH = Path("freight_forecast_model_v3.joblib")

print("=== PHASE 1: DATA & MODEL AUDIT ===")

# Model v3 verification
model_bytes = MODEL_PATH.read_bytes()
model_hash = hashlib.sha256(model_bytes).hexdigest()
print(f"Model v3 Path: {MODEL_PATH}")
print(f"Model v3 Size: {len(model_bytes)} bytes")
print(f"Model v3 SHA-256: {model_hash}")
expected_hash = "71fbb870bb1f555d73a51ed7d83fb5a877cc4405ce54d1fe18407c9ce37c46a8"
assert model_hash == expected_hash, f"Hash mismatch! Expected {expected_hash}, got {model_hash}"
print("Model SHA-256 VERIFIED identical to reference hash.")

model = joblib.load(MODEL_PATH)
print(f"Model type: {type(model).__name__}")
print(f"Pipeline steps: {[name for name, _ in model.steps]}")
print(f"Input features ({len(model.feature_names_in_)}): {list(model.feature_names_in_)}")

# Data inspection
df = pd.read_csv(DATA_PATH)
print(f"\nTotal rows in dataset: {len(df)}")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")

# Missing values
missing = df.isnull().sum().to_dict()
print(f"Missing values per column: {missing}")
assert sum(missing.values()) == 0, "Found unexpected missing values!"

# Duplicates
dup_count = df.duplicated().sum()
print(f"Duplicate rows: {dup_count}")

# Dates and periods
dates = sorted(df["date"].unique())
print(f"Unique dates count: {len(dates)}")
print(f"Date range: {dates[0]} to {dates[-1]}")
print(f"Dates list: {dates}")

# Canonical combinations
combos = df.groupby(["origin", "destination", "commodity", "vessel_type"]).size()
print(f"\nCanonical combinations ({len(combos)}):")
for (o, d, c, v), count in combos.items():
    print(f"  {o} -> {d} | {c} | {v}: {count} observations")

# Check time series continuity per combination
print("\nChecking continuity of monthly time series per combination:")
all_continuous = True
for (o, d, c, v), grp in df.groupby(["origin", "destination", "commodity", "vessel_type"]):
    grp_sorted = grp.sort_values("date")
    combo_dates = grp_sorted["date"].tolist()
    is_same = combo_dates == dates
    print(f"  {o[:12]}.. | {v:8} | Exact 22 consecutive months? {is_same}")
    if not is_same:
        all_continuous = False

print(f"All 5 combinations form continuous monthly time series: {all_continuous}")

# Target and Feature distributions
print("\n--- Summary Statistics (Numericals) ---")
num_cols = df.select_dtypes(include=[np.number]).columns
stats = df[num_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
print(stats.to_string())

# Target Delta distribution
df["delta"] = df["next_month_freight_usd_per_tonne"] - df["current_freight_usd_per_tonne"]
print("\n--- Target Delta (next_month - current_freight) Distribution ---")
print(df["delta"].describe().to_string())

# Correlation of delta with features
corrs = df[num_cols].apply(lambda c: df["delta"].corr(c))
print("\n--- Correlation with Target Delta ---")
print(corrs.sort_values(ascending=False).to_string())
