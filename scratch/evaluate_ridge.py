import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

df = pd.read_csv("data/master_freight_training_expanded_v1.csv")
df["date_dt"] = pd.to_datetime(df["date"])

def build_ridge_pipeline(alpha=10.0):
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

# Let's perform walk-forward expanding window validation across all 5 routes
# There are 22 months: 2024-02 to 2025-11.
# Starting window: 12 months (indices 0..11, i.e. 2024-02 to 2025-01)
# Testing on months 12 to 21 (10 test periods)

dates = sorted(df["date_dt"].unique())
print(f"Total unique monthly dates: {len(dates)}")

all_actuals = []
all_ridge_preds = []
all_pers_preds = []
all_routes = []
all_dates = []

for t in range(12, len(dates)):
    test_date = dates[t]
    train_dates = dates[:t]
    
    train_df = df[df["date_dt"].isin(train_dates)].copy()
    test_df = df[df["date_dt"] == test_date].copy()
    
    train_delta = train_df[TARGET].values - train_df["current_freight_usd_per_tonne"].values
    
    model = build_ridge_pipeline(alpha=10.0)
    model.fit(train_df[FEATURES], train_delta)
    
    raw_delta = model.predict(test_df[FEATURES])
    clipped_delta = np.clip(raw_delta, -4.0, 4.0)
    ridge_preds = np.maximum(1.0, test_df["current_freight_usd_per_tonne"].values + clipped_delta)
    
    actuals = test_df[TARGET].values
    pers = test_df["current_freight_usd_per_tonne"].values
    
    all_actuals.extend(actuals)
    all_ridge_preds.extend(ridge_preds)
    all_pers_preds.extend(pers)
    all_routes.extend(test_df["origin"] + " | " + test_df["vessel_type"])
    all_dates.extend([str(test_date)[:10]] * len(test_df))

actuals = np.array(all_actuals)
ridge_preds = np.array(all_ridge_preds)
pers_preds = np.array(all_pers_preds)

print("\n--- WALK-FORWARD (EXPANDING WINDOW 12->22) RESULTS ---")
print(f"Total out-of-sample evaluations: {len(actuals)} (10 months x 5 routes)")

# Metrics
ridge_mae = mean_absolute_error(actuals, ridge_preds)
ridge_rmse = np.sqrt(mean_squared_error(actuals, ridge_preds))
ridge_dir = np.mean(np.sign(actuals - pers_preds) == np.sign(ridge_preds - pers_preds)) * 100

pers_mae = mean_absolute_error(actuals, pers_preds)
pers_rmse = np.sqrt(mean_squared_error(actuals, pers_preds))

print(f"Persistence Baseline:")
print(f"  MAE:  {pers_mae:.4f} USD/t")
print(f"  RMSE: {pers_rmse:.4f} USD/t")
print(f"  Directional Accuracy: N/A (predicts 0 delta)")

print(f"\nRidge Model (v3):")
print(f"  MAE:  {ridge_mae:.4f} USD/t")
print(f"  RMSE: {ridge_rmse:.4f} USD/t")
print(f"  Directional Accuracy: {ridge_dir:.1f}%")

# Per-route breakdown
print("\n--- PER-ROUTE BREAKDOWN (Ridge vs Persistence) ---")
res_df = pd.DataFrame({
    "route": all_routes,
    "date": all_dates,
    "actual": actuals,
    "persistence": pers_preds,
    "ridge": ridge_preds,
})

for route, rgroup in res_df.groupby("route"):
    r_act = rgroup["actual"].values
    r_pers = rgroup["persistence"].values
    r_rid = rgroup["ridge"].values
    
    r_pers_mae = mean_absolute_error(r_act, r_pers)
    r_rid_mae = mean_absolute_error(r_act, r_rid)
    r_rid_dir = np.mean(np.sign(r_act - r_pers) == np.sign(r_rid - r_pers)) * 100
    print(f"{route:40} | Pers MAE: {r_pers_mae:.4f} | Ridge MAE: {r_rid_mae:.4f} | Ridge DirAcc: {r_rid_dir:.1f}%")
