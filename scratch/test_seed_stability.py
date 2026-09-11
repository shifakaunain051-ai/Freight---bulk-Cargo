"""Test stability of GradientBoosting and RandomForest across random seeds.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

DATA_PATH = Path("data/master_freight_training_expanded_v1.csv")
df_freight = pd.read_csv(DATA_PATH)
df_freight["date_dt"] = pd.to_datetime(df_freight["date"])

corridor_dfs = []
for (o, d, c, v), grp in df_freight.groupby(["origin", "destination", "commodity", "vessel_type"]):
    grp = grp.sort_values("date_dt").copy()
    grp["bdi_1m_momentum"] = grp["bdi"].pct_change(1)
    grp["bdi_3m_momentum"] = grp["bdi"].pct_change(3)
    grp["vlsfo_1m_momentum"] = grp["vlsfo_usd_per_tonne"].pct_change(1)
    grp["vlsfo_3m_momentum"] = grp["vlsfo_usd_per_tonne"].pct_change(3)
    corridor_dfs.append(grp)

df_all = pd.concat(corridor_dfs).sort_values(["date_dt", "origin", "vessel_type"]).reset_index(drop=True)
df_all["delta"] = df_all["next_month_freight_usd_per_tonne"] - df_all["current_freight_usd_per_tonne"]

BASE_CAT = ["origin", "destination", "commodity", "vessel_type"]
BASE_NUM = [
    "bdi", "vlsfo_usd_per_tonne", "coal_price_usd_per_mt", "iron_ore_price_usd_per_dmt",
    "wind_kmh", "wave_height_m", "cyclone_risk", "weather_delay_days", "current_freight_usd_per_tonne"
]
MACRO_MOM = ["bdi_1m_momentum", "bdi_3m_momentum", "vlsfo_1m_momentum", "vlsfo_3m_momentum"]
FEATURES = BASE_CAT + BASE_NUM + MACRO_MOM

dates = sorted(df_all["date_dt"].unique())
folds = []
for test_idx in range(17, 22):
    test_date = dates[test_idx]
    train_dates = dates[:test_idx]
    folds.append((test_idx - 16, train_dates, test_date))

seeds = [0, 1, 7, 42, 100, 2024]
print("Testing seed stability for GradientBoostingRegressor and RandomForestRegressor:")
print(f"{'Seed':6} | {'GBR Mean MAE':14} | {'GBR Fold MAEs':32} | {'RF Mean MAE':14} | {'RF Fold MAEs':32}")
print("-" * 105)

for s in seeds:
    # GBR
    gbr_maes = []
    rf_maes = []
    for fold_num, tr_d, te_d in folds:
        tr_df = df_all[df_all["date_dt"].isin(tr_d)].copy()
        te_df = df_all[df_all["date_dt"] == te_d].copy()
        
        prep = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), BASE_CAT),
            ("num", SimpleImputer(strategy="median"), BASE_NUM + MACRO_MOM)
        ])
        
        gbr = Pipeline([
            ("prep", prep),
            ("reg", GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=s))
        ])
        gbr.fit(tr_df[FEATURES], tr_df["delta"])
        pred_gbr = te_df["current_freight_usd_per_tonne"].values + np.clip(gbr.predict(te_df[FEATURES]), -4.0, 4.0)
        gbr_maes.append(mean_absolute_error(te_df["next_month_freight_usd_per_tonne"], pred_gbr))
        
        rf = Pipeline([
            ("prep", prep),
            ("reg", RandomForestRegressor(n_estimators=100, max_depth=4, random_state=s))
        ])
        rf.fit(tr_df[FEATURES], tr_df["delta"])
        pred_rf = te_df["current_freight_usd_per_tonne"].values + np.clip(rf.predict(te_df[FEATURES]), -4.0, 4.0)
        rf_maes.append(mean_absolute_error(te_df["next_month_freight_usd_per_tonne"], pred_rf))
        
    gbr_str = ", ".join([f"{m:.2f}" for m in gbr_maes])
    rf_str = ", ".join([f"{m:.2f}" for m in rf_maes])
    print(f"{s:6} | {np.mean(gbr_maes):14.4f} | [{gbr_str:30}] | {np.mean(rf_maes):14.4f} | [{rf_str:30}]")
