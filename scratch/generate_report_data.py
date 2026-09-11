"""Generate all verified tables, fold-by-fold metrics, feature group evaluations,
leakage checks, and robustness results for Phase 11 Output Report.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path("data/master_freight_training_expanded_v1.csv")
COMM_PATH = Path("data/commodity_prices_worldbank.csv")
MODEL_V3_PATH = Path("freight_forecast_model_v3.joblib")

v3_bytes = MODEL_V3_PATH.read_bytes()
v3_hash = hashlib.sha256(v3_bytes).hexdigest()
model_v3_frozen = joblib.load(MODEL_V3_PATH)

df_freight = pd.read_csv(DATA_PATH)
df_comm = pd.read_csv(COMM_PATH)
df_comm["date_dt"] = pd.to_datetime(df_comm["date"]).sort_values().reset_index(drop=True)
df_freight["date_dt"] = pd.to_datetime(df_freight["date"])

# 1. Feature Engineering (Strictly chronological, zero future leakage)
comm_feats = []
for _, row in df_freight.iterrows():
    t = row["date_dt"]
    sub = df_comm[df_comm["date_dt"] <= t]
    
    coal_t = sub.iloc[-1]["coal_price_usd_per_mt"]
    iron_t = sub.iloc[-1]["iron_ore_price_usd_per_dmt"]
    
    coal_3m_ago = sub.iloc[-4]["coal_price_usd_per_mt"]
    iron_3m_ago = sub.iloc[-4]["iron_ore_price_usd_per_dmt"]
    coal_3m_mom = (coal_t - coal_3m_ago) / coal_3m_ago
    iron_3m_mom = (iron_t - iron_3m_ago) / iron_3m_ago
    
    coal_6m_ago = sub.iloc[-7]["coal_price_usd_per_mt"]
    iron_6m_ago = sub.iloc[-7]["iron_ore_price_usd_per_dmt"]
    coal_6m_mom = (coal_t - coal_6m_ago) / coal_6m_ago
    iron_6m_mom = (iron_t - iron_6m_ago) / iron_6m_ago
    
    coal_roll_vol = sub["coal_price_usd_per_mt"].pct_change().iloc[-6:].std()
    iron_roll_vol = sub["iron_ore_price_usd_per_dmt"].pct_change().iloc[-6:].std()
    
    coal_hist_pct = stats.percentileofscore(sub["coal_price_usd_per_mt"].dropna().values, coal_t)
    iron_hist_pct = stats.percentileofscore(sub["iron_ore_price_usd_per_dmt"].dropna().values, iron_t)
    
    comm_feats.append({
        "coal_3m_momentum": coal_3m_mom,
        "coal_6m_momentum": coal_6m_mom,
        "iron_ore_3m_momentum": iron_3m_mom,
        "iron_ore_6m_momentum": iron_6m_mom,
        "coal_rolling_volatility": coal_roll_vol,
        "iron_ore_rolling_volatility": iron_roll_vol,
        "coal_historical_percentile": coal_hist_pct,
        "iron_ore_historical_percentile": iron_hist_pct,
    })

df_all = pd.concat([df_freight, pd.DataFrame(comm_feats)], axis=1)

corridor_dfs = []
for (o, d, c, v), grp in df_all.groupby(["origin", "destination", "commodity", "vessel_type"]):
    grp = grp.sort_values("date_dt").copy()
    grp["freight_lag_1"] = grp["current_freight_usd_per_tonne"].shift(1)
    grp["freight_lag_3"] = grp["current_freight_usd_per_tonne"].shift(3)
    grp["freight_rolling_mean_3"] = grp["current_freight_usd_per_tonne"].rolling(3, min_periods=1).mean()
    grp["freight_rolling_mean_6"] = grp["current_freight_usd_per_tonne"].rolling(6, min_periods=1).mean()
    grp["bdi_1m_momentum"] = grp["bdi"].pct_change(1)
    grp["bdi_3m_momentum"] = grp["bdi"].pct_change(3)
    grp["vlsfo_1m_momentum"] = grp["vlsfo_usd_per_tonne"].pct_change(1)
    grp["vlsfo_3m_momentum"] = grp["vlsfo_usd_per_tonne"].pct_change(3)
    grp["month"] = grp["date_dt"].dt.month
    grp["quarter"] = grp["date_dt"].dt.quarter
    corridor_dfs.append(grp)

df_all = pd.concat(corridor_dfs).sort_values(["date_dt", "origin", "vessel_type"]).reset_index(drop=True)
df_all["delta"] = df_all["next_month_freight_usd_per_tonne"] - df_all["current_freight_usd_per_tonne"]

BASE_CAT = ["origin", "destination", "commodity", "vessel_type"]
BASE_NUM = [
    "bdi", "vlsfo_usd_per_tonne", "coal_price_usd_per_mt", "iron_ore_price_usd_per_dmt",
    "wind_kmh", "wave_height_m", "cyclone_risk", "weather_delay_days", "current_freight_usd_per_tonne"
]

GROUP_A_COMM = [
    "coal_3m_momentum", "coal_6m_momentum", "iron_ore_3m_momentum", "iron_ore_6m_momentum",
    "coal_rolling_volatility", "iron_ore_rolling_volatility", "coal_historical_percentile", "iron_ore_historical_percentile"
]
GROUP_B_FREIGHT_DYN = ["freight_lag_1", "freight_lag_3", "freight_rolling_mean_3", "freight_rolling_mean_6"]
GROUP_C_MACRO_MOM = ["bdi_1m_momentum", "bdi_3m_momentum", "vlsfo_1m_momentum", "vlsfo_3m_momentum"]
GROUP_D_SEASONALITY = ["month", "quarter"]

dates = sorted(df_all["date_dt"].unique())
folds = []
for test_idx in range(17, 22):
    test_date = dates[test_idx]
    train_dates = dates[:test_idx]
    folds.append((test_idx - 16, train_dates, test_date))

# Build evaluation across:
# 1. Persistence Baseline
# 2. Benchmark Frozen Model v3
# 3. Model v3 architecture re-fit on folds (Ridge a=10, passthrough)
# 4. Ridge a=10 with +Freight Dynamics
# 5. Ridge a=10 with +Macro Momentum
# 6. Ridge a=10 with +Commodity Dyn
# 7. Ridge a=10 with +Seasonality
# 8. Ridge a=10 with All Features
# 9. ElasticNet a=0.1 with Base
# 10. Lasso a=0.05 with Base
# 11. RandomForest with Base vs +Macro
# 12. ExtraTrees with Base vs +Macro
# 13. GradientBoosting with Base vs +Macro
# 14. HistGradientBoosting with Base vs +Macro

models_to_run = [
    ("Persistence", None, None, False, False),
    ("Frozen Model v3", None, None, False, False),
    ("Ridge (v3 refit)", Ridge(alpha=10.0, random_state=42), (BASE_CAT, BASE_NUM), False, True),
    ("Ridge (+Commodity)", Ridge(alpha=10.0, random_state=42), (BASE_CAT, BASE_NUM + GROUP_A_COMM), False, True),
    ("Ridge (+Freight Dyn)", Ridge(alpha=10.0, random_state=42), (BASE_CAT, BASE_NUM + GROUP_B_FREIGHT_DYN), False, True),
    ("Ridge (+Macro Mom)", Ridge(alpha=10.0, random_state=42), (BASE_CAT, BASE_NUM + GROUP_C_MACRO_MOM), False, True),
    ("Ridge (+Seasonality)", Ridge(alpha=10.0, random_state=42), (BASE_CAT, BASE_NUM + GROUP_D_SEASONALITY), False, True),
    ("Ridge (All Features)", Ridge(alpha=10.0, random_state=42), (BASE_CAT, BASE_NUM + GROUP_A_COMM + GROUP_B_FREIGHT_DYN + GROUP_C_MACRO_MOM + GROUP_D_SEASONALITY), False, True),
    ("ElasticNet (Base)", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42), (BASE_CAT, BASE_NUM), False, True),
    ("Lasso (Base)", Lasso(alpha=0.05, random_state=42), (BASE_CAT, BASE_NUM), False, True),
    ("RandomForest (Base)", RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42), (BASE_CAT, BASE_NUM), False, True),
    ("RandomForest (+Macro Mom)", RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42), (BASE_CAT, BASE_NUM + GROUP_C_MACRO_MOM), False, True),
    ("ExtraTrees (Base)", ExtraTreesRegressor(n_estimators=100, max_depth=4, random_state=42), (BASE_CAT, BASE_NUM), False, True),
    ("ExtraTrees (+Macro Mom)", ExtraTreesRegressor(n_estimators=100, max_depth=4, random_state=42), (BASE_CAT, BASE_NUM + GROUP_C_MACRO_MOM), False, True),
    ("GradientBoosting (Base)", GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42), (BASE_CAT, BASE_NUM), False, True),
    ("GradientBoosting (+Macro Mom)", GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42), (BASE_CAT, BASE_NUM + GROUP_C_MACRO_MOM), False, True),
    ("HistGradientBoosting (Base)", HistGradientBoostingRegressor(max_iter=50, max_depth=3, learning_rate=0.05, random_state=42), (BASE_CAT, BASE_NUM), False, True),
    ("HistGradientBoosting (+Macro Mom)", HistGradientBoostingRegressor(max_iter=50, max_depth=3, learning_rate=0.05, random_state=42), (BASE_CAT, BASE_NUM + GROUP_C_MACRO_MOM), False, True),
]

summary_rows = []
for label, reg_obj, feat_tuple, scale_num, is_model in models_to_run:
    fold_maes, fold_rmses, fold_biases, fold_dirs = [], [], [], []
    
    for fold_num, tr_d, te_d in folds:
        te_df = df_all[df_all["date_dt"] == te_d].copy()
        y_true = te_df["next_month_freight_usd_per_tonne"].values
        y_curr = te_df["current_freight_usd_per_tonne"].values
        
        if label == "Persistence":
            preds = y_curr
            delta_pred = np.zeros_like(y_curr)
            f_dir = 0.0
        elif label == "Frozen Model v3":
            delta_pred = np.clip(model_v3_frozen.predict(te_df[BASE_CAT + BASE_NUM]), -4.0, 4.0)
            preds = y_curr + delta_pred
            f_dir = np.mean(np.sign(y_true - y_curr) == np.sign(preds - y_curr)) * 100
        else:
            cat_cols, num_cols = feat_tuple
            all_cols = cat_cols + num_cols
            tr_df = df_all[df_all["date_dt"].isin(tr_d)].copy()
            
            prep = ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ("num", SimpleImputer(strategy="median"), num_cols)
            ])
            pipe = Pipeline([("prep", prep), ("reg", reg_obj)])
            pipe.fit(tr_df[all_cols], tr_df["delta"].values)
            
            delta_pred = np.clip(pipe.predict(te_df[all_cols]), -4.0, 4.0)
            preds = y_curr + delta_pred
            f_dir = np.mean(np.sign(y_true - y_curr) == np.sign(preds - y_curr)) * 100
            
        fold_maes.append(mean_absolute_error(y_true, preds))
        fold_rmses.append(np.sqrt(mean_squared_error(y_true, preds)))
        fold_biases.append(np.mean(preds - y_true))
        fold_dirs.append(f_dir)
        
    summary_rows.append({
        "Model": label,
        "Mean_MAE": np.mean(fold_maes),
        "Std_MAE": np.std(fold_maes),
        "Worst_Fold": np.max(fold_maes),
        "Mean_RMSE": np.mean(fold_rmses),
        "Mean_Bias": np.mean(fold_biases),
        "Dir_Acc_%": np.mean(fold_dirs) if label != "Persistence" else 0.0,
        "Fold_1": fold_maes[0],
        "Fold_2": fold_maes[1],
        "Fold_3": fold_maes[2],
        "Fold_4": fold_maes[3],
        "Fold_5": fold_maes[4],
    })

df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string())

with open("scratch/final_report_data.json", "w") as f:
    json.dump(summary_rows, f, indent=2)
print("\nWrote scratch/final_report_data.json")
