"""Comprehensive Experimentation Engine for NaviFreight Model Improvement Research.
Executes Phases 3 to 10 in strict walk-forward temporal cross-validation.
"""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

# Verify Model v3 hash
v3_bytes = MODEL_V3_PATH.read_bytes()
v3_hash = hashlib.sha256(v3_bytes).hexdigest()
assert v3_hash == "71fbb870bb1f555d73a51ed7d83fb5a877cc4405ce54d1fe18407c9ce37c46a8"
model_v3_frozen = joblib.load(MODEL_V3_PATH)

# Load data
df_freight = pd.read_csv(DATA_PATH)
df_comm = pd.read_csv(COMM_PATH)

df_comm["date_dt"] = pd.to_datetime(df_comm["date"])
df_comm = df_comm.sort_values("date_dt").reset_index(drop=True)

df_freight["date_dt"] = pd.to_datetime(df_freight["date"])
df_freight = df_freight.sort_values(["origin", "destination", "commodity", "vessel_type", "date_dt"]).reset_index(drop=True)

# Phase 3: Strictly temporal commodity features (date <= t)
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

df_comm_feats = pd.DataFrame(comm_feats)
df_all = pd.concat([df_freight, df_comm_feats], axis=1)

# Phase 4: Freight / Macro temporal features
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

# Feature Definitions
BASE_CAT = ["origin", "destination", "commodity", "vessel_type"]
BASE_NUM = [
    "bdi", "vlsfo_usd_per_tonne", "coal_price_usd_per_mt", "iron_ore_price_usd_per_dmt",
    "wind_kmh", "wave_height_m", "cyclone_risk", "weather_delay_days", "current_freight_usd_per_tonne"
]
BASE_FEATURES = BASE_CAT + BASE_NUM

GROUP_A_COMM = [
    "coal_3m_momentum", "coal_6m_momentum", "iron_ore_3m_momentum", "iron_ore_6m_momentum",
    "coal_rolling_volatility", "iron_ore_rolling_volatility", "coal_historical_percentile", "iron_ore_historical_percentile"
]

GROUP_B_FREIGHT_DYN = [
    "freight_lag_1", "freight_lag_3", "freight_rolling_mean_3", "freight_rolling_mean_6"
]

GROUP_C_MACRO_MOM = [
    "bdi_1m_momentum", "bdi_3m_momentum", "vlsfo_1m_momentum", "vlsfo_3m_momentum"
]

GROUP_D_SEASONALITY = [
    "month", "quarter"
]

FEATURE_SETS = {
    "Base_v3": (BASE_CAT, BASE_NUM),
    "+Commodity_Dyn": (BASE_CAT, BASE_NUM + GROUP_A_COMM),
    "+Freight_Dynamics": (BASE_CAT, BASE_NUM + GROUP_B_FREIGHT_DYN),
    "+Macro_Momentum": (BASE_CAT, BASE_NUM + GROUP_C_MACRO_MOM),
    "+Seasonality": (BASE_CAT, BASE_NUM + GROUP_D_SEASONALITY),
    "All_Engineered": (BASE_CAT, BASE_NUM + GROUP_A_COMM + GROUP_B_FREIGHT_DYN + GROUP_C_MACRO_MOM + GROUP_D_SEASONALITY)
}

# 5 Expanding Folds (Months 18 to 22)
dates = sorted(df_all["date_dt"].unique())
folds = []
for test_idx in range(17, 22):
    test_date = dates[test_idx]
    train_dates = dates[:test_idx]
    folds.append((test_idx - 16, train_dates, test_date))

def build_model_pipeline(model_cls, kwargs, cat_cols, num_cols, scale_num=True):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_num:
        num_steps.append(("scaler", StandardScaler()))
    prep = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", Pipeline(num_steps), num_cols)
        ]
    )
    return Pipeline([("prep", prep), ("reg", model_cls(**kwargs))])

# Model candidates to test
CANDIDATE_ALGORITHMS = {
    "Ridge_a10 (v3 cfg)": (Ridge, {"alpha": 10.0, "random_state": 42}, True),
    "Ridge_a1.0": (Ridge, {"alpha": 1.0, "random_state": 42}, True),
    "ElasticNet_a0.1": (ElasticNet, {"alpha": 0.1, "l1_ratio": 0.5, "random_state": 42}, True),
    "Lasso_a0.05": (Lasso, {"alpha": 0.05, "random_state": 42}, True),
    "RandomForest": (RandomForestRegressor, {"n_estimators": 100, "max_depth": 4, "random_state": 42}, False),
    "ExtraTrees": (ExtraTreesRegressor, {"n_estimators": 100, "max_depth": 4, "random_state": 42}, False),
    "GradientBoosting": (GradientBoostingRegressor, {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.05, "random_state": 42}, False),
    "HistGradientBoosting": (HistGradientBoostingRegressor, {"max_iter": 50, "max_depth": 3, "learning_rate": 0.05, "random_state": 42}, False),
}

# Evaluation function
def evaluate_walk_forward():
    results = {}
    
    # Baseline 1: Persistence
    pers_fold_maes = []
    pers_fold_rmses = []
    pers_fold_biases = []
    pers_dir_accs = []
    
    # Baseline 2: Frozen Model v3 Artifact
    frozen_fold_maes = []
    frozen_fold_rmses = []
    frozen_fold_biases = []
    frozen_dir_accs = []
    
    for fold_num, tr_d, te_d in folds:
        te_df = df_all[df_all["date_dt"] == te_d].copy()
        y_true = te_df["next_month_freight_usd_per_tonne"].values
        y_curr = te_df["current_freight_usd_per_tonne"].values
        
        # Persistence: pred = current
        pers_preds = y_curr
        pers_fold_maes.append(mean_absolute_error(y_true, pers_preds))
        pers_fold_rmses.append(np.sqrt(mean_squared_error(y_true, pers_preds)))
        pers_fold_biases.append(np.mean(pers_preds - y_true))
        pers_dir_accs.append(np.nan) # Persistence predicts 0 delta
        
        # Frozen Model v3:
        pred_delta_v3 = model_v3_frozen.predict(te_df[BASE_FEATURES])
        pred_delta_v3 = np.clip(pred_delta_v3, -4.0, 4.0)
        v3_preds = y_curr + pred_delta_v3
        frozen_fold_maes.append(mean_absolute_error(y_true, v3_preds))
        frozen_fold_rmses.append(np.sqrt(mean_squared_error(y_true, v3_preds)))
        frozen_fold_biases.append(np.mean(v3_preds - y_true))
        true_dir = np.sign(y_true - y_curr)
        pred_dir = np.sign(v3_preds - y_curr)
        frozen_dir_accs.append(np.mean(true_dir == pred_dir) * 100)
        
    results["Baseline_Persistence"] = {
        "fold_maes": pers_fold_maes,
        "mean_mae": np.mean(pers_fold_maes),
        "std_mae": np.std(pers_fold_maes),
        "worst_fold_mae": np.max(pers_fold_maes),
        "mean_rmse": np.mean(pers_fold_rmses),
        "mean_bias": np.mean(pers_fold_biases),
        "dir_acc": 0.0,
    }
    
    results["Benchmark_Frozen_Model_v3"] = {
        "fold_maes": frozen_fold_maes,
        "mean_mae": np.mean(frozen_fold_maes),
        "std_mae": np.std(frozen_fold_maes),
        "worst_fold_mae": np.max(frozen_fold_maes),
        "mean_rmse": np.mean(frozen_fold_rmses),
        "mean_bias": np.mean(frozen_fold_biases),
        "dir_acc": np.mean(frozen_dir_accs),
    }

    # Now evaluate all Candidate Models across all Feature Sets
    model_matrix = []
    
    for fset_name, (cat_cols, num_cols) in FEATURE_SETS.items():
        all_cols = cat_cols + num_cols
        
        for algo_name, (model_cls, kwargs, scale_num) in CANDIDATE_ALGORITHMS.items():
            key = f"{algo_name} | {fset_name}"
            fold_maes, fold_rmses, fold_biases, dir_accs = [], [], [], []
            
            for fold_num, tr_d, te_d in folds:
                tr_df = df_all[df_all["date_dt"].isin(tr_d)].copy()
                te_df = df_all[df_all["date_dt"] == te_d].copy()
                
                y_tr_delta = tr_df["delta"].values
                y_te_true = te_df["next_month_freight_usd_per_tonne"].values
                y_te_curr = te_df["current_freight_usd_per_tonne"].values
                
                # Fit inside fold strictly!
                pipe = build_model_pipeline(model_cls, kwargs, cat_cols, num_cols, scale_num=scale_num)
                pipe.fit(tr_df[all_cols], y_tr_delta)
                
                # Predict delta and add current freight
                pred_delta = pipe.predict(te_df[all_cols])
                pred_delta = np.clip(pred_delta, -4.0, 4.0)
                preds = y_te_curr + pred_delta
                
                f_mae = mean_absolute_error(y_te_true, preds)
                f_rmse = np.sqrt(mean_squared_error(y_te_true, preds))
                f_bias = np.mean(preds - y_te_true)
                
                true_dir = np.sign(y_te_true - y_te_curr)
                pred_dir = np.sign(preds - y_te_curr)
                f_dir = np.mean(true_dir == pred_dir) * 100
                
                fold_maes.append(f_mae)
                fold_rmses.append(f_rmse)
                fold_biases.append(f_bias)
                dir_accs.append(f_dir)
                
            mean_mae = np.mean(fold_maes)
            std_mae = np.std(fold_maes)
            worst_mae = np.max(fold_maes)
            mean_rmse = np.mean(fold_rmses)
            mean_bias = np.mean(fold_biases)
            mean_dir = np.mean(dir_accs)
            
            pers_impr = ((results["Baseline_Persistence"]["mean_mae"] - mean_mae) / results["Baseline_Persistence"]["mean_mae"]) * 100
            v3_impr = ((results["Benchmark_Frozen_Model_v3"]["mean_mae"] - mean_mae) / results["Benchmark_Frozen_Model_v3"]["mean_mae"]) * 100
            
            res_dict = {
                "Algorithm": algo_name,
                "Feature_Set": fset_name,
                "Mean_MAE": round(mean_mae, 4),
                "Std_MAE": round(std_mae, 4),
                "Worst_Fold_MAE": round(worst_mae, 4),
                "Mean_RMSE": round(mean_rmse, 4),
                "Mean_Bias": round(mean_bias, 4),
                "Dir_Acc_%": round(mean_dir, 1),
                "vs_Persistence_%": round(pers_impr, 2),
                "vs_Model_v3_%": round(v3_impr, 2),
                "Fold_MAEs": [round(m, 4) for m in fold_maes]
            }
            model_matrix.append(res_dict)
            results[key] = res_dict
            
    return results, model_matrix

print("Running walk-forward evaluation across all model candidates and feature groups...")
results, model_matrix = evaluate_walk_forward()
df_res = pd.DataFrame(model_matrix)
df_res = df_res.sort_values("Mean_MAE").reset_index(drop=True)

print("\n" + "=" * 90)
print("TOP 15 CANDIDATE CONFIGURATIONS (by Mean Walk-Forward MAE)")
print("=" * 90)
print(df_res[["Algorithm", "Feature_Set", "Mean_MAE", "Std_MAE", "Worst_Fold_MAE", "Mean_RMSE", "Dir_Acc_%", "vs_Model_v3_%"]].head(15).to_string())

print("\n" + "=" * 90)
print("BENCHMARKS COMPARISON")
print("=" * 90)
pers = results["Baseline_Persistence"]
v3 = results["Benchmark_Frozen_Model_v3"]
print(f"Persistence Baseline: Mean MAE = {pers['mean_mae']:.4f} | Worst Fold = {pers['worst_fold_mae']:.4f} | RMSE = {pers['mean_rmse']:.4f}")
print(f"Frozen Model v3 Benchmark: Mean MAE = {v3['mean_mae']:.4f} | Worst Fold = {v3['worst_fold_mae']:.4f} | RMSE = {v3['mean_rmse']:.4f} | Dir Acc = {v3['dir_acc']:.1f}%")
print(f"Frozen Model v3 Fold MAEs: {[round(m, 4) for m in v3['fold_maes']]}")

# Save detailed results to json in scratch
with open("scratch/walk_forward_results.json", "w") as f:
    json.dump(model_matrix, f, indent=2)
print("\nSaved walk-forward results to scratch/walk_forward_results.json")
