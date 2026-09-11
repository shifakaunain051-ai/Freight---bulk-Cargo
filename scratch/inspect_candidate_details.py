"""Inspect candidate model predictions vs Model v3.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from scipy import stats

DATA_PATH = Path("data/master_freight_training_expanded_v1.csv")
COMM_PATH = Path("data/commodity_prices_worldbank.csv")
MODEL_V3_PATH = Path("freight_forecast_model_v3.joblib")
v3_model = joblib.load(MODEL_V3_PATH)

df_freight = pd.read_csv(DATA_PATH)
df_comm = pd.read_csv(COMM_PATH)
df_comm["date_dt"] = pd.to_datetime(df_comm["date"]).sort_values().reset_index(drop=True)
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

print("Evaluating GradientBoosting with +Macro_Momentum fold by fold:")
for test_idx in range(17, 22):
    test_date = dates[test_idx]
    train_dates = dates[:test_idx]
    
    tr_df = df_all[df_all["date_dt"].isin(train_dates)].copy()
    te_df = df_all[df_all["date_dt"] == test_date].copy()
    
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), BASE_CAT),
        ("num", SimpleImputer(strategy="median"), BASE_NUM + MACRO_MOM)
    ])
    gbr = Pipeline([
        ("prep", prep),
        ("reg", GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42))
    ])
    gbr.fit(tr_df[FEATURES], tr_df["delta"])
    
    pred_delta_gbr = np.clip(gbr.predict(te_df[FEATURES]), -4.0, 4.0)
    preds_gbr = te_df["current_freight_usd_per_tonne"].values + pred_delta_gbr
    
    y_true = te_df["next_month_freight_usd_per_tonne"].values
    y_curr = te_df["current_freight_usd_per_tonne"].values
    
    print(f"\nFold Month {test_date.strftime('%Y-%m')}:")
    for i, (_, row) in enumerate(te_df.iterrows()):
        t_d = y_true[i] - y_curr[i]
        g_d = pred_delta_gbr[i]
        err = abs(y_true[i] - preds_gbr[i])
        print(f"  {row['origin'][:12]:12} | {row['vessel_type']:8} | Curr: ${y_curr[i]:5.2f} | True: ${y_true[i]:5.2f} (d={t_d:+5.2f}) | GBR: ${preds_gbr[i]:5.2f} (d={g_d:+5.2f}, Err: {err:.2f})")
