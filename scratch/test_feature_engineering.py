"""Test script to implement and verify Phases 3 & 4 feature engineering.
Ensures zero future leakage.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

DATA_PATH = Path("data/master_freight_training_expanded_v1.csv")
COMM_PATH = Path("data/commodity_prices_worldbank.csv")

df_freight = pd.read_csv(DATA_PATH)
df_comm = pd.read_csv(COMM_PATH)

df_comm["date_dt"] = pd.to_datetime(df_comm["date"])
df_comm = df_comm.sort_values("date_dt").reset_index(drop=True)

df_freight["date_dt"] = pd.to_datetime(df_freight["date"])
df_freight = df_freight.sort_values(["origin", "destination", "commodity", "vessel_type", "date_dt"]).reset_index(drop=True)

# 1. Commodity Feature Engineering (Phase 3)
# For each freight observation at date t, use ONLY df_comm[df_comm['date_dt'] <= t]
comm_features_list = []
for idx, row in df_freight.iterrows():
    t = row["date_dt"]
    sub = df_comm[df_comm["date_dt"] <= t].copy()
    
    # Must have at least 12 months of history
    assert len(sub) >= 12, f"Insufficient commodity history for {t}"
    
    # Prices at t
    coal_t = sub.iloc[-1]["coal_price_usd_per_mt"]
    iron_t = sub.iloc[-1]["iron_ore_price_usd_per_dmt"]
    
    # 3m momentum: (P_t - P_{t-3}) / P_{t-3}
    coal_3m_ago = sub.iloc[-4]["coal_price_usd_per_mt"]
    iron_3m_ago = sub.iloc[-4]["iron_ore_price_usd_per_dmt"]
    coal_3m_mom = (coal_t - coal_3m_ago) / coal_3m_ago
    iron_3m_mom = (iron_t - iron_3m_ago) / iron_3m_ago
    
    # 6m momentum: (P_t - P_{t-6}) / P_{t-6}
    coal_6m_ago = sub.iloc[-7]["coal_price_usd_per_mt"]
    iron_6m_ago = sub.iloc[-7]["iron_ore_price_usd_per_dmt"]
    coal_6m_mom = (coal_t - coal_6m_ago) / coal_6m_ago
    iron_6m_mom = (iron_t - iron_6m_ago) / iron_6m_ago
    
    # 6m rolling volatility (std of monthly percentage returns over last 6 months)
    coal_returns_6m = sub["coal_price_usd_per_mt"].pct_change().iloc[-6:]
    iron_returns_6m = sub["iron_ore_price_usd_per_dmt"].pct_change().iloc[-6:]
    coal_roll_vol = coal_returns_6m.std()
    iron_roll_vol = iron_returns_6m.std()
    
    # Historical expanding percentile: percentile of P_t among all prices <= t
    coal_all = sub["coal_price_usd_per_mt"].dropna().values
    iron_all = sub["iron_ore_price_usd_per_dmt"].dropna().values
    coal_hist_pctile = stats.percentileofscore(coal_all, coal_t)
    iron_hist_pctile = stats.percentileofscore(iron_all, iron_t)
    
    comm_features_list.append({
        "coal_3m_momentum": coal_3m_mom,
        "coal_6m_momentum": coal_6m_mom,
        "iron_ore_3m_momentum": iron_3m_mom,
        "iron_ore_6m_momentum": iron_6m_mom,
        "coal_rolling_volatility": coal_roll_vol,
        "iron_ore_rolling_volatility": iron_roll_vol,
        "coal_historical_percentile": coal_hist_pctile,
        "iron_ore_historical_percentile": iron_hist_pctile,
    })

df_comm_feats = pd.DataFrame(comm_features_list)
print(f"Generated {len(df_comm_feats)} rows of commodity features.")
print(df_comm_feats.describe().T[["mean", "std", "min", "50%", "max"]].to_string())

# 2. Freight and Macro Feature Engineering (Phase 4)
# Group by corridor to compute freight lags and rolling means
df_freight = pd.concat([df_freight, df_comm_feats], axis=1)

macro_features_list = []
for (o, d, c, v), grp in df_freight.groupby(["origin", "destination", "commodity", "vessel_type"]):
    grp = grp.sort_values("date_dt").copy()
    
    # Freight lags & rolling means
    grp["freight_lag_1"] = grp["current_freight_usd_per_tonne"].shift(1)
    grp["freight_lag_3"] = grp["current_freight_usd_per_tonne"].shift(3)
    grp["freight_rolling_mean_3"] = grp["current_freight_usd_per_tonne"].rolling(3, min_periods=1).mean()
    grp["freight_rolling_mean_6"] = grp["current_freight_usd_per_tonne"].rolling(6, min_periods=1).mean()
    
    # Macro momentum
    grp["bdi_1m_momentum"] = grp["bdi"].pct_change(1)
    grp["bdi_3m_momentum"] = grp["bdi"].pct_change(3)
    grp["vlsfo_1m_momentum"] = grp["vlsfo_usd_per_tonne"].pct_change(1)
    grp["vlsfo_3m_momentum"] = grp["vlsfo_usd_per_tonne"].pct_change(3)
    
    # Seasonality
    grp["month"] = grp["date_dt"].dt.month
    grp["quarter"] = grp["date_dt"].dt.quarter
    
    macro_features_list.append(grp)

df_engineered = pd.concat(macro_features_list).sort_index()
print(f"\nEngineered dataset shape: {df_engineered.shape}")
print("Null counts in new features:")
new_cols = [
    "coal_3m_momentum", "coal_6m_momentum", "iron_ore_3m_momentum", "iron_ore_6m_momentum",
    "coal_rolling_volatility", "iron_ore_rolling_volatility", "coal_historical_percentile", "iron_ore_historical_percentile",
    "freight_lag_1", "freight_lag_3", "freight_rolling_mean_3", "freight_rolling_mean_6",
    "bdi_1m_momentum", "bdi_3m_momentum", "vlsfo_1m_momentum", "vlsfo_3m_momentum",
    "month", "quarter"
]
print(df_engineered[new_cols].isnull().sum().to_dict())
