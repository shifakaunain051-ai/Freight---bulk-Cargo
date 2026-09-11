"""Phase 9: Robustness Tests for Candidate Models vs Model v3.
Stress tests under extreme conditions, shocks, and edge cases.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

DATA_PATH = Path("data/master_freight_training_expanded_v1.csv")
MODEL_V3_PATH = Path("freight_forecast_model_v3.joblib")
v3_model = joblib.load(MODEL_V3_PATH)

df_freight = pd.read_csv(DATA_PATH)
df_freight["date_dt"] = pd.to_datetime(df_freight["date"])

# Compute macro momentum
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
FEATURES_MOM = BASE_CAT + BASE_NUM + MACRO_MOM

# Train candidate GBR and RF on full dataset for fair inference comparison
prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), BASE_CAT),
    ("num", SimpleImputer(strategy="median"), BASE_NUM + MACRO_MOM)
])

gbr_pipe = Pipeline([
    ("prep", prep),
    ("reg", GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42))
])
gbr_pipe.fit(df_all[FEATURES_MOM], df_all["delta"])

rf_pipe = Pipeline([
    ("prep", prep),
    ("reg", RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42))
])
rf_pipe.fit(df_all[FEATURES_MOM], df_all["delta"])

# Define Stress Test Scenarios
base_scenario = {
    "origin": "Hay Point",
    "destination": "East Coast India",
    "commodity": "Coal",
    "vessel_type": "Capesize",
    "current_freight_usd_per_tonne": 14.00,
    "bdi": 1700,
    "vlsfo_usd_per_tonne": 630.0,
    "coal_price_usd_per_mt": 125.0,
    "iron_ore_price_usd_per_dmt": 105.0,
    "wind_kmh": 30.0,
    "wave_height_m": 2.2,
    "cyclone_risk": 2,
    "weather_delay_days": 1.0,
    "bdi_1m_momentum": 0.05,
    "bdi_3m_momentum": 0.10,
    "vlsfo_1m_momentum": 0.02,
    "vlsfo_3m_momentum": 0.04
}

scenarios = {
    "Baseline Normal": base_scenario.copy(),
    "High Freight Shock ($35/t)": {**base_scenario, "current_freight_usd_per_tonne": 35.0},
    "Low Freight Shock ($4/t)": {**base_scenario, "current_freight_usd_per_tonne": 4.0},
    "Extreme High BDI (4500)": {**base_scenario, "bdi": 4500, "bdi_1m_momentum": 0.80, "bdi_3m_momentum": 1.50},
    "Severe BDI Crash (500)": {**base_scenario, "bdi": 500, "bdi_1m_momentum": -0.60, "bdi_3m_momentum": -0.75},
    "Extreme High VLSFO ($1100/t)": {**base_scenario, "vlsfo_usd_per_tonne": 1100.0, "vlsfo_1m_momentum": 0.50},
    "Severe Weather / Cyclone 5": {**base_scenario, "cyclone_risk": 5, "weather_delay_days": 8.0, "wind_kmh": 85.0, "wave_height_m": 6.5},
    "Extreme Commodity Spike (Coal $350)": {**base_scenario, "coal_price_usd_per_mt": 350.0},
    "Missing Macro Momentum (NaNs)": {**base_scenario, "bdi_1m_momentum": np.nan, "bdi_3m_momentum": np.nan, "vlsfo_1m_momentum": np.nan, "vlsfo_3m_momentum": np.nan},
}

print(f"{'Scenario':36} | {'Model v3 (Ridge)':18} | {'GBR (+MacroMom)':18} | {'RF (+MacroMom)':18}")
print("-" * 96)

for name, sc in scenarios.items():
    df_sc = pd.DataFrame([sc])
    
    # Model v3
    v3_delta = np.clip(v3_model.predict(df_sc[list(v3_model.feature_names_in_)])[0], -4.0, 4.0)
    v3_pred = df_sc["current_freight_usd_per_tonne"].values[0] + v3_delta
    
    # GBR
    gbr_delta = np.clip(gbr_pipe.predict(df_sc[FEATURES_MOM])[0], -4.0, 4.0)
    gbr_pred = df_sc["current_freight_usd_per_tonne"].values[0] + gbr_delta
    
    # RF
    rf_delta = np.clip(rf_pipe.predict(df_sc[FEATURES_MOM])[0], -4.0, 4.0)
    rf_pred = df_sc["current_freight_usd_per_tonne"].values[0] + rf_delta
    
    print(f"{name:36} | ${v3_pred:6.2f} (d={v3_delta:+5.2f}) | ${gbr_pred:6.2f} (d={gbr_delta:+5.2f}) | ${rf_pred:6.2f} (d={rf_delta:+5.2f})")

