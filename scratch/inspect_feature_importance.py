"""Inspect feature importances for GradientBoosting and RandomForest.
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

prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), BASE_CAT),
    ("num", SimpleImputer(strategy="median"), BASE_NUM + MACRO_MOM)
])

gbr = Pipeline([
    ("prep", prep),
    ("reg", GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42))
])
gbr.fit(df_all[FEATURES], df_all["delta"])

cat_encoder = gbr.named_steps["prep"].named_transformers_["cat"]
cat_names = list(cat_encoder.get_feature_names_out(BASE_CAT))
num_names = BASE_NUM + MACRO_MOM
all_feat_names = cat_names + num_names

importances = gbr.named_steps["reg"].feature_importances_
df_imp = pd.DataFrame({"Feature": all_feat_names, "Importance": importances})
df_imp = df_imp.sort_values("Importance", ascending=False).reset_index(drop=True)

print("Top 15 Feature Importances in GradientBoostingRegressor:")
print(df_imp.head(15).to_string())
