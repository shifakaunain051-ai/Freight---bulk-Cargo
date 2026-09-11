import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge

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
dates = sorted(df["date_dt"].unique())

canonical_routes = [
    ("Australia West Coast", "East Coast India", "Iron Ore", "Capesize"),
    ("Hay Point", "East Coast India", "Coal", "Capesize"),
    ("Hay Point", "East Coast India", "Coal", "Panamax"),
    ("Taboneo", "East Coast India", "Thermal Coal", "Panamax"),
    ("Taboneo", "East Coast India", "Thermal Coal", "Supramax"),
]

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

# 1. Evaluate Ridge and Persistence
ridge_preds_all, pers_preds_all, actuals_all = [], [], []

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
    
    actuals_all.extend(test_df[TARGET].values)
    pers_preds_all.extend(test_df["current_freight_usd_per_tonne"].values)
    ridge_preds_all.extend(ridge_preds)

actuals_arr = np.array(actuals_all)
pers_arr = np.array(pers_preds_all)
ridge_arr = np.array(ridge_preds_all)

print("=== BASELINE & CURRENT MODEL ===")
print(f"Persistence Baseline: MAE = {mean_absolute_error(actuals_arr, pers_arr):.4f}, RMSE = {np.sqrt(mean_squared_error(actuals_arr, pers_arr)):.4f}")
ridge_dir = np.mean(np.sign(actuals_arr - pers_arr) == np.sign(ridge_arr - pers_arr)) * 100
print(f"Ridge Model v3:      MAE = {mean_absolute_error(actuals_arr, ridge_arr):.4f}, RMSE = {np.sqrt(mean_squared_error(actuals_arr, ridge_arr)):.4f}, DirAcc = {ridge_dir:.1f}%")

# 2. Test ARIMA grid (p, d, q) across routes
orders = [
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 0),
    (0, 1, 2),
    (2, 0, 0),
    (0, 1, 0), # random walk
]

print("\n=== EVALUATING PURE ARIMA MODELS (Per-route walk-forward) ===")
arima_results = {}

for order in orders:
    order_preds = []
    order_actuals = []
    order_pers = []
    
    for (o, d, c, v) in canonical_routes:
        route_df = df[(df["origin"] == o) & (df["destination"] == d) & (df["commodity"] == c) & (df["vessel_type"] == v)].sort_values("date_dt").reset_index(drop=True)
        # 22 observations
        for t in range(12, len(route_df)):
            # training observations up to t: route_df.iloc[:t]
            # test observation: route_df.iloc[t]
            # Notice target to predict is current_freight at t (which is next_month_freight of t-1)
            # route_df["current_freight_usd_per_tonne"].iloc[:t] has length t
            # we want to forecast step 1 ahead, which is current_freight at step t!
            y_train = route_df["current_freight_usd_per_tonne"].iloc[:t].values
            y_test = route_df["current_freight_usd_per_tonne"].iloc[t]
            pers_test = route_df["current_freight_usd_per_tonne"].iloc[t-1]
            
            try:
                mod = ARIMA(y_train, order=order)
                res = mod.fit()
                pred = float(res.forecast(steps=1)[0])
                pred = max(1.0, pred)
            except Exception as e:
                pred = pers_test
                
            order_preds.append(pred)
            order_actuals.append(y_test)
            order_pers.append(pers_test)
            
    preds_np = np.array(order_preds)
    acts_np = np.array(order_actuals)
    pers_np = np.array(order_pers)
    
    mae = mean_absolute_error(acts_np, preds_np)
    rmse = np.sqrt(mean_squared_error(acts_np, preds_np))
    dir_acc = np.mean(np.sign(acts_np - pers_np) == np.sign(preds_np - pers_np)) * 100
    arima_results[order] = (mae, rmse, dir_acc)
    print(f"ARIMA {str(order):10} : MAE = {mae:.4f} | RMSE = {rmse:.4f} | DirAcc = {dir_acc:5.1f}%")

# 3. Test SARIMAX with seasonality s=12 (if possible)
print("\n=== EVALUATING SARIMA(p,d,q)x(P,D,Q,12) ===")
seasonal_orders = [
    ((1, 0, 0), (1, 0, 0, 12)),
    ((1, 1, 0), (1, 0, 0, 12)),
    ((0, 1, 1), (0, 0, 1, 12)),
    ((1, 0, 0), (0, 1, 0, 12)),
    ((0, 1, 0), (0, 1, 0, 12)),
]

for order, s_order in seasonal_orders:
    s_preds, s_actuals, s_pers = [], [], []
    failed_fits = 0
    for (o, d, c, v) in canonical_routes:
        route_df = df[(df["origin"] == o) & (df["destination"] == d) & (df["commodity"] == c) & (df["vessel_type"] == v)].sort_values("date_dt").reset_index(drop=True)
        for t in range(12, len(route_df)):
            y_train = route_df["current_freight_usd_per_tonne"].iloc[:t].values
            y_test = route_df["current_freight_usd_per_tonne"].iloc[t]
            pers_test = route_df["current_freight_usd_per_tonne"].iloc[t-1]
            try:
                mod = SARIMAX(y_train, order=order, seasonal_order=s_order, enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit(disp=False)
                pred = float(res.forecast(steps=1)[0])
                pred = max(1.0, pred)
            except Exception as e:
                failed_fits += 1
                pred = pers_test
            s_preds.append(pred)
            s_actuals.append(y_test)
            s_pers.append(pers_test)
    preds_np = np.array(s_preds)
    acts_np = np.array(s_actuals)
    pers_np = np.array(s_pers)
    mae = mean_absolute_error(acts_np, preds_np)
    rmse = np.sqrt(mean_squared_error(acts_np, preds_np))
    dir_acc = np.mean(np.sign(acts_np - pers_np) == np.sign(preds_np - pers_np)) * 100
    print(f"SARIMA {str(order):10} x {str(s_order):15} (failed={failed_fits}): MAE = {mae:.4f} | RMSE = {rmse:.4f} | DirAcc = {dir_acc:5.1f}%")

# 4. Test ARIMAX with Exogenous Variables
print("\n=== EVALUATING ARIMAX WITH EXOGENOUS VARIABLES ===")
# Exogenous variables available at forecast origin (month t-1 when predicting freight at month t)
# Or contemporaneous market signals
exog_candidates = [
    ["bdi"],
    ["bdi", "vlsfo_usd_per_tonne"],
    ["bdi", "weather_delay_days"],
    ["bdi", "vlsfo_usd_per_tonne", "coal_price_usd_per_mt", "iron_ore_price_usd_per_dmt"],
    ["weather_delay_days", "cyclone_risk"],
    ["bdi", "weather_delay_days", "cyclone_risk"],
]

for ex_cols in exog_candidates:
    for order in [(1, 0, 0), (1, 1, 0), (0, 1, 1)]:
        arimax_preds, arimax_actuals, arimax_pers = [], [], []
        failed = 0
        for (o, d, c, v) in canonical_routes:
            route_df = df[(df["origin"] == o) & (df["destination"] == d) & (df["commodity"] == c) & (df["vessel_type"] == v)].sort_values("date_dt").reset_index(drop=True)
            for t in range(12, len(route_df)):
                y_train = route_df["current_freight_usd_per_tonne"].iloc[:t].values
                y_test = route_df["current_freight_usd_per_tonne"].iloc[t]
                pers_test = route_df["current_freight_usd_per_tonne"].iloc[t-1]
                
                # Exogenous variables available at forecast origin:
                # To predict freight at step t from history up to t-1:
                # The exogenous data up to t-1 is route_df[ex_cols].iloc[:t-1]
                # And the exogenous data at forecast origin (t-1) for step t is route_df[ex_cols].iloc[t-1:t]
                # Wait! Let's verify:
                # In ARIMAX regression: y_k = X_k * beta + eta_k.
                # If X_k is known at time k (e.g. market conditions during period k):
                # When forecasting period t, is X_t available at time t-1? NO!
                # Unless X is lagged: X_lag_k = X_{k-1}.
                # Let's test both lagged exog (leakage-safe) and current exog.
                X_train_lagged = route_df[ex_cols].iloc[:t-1].values
                y_train_lagged = y_train[1:] # aligned with X_{k-1}
                X_test_lagged = route_df[ex_cols].iloc[t-1:t].values # available at forecast origin t-1!
                
                try:
                    mod = SARIMAX(y_train_lagged, exog=X_train_lagged, order=order, enforce_stationarity=False, enforce_invertibility=False)
                    res = mod.fit(disp=False)
                    pred = float(res.forecast(steps=1, exog=X_test_lagged)[0])
                    pred = max(1.0, pred)
                except Exception as e:
                    failed += 1
                    pred = pers_test
                    
                arimax_preds.append(pred)
                arimax_actuals.append(y_test)
                arimax_pers.append(pers_test)
                
        preds_np = np.array(arimax_preds)
        acts_np = np.array(arimax_actuals)
        pers_np = np.array(arimax_pers)
        mae = mean_absolute_error(acts_np, preds_np)
        rmse = np.sqrt(mean_squared_error(acts_np, preds_np))
        dir_acc = np.mean(np.sign(acts_np - pers_np) == np.sign(preds_np - pers_np)) * 100
        print(f"ARIMAX {str(order):10} + {str(ex_cols):45} (failed={failed}): MAE = {mae:.4f} | RMSE = {rmse:.4f} | DirAcc = {dir_acc:5.1f}%")
