import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv("data/master_freight_training_expanded_v1.csv")
df["date_dt"] = pd.to_datetime(df["date"])

canonical_routes = [
    ("Australia West Coast", "East Coast India", "Iron Ore", "Capesize"),
    ("Hay Point", "East Coast India", "Coal", "Capesize"),
    ("Hay Point", "East Coast India", "Coal", "Panamax"),
    ("Taboneo", "East Coast India", "Thermal Coal", "Panamax"),
    ("Taboneo", "East Coast India", "Thermal Coal", "Supramax"),
]

# Test ARIMA(2,0,0) with and without exog
configs = [
    ("ARIMA (2,0,0)", (2, 0, 0), []),
    ("ARIMA (0,1,1)", (0, 1, 1), []),
    ("ARIMA (1,1,0)", (1, 1, 0), []),
    ("ARIMAX (2,0,0) + [bdi]", (2, 0, 0), ["bdi"]),
    ("ARIMAX (2,0,0) + [bdi, vlsfo]", (2, 0, 0), ["bdi", "vlsfo_usd_per_tonne"]),
    ("ARIMAX (0,1,1) + [bdi]", (0, 1, 1), ["bdi"]),
    ("ARIMAX (0,1,1) + [weather_delay]", (0, 1, 1), ["weather_delay_days"]),
]

print("=== DETAILED COMPARISON ON EXPANDING WINDOW (12 -> 22) ===")

for label, order, ex_cols in configs:
    preds_all, actuals_all, pers_all = [], [], []
    failed = 0
    route_stats = {}
    
    for (o, d, c, v) in canonical_routes:
        route_key = f"{o} | {v}"
        route_df = df[(df["origin"] == o) & (df["destination"] == d) & (df["commodity"] == c) & (df["vessel_type"] == v)].sort_values("date_dt").reset_index(drop=True)
        r_preds, r_actuals, r_pers = [], [], []
        
        for t in range(12, len(route_df)):
            y_train = route_df["current_freight_usd_per_tonne"].iloc[:t].values
            y_test = route_df["current_freight_usd_per_tonne"].iloc[t]
            pers_test = route_df["current_freight_usd_per_tonne"].iloc[t-1]
            
            if ex_cols:
                # Lagged exog available at forecast origin
                X_tr = route_df[ex_cols].iloc[:t-1].values
                y_tr = y_train[1:]
                X_te = route_df[ex_cols].iloc[t-1:t].values
            else:
                X_tr = None
                y_tr = y_train
                X_te = None
                
            try:
                mod = SARIMAX(y_tr, exog=X_tr, order=order, enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit(disp=False)
                pred = float(res.forecast(steps=1, exog=X_te)[0])
                pred = max(1.0, pred)
            except Exception as e:
                failed += 1
                pred = pers_test
                
            r_preds.append(pred)
            r_actuals.append(y_test)
            r_pers.append(pers_test)
            
        r_mae = mean_absolute_error(r_actuals, r_preds)
        r_dir = np.mean(np.sign(np.array(r_actuals) - np.array(r_pers)) == np.sign(np.array(r_preds) - np.array(r_pers))) * 100
        route_stats[route_key] = (r_mae, r_dir)
        preds_all.extend(r_preds)
        actuals_all.extend(r_actuals)
        pers_all.extend(r_pers)
        
    p_np = np.array(preds_all)
    a_np = np.array(actuals_all)
    pe_np = np.array(pers_all)
    
    tot_mae = mean_absolute_error(a_np, p_np)
    tot_rmse = np.sqrt(mean_squared_error(a_np, p_np))
    tot_dir = np.mean(np.sign(a_np - pe_np) == np.sign(p_np - pe_np)) * 100
    
    print(f"\n{label:35}: Overall MAE={tot_mae:.4f}, RMSE={tot_rmse:.4f}, DirAcc={tot_dir:.1f}% (failed={failed})")
    for rk, (rm, rd) in route_stats.items():
        print(f"   {rk:35} -> MAE: {rm:.4f}, DirAcc: {rd:.1f}%")
