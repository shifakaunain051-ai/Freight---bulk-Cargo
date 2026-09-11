# MODEL V3 INTEGRATION REPORT
## Production Activation & Benchmark Verification for Model v3

---

## 1. Model Architecture

**Model v3** implements a **Bounded Residual Ridge Regression** architecture:

$$\hat{y}_{t+1} = \text{current\_freight\_usd\_per\_tonne}_t + \text{clip}(\widehat{\Delta y}, -4.0, 4.0)$$

where:
- $\widehat{\Delta y} = \mathbf{w}^T \mathbf{x} + b$ is the predicted 1-month freight rate change (USD/tonne).
- Preprocessing: `ColumnTransformer` with `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` on the 4 categorical features, and `passthrough` on the 9 numerical features.
- Estimator: `sklearn.linear_model.Ridge(alpha=10.0, random_state=42)` fitted on $\Delta y = y_{t+1} - y_t$.
- Guardrail: $[-4.0, +4.0]\text{ USD/t}$ (training-derived defensive boundary).

---

## 2. Training Dataset

- **Primary Dataset**: `data/master_freight_training_expanded_v1.csv`
- **Total Rows**: 110 genuine real historical observations (22 consecutive monthly periods, `2024-02-01` to `2025-11-01`).
- **Trade Combinations**: 5 fixed trade lanes (22 rows per combination).
- **Missing Data**: 0 missing values across all columns.
- **Synthetic Data**: Excluded from training (`master_freight_training_synthetic_v2.csv` remains quarantined).

---

## 3. Training Configuration

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge

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

pipeline = Pipeline([
    ("prep", prep),
    ("model", Ridge(alpha=10.0, random_state=42)),
])
```

---

## 4. Feature Contract

The active contract contains exactly **13 features** (`cargo_tonnes` is excluded):

1. `origin` (categorical)
2. `destination` (categorical)
3. `commodity` (categorical)
4. `vessel_type` (categorical)
5. `bdi` (numeric)
6. `vlsfo_usd_per_tonne` (numeric)
7. `coal_price_usd_per_mt` (numeric)
8. `iron_ore_price_usd_per_dmt` (numeric)
9. `wind_kmh` (numeric)
10. `wave_height_m` (numeric)
11. `cyclone_risk` (numeric)
12. `weather_delay_days` (numeric)
13. `current_freight_usd_per_tonne` (numeric)

Target: `next_month_freight_usd_per_tonne`

---

## 5. Model Artifact Details & Hash

| Property | Value |
|---|---|
| **Artifact Path** | `freight_forecast_model_v3.joblib` |
| **File Size** | `4,163 bytes` |
| **SHA-256 Checksum** | `71fbb870bb1f555d73a51ed7d83fb5a877cc4405ce54d1fe18407c9ce37c46a8` |
| **Historical Models Preserved** | `freight_forecast_model_final.joblib` (UNTOUCHED)<br>`freight_forecast_model_v1.joblib` (UNTOUCHED) |

---

## 6. Validation Metrics (Out-of-Sample Holdout: Months 18–22, 25 Obs)

- **Holdout MAE**: **0.4730 USD/tonne**
- **Holdout RMSE**: **0.5810 USD/tonne**
- **Holdout $R^2$**: **0.9655**
- **Holdout MAPE**: **3.47%**
- **Directional Accuracy**: **60.0%**
- **Mean Signed Bias**: **−0.1653 USD/tonne**

---

## 7. Comparison with Persistence Baseline

| Metric | Persistence Baseline | Model v3 | Improvement |
|---|---|---|---|
| **MAE** | 0.8280 USD/t | **0.4730 USD/t** | **−42.88% error reduction** |
| **RMSE** | 1.0750 USD/t | **0.5810 USD/t** | **−45.95% error reduction** |
| **$R^2$** | 0.8819 | **0.9655** | **+0.0836** |
| **Directional Accuracy** | 0.0% | **60.0%** | **+60.0% points** |

---

## 8. Comparison with Previous Production Model (Random Forest)

| Model | Training Dataset | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Status on Real Data |
|---|---|---|---|---|---|
| **Model v3 [ACTIVE]** | **110 Real Rows** | **0.4730** | **0.5810** | **0.9655** | **Beats Persistence by 42.9%** |
| **Persistence Baseline** | — | 0.8280 | 1.0750 | 0.8819 | Baseline |
| **Model Final (RF)** | 1,110 Synthetic v2 Rows | 1.2058 | 1.3765 | 0.8064 | +45.6% Worse than Persistence |
| **Random Forest (Real)** | 85 Real Rows | 1.2390 | 1.5144 | 0.7656 | +49.6% Worse than Persistence |

Model v3 outperforms the previous production Random Forest model on **22 out of 25 holdout observations (88.0%)**.

---

## 9. API Integration Changes

1. `backend/predict.py`:
   - Updated default `MODEL_PATH` to `freight_forecast_model_v3.joblib`.
   - Added automatic model detection and residual inference logic:
     $$\text{forecast} = \text{current\_freight} + \text{clip}(\text{pred\_delta}, -4.0, 4.0)$$
   - Implemented `get_model_metadata()` for dynamic introspection.
2. `backend/main.py`:
   - Updated application version to `3.0.0`.
   - Updated description to reflect Model v3 (Bounded Residual Ridge Regression).
   - Dynamic `/model/info` endpoint returning live model metadata.

---

## 10. Test Results

- **Test Suite 1 (`backend/test_suite.py`)**: 19 tests passing 100% in 0.098s.
- **Test Suite 2 (`backend/test_v3_model.py`)**: 4 tests passing 100% in 0.073s.
- Tested:
  - Artifact loading and feature inspection
  - 13-feature contract validation
  - Inference on all 5 supported combinations
  - API `POST /predict` end-to-end
  - Change percentage calculation
  - Weather risk classification (`LOW`/`MEDIUM`/`HIGH`)
  - Recommendation logic (`CHARTER NOW`/`WAIT`/`MONITOR`)
  - 422 error validation for missing required fields
  - Data provenance tracking

---

## 11. Model Info Verification (`GET /model/info`)

```json
{
  "model": "freight_forecast_model_v3",
  "version": "3.0.0",
  "algorithm": "Bounded Residual Ridge Regression",
  "alpha": 10.0,
  "residual_guardrail_usd_per_tonne": [-4.0, 4.0],
  "features": 13,
  "feature_names": [
    "origin", "destination", "commodity", "vessel_type",
    "bdi", "vlsfo_usd_per_tonne", "coal_price_usd_per_mt",
    "iron_ore_price_usd_per_dmt", "wind_kmh", "wave_height_m",
    "cyclone_risk", "weather_delay_days", "current_freight_usd_per_tonne"
  ],
  "excludes_cargo_tonnes": true,
  "model_file": "freight_forecast_model_v3.joblib",
  "training_dataset": "master_freight_training_expanded_v1.csv (110 real observations)",
  "synthetic_data_used": false
}
```

---

## 12. Known Limitations

1. **Dataset Scope**: Model v3 is calibrated on 110 real monthly observations across 22 consecutive months. Continued ingestion of real fixture data will refine long-term seasonal trends.
2. **Geographic Coverage**: Primary training covers 5 key bulk trade lanes to East Coast India. Unseen origins/destinations use one-hot unknown encoding fallback.

---

## 13. Rollback Procedure

All legacy models are preserved in the repository root:
- `freight_forecast_model_final.joblib` (Model v2 Random Forest)
- `freight_forecast_model_v1.joblib` (Model v1)

To rollback immediately to Model v2, update `backend/predict.py`:
```python
MODEL_PATH = REPO_ROOT / "freight_forecast_model_final.joblib"
```
The backend automatically detects legacy level-prediction models and operates seamlessly.

---

*Model v3 successfully trained, verified, and activated in production.*
