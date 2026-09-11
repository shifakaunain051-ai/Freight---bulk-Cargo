# PHASE 1 — FORECAST EXPLAINABILITY REPORT
## Closed-Form Mathematical Attribution & Explainability Layer for Model v3

---

## 1. Executive Summary

This report documents the design, verification, and implementation of the **Forecast Explainability Layer** for the Freight Intelligence Platform in preparation for the September 10 demonstration.

Without modifying the trained model artifact, altering inference behavior, or retraining the pipeline, we engineered a closed-form mathematical decomposition layer that answers:

> **"Why did the model predict this freight rate?"**

Every prediction now exposes:
1. **Dynamic Natural Language Summary**: A natural, context-aware forecast summary suitable for an executive dashboard.
2. **Ranked Feature Drivers**: Exact additive linear contributions ($\text{USD/tonne}$) for all market, weather, and trade corridor variables.
3. **Mathematical Anchor Decomposition**: Complete breakdown of base freight anchor, raw delta, model intercept, guardrail clipping, and physical floor state.

---

## 2. Current Architecture Discovered

| Component | Active Implementation | Verification Evidence |
|---|---|---|
| **Model Artifact** | [`freight_forecast_model_v3.joblib`](file:///c:/Users/junai/OneDrive/Documents/GitHub/Freight---bulk-Cargo/freight_forecast_model_v3.joblib) | SHA-256: `71fbb870bb1f555d73a51ed7d83fb5a877cc4405ce54d1fe18407c9ce37c46a8` |
| **Pipeline Class** | `sklearn.pipeline.Pipeline` | Steps: `[('prep', ColumnTransformer(...)), ('model', Ridge(alpha=10.0))]` |
| **Categorical Preprocessing** | `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` | 4 categoricals: `origin`, `destination`, `commodity`, `vessel_type` (10 one-hot columns) |
| **Numerical Preprocessing** | `passthrough` | 9 numerics: `bdi`, `vlsfo`, `coal_price`, `iron_ore_price`, `wind_kmh`, `wave_height_m`, `cyclone_risk`, `weather_delay_days`, `current_freight` |
| **Model Formulation** | Residual Ridge Regression ($\alpha=10.0$) | Intercept: $\beta_0 = -21.7341\text{ USD/t}$; 19 coefficients |
| **Inference Formula** | Level prediction with guardrails | $\hat{y}_{t+1} = \max(1.0, y_{\text{current}} + \text{clip}(\widehat{\Delta y}, -4.0, 4.0))$ |

---

## 3. Mathematical Attribution Methodology

Because the preprocessing pipeline uses standard One-Hot Encoding and direct passthrough for numeric features (with no non-linear kernels or polynomial expansions), the raw predicted residual delta $\widehat{\Delta y}$ is **strictly linear and additive**:

$$\widehat{\Delta y} = \beta_0 + \sum_{j=1}^{M} \beta_j \cdot x'_j$$

Where:
- $\beta_0 = -21.7341$ (global baseline intercept).
- For numerical features (e.g. VLSFO bunker price $x_k = \$638.00/\text{t}$ with $\beta_k = +0.0872$):
  $$\text{Contribution}_k = \beta_k \cdot x_k = (+0.0872) \times 638.00 = +55.63\text{ USD/t}$$
- For active categorical features (e.g. $\text{origin} = \text{"Hay Point"}$ with $\beta = +0.0714$):
  $$\text{Contribution}_{\text{origin}} = \beta_{\text{origin}=\text{Hay Point}} \times 1.0 = +0.0714\text{ USD/t}$$

### Numerical Precision Verification:
Across all tested samples:
$$\left| \widehat{\Delta y}_{\text{predict()}} - \left( \beta_0 + \sum \text{Contribution}_j \right) \right| \le 2.33 \times 10^{-15}\text{ USD/tonne}$$

This confirms that the explainability layer is **100% exact, closed-form, and non-fabricated**.

---

## 4. Model Coefficients Reference

| Feature | Feature Label | Unit | Ridge Coefficient $\beta$ | Sample Value | Contribution ($\text{USD/t}$) | Effect |
|---|---|---|---|---|---|---|
| `vlsfo_usd_per_tonne` | VLSFO Bunker Price | USD/tonne | `+0.0872` | 638.00 | `+55.6251` | Positive |
| `bdi` | Baltic Dry Index (BDI) | points | `-0.0116` | 1560.00 | `-18.0857` | Negative |
| `iron_ore_price_usd_per_dmt` | Iron Ore Price | USD/dmt | `-0.0630` | 124.00 | `-7.8149` | Negative |
| `coal_price_usd_per_mt` | Coal Benchmark Price | USD/MT | `-0.0439` | 124.00 | `-5.4492` | Negative |
| `wind_kmh` | Wind Speed | km/h | `-0.0888` | 32.00 | `-2.8405` | Negative |
| `cyclone_risk` | Cyclone Risk Score | 0-5 | `+0.6450` | 2.00 | `+1.2899` | Positive |
| `current_freight_usd_per_tonne` | Current Freight (Mean Reversion) | USD/tonne | `-0.0341` | 16.50 | `-0.5626` | Negative |
| `wave_height_m` | Significant Wave Height | m | `+0.2256` | 2.00 | `+0.4512` | Positive |
| `origin` | Loading Port (Hay Point) | — | `+0.0714` | Hay Point | `+0.0714` | Positive |
| `commodity` | Cargo Commodity (Coal) | — | `+0.0714` | Coal | `+0.0714` | Positive |
| `weather_delay_days` | Weather Delay Estimate | days | `-0.0997` | 0.50 | `-0.0498` | Negative |
| `vessel_type` | Vessel Class (Panamax) | — | `+0.0017` | Panamax | `+0.0017` | Positive |
| `destination` | Discharge Port (East Coast India) | — | `+0.0000` | East Coast India | `0.0000` | Neutral |

---

## 5. API Response Schema Changes

The `/predict` response was non-breakingly extended to include an optional `explanation` object:

```json
{
  "predicted_next_month_freight_usd_per_tonne": 17.47,
  "current_freight_usd_per_tonne": 16.50,
  "forecast_change_percent": 5.90,
  "risk_level": "LOW",
  "recommendation": "CHARTER NOW",
  "reason": "Forecast indicates freight rates will rise by 5.90%. Lock in current rates before they increase.",
  "sources": {
    "origin": "user",
    "destination": "user",
    "commodity": "user",
    "vessel_type": "user",
    "current_freight_usd_per_tonne": "user",
    "bdi": "user",
    "vlsfo_usd_per_tonne": "user",
    "coal_price_usd_per_mt": "user",
    "iron_ore_price_usd_per_dmt": "user",
    "wind_kmh": "user",
    "wave_height_m": "user",
    "cyclone_risk": "user",
    "weather_delay_days": "user"
  },
  "explanation": {
    "summary": "Freight is projected to increase from $16.50/t to $17.47/t (+5.90%). The model's upward forecast (+$0.97/t) is primarily driven by VLSFO Bunker Price. Downward pressure from Baltic Dry Index (BDI) partially offset this increase.",
    "drivers": [
      {
        "feature": "vlsfo_usd_per_tonne",
        "feature_label": "VLSFO Bunker Price",
        "value": 638.0,
        "unit": "USD/tonne",
        "coefficient": 0.0872,
        "contribution_usd_per_tonne": 55.6251,
        "effect": "positive",
        "source": "model"
      },
      {
        "feature": "bdi",
        "feature_label": "Baltic Dry Index (BDI)",
        "value": 1560.0,
        "unit": "points",
        "coefficient": -0.0116,
        "contribution_usd_per_tonne": -18.0857,
        "effect": "negative",
        "source": "model"
      },
      {
        "feature": "cyclone_risk",
        "feature_label": "Cyclone Risk Score",
        "value": 2.0,
        "unit": "0-5",
        "coefficient": 0.645,
        "contribution_usd_per_tonne": 1.2899,
        "effect": "positive",
        "source": "model"
      }
    ],
    "anchor": {
      "current_freight_usd_per_tonne": 16.50,
      "predicted_next_month_freight_usd_per_tonne": 17.47,
      "raw_predicted_delta_usd_per_tonne": 0.974,
      "bounded_delta_usd_per_tonne": 0.974,
      "model_intercept": -21.7341,
      "residual_guardrail_applied": false,
      "physical_floor_applied": false
    }
  }
}
```

---

## 6. Files Modified

| File | Status | Description |
|---|---|---|
| [`backend/schemas.py`](file:///c:/Users/junai/OneDrive/Documents/GitHub/Freight---bulk-Cargo/backend/schemas.py) | **MODIFIED** | Added Pydantic schemas: `ExplanationDriver`, `ExplanationAnchor`, `PredictionExplanation`. Added `explanation` field to `FreightResponse`. |
| [`backend/predict.py`](file:///c:/Users/junai/OneDrive/Documents/GitHub/Freight---bulk-Cargo/backend/predict.py) | **MODIFIED** | Added `compute_explanation()` deriving closed-form linear feature contributions and dynamic natural language summaries. |
| [`backend/test_suite.py`](file:///c:/Users/junai/OneDrive/Documents/GitHub/Freight---bulk-Cargo/backend/test_suite.py) | **MODIFIED** | Added `TestForecastExplainability` test class covering numerical consistency and 5-route coverage (19 total tests passing). |

---

## 7. Verification & Automated Test Results

| Test Suite | Total Tests | Pass Rate | Execution Time | Coverage Details |
|---|---|---|---|---|
| `backend/test_suite.py` | 19 | **100% PASS** | 0.24s | Mathematical attribution, BDI constraints, 5-route predictions, explanation schemas. |
| `backend/test_v3_model.py` | 4 | **100% PASS** | 0.13s | End-to-end API and direct pipeline integration across all 5 canonical routes. |
| `backend/test_concurrency.py` | 1 | **100% PASS** | 0.78s | Multithreaded SQLite WAL concurrency under load. |
| **Total Test Suite** | **24** | **100% PASS** | **1.15s** | **All tests passing cleanly.** |

---

## 8. Checksum & Zero-Modification Verification

- **Model Checksum Before**: `71fbb870bb1f555d73a51ed7d83fb5a877cc4405ce54d1fe18407c9ce37c46a8`
- **Model Checksum After**: `71fbb870bb1f555d73a51ed7d83fb5a877cc4405ce54d1fe18407c9ce37c46a8` (**100% UNCHANGED**)
- **Retraining**: **None** (Zero retraining performed).
- **Synthetic Data**: **None** (Quarantine preserved).

---

*Phase 1 Complete. Ready for demonstration review.*
