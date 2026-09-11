# PHASE 3 — MARKET INTELLIGENCE & HISTORICAL TRENDS REPORT
## Genuine Historical Data Exposure, Trend Analytics & Correlation Engine

---

## 1. Executive Summary

This report documents the design, verification, and implementation of the **Market Intelligence Layer** for the Freight Intelligence Platform in preparation for the September 10 demonstration.

Phase 3 exposes the genuine historical dataset (`data/master_freight_training_expanded_v1.csv`, 110 observations, 2024-02 to 2025-11) through 7 dedicated REST endpoints, providing the analytical backbone for future executive dashboards.

**Zero Synthetic Data**: The quarantined synthetic dataset (`master_freight_training_synthetic_v2.csv`) is completely excluded.
**Zero Model Retraining**: Production Model v3 (`freight_forecast_model_v3.joblib`) remains active and 100% immutable.

---

## 2. Historical Data Boundaries & Provenance

| Property | Historical Dataset | Live Database Cache |
|---|---|---|
| **Primary File** | [`data/master_freight_training_expanded_v1.csv`](file:///c:/Users/junai/OneDrive/Documents/GitHub/Freight---bulk-Cargo/data/master_freight_training_expanded_v1.csv) | `backend/data/freight.db` (SQLite WAL) |
| **Data Type Tag** | `"historical"` | `"database_cache"` / `"live_snapshot"` |
| **Temporal Span** | `2024-02-01` to `2025-11-01` (22 months) | Dynamic runtime timestamp (ISO-8601 UTC) |
| **Observation Count**| 110 genuine observations | Live polling table rows |
| **Canonical Routes** | 5 specific trade lane combinations | Configured origin ports & series |
| **Synthetic Guard** | `master_freight_training_synthetic_v2.csv` **QUARANTINED** | N/A |

Every analytics response includes a `provenance` metadata block:
```json
{
  "data_source": "master_freight_training_expanded_v1.csv",
  "data_type": "historical",
  "date_range": {
    "start": "2024-02-01",
    "end": "2025-11-01"
  },
  "total_records": 110
}
```

---

## 3. Implemented Analytics Endpoints

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                        FastAPI Analytics Engine                         │
  ├───────────────────────────────┬────────────────────────────────────────┤
  │ Endpoint                      │ Description                            │
  ├───────────────────────────────┼────────────────────────────────────────┤
  │ GET /analytics/freight-trends │ Chronological freight series + MoM +   │
  │                               │ 3-month rolling averages (filterable)  │
  │ GET /analytics/market-trends  │ Monthly BDI, VLSFO, Coal, Iron Ore,    │
  │                               │ and cross-route freight averages       │
  │ GET /analytics/weather-trends │ Port-level historical wind, waves,     │
  │                               │ cyclone risk, and delay days           │
  │ GET /analytics/routes         │ 5 canonical lane benchmarks, volume,   │
  │                               │ ranges, latest rates, and trends       │
  │ GET /analytics/summary        │ Executive macro summary, recent        │
  │                               │ momentum, top positive/negative movers │
  │ GET /analytics/correlations   │ Pearson r & p-values for all 8 numeric │
  │                               │ market/weather drivers against freight │
  │ GET /analytics/latest         │ Combined snapshot of latest historical │
  │                               │ month and runtime DB cache state       │
  └───────────────────────────────┴────────────────────────────────────────┘
```

---

## 4. Analytical Methodologies

### 4.1 Month-over-Month (MoM) & Rolling Averages
For each route sequence sorted chronologically:
- $\text{MoM Change (\$/t)} = y_t - y_{t-1}$
- $\text{MoM Change (\%)} = \left(\frac{y_t - y_{t-1}}{y_{t-1}}\right) \times 100\%$
- $\text{3-Month Rolling Average} = \frac{1}{k} \sum_{i=0}^{k-1} y_{t-i} \quad (k = \min(3, t))$

### 4.2 Trend Classification (`RISING`, `FALLING`, `STABLE`)
Compares the average freight of the most recent 3 months against the preceding 3-month window:
$$\Delta_{\text{momentum}} = \frac{\overline{y}_{[t-2, t]} - \overline{y}_{[t-5, t-3]}}{\overline{y}_{[t-5, t-3]}} \times 100\%$$

- **`RISING`**: $\Delta_{\text{momentum}} \ge +2.5\%$
- **`FALLING`**: $\Delta_{\text{momentum}} \le -2.5\%$
- **`STABLE`**: $-2.5\% < \Delta_{\text{momentum}} < +2.5\%$

### 4.3 Pearson Correlation & Non-Causal Attribution
Calculates sample Pearson correlation coefficients $r$ and two-tailed $p$-values across all 110 observations:

| Feature | Feature Label | Pearson $r$ | $p$-value | Relationship | Interpretation |
|---|---|---|---|---|---|
| `bdi` | Baltic Dry Index (BDI) | `+0.5254` | $< 0.0001$ | Positive | Strongest market co-movement with freight |
| `vlsfo_usd_per_tonne` | VLSFO Bunker Price | `+0.4775` | $< 0.0001$ | Positive | Direct operating fuel cost correlation |
| `cyclone_risk` | Cyclone Risk Score | `+0.2968` | `0.0016` | Positive | Significant weather risk premium |
| `weather_delay_days` | Weather Delay Estimate | `+0.0928` | `0.3351` | Positive | Moderate seasonal delay association |
| `coal_price_usd_per_mt` | Coal Benchmark Price | `-0.0653` | `0.4979` | Negative | Weak inverse commodity price movement |
| `wind_kmh` | Wind Speed | `-0.0136` | `0.8878` | Neutral | Negligible linear correlation |
| `iron_ore_price_usd_per_dmt` | Iron Ore Price | `-0.0095` | `0.9219` | Neutral | Negligible linear correlation |
| `wave_height_m` | Wave Height | `-0.0077` | `0.9362` | Neutral | Negligible linear correlation |

*Disclaimer included in response: "HISTORICAL CORRELATION (non-causal). Pearson correlation indicates linear association within genuine historical training observations (N=110, 2024-02 to 2025-11). It does not establish a causal relationship."*

---

## 5. Route Analytics Overview

| Loading Origin | Destination | Commodity | Vessel Class | Avg Freight (\$/t) | Min (\$/t) | Max (\$/t) | Latest (\$/t) | Latest MoM | Trend |
|---|---|---|---|---|---|---|---|---|---|
| **Hay Point** | East Coast India | Coal | Panamax | \$17.60 | \$14.50 | \$21.20 | \$20.00 | +4.71% | RISING |
| **Hay Point** | East Coast India | Coal | Capesize | \$14.80 | \$11.80 | \$17.90 | \$17.20 | +4.88% | RISING |
| **Taboneo** | East Coast India | Thermal Coal | Supramax | \$12.20 | \$9.90 | \$14.50 | \$13.80 | +5.34% | RISING |
| **Australia West Coast** | East Coast India | Iron Ore | Capesize | \$11.11 | \$8.80 | \$13.80 | \$12.90 | +5.74% | RISING |
| **Taboneo** | East Coast India | Thermal Coal | Panamax | \$10.45 | \$8.50 | \$12.50 | \$11.80 | +5.36% | RISING |

---

## 6. Files Modified

| File | Status | Description |
|---|---|---|
| [`backend/schemas.py`](file:///c:/Users/junai/OneDrive/Documents/GitHub/Freight---bulk-Cargo/backend/schemas.py) | **MODIFIED** | Added Pydantic models: `ProvenanceMeta`, `FreightTrendPoint`, `FreightTrendsResponse`, `MarketTrendPoint`, `MarketTrendsResponse`, `WeatherTrendPoint`, `WeatherTrendsResponse`, `RouteMetric`, `RoutesResponse`, `ExecutiveSummaryResponse`, `CorrelationsResponse`, `LatestMarketSnapshotResponse`. |
| [`backend/services/analytics_service.py`](file:///c:/Users/junai/OneDrive/Documents/GitHub/Freight---bulk-Cargo/backend/services/analytics_service.py) | **NEW** | Deterministic analytics engine performing data loading, filtering, MoM calculations, rolling averages, Pearson correlations, and summary generation. |
| [`backend/main.py`](file:///c:/Users/junai/OneDrive/Documents/GitHub/Freight---bulk-Cargo/backend/main.py) | **MODIFIED** | Added 7 `/analytics/*` REST endpoints with typed schemas and structured 422 error handlers. |
| [`backend/test_suite.py`](file:///c:/Users/junai/OneDrive/Documents/GitHub/Freight---bulk-Cargo/backend/test_suite.py) | **MODIFIED** | Added `TestMarketIntelligenceAnalytics` class covering all 7 endpoints, query filters, correlation math, and quarantine assertions (34 tests in suite). |

---

## 7. Verification & Automated Test Results

| Test Suite | Total Tests | Pass Rate | Execution Time | Coverage Details |
|---|---|---|---|---|
| `backend/test_suite.py` | 34 | **100% PASS** | 0.52s | Model contract, explainability math, scenario shocks, and all 7 analytics endpoints. |
| `backend/test_v3_model.py` | 4 | **100% PASS** | 0.14s | Production route integration and model introspection. |
| `backend/test_concurrency.py` | 1 | **100% PASS** | 0.79s | SQLite WAL multithreaded concurrency. |
| **Total Automated Tests** | **39** | **100% PASS** | **1.45s** | **100% clean test execution.** |

---

## 8. Checksum & Immutability Verification

- **Model Artifact**: [`freight_forecast_model_v3.joblib`](file:///c:/Users/junai/OneDrive/Documents/GitHub/Freight---bulk-Cargo/freight_forecast_model_v3.joblib)
- **SHA-256 Checksum**: `71fbb870bb1f555d73a51ed7d83fb5a877cc4405ce54d1fe18407c9ce37c46a8` (**100% UNCHANGED**).
- **Retraining**: **None** (Zero retraining performed).
- **Synthetic Data**: **None** (Quarantine strictly maintained).

---

*Phase 3 Complete. Ready for hackathon demonstration.*
