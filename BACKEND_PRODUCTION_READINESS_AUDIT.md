# BACKEND & DATA PIPELINE PRODUCTION READINESS AUDIT
## Comprehensive Architectural, Data Integrity, and Safety Audit

---

## 1. Executive Summary

This audit evaluates the **Freight Intelligence Platform** backend (`FastAPI`), data ingestion layers (`SQLite`, `Open-Meteo`, `MarketDataProvider`), and inference pipeline following the successful activation of **Model v3** (`freight_forecast_model_v3.joblib`).

### Overall System Posture:
- **Core ML & Inference**: **STRONG**. Model v3 (Bounded Residual Ridge Regression) is cleanly integrated, reproducible, and provides superior real-world performance ($\text{MAE} = 0.4730\text{ USD/t}$ vs Persistence $0.8280$).
- **API & Validation**: **GOOD**. Pydantic v2 schemas enforce strict types, bounds, and deterministic 13-feature input contracts without `cargo_tonnes`.
- **Data Ingestion & Safety**: **MODERATE**. While the placeholder market provider and weather fetchers prevent data fabrication, key production gaps exist in **staleness detection**, **geographic coverage** (missing `Australia West Coast` in weather coordinates), **SQLite concurrency** (missing WAL mode), **negative freight clamping**, and **prediction audit logging**.

---

## 2. Current Architecture Overview

```
[ HTTP Client / Frontend ]
           │
           ▼
[ FastAPI Application (main.py) ]  ──> GET /model/info (Dynamic Metadata)
           │
           ▼
[ Forecast Service (forecast_service.py) ]
   ├── Merge User Inputs (User input always takes priority)
   ├── Query SQLite DB for missing weather/market fields
   ├── Enforce Zero-Fabrication (Fail with 422 if required data is missing)
   └── Track Field Provenance ("user" vs "weather_db[Port@Timestamp]")
           │
           ▼
[ Prediction Layer (predict.py) ]
   ├── Input Validation (13-feature DataFrame construction)
   ├── Model v3 Pipeline: ColumnTransformer + Ridge(alpha=10.0)
   ├── Delta Calculation: predicted_delta = Ridge(X)
   ├── Defensive Guardrail: clip(predicted_delta, -4.0, +4.0)
   ├── Level Forecast: forecast = current_freight + bounded_delta
   ├── Market Momentum: change_percent = (forecast - current) / current * 100
   ├── Weather Risk Classification: compute_risk_level() -> LOW/MEDIUM/HIGH
   └── Recommendation Engine: compute_recommendation() -> CHARTER NOW/WAIT/MONITOR
           │
           ▼
[ SQLite Storage Layer (database.py) ]
   ├── weather_data (port snapshots)
   ├── market_data (series quotes)
   ├── freight_observations (route rate observations)
   └── data_status (category update timestamps)
```

---

## 3. Model Integration Audit

| Dimension | Production Specification | Current Implementation | Status |
|---|---|---|---|
| **Active Artifact** | `freight_forecast_model_v3.joblib` | `MODEL_PATH = REPO_ROOT / "freight_forecast_model_v3.joblib"` | ✅ Verified |
| **Model Algorithm** | Bounded Residual Ridge Regression | `ColumnTransformer` + `Ridge(alpha=10.0)` | ✅ Verified |
| **Residual Formulation** | $\hat{y} = y_{\text{current}} + \Delta y$ | `current + bounded_delta` | ✅ Verified |
| **Defensive Guardrail** | $[-4.0, +4.0]\text{ USD/t}$ | `max(-4.0, min(4.0, raw_delta))` | ✅ Verified |
| **Feature Count & Names** | Exactly 13 canonical features | 13 features matching `model.feature_names_in_` | ✅ Verified |
| **Excluded Features** | `cargo_tonnes` omitted | Absent from all schemas and pipelines | ✅ Verified |
| **Rollback Capability** | Seamless fallback to v2/v1 | Legacy detection logic in `predict.py` | ✅ Verified |
| **Dynamic Metadata** | `/model/info` reports v3 parameters | Returns live metadata from `get_model_metadata()` | ✅ Verified |

---

## 4. API Contract Audit (`POST /predict`)

### 4.1 Input Validation Rules
- **Required User Inputs (5)**: `origin`, `destination`, `commodity`, `vessel_type`, `current_freight_usd_per_tonne`.
- **Optional Inputs (8)**: `bdi`, `vlsfo_usd_per_tonne`, `coal_price_usd_per_mt`, `iron_ore_price_usd_per_dmt`, `wind_kmh`, `wave_height_m`, `cyclone_risk`, `weather_delay_days`.

### 4.2 Gaps Identified:
1. **Unbounded BDI Input**: `bdi` is defined as `Optional[float] = Field(default=None)`. It lacks a `ge=0` or `gt=0` constraint. Negative or zero BDI is physically impossible and should fail validation.
2. **Open String Categoricals**: `origin`, `destination`, `commodity`, `vessel_type` are unconstrained strings. While `OneHotEncoder(handle_unknown="ignore")` prevents runtime crashes, invalid entries (e.g. `origin="Mars"`, `commodity=""`) are silently treated as unknown without informative feedback.
3. **Pydantic Finite Value Enforcement**: `float('inf')` and `float('nan')` are not explicitly blocked with `allow_inf_nan=False`.

---

## 5. Data Provenance Audit

### 5.1 Provenance Hierarchy:
1. `USER`: Value supplied directly in the HTTP request payload.
2. `DATABASE CACHE`: Value retrieved from SQLite (`weather_db[Port@Timestamp]` or `market_db[Series@Timestamp]`).
3. `DERIVED`: Values calculated deterministically (`cyclone_risk`, `weather_delay_days`, `forecast_change_percent`).
4. `UNAVAILABLE`: Missing from user and DB $\rightarrow$ Triggers HTTP 422 (zero fabrication).

### 5.2 Provenance Gaps:
- **No Staleness Flagging**: Provenance strings report timestamp (`weather_db[Hay Point@2026-08-27T17:14:09Z]`), but the API does not calculate elapsed age or issue a warning if the cached record is days or weeks old.

---

## 6. Weather Data Pipeline Audit

### 6.1 Open-Meteo Integration:
- **Primary Endpoint**: `https://api.open-meteo.com/v1/forecast` (current wind speed, temperature).
- **Marine Endpoint**: `https://marine-api.open-meteo.com/v1/marine` (current wave height).
- **Archive Fallback**: `https://archive-api.open-meteo.com/v1/archive` (historical backup if rate-limited).
- **Coordinate Mapping**:
  - `Hay Point`: `(-21.37, 149.32)`
  - `Taboneo`: `(-3.65, 114.85)`
  - `Visakhapatnam`: `(17.68, 83.27)`
  - `Paradip`: `(20.32, 86.70)`

### 6.2 Weather Pipeline Gaps:
1. **Missing Port in Coordinates Table**: `Australia West Coast` (Dampier / Port Hedland `[-20.32, 118.57]`) is one of the 5 core trade routes in the training set and API contract, but is **missing from `PORTS` in `weather.py`**. If a user submits a request for `Australia West Coast` without supplying weather, the database lookup fails.
2. **Archive Fallback Latency**: If the forecast API times out, the archive fallback requests 2 full days of hourly data over HTTP, which takes ~1.5 seconds.
3. **No Dynamic Update Trigger**: Weather is only refreshed when `update_data.py` is invoked manually or via cron. There is no auto-refresh on cache expiration.

---

## 7. Market Data Pipeline Audit

### 7.1 Provider Interface (`backend/data/market.py`):
- `MarketDataProvider` protocol defines `fetch_series(series: str)` and `fetch_all()`.
- `PlaceholderMarketDataProvider`: Returns `None` for all 4 series.
- Zero synthetic or default values are substituted: **100% compliant with zero-fabrication directive**.

### 7.2 Market Pipeline Gaps:
- **Standardized Error Representation**: When market data is missing, `forecast_service.py` raises a generic `ValueError` string. In production, this should return a structured JSON error body with machine-readable error codes (e.g. `{"error_code": "MARKET_DATA_MISSING", "missing_fields": ["bdi", "vlsfo_usd_per_tonne"]}`).

---

## 8. Freight Observation Pipeline Audit

- `current_freight_usd_per_tonne` is configured as **strictly required from the user** in `schemas.py` and `forecast_service.py`.
- `freight_observations` table stores historical observations with `(origin, destination, commodity, vessel_type, current_freight_usd_per_tonne, observed_at)`.
- **Freshness Hazard**: If `_fill_freight` is ever re-enabled for user convenience, using an observation from an outdated date without a Time-To-Live (TTL) constraint would introduce severe anchoring bias.

---

## 9. SQLite Database Architecture Audit

### 9.1 Schema & Indexes:
- Tables: `weather_data`, `market_data`, `freight_observations`, `data_status`.
- Indexes: `idx_weather_port_time`, `idx_market_series_time`, `idx_freight_route`.

### 9.2 Database Gaps:
1. **No WAL (Write-Ahead Logging) Mode**: The database initializes with default rollback journal mode. Concurrent API reads during an `update_data.py` write run the risk of `sqlite3.OperationalError: database is locked`.
2. **Deprecated Timestamp Call**: `database.py` uses `datetime.utcnow()`, which is deprecated in modern Python. Should use `datetime.now(timezone.utc)`.
3. **No Automatic Pruning**: Appends historical rows indefinitely without retention policies.

---

## 10. Prediction Safety Audit

### 10.1 Safety Bounds & Checks:
- **Residual Delta Guardrail**: $\widehat{\Delta y} \in [-4.0, +4.0]\text{ USD/t}$ (active).
- **Division-by-Zero Protection**: `current_freight_usd_per_tonne > 0` enforced by schema.

### 10.2 Prediction Safety Gaps:
1. **No Absolute Floor on Level Forecast**: If an anomalous input has $y_{\text{current}} = 2.0\text{ USD/t}$ and $\widehat{\Delta y} = -4.0\text{ USD/t}$, the model outputs $-2.0\text{ USD/t}$. Ocean freight rates cannot be negative. A physical floor ($\hat{y} \ge 1.0\text{ USD/t}$) must be enforced.
2. **Unknown Category Behavior**: An unrecognised port or commodity is mapped to all zeros by the OneHotEncoder. The output receives only the baseline intercept and numerical features without warning.

---

## 11. Recommendation Engine Audit

### 11.1 Threshold Verification:

$$\text{Risk Level} = \begin{cases} \text{HIGH} & \text{if } \text{cyclone\_risk} \ge 4.0 \lor \text{weather\_delay\_days} \ge 2.5 \\ \text{MEDIUM} & \text{if } \text{cyclone\_risk} \ge 3.0 \lor \text{weather\_delay\_days} \ge 1.0 \\ \text{LOW} & \text{otherwise} \end{cases}$$

$$\text{Recommendation} = \begin{cases} \text{CHARTER NOW} & \text{if } \Delta\% \ge +5.0\% \lor \text{Risk} = \text{HIGH} \\ \text{WAIT} & \text{if } \Delta\% \le -5.0\% \land \text{Risk} \ne \text{HIGH} \\ \text{MONITOR} & \text{if } -5.0\% < \Delta\% < +5.0\% \land \text{Risk} \ne \text{HIGH} \end{cases}$$

- All boundary conditions ($\pm 5.0\%$, risk 3/4, delay 1.0/2.5) are logically sound and properly prioritize safety over price drops.

---

## 12. Error Handling Audit

- **Open-Meteo Outage**: Handled cleanly with fallback and error reporting.
- **Missing Database**: Handled cleanly with auto-initialization in startup lifespan.
- **Missing Model File**: Handled cleanly with immediate fail-fast on startup.
- **Gaps**: Unhandled exceptions in `/predict` produce generic unformatted 500 errors instead of structured error payloads with request tracking IDs.

---

## 13. Performance & Concurrency Audit

- **Model Inference**: Single prediction executes in **$< 1.5\text{ ms}$**.
- **Memory Footprint**: `freight_forecast_model_v3.joblib` is only **4.1 KB** (vs 2.08 MB for legacy RF).
- **Concurrency**: High for read-only inference. Database write concurrency requires WAL mode.

---

## 14. Test Coverage Audit

- **Current Suites**:
  - `backend/test_suite.py`: 19 tests covering 13-feature contract, metadata, `/predict` on all 5 routes, provenance, and 422 error validation.
  - `backend/test_v3_model.py`: 4 tests covering Model v3 artifact, inference bounds, and API integration.
- **Testing Gaps**:
  1. No boundary tests for exact recommendation thresholds ($\pm 5.00\%$, delay $0.99$ vs $1.00$, delay $2.49$ vs $2.50$).
  2. No tests verifying negative freight rate prevention.
  3. No tests for SQLite database locking under simulated concurrent read/write.
  4. No tests verifying missing weather handling specifically for `Australia West Coast`.

---

## 15. Documentation Audit

- `README.md`, `MODEL_V3_FINAL_VALIDATION.md`, and `MODEL_V3_INTEGRATION_REPORT.md` are up-to-date with Model v3.
- **Minor Stale Docstrings**:
  - `backend/schemas.py`: Header docstring references `FINAL model (freight_forecast_model_final.joblib)`.
  - `backend/test_example.py`: Header references `(FINAL model)`.

---

## 16. Critical Issues

### Issue C1: Missing Port in Weather Coordinates Table
- **Severity**: **CRITICAL**
- **File**: [`backend/data/weather.py`](file:///c:/Users/junai/OneDrive/Documents/GitHub/Freight---bulk-Cargo/backend/data/weather.py#L30-L35)
- **Problem**: `PORTS` dictionary contains only 4 ports (`Hay Point`, `Taboneo`, `Visakhapatnam`, `Paradip`). `Australia West Coast` (Dampier/Port Hedland) is one of the 5 canonical trade routes in the training dataset and API contract, but is missing from `PORTS`.
- **Impact**: Any user requesting a forecast for `Australia West Coast` without entering weather will fail with a 422 error because weather cannot be fetched or cached for this region.
- **Recommended Fix**: Add `"Australia West Coast": (-20.32, 118.57)` to `PORTS` in `weather.py`.

### Issue C2: Missing Absolute Floor on Level Forecast
- **Severity**: **CRITICAL**
- **File**: [`backend/predict.py`](file:///c:/Users/junai/OneDrive/Documents/GitHub/Freight---bulk-Cargo/backend/predict.py#L130-L145)
- **Problem**: The forecast calculation $\hat{y} = y_{\text{current}} + \text{clip}(\widehat{\Delta y}, -4.0, 4.0)$ does not enforce a non-negative floor. If spot freight is low ($<\$4.00/t$) and market signals predict a decrease, the forecast could produce a negative freight rate.
- **Impact**: Negative freight rates are physically impossible and violate maritime domain rules.
- **Recommended Fix**: Apply a non-negotiable floor clamp: `predicted = max(1.0, current + bounded_delta)`.

---

## 17. High & Medium-Priority Issues

### Issue M1: SQLite Rollback Journal Mode (Lack of WAL Mode)
- **Severity**: **HIGH**
- **File**: [`backend/data/database.py`](file:///c:/Users/junai/OneDrive/Documents/GitHub/Freight---bulk-Cargo/backend/data/database.py#L69-L87)
- **Problem**: SQLite connections operate in default journal mode without Write-Ahead Logging (WAL) enabled.
- **Impact**: Database writes (e.g. during weather updates) will lock the database file, causing incoming `/predict` or `/data/latest` reads to fail with `database is locked`.
- **Recommended Fix**: Execute `PRAGMA journal_mode=WAL;` and `PRAGMA synchronous=NORMAL;` inside `init_db()` and set `timeout=10.0` in `sqlite3.connect()`.

### Issue M2: Absence of Data Freshness / Staleness Warnings
- **Severity**: **MEDIUM**
- **File**: [`backend/services/forecast_service.py`](file:///c:/Users/junai/OneDrive/Documents/GitHub/Freight---bulk-Cargo/backend/services/forecast_service.py#L50-L88)
- **Problem**: When cached weather or market data is auto-filled from SQLite, there is no validation of elapsed time since `fetched_at`.
- **Impact**: Multi-week-old cached weather could silently be applied to current forecasts.
- **Recommended Fix**: Add a configurable max-age check (e.g. 24 hours for weather). If cached data exceeds max-age, flag as stale in `sources` or return a warning header.

### Issue M3: Unbounded Baltic Dry Index (BDI) Schema Validation
- **Severity**: **MEDIUM**
- **File**: [`backend/schemas.py`](file:///c:/Users/junai/OneDrive/Documents/GitHub/Freight---bulk-Cargo/backend/schemas.py#L40)
- **Problem**: `bdi: Optional[float] = Field(default=None)` lacks a `gt=0` constraint.
- **Impact**: Negative or zero values can be submitted without triggering schema validation.
- **Recommended Fix**: Change to `bdi: Optional[float] = Field(default=None, gt=0, description="Baltic Dry Index value")`.

### Issue M4: Missing Prediction Logging / Telemetry
- **Severity**: **MEDIUM**
- **File**: [`backend/main.py`](file:///c:/Users/junai/OneDrive/Documents/GitHub/Freight---bulk-Cargo/backend/main.py)
- **Problem**: The backend has zero persistent logging of generated predictions.
- **Impact**: Operational monitoring, prediction tracking, model drift auditing, and historical accuracy validation are impossible without post-hoc user logs.
- **Recommended Fix**: Add an asynchronous SQLite table `prediction_logs` recording timestamp, route, inputs hash, predicted rate, risk band, and latency.

---

## 18. Low-Priority Issues

### Issue L1: Deprecated `datetime.utcnow()`
- **Severity**: **LOW**
- **File**: [`backend/data/database.py`](file:///c:/Users/junai/OneDrive/Documents/GitHub/Freight---bulk-Cargo/backend/data/database.py#L89-L91)
- **Problem**: `datetime.utcnow()` is deprecated in Python 3.12+.
- **Recommended Fix**: Replace with `datetime.now(timezone.utc).isoformat(timespec="seconds")`.

### Issue L2: Residual Stale Docstrings
- **Severity**: **LOW**
- **File**: [`backend/schemas.py`](file:///c:/Users/junai/OneDrive/Documents/GitHub/Freight---bulk-Cargo/backend/schemas.py#L9) and [`backend/test_example.py`](file:///c:/Users/junai/OneDrive/Documents/GitHub/Freight---bulk-Cargo/backend/test_example.py#L1)
- **Problem**: Minor docstring mentions of legacy `FINAL model`.
- **Recommended Fix**: Update docstrings to reference `Model v3`.

---

## 19. Recommended Implementation Order

```
Phase 1: Critical Safety & Correctness (Immediate)
  ├── 1. Add "Australia West Coast" coordinates to PORTS in weather.py
  └── 2. Add physical non-negative level floor (max(1.0, ...)) in predict.py

Phase 2: Database Concurrency & API Hardening
  ├── 3. Enable WAL mode & 10s busy timeout in database.py
  ├── 4. Add gt=0 constraint to bdi in schemas.py
  └── 5. Replace deprecated datetime.utcnow() with datetime.now(timezone.utc)

Phase 3: Operational Monitoring & Quality of Service
  ├── 6. Add staleness age tracking and warning headers in forecast_service.py
  ├── 7. Add prediction_logs SQLite audit table
  └── 8. Expand test suite with boundary and concurrency test cases
```

---

*Audit completed. No production code was modified during this audit.*
