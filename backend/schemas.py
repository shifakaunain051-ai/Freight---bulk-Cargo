"""Pydantic request/response schemas for the freight forecasting API, Market Intelligence, and Dashboard."""

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class FreightRequest(BaseModel):
    """Forecast request for Model v3 (freight_forecast_model_v3.joblib).

    Model v3 uses 13 input features (NO cargo_tonnes - it was
    intentionally excluded because cargo values were representative
    vessel capacities, not observed shipment quantities).

    Required fields (identity + current freight):
        origin, destination, commodity, vessel_type, current_freight_usd_per_tonne

    Optional fields (filled from the data layer if omitted):
        bdi, vlsfo_usd_per_tonne, coal_price_usd_per_mt,
        iron_ore_price_usd_per_dmt, wind_kmh, wave_height_m,
        cyclone_risk, weather_delay_days

    If an optional field cannot be resolved from the database AND the user
    has not supplied it, the API returns a structured 422 listing the missing fields.
    No values are fabricated.
    """

    # --- identity (required) ---
    origin: str = Field(..., min_length=1, description="Loading port/region, e.g. 'Hay Point'")
    destination: str = Field(..., min_length=1, description="Discharge port/region, e.g. 'East Coast India'")
    commodity: str = Field(..., min_length=1, description="Cargo commodity, e.g. 'Coal'")
    vessel_type: str = Field(..., min_length=1, description="Vessel class, e.g. 'Panamax'")

    # --- current freight (required - the model's primary numeric signal) ---
    current_freight_usd_per_tonne: float = Field(
        ..., gt=0, description="Current freight rate (USD/tonne)"
    )

    # --- market (optional, filled from market_data if omitted) ---
    bdi: Optional[float] = Field(
        default=None, gt=0, description="Baltic Dry Index value (>0)"
    )
    vlsfo_usd_per_tonne: Optional[float] = Field(
        default=None, ge=0, description="VLSFO bunker price (USD/tonne)"
    )
    coal_price_usd_per_mt: Optional[float] = Field(
        default=None, ge=0, description="Coal benchmark price (USD/MT)"
    )
    iron_ore_price_usd_per_dmt: Optional[float] = Field(
        default=None, ge=0, description="Iron ore price (USD/dmt)"
    )

    # --- weather (optional, filled from weather_data for the origin port) ---
    wind_kmh: Optional[float] = Field(default=None, ge=0, description="Wind speed (km/h)")
    wave_height_m: Optional[float] = Field(default=None, ge=0, description="Wave height (m)")
    cyclone_risk: Optional[float] = Field(
        default=None, ge=0, le=5, description="Cyclone risk score 0-5"
    )
    weather_delay_days: Optional[float] = Field(
        default=None, ge=0, description="Expected weather delay (days)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "origin": "Hay Point",
                "destination": "East Coast India",
                "commodity": "Coal",
                "vessel_type": "Panamax",
                "current_freight_usd_per_tonne": 16.5,
                "bdi": 1560,
                "vlsfo_usd_per_tonne": 638,
                "coal_price_usd_per_mt": 124,
                "iron_ore_price_usd_per_dmt": 124,
                "wind_kmh": 32,
                "wave_height_m": 2.0,
                "cyclone_risk": 2,
                "weather_delay_days": 0.5,
            }
        }
    }


class ExplanationDriver(BaseModel):
    """Additive feature contribution derived from the model pipeline."""

    feature: str = Field(..., description="Feature identifier")
    feature_label: str = Field(..., description="Human-readable feature label")
    value: Union[float, str] = Field(..., description="Observed or input feature value")
    unit: str = Field(default="", description="Unit of measurement (e.g. USD/tonne, points, km/h)")
    coefficient: float = Field(..., description="Ridge regression model coefficient")
    contribution_usd_per_tonne: float = Field(..., description="Additive linear contribution to delta (USD/tonne)")
    effect: Literal["positive", "negative", "neutral"] = Field(..., description="Direction of contribution")
    source: Literal["model", "context", "market", "weather"] = Field(default="model", description="Contribution origin")


class ExplanationAnchor(BaseModel):
    """Mathematical decomposition of the prediction anchor."""

    current_freight_usd_per_tonne: float = Field(..., description="Anchor spot freight rate (USD/tonne)")
    predicted_next_month_freight_usd_per_tonne: float = Field(..., description="Forecasted level rate (USD/tonne)")
    raw_predicted_delta_usd_per_tonne: float = Field(..., description="Raw residual/time-series delta")
    bounded_delta_usd_per_tonne: float = Field(..., description="Residual delta after guardrail/sanity checks")
    model_intercept: float = Field(..., description="Global baseline intercept term")
    residual_guardrail_applied: bool = Field(..., description="Whether clipping/sanity boundary was triggered")
    physical_floor_applied: bool = Field(..., description="Whether >= 1.0 USD/tonne floor was triggered")
    predicted_freight_usd_per_tonne: Optional[float] = Field(default=None, description="Forecasted level rate (USD/tonne)")
    forecast_change_usd_per_tonne: Optional[float] = Field(default=None, description="Forecast change in USD/tonne")
    forecast_change_percent: Optional[float] = Field(default=None, description="Forecast change in percent")
    direction: Optional[str] = Field(default=None, description="Forecast direction (UP/DOWN/STABLE)")
    route_matched: Optional[str] = Field(default=None, description="Canonical route matched")
    canonical_order: Optional[list[int]] = Field(default=None, description="ARIMA (p, d, q) order")
    canonical_seasonal_order: Optional[list[int]] = Field(default=None, description="SARIMA seasonal order")

    model_config = {"extra": "allow"}


class PredictionExplanation(BaseModel):
    """Transparent mathematical and natural language explanation of the forecast."""

    summary: str = Field(..., description="Concise human-readable explanation of forecast drivers")
    drivers: list[ExplanationDriver] = Field(
        default_factory=list, description="Ranked feature-level additive contributions"
    )
    anchor: ExplanationAnchor = Field(..., description="Mathematical anchoring decomposition")


class FreightResponse(BaseModel):
    """Forecast result returned by the /predict endpoint."""

    predicted_next_month_freight_usd_per_tonne: float = Field(
        ..., description="Model forecast for next-month freight rate (USD/tonne)"
    )
    current_freight_usd_per_tonne: float = Field(
        ..., description="Current freight rate (USD/tonne)"
    )
    forecast_change_percent: float = Field(
        ..., description="Percent change of forecast vs current rate"
    )
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(..., description="Weather risk band")
    recommendation: Literal["CHARTER NOW", "WAIT", "MONITOR"] = Field(
        ..., description="Actionable chartering recommendation"
    )
    reason: str = Field(..., description="Human-readable explanation of the recommendation")
    predicted_freight_usd_per_tonne: Optional[float] = Field(
        default=None, description="Model forecast for next-month freight rate (USD/tonne)"
    )
    forecast_change_usd_per_tonne: Optional[float] = Field(
        default=None, description="Forecast delta in USD/tonne"
    )
    direction: Optional[Literal["UP", "DOWN", "STABLE"]] = Field(
        default=None, description="Forecast movement direction"
    )
    model_name: Optional[str] = Field(
        default=None, description="Active model identifier"
    )
    model_version: Optional[str] = Field(
        default=None, description="Active model semantic version"
    )
    training_data_end_date: Optional[str] = Field(
        default=None, description="End date of historical training observations"
    )
    forecast_timestamp: Optional[str] = Field(
        default=None, description="Timestamp of forecast generation"
    )
    sources: dict[str, str] = Field(
        default_factory=dict,
        description="Provenance of each model input: 'user' or a DB reference with freshness info.",
    )
    explanation: Optional[PredictionExplanation] = Field(
        default=None,
        description="Transparent mathematical breakdown and natural language drivers of the prediction.",
    )

    model_config = {"extra": "allow"}


# --------------------------------------------------------------------------- #
# Scenario Analysis Schemas (Phase 2)
# --------------------------------------------------------------------------- #
class ScenarioModifications(BaseModel):
    """Simulated parameter shifts. Supports both absolute overrides and percentage shocks."""

    bdi: Optional[float] = Field(default=None, gt=0, description="Simulated absolute BDI (>0)")
    vlsfo_usd_per_tonne: Optional[float] = Field(
        default=None, ge=0, description="Simulated absolute VLSFO bunker price (USD/tonne)"
    )
    coal_price_usd_per_mt: Optional[float] = Field(
        default=None, ge=0, description="Simulated absolute Coal benchmark price (USD/MT)"
    )
    iron_ore_price_usd_per_dmt: Optional[float] = Field(
        default=None, ge=0, description="Simulated absolute Iron ore price (USD/dmt)"
    )
    wind_kmh: Optional[float] = Field(
        default=None, ge=0, description="Simulated absolute Wind speed (km/h)"
    )
    wave_height_m: Optional[float] = Field(
        default=None, ge=0, description="Simulated absolute Wave height (m)"
    )
    cyclone_risk: Optional[float] = Field(
        default=None, ge=0, le=5, description="Simulated absolute Cyclone risk score (0-5)"
    )
    weather_delay_days: Optional[float] = Field(
        default=None, ge=0, description="Simulated absolute Weather delay estimate (days)"
    )
    current_freight_usd_per_tonne: Optional[float] = Field(
        default=None, gt=0, description="Simulated absolute Current base freight (USD/tonne)"
    )

    bdi_change_percent: Optional[float] = Field(
        default=None, description="Percentage change in BDI (e.g. +20.0 for +20%)"
    )
    vlsfo_change_percent: Optional[float] = Field(
        default=None, description="Percentage change in VLSFO price (e.g. +10.0 for +10%)"
    )
    coal_price_change_percent: Optional[float] = Field(
        default=None, description="Percentage change in Coal price (e.g. -5.0 for -5%)"
    )
    iron_ore_price_change_percent: Optional[float] = Field(
        default=None, description="Percentage change in Iron Ore price (e.g. +5.0 for +5%)"
    )
    wind_change_percent: Optional[float] = Field(
        default=None, description="Percentage change in Wind speed (e.g. +30.0 for +30%)"
    )
    wave_height_change_percent: Optional[float] = Field(
        default=None, description="Percentage change in Wave height (e.g. +25.0 for +25%)"
    )
    cyclone_risk_change: Optional[float] = Field(
        default=None, description="Point change in Cyclone risk (e.g. +2.0 to move 2 -> 4)"
    )
    weather_delay_change_percent: Optional[float] = Field(
        default=None, description="Percentage change in Weather delay (e.g. +50.0 for +50%)"
    )
    current_freight_change_percent: Optional[float] = Field(
        default=None, description="Percentage change in Current freight (e.g. -10.0 for -10%)"
    )


class ScenarioRequest(FreightRequest):
    """What-if scenario request inheriting baseline inputs plus scenario modifications."""

    scenario_changes: Optional[ScenarioModifications] = Field(
        default=None,
        description="Scenario parameter modifications or relative shocks.",
    )


class ScenarioChangeItem(BaseModel):
    """Detailed record of a single changed input parameter."""

    feature: str = Field(..., description="Feature identifier")
    feature_label: str = Field(..., description="Human-readable feature name")
    baseline: float = Field(..., description="Baseline input value")
    scenario: float = Field(..., description="Simulated scenario input value")
    absolute_change: float = Field(..., description="Scenario minus baseline value")
    percentage_change: Optional[float] = Field(
        default=None, description="Percentage shift relative to baseline"
    )
    unit: str = Field(default="", description="Unit of measurement")


class ScenarioImpact(BaseModel):
    """Comparative impact metrics between baseline and scenario predictions."""

    difference_usd_per_tonne: float = Field(
        ..., description="Scenario predicted rate minus baseline predicted rate (USD/tonne)"
    )
    difference_percent: float = Field(
        ..., description="Percentage difference in forecast vs baseline forecast"
    )
    baseline_change_percent: float = Field(
        ..., description="Baseline forecast change percent vs base freight"
    )
    scenario_change_percent: float = Field(
        ..., description="Scenario forecast change percent vs base freight"
    )
    risk_level_shift: str = Field(
        ..., description="Risk level change, e.g. 'LOW -> HIGH' or 'LOW (unchanged)'"
    )
    recommendation_shift: str = Field(
        ..., description="Recommendation change, e.g. 'MONITOR -> CHARTER NOW'"
    )


class ScenarioResponse(BaseModel):
    """Complete response returned by POST /predict/scenario."""

    summary: str = Field(..., description="Executive natural language scenario summary")
    baseline: FreightResponse = Field(..., description="Baseline prediction & explainability")
    scenario: FreightResponse = Field(..., description="Simulated scenario prediction & explainability")
    impact: ScenarioImpact = Field(..., description="Comparative delta impact metrics")
    changes: list[ScenarioChangeItem] = Field(
        default_factory=list, description="Applied scenario modifications"
    )


# --------------------------------------------------------------------------- #
# Market Intelligence & Analytics Schemas (Phase 3)
# --------------------------------------------------------------------------- #
class ProvenanceMeta(BaseModel):
    """Data provenance metadata ensuring transparency of historical analytics."""

    data_source: str = Field(
        default="master_freight_training_expanded_v1.csv",
        description="Underlying primary data file",
    )
    data_type: Literal["historical", "database_cache", "live_snapshot"] = Field(
        default="historical",
        description="Data context: genuine historical dataset vs runtime DB cache",
    )
    date_range: dict[str, str] = Field(
        default_factory=lambda: {"start": "2024-02-01", "end": "2025-11-01"},
        description="Chronological span of available observations",
    )
    total_records: int = Field(..., description="Total observation count in the query result")


class FreightTrendPoint(BaseModel):
    """Single monthly freight rate observation with month-over-month and rolling indicators."""

    date: str = Field(..., description="Observation month (YYYY-MM-DD)")
    origin: str = Field(..., description="Loading port/region")
    destination: str = Field(..., description="Discharge port/region")
    commodity: str = Field(..., description="Cargo commodity")
    vessel_type: str = Field(..., description="Vessel class")
    freight_rate_usd_per_tonne: float = Field(
        ..., description="Observed historical freight rate (USD/tonne)"
    )
    previous_month_freight: Optional[float] = Field(
        default=None, description="Previous month freight rate (USD/tonne)"
    )
    mom_change_usd: Optional[float] = Field(
        default=None, description="Month-over-month absolute rate change (USD/tonne)"
    )
    mom_change_percent: Optional[float] = Field(
        default=None, description="Month-over-month percentage change"
    )
    rolling_3m_avg: Optional[float] = Field(
        default=None, description="3-month rolling average freight rate (USD/tonne)"
    )


class FreightTrendsResponse(BaseModel):
    """Response returned by GET /analytics/freight-trends."""

    provenance: ProvenanceMeta
    filters_applied: dict[str, Optional[str]]
    series: list[FreightTrendPoint]


class MarketTrendPoint(BaseModel):
    """Temporal macroeconomic and fuel price observation for a specific month."""

    date: str = Field(..., description="Observation month (YYYY-MM-DD)")
    bdi: float = Field(..., description="Baltic Dry Index benchmark")
    vlsfo_usd_per_tonne: float = Field(..., description="VLSFO bunker fuel price (USD/tonne)")
    coal_price_usd_per_mt: float = Field(..., description="Coal benchmark price (USD/MT)")
    iron_ore_price_usd_per_dmt: float = Field(..., description="Iron Ore benchmark price (USD/dmt)")
    average_freight_usd_per_tonne: float = Field(
        ..., description="Cross-route average observed freight for this month (USD/tonne)"
    )
    min_freight_usd_per_tonne: float = Field(
        ..., description="Minimum freight observed across routes (USD/tonne)"
    )
    max_freight_usd_per_tonne: float = Field(
        ..., description="Maximum freight observed across routes (USD/tonne)"
    )


class MarketTrendsResponse(BaseModel):
    """Response returned by GET /analytics/market-trends."""

    provenance: ProvenanceMeta
    series: list[MarketTrendPoint]


class WeatherTrendPoint(BaseModel):
    """Historical weather observation for a specific origin port and month."""

    date: str = Field(..., description="Observation month (YYYY-MM-DD)")
    origin: str = Field(..., description="Port or coastal loading region")
    wind_kmh: float = Field(..., description="Wind speed (km/h)")
    wave_height_m: float = Field(..., description="Significant wave height (m)")
    cyclone_risk: float = Field(..., description="Cyclone risk score (0-5)")
    weather_delay_days: float = Field(..., description="Estimated weather delay (days)")


class WeatherTrendsResponse(BaseModel):
    """Response returned by GET /analytics/weather-trends."""

    provenance: ProvenanceMeta
    origin_filter: Optional[str]
    series: list[WeatherTrendPoint]


class RouteMetric(BaseModel):
    """Summary statistics for a canonical trade lane."""

    origin: str
    destination: str
    commodity: str
    vessel_type: str
    observation_count: int
    first_date: str
    last_date: str
    average_freight: float
    minimum_freight: float
    maximum_freight: float
    latest_freight: float
    latest_monthly_change: float
    latest_monthly_change_percent: float
    trend: Literal["RISING", "FALLING", "STABLE"]


class RoutesResponse(BaseModel):
    """Response returned by GET /analytics/routes."""

    provenance: ProvenanceMeta
    routes: list[RouteMetric]


class MarketStateSnapshot(BaseModel):
    """Snapshot of latest macroeconomic variables."""

    bdi: float
    vlsfo_usd_per_tonne: float
    coal_price_usd_per_mt: float
    iron_ore_price_usd_per_dmt: float
    average_freight_usd_per_tonne: float


class TrendClassification(BaseModel):
    """Overall market trend evaluation."""

    freight_trend_classification: Literal["RISING", "FALLING", "STABLE"]
    methodology: str
    recent_3m_avg_freight: float
    prior_3m_avg_freight: float
    shift_percent: float


class ExecutiveSummaryResponse(BaseModel):
    """Executive market snapshot returned by GET /analytics/summary."""

    provenance: ProvenanceMeta
    latest_date: str
    market_state: MarketStateSnapshot
    recent_trends: TrendClassification
    strongest_positive_mover: Optional[RouteMetric]
    strongest_negative_mover: Optional[RouteMetric]
    tracked_routes_count: int


class CorrelationItem(BaseModel):
    """Pearson correlation record for a specific market or weather variable against freight."""

    feature: str
    feature_label: str
    correlation: float
    p_value: float
    sample_count: int
    relationship: Literal["positive", "negative", "neutral"]
    interpretation: str


class CorrelationsResponse(BaseModel):
    """Response returned by GET /analytics/correlations."""

    provenance: ProvenanceMeta
    target_variable: str
    correlations: list[CorrelationItem]
    disclaimer: str


class LatestMarketSnapshotResponse(BaseModel):
    """Combined dashboard snapshot returned by GET /analytics/latest."""

    provenance: ProvenanceMeta
    latest_historical_date: str
    market_macro: MarketStateSnapshot
    weather_by_origin: list[WeatherTrendPoint]
    freight_by_route: list[RouteMetric]
    live_database_cache_status: dict[str, Any]


# --------------------------------------------------------------------------- #
# Dashboard Aggregation Schemas (Phase 4)
# --------------------------------------------------------------------------- #
class MarketOverviewSection(BaseModel):
    """Dashboard market macro snapshot and trend momentum."""

    latest_date: str = Field(..., description="Observation month (YYYY-MM-DD)")
    bdi: float = Field(..., description="Baltic Dry Index benchmark")
    vlsfo_usd_per_tonne: float = Field(..., description="VLSFO bunker fuel price (USD/tonne)")
    coal_price_usd_per_mt: float = Field(..., description="Coal benchmark price (USD/MT)")
    iron_ore_price_usd_per_dmt: float = Field(..., description="Iron ore benchmark price (USD/dmt)")
    average_freight_usd_per_tonne: float = Field(
        ..., description="Cross-route average spot freight (USD/tonne)"
    )
    freight_trend_classification: Literal["RISING", "FALLING", "STABLE"] = Field(
        ..., description="3-month momentum direction"
    )
    recent_3m_avg_freight: float = Field(..., description="Recent 3-month rolling average freight")
    prior_3m_avg_freight: float = Field(..., description="Prior 3-month rolling average freight")
    shift_percent: float = Field(..., description="Momentum change percentage")
    source_context: str = Field(default="historical_training_dataset")


class RouteRankings(BaseModel):
    """Rankings across canonical trade corridors."""

    highest_freight: RouteMetric
    lowest_freight: RouteMetric
    strongest_momentum: RouteMetric
    weakest_momentum: RouteMetric


class RoutesOverviewSection(BaseModel):
    """Dashboard trade corridors and comparative rankings."""

    canonical_lanes: list[RouteMetric] = Field(..., description="5 canonical routes")
    rankings: RouteRankings = Field(..., description="Extrema and momentum rankings")


class WeatherPortOverview(BaseModel):
    """Port sea state and risk evaluation."""

    port: str
    wind_kmh: float
    wave_height_m: float
    cyclone_risk: float
    weather_delay_days: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    status: Literal["available", "stale", "unavailable"]
    latest_observation_date: str
    live_cache_age_hours: Optional[float] = None


class WeatherOverviewSection(BaseModel):
    """Dashboard weather and maritime risk state."""

    status: Literal["available", "stale", "unavailable"]
    observation_date: str
    ports: list[WeatherPortOverview]


class RouteForecastItem(BaseModel):
    """Next-month freight forecast for a canonical trade lane."""

    origin: str
    destination: str
    commodity: str
    vessel_type: str
    current_freight_usd_per_tonne: float
    predicted_next_month_freight_usd_per_tonne: float
    forecast_change_percent: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    recommendation: Literal["CHARTER NOW", "WAIT", "MONITOR"]
    top_driver: str


class ForecastOverviewSection(BaseModel):
    """Fleet-wide forward forecast intelligence generated by Model v3."""

    reference_summary: str
    route_forecasts: list[RouteForecastItem]


class MarketSignalItem(BaseModel):
    """Deterministic market or weather intelligence signal."""

    type: str = Field(..., description="Signal classification type")
    severity: Literal["LOW", "MEDIUM", "HIGH"] = Field(..., description="Alert severity")
    title: str = Field(..., description="Executive signal headline")
    description: str = Field(..., description="Detailed deterministic explanation")
    evidence: dict[str, Any] = Field(default_factory=dict, description="Supporting metrics")


class DataQualitySection(BaseModel):
    """Data health, provenance, and storage architecture status."""

    historical_dataset_verified: bool
    historical_records_count: int
    historical_date_range: dict[str, str]
    synthetic_data_used: bool
    synthetic_dataset_quarantined: bool
    live_database_connected: bool
    database_journal_mode: str
    weather_last_updated: Optional[str] = None
    market_last_updated: Optional[str] = None
    overall_health_status: Literal["HEALTHY", "DEGRADED"]


class ValidationEvidence(BaseModel):
    """Empirical validation metrics on clean out-of-sample holdout."""

    holdout_mae_usd_per_tonne: float
    persistence_mae_usd_per_tonne: float
    holdout_observations: int
    directional_accuracy_percent: float
    evaluation_type: str


class ModelOverviewSection(BaseModel):
    """Production Model metadata and architectural specifications."""

    model_name: str
    version: str
    algorithm: str
    alpha: Optional[float] = None
    order: Optional[list[int]] = None
    seasonal_order: Optional[list[int]] = None
    features_count: int
    excludes_cargo_tonnes: bool
    synthetic_data_used: bool
    residual_guardrail_usd_per_tonne: list[float]
    physical_floor_usd_per_tonne: float
    validation_evidence: ValidationEvidence
    model_config = {"extra": "allow"}


class DashboardProvenance(BaseModel):
    """Complete provenance metadata for the dashboard overview."""

    historical_dataset: dict[str, Any]
    synthetic_data_used: bool
    model_artifact: str
    model_sha256: str


class DashboardOverviewResponse(BaseModel):
    """Comprehensive executive dashboard response returned by GET /dashboard/overview."""

    market: MarketOverviewSection
    routes: RoutesOverviewSection
    weather: WeatherOverviewSection
    forecast: ForecastOverviewSection
    signals: list[MarketSignalItem]
    data_quality: DataQualitySection
    model: ModelOverviewSection
    provenance: DashboardProvenance


class ErrorResponse(BaseModel):
    """Structured error payload for failed requests."""

    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable summary of the error")
    missing_fields: Optional[list[str]] = Field(
        default=None, description="List of missing feature names if applicable"
    )
    detail: Optional[str] = Field(default=None, description="Detailed troubleshooting instruction")


# --------------------------------------------------------------------------- #
# /data endpoints
# --------------------------------------------------------------------------- #
class DataStatus(BaseModel):
    """Per-category last-updated timestamps."""

    weather: Optional[str] = Field(None, description="Last weather update (ISO-8601 UTC) or null")
    market: Optional[str] = Field(None, description="Last market update (ISO-8601 UTC) or null")
    freight: Optional[str] = Field(None, description="Last freight observation (ISO-8601 UTC) or null")


class WeatherSnapshot(BaseModel):
    port: str
    latitude: float
    longitude: float
    wind_kmh: Optional[float] = None
    wave_height_m: Optional[float] = None
    cyclone_risk: Optional[float] = None
    weather_delay_days: Optional[float] = None
    temperature_c: Optional[float] = None
    fetched_at: str


class MarketQuoteOut(BaseModel):
    series: str
    value: Optional[float] = None
    unit: Optional[str] = None
    source: Optional[str] = None
    fetched_at: str


class LatestData(BaseModel):
    """Latest stored data across all categories."""

    weather: list[WeatherSnapshot] = Field(
        default_factory=list, description="Latest weather row per port"
    )
    market: list[MarketQuoteOut] = Field(
        default_factory=list, description="Latest quote per market series"
    )
    freight_observations_count: int = Field(
        0, description="Total number of stored freight observations"
    )
    latest_freight_observation: Optional[dict] = Field(
        None, description="Most recent freight observation overall (if any)"
    )


# --------------------------------------------------------------------------- #
# Procurement Valuation schemas
# --------------------------------------------------------------------------- #
class ProcurementValuationResponse(BaseModel):
    """Structured valuation and procurement decision signal for a commodity."""

    commodity: str = Field(..., description="Canonical commodity name (e.g. Coal, Iron Ore)")
    benchmark_price_usd_per_mt: float = Field(..., description="Latest available benchmark price")
    unit: str = Field(..., description="Quotation unit (e.g. USD/mt or USD/dmt)")
    benchmark_date: str = Field(..., description="Date of the benchmark observation (YYYY-MM-DD)")
    percentile: float = Field(..., description="Historical percentile of benchmark price (0-100)")
    momentum_3m_pct: float = Field(..., description="Trailing 3-month price change percentage")
    volatility_3m_pct: float = Field(..., description="Trailing 3-month standard deviation of monthly returns")
    signal: str = Field(..., description="Procurement recommendation: BUY | MONITOR | WAIT")
    signal_score: float = Field(..., description="Deterministic decision score (-100 to +100)")
    reasons: list[str] = Field(..., description="Explanatory decision rationale bullets")
    source: str = Field(default="World Bank Pink Sheet", description="Benchmark data source")
    data_freshness: str = Field(..., description="Data observation period and publication status")


class ProcurementOverviewResponse(BaseModel):
    """Unified procurement valuation overview across all supported commodities."""

    commodities: list[ProcurementValuationResponse] = Field(
        ..., description="List of commodity procurement valuations"
    )
    total_supported: int = Field(..., description="Count of evaluated commodities")
    source: str = Field(default="World Bank Pink Sheet")
    evaluated_at: str = Field(..., description="UTC ISO timestamp of evaluation")


# --------------------------------------------------------------------------- #
# Vessel Suitability & Chartering Optimization schemas
# --------------------------------------------------------------------------- #
class VesselOptimizationRequest(BaseModel):
    """Request payload for vessel suitability and chartering optimization."""

    origin: str = Field(..., description="Origin port or region (e.g. 'Hay Point', 'Taboneo')")
    destination: str = Field(..., description="Discharge port or region (e.g. 'Dhamra', 'Paradip', 'East Coast India')")
    commodity: str = Field(..., description="Cargo commodity (e.g. 'Coal', 'Thermal Coal', 'Iron Ore')")
    cargo_tonnes: float = Field(..., description="Intended shipment cargo volume in metric tonnes (strictly > 0)")
    current_freight_usd_per_tonne: Optional[float] = Field(
        None, description="Optional current freight benchmark rate (if omitted, latest canonical benchmark is used)"
    )


class VesselEvaluationItem(BaseModel):
    """Evaluation result for an individual vessel class."""

    vessel_type: str = Field(..., description="Vessel class (Capesize, Panamax, Supramax, Handysize)")
    standard_dwt: float = Field(..., description="Baltic Exchange standard deadweight tonnage")
    draft_m: float = Field(..., description="Vessel fully laden draft in meters")
    cargo_tonnes: float = Field(..., description="Evaluated cargo volume in metric tonnes")
    utilization_pct: float = Field(..., description="Cargo capacity utilization percentage")
    predicted_freight_usd_per_tonne: Optional[float] = Field(
        None, description="Model v3 forecasted freight rate (USD/t)"
    )
    estimated_freight_outlay_usd: Optional[float] = Field(
        None, description="Total estimated freight cost (rate * tonnes)"
    )
    forecast_change_percent: Optional[float] = Field(
        None, description="Model v3 forecast percent change vs current freight"
    )
    forecast_recommendation: Optional[str] = Field(
        None, description="Model v3 chartering recommendation ('CHARTER NOW', 'WAIT', 'MONITOR')"
    )
    forecast_risk_level: Optional[str] = Field(
        None, description="Model v3 weather risk band ('LOW', 'MEDIUM', 'HIGH')"
    )
    port_compatible: bool = Field(..., description="Physical draft and berth feasibility at discharge port")
    port_fit_label: Optional[str] = Field(None, description="Human-readable port draft feasibility summary")
    cargo_fit: bool = Field(..., description="Cargo volume compatibility with vessel deadweight")
    cargo_fit_label: Optional[str] = Field(None, description="Human-readable cargo utilization fit summary")
    corridor_supported: bool = Field(..., description="Whether vessel class operates on this trade lane")
    eligible: bool = Field(..., description="Overall eligibility gate (port_compatible AND cargo_fit AND corridor_supported)")
    suitability_score: float = Field(..., description="Suitability score on a 0-100 scale")
    reasons: list[str] = Field(..., description="Detailed feasibility and scoring rationale bullets")


class VesselOptimizationResponse(BaseModel):
    """Optimization decision and comparative vessel evaluations."""

    origin: str = Field(..., description="Loading port or region")
    destination: str = Field(..., description="Discharge port or region")
    commodity: str = Field(..., description="Cargo commodity")
    cargo_tonnes: float = Field(..., description="Cargo volume in metric tonnes")
    status: str = Field(..., description="'OPTIMIZED' or 'NO_SUITABLE_VESSEL'")
    recommended_vessel: Optional[str] = Field(None, description="Recommended vessel class (or null if none eligible)")
    recommendation_reason: str = Field(..., description="Comprehensive decision rationale")
    evaluated_vessels: list[VesselEvaluationItem] = Field(..., description="Comparative evaluations across vessel classes")
    source_data: dict[str, str] = Field(default_factory=dict, description="Attributed source datasets")


# --------------------------------------------------------------------------- #
# Decision Orchestration Schemas
# --------------------------------------------------------------------------- #
class DecisionProcurementSummary(BaseModel):
    """Procurement valuation summary embedded in decision orchestration response."""

    signal: str = Field(..., description="Procurement recommendation: BUY | MONITOR | WAIT")
    benchmark_price_usd_per_mt: float = Field(..., description="Latest available commodity benchmark price")
    unit: str = Field(..., description="Quotation unit (e.g. USD/mt or USD/dmt)")
    benchmark_date: str = Field(..., description="Date of the benchmark observation (YYYY-MM-DD)")
    percentile: float = Field(..., description="Historical percentile of benchmark price (0-100)")
    momentum_3m_pct: float = Field(..., description="Trailing 3-month price change percentage")
    data_freshness: str = Field(..., description="Data observation period and publication status")
    reasons: list[str] = Field(default_factory=list, description="Procurement valuation rationale bullets")


class DecisionVesselSummary(BaseModel):
    """Vessel suitability & freight outlay summary embedded in decision orchestration response."""

    recommended_vessel: Optional[str] = Field(None, description="Recommended vessel class (or null if none eligible)")
    status: str = Field(..., description="'OPTIMIZED' or 'NO_SUITABLE_VESSEL'")
    predicted_freight_usd_per_tonne: Optional[float] = Field(None, description="Model v3 forecasted freight rate (USD/t)")
    estimated_freight_outlay_usd: Optional[float] = Field(None, description="Total estimated freight cost (rate * tonnes)")
    suitability_score: Optional[float] = Field(None, description="Suitability score on a 0-100 scale")
    reasons: list[str] = Field(default_factory=list, description="Vessel suitability and feasibility bullets")
    evaluated_vessels: list[VesselEvaluationItem] = Field(
        default_factory=list, description="Comparative evaluations across vessel classes"
    )


class DecisionLandedCostSummary(BaseModel):
    """Delivered commodity acquisition cost economics combining commodity benchmark and ocean freight."""

    commodity_fob_usd: Optional[float] = Field(
        None, description="Procurement benchmark commodity price FOB"
    )
    ocean_freight_usd_per_tonne: Optional[float] = Field(
        None, description="Forecasted ocean freight rate (USD/t)"
    )
    estimated_landed_cost_usd: Optional[float] = Field(
        None, description="Estimated delivered commodity acquisition cost (FOB + Ocean Freight)"
    )
    commodity_unit: str = Field(..., description="Quotation unit for commodity (USD/mt or USD/dmt)")
    freight_unit: str = Field(default="USD/t", description="Quotation unit for ocean freight (USD/t)")
    landed_cost_unit: str = Field(..., description="Delivered quotation unit (USD/mt or USD/dmt)")
    estimated_total_landed_outlay_usd: Optional[float] = Field(
        None, description="Total estimated cargo acquisition + ocean freight outlay for parcel"
    )
    formula: str = Field(
        default="Landed Cost = Commodity FOB + Ocean Freight",
        description="Applied landed cost calculation formula",
    )
    provenance: dict[str, str] = Field(
        default_factory=dict, description="Source provenance for commodity and freight components"
    )
    reasons: list[str] = Field(
        default_factory=list, description="Landed cost calculation breakdown and explanation"
    )


class DecisionAnalysisRequest(BaseModel):
    """Request payload for end-to-end procurement and chartering decision orchestration."""

    commodity: str = Field(..., description="Cargo commodity (e.g. 'Coal', 'Iron Ore')")
    origin: str = Field(..., description="Origin port or region (e.g. 'Hay Point', 'Taboneo')")
    destination: str = Field(..., description="Discharge port or region (e.g. 'Dhamra', 'Paradip', 'East Coast India')")
    cargo_tonnes: float = Field(..., description="Intended shipment cargo volume in metric tonnes (strictly > 0)")
    current_freight_usd_per_tonne: Optional[float] = Field(
        None, description="Optional current freight benchmark rate (USD/tonne) to override baseline"
    )


class DecisionAnalysisResponse(BaseModel):
    """Unified operational procurement and chartering strategy decision response."""

    commodity: str = Field(..., description="Canonical commodity name")
    origin: str = Field(..., description="Loading port or trade corridor")
    destination: str = Field(..., description="Discharge port or region")
    cargo_tonnes: float = Field(..., description="Cargo volume in metric tonnes")
    procurement: DecisionProcurementSummary = Field(..., description="Commodity valuation and market signal")
    vessel: DecisionVesselSummary = Field(..., description="Vessel suitability and freight forecast economics")
    landed_cost: DecisionLandedCostSummary = Field(
        ..., description="Delivered commodity acquisition cost economics (FOB + Freight)"
    )
    charter_decision: str = Field(
        ..., description="Freight timing recommendation: CHARTER NOW | WAIT TO CHARTER | MONITOR FREIGHT | NO SUITABLE VESSEL"
    )
    overall_strategy: str = Field(
        ..., description="Combined operational directive (e.g. 'BUY CARGO — CHARTER NOW')"
    )
    weather_override: bool = Field(
        ..., description="Whether elevated weather risk changed the charter timing decision to CHARTER NOW"
    )
    weather_risk_level: str = Field(
        ..., description="Maritime weather risk band: LOW | MEDIUM | HIGH"
    )
    decision_reasons: list[str] = Field(
        ..., description="Structured transparent explanations synthesizing all layers"
    )



