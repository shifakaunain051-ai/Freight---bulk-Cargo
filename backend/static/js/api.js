/**
 * Freight Intelligence API Client
 * High-reliability typed interface to FastAPI backend services.
 */

const API_BASE = "";

class APIError extends Error {
  constructor(status, data) {
    super(data.message || "An unexpected error occurred.");
    this.name = "APIError";
    this.status = status;
    this.errorCode = data.error_code || "UNKNOWN_ERROR";
    this.missingFields = data.missing_fields || [];
    this.detail = data.detail || null;
  }
}

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const defaultHeaders = {
    "Accept": "application/json",
    "Content-Type": "application/json",
  };

  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        ...defaultHeaders,
        ...(options.headers || {}),
      },
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new APIError(response.status, data);
    }

    return data;
  } catch (err) {
    if (err instanceof APIError) {
      throw err;
    }
    throw new APIError(0, {
      error_code: "NETWORK_ERROR",
      message: err.message || "Failed to communicate with backend server.",
    });
  }
}

export const API = {
  // Decision Orchestration & Evaluation (MVP Core)
  async analyzeDecision(payload) {
    return request("/decision/analyze", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  async getProcurementOverview() {
    return request("/procurement/overview");
  },

  // Phase 4: Unified Dashboard Aggregation
  async getDashboardOverview() {
    return request("/dashboard/overview");
  },

  // Phase 1 & 2: Inference & Scenario Simulation
  async predict(payload) {
    return request("/predict", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  async predictScenario(payload) {
    return request("/predict/scenario", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  // Phase 3: Market Intelligence & Analytics
  async getFreightTrends(params = {}) {
    const query = new URLSearchParams();
    if (params.origin) query.set("origin", params.origin);
    if (params.destination) query.set("destination", params.destination);
    if (params.commodity) query.set("commodity", params.commodity);
    if (params.vessel_type) query.set("vessel_type", params.vessel_type);
    const qs = query.toString() ? `?${query.toString()}` : "";
    return request(`/analytics/freight-trends${qs}`);
  },

  async getMarketTrends() {
    return request("/analytics/market-trends");
  },

  async getWeatherTrends(origin = null) {
    const qs = origin ? `?origin=${encodeURIComponent(origin)}` : "";
    return request(`/analytics/weather-trends${qs}`);
  },

  async getRoutesAnalytics() {
    return request("/analytics/routes");
  },

  async getExecutiveSummary() {
    return request("/analytics/summary");
  },

  async getCorrelations() {
    return request("/analytics/correlations");
  },

  async getLatestSnapshot() {
    return request("/analytics/latest");
  },

  // System & Model Status
  async getModelInfo() {
    return request("/model/info");
  },

  async getDataStatus() {
    return request("/data/status");
  },

  async getLatestData() {
    return request("/data/latest");
  },

  async getTelemetry(limit = 20) {
    return request(`/data/telemetry?limit=${limit}`);
  },

  async getHealth() {
    return request("/health");
  },
};
