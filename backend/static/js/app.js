/**
 * NaviFreight — Application Controller
 * Simplified User-Facing Language (Professional Logistics Focus)
 * Zero user-facing weather models, zero technical ML jargon.
 */

import { API } from "./api.js";
import { Charts } from "./charts.js";

// Canonical trade corridors configuration for intuitive form defaults
const CANONICAL_CORRIDORS = {
  "Coal": {
    defaultOrigin: "Hay Point",
    defaultDest: "Dhamra",
    defaultVolume: 150000,
    validOrigins: ["Hay Point"],
  },
  "Iron Ore": {
    defaultOrigin: "Australia West Coast",
    defaultDest: "Dhamra",
    defaultVolume: 150000,
    validOrigins: ["Australia West Coast"],
  },
  "Thermal Coal": {
    defaultOrigin: "Taboneo",
    defaultDest: "Dhamra",
    defaultVolume: 75000,
    validOrigins: ["Taboneo"],
  },
};

// Global State
const appState = {
  currentTab: "view-cargo-decision",
  latestDecision: null,
  activeOrigin: "Hay Point",
  activeDestination: "Dhamra",
  activeCommodity: "Coal",
  activeCargoTonnes: 150000,
};

// DOM Initialization
document.addEventListener("DOMContentLoaded", async () => {
  initTabs();
  initFormInteractions();
  initBackToTop();
  
  // Initial analysis with standard default shipment (no overlay/auto-scroll on initial load)
  await runShipmentAnalysis({ scrollOnSuccess: false, showAnalysisOverlay: false });
});

// ---------------------------------------------------------------------------
// 1. Tab Switching (Exactly Two Tabs)
// ---------------------------------------------------------------------------
function initTabs() {
  const tabCargo = document.getElementById("tab-cargo");
  const tabAudit = document.getElementById("tab-audit");
  const viewCargo = document.getElementById("view-cargo-decision");
  const viewAudit = document.getElementById("view-audit-evidence");

  function switchTab(targetTab) {
    if (targetTab === "cargo") {
      tabCargo.classList.add("active");
      tabAudit.classList.remove("active");
      viewCargo.classList.remove("hidden");
      viewAudit.classList.add("hidden");
      appState.currentTab = "view-cargo-decision";
    } else {
      tabAudit.classList.add("active");
      tabCargo.classList.remove("active");
      viewAudit.classList.remove("hidden");
      viewCargo.classList.add("hidden");
      appState.currentTab = "view-audit-evidence";
    }
  }

  tabCargo.addEventListener("click", () => switchTab("cargo"));
  tabAudit.addEventListener("click", () => switchTab("audit"));
}

// ---------------------------------------------------------------------------
// 2. Form Interactions & Canonical Corridors Synchronization
// ---------------------------------------------------------------------------
function initFormInteractions() {
  const selectCommodity = document.getElementById("input-commodity");
  const selectOrigin = document.getElementById("input-origin");
  const inputCargo = document.getElementById("input-cargo-tonnes");
  const formShipment = document.getElementById("form-shipment");

  // Synchronize Origin when Commodity changes
  selectCommodity.addEventListener("change", (e) => {
    const comm = e.target.value;
    if (CANONICAL_CORRIDORS[comm]) {
      selectOrigin.value = CANONICAL_CORRIDORS[comm].defaultOrigin;
      inputCargo.value = CANONICAL_CORRIDORS[comm].defaultVolume;
    }
  });

  // Synchronize Commodity when Origin changes
  selectOrigin.addEventListener("change", (e) => {
    const orig = e.target.value;
    if (orig === "Australia West Coast") {
      selectCommodity.value = "Iron Ore";
      inputCargo.value = 150000;
    } else if (orig === "Taboneo") {
      selectCommodity.value = "Thermal Coal";
      inputCargo.value = 75000;
    } else if (orig === "Hay Point") {
      selectCommodity.value = "Coal";
      inputCargo.value = 150000;
    }
  });

  // Form Submit Handler (Scrolls to results on successful user submission)
  formShipment.addEventListener("submit", async (e) => {
    e.preventDefault();
    await runShipmentAnalysis({ scrollOnSuccess: true, showAnalysisOverlay: true });
  });
}

// ---------------------------------------------------------------------------
// 3. Shipment Decision Execution & Rendering
// ---------------------------------------------------------------------------
async function runShipmentAnalysis(opts = { scrollOnSuccess: false, showAnalysisOverlay: false }) {
  const btnAnalyze = document.getElementById("btn-analyze");
  const spinner = document.getElementById("btn-analyze-spinner");
  const btnText = document.getElementById("btn-analyze-text");
  const errorContainer = document.getElementById("decision-error");
  const errorMessage = document.getElementById("decision-error-message");

  const commodity = document.getElementById("input-commodity").value;
  const origin = document.getElementById("input-origin").value;
  const destination = document.getElementById("input-destination").value;
  const cargoInputEl = document.getElementById("input-cargo-tonnes");
  const rawInput = (cargoInputEl.value || "").trim().replace(/,/g, "");

  function showValidationError(msg) {
    btnAnalyze.disabled = false;
    spinner.classList.add("hidden");
    btnText.textContent = "Analyze Shipment";
    errorMessage.textContent = msg;
    errorContainer.classList.remove("hidden");
    cargoInputEl.focus();
  }

  // 1. Empty Check
  if (!rawInput) {
    showValidationError("Please enter a cargo volume in metric tonnes (e.g. 12345).");
    return;
  }

  // 2. Non-numeric / letters check
  if (!/^-?\d+(\.\d+)?$/.test(rawInput)) {
    showValidationError("Please enter a valid whole number (digits only, e.g. 46038).");
    return;
  }

  // 3. Decimal check (reject decimals with specific message)
  if (rawInput.includes(".")) {
    showValidationError("Please enter a whole number of tonnes (no decimals).");
    return;
  }

  const cargoTonnes = parseInt(rawInput, 10);

  // 4. Positive check
  if (isNaN(cargoTonnes) || cargoTonnes <= 0) {
    showValidationError("Cargo volume must be a positive number.");
    return;
  }

  // 5. Minimum cargo limit (10,000 mt = smallest standard Handysize parcel)
  if (cargoTonnes < 10000) {
    showValidationError("Cargo volume must be at least 10,000 tonnes (smallest standard Handysize parcel).");
    return;
  }

  // 6. Maximum cargo limit (200,000 mt = Capesize physical maximum)
  if (cargoTonnes > 200000) {
    showValidationError("Cargo volume cannot exceed 200,000 tonnes (maximum Capesize capacity).");
    return;
  }

  appState.activeCommodity = commodity;
  appState.activeOrigin = origin;
  appState.activeDestination = destination;
  appState.activeCargoTonnes = cargoTonnes;

  // 1. START ANALYSIS STATE
  // Immediately disable the Analyze Shipment button to prevent duplicate requests
  btnAnalyze.disabled = true;
  spinner.classList.remove("hidden");
  btnText.textContent = "Analyzing...";
  errorContainer.classList.add("hidden");

  try {
    const payload = {
      origin,
      destination,
      commodity,
      cargo_tonnes: cargoTonnes,
    };

    let decision;
    if (opts && opts.showAnalysisOverlay) {
      showLoadingCard();

      let apiFinished = false;
      let apiError = null;
      let apiResult = null;

      // Start the real backend call immediately (no duplicate requests)
      API.analyzeDecision(payload)
        .then((res) => {
          apiResult = res;
          apiFinished = true;
        })
        .catch((err) => {
          apiError = err;
          apiFinished = true;
        });

      // Sequence through all 5 statements (~600ms each, total ~3s)
      for (let i = 0; i < ANALYSIS_SENTENCES.length; i++) {
        updateLoadingSentence(ANALYSIS_SENTENCES[i]);

        // Display each sentence for ~600ms (check every 50ms for error exit)
        let elapsed = 0;
        while (elapsed < 600) {
          if (apiError) break;
          await new Promise((r) => setTimeout(r, 50));
          elapsed += 50;
        }

        if (apiError) break;
      }

      // If backend is still running after all 5 statements have shown,
      // stay on "Calculating landed cost ..." with the animated dots until it arrives
      if (!apiError && !apiFinished) {
        while (!apiFinished) {
          await new Promise((r) => setTimeout(r, 50));
        }
      }

      hideLoadingCard();

      if (apiError) {
        throw apiError;
      }
      decision = apiResult;
    } else {
      decision = await API.analyzeDecision(payload);
    }

    appState.latestDecision = decision;

    // Display the actual recommendation and results
    renderRecommendation(decision);
    renderDecisionSummary(decision);
    renderWhyReasons(decision);
    renderLandedCost(decision);
    renderVesselOptions(decision);
    await renderFreightHistory(origin, commodity, decision.vessel ? decision.vessel.recommended_vessel : null);
    syncAuditScreen(decision);

    // Smoothly scroll the page so the Recommendation heading is visible near top of viewport
    if (opts && opts.scrollOnSuccess) {
      setTimeout(() => {
        scrollToResults();
      }, 50);
    }

  } catch (err) {
    console.error("Analysis error:", err);
    // Error Handling: remove loading card, show simple error message, do not scroll
    hideLoadingCard();
    errorContainer.classList.remove("hidden");
    errorMessage.textContent = "Unable to analyze this shipment. Please try again.";
  } finally {
    // Re-enable Analyze Shipment button
    btnAnalyze.disabled = false;
    spinner.classList.add("hidden");
    btnText.textContent = "Analyze Shipment";
  }
}

// ---------------------------------------------------------------------------
// 4. Section Renderers (Simple Professional Language)
// Standardized Logistics Decision Terminology Helpers
function getCargoDecisionText(signal) {
  if (signal === "BUY") return "BUY CARGO";
  if (signal === "WAIT") return "WAIT TO BUY";
  return "MONITOR CARGO";
}

function getFreightDecisionText(charterDecision) {
  if (charterDecision === "CHARTER NOW") return "CHARTER NOW";
  if (charterDecision === "WAIT TO CHARTER") return "WAIT TO CHARTER";
  return "MONITOR FREIGHT";
}

// ---------------------------------------------------------------------------
// 4. Section Renderers (Simple Professional Language)
// ---------------------------------------------------------------------------

/**
 * 2. Recommendation Hero Card (Two Separate Decisions: Cargo & Freight)
 */
function renderRecommendation(data) {
  const twoDecisionsContainer = document.getElementById("recommendation-two-decisions");
  const infeasibleContainer = document.getElementById("recommendation-infeasible");
  const recCargoEl = document.getElementById("rec-cargo-decision");
  const recFreightEl = document.getElementById("rec-freight-decision");
  const explEl = document.getElementById("strategy-explanation");
  const tsEl = document.getElementById("strategy-timestamp");

  const vRec = data.vessel || {};
  const isNoVessel = vRec.status === "NO_SUITABLE_VESSEL" || (data.overall_strategy || "").includes("NO SUITABLE VESSEL");

  if (isNoVessel) {
    if (twoDecisionsContainer) twoDecisionsContainer.classList.add("hidden");
    if (infeasibleContainer) infeasibleContainer.classList.remove("hidden");
    explEl.textContent = "This shipment cannot be handled by a suitable vessel for the selected route. Try a smaller cargo volume or another destination port.";
  } else {
    if (twoDecisionsContainer) twoDecisionsContainer.classList.remove("hidden");
    if (infeasibleContainer) infeasibleContainer.classList.add("hidden");

    const cargoDecision = getCargoDecisionText(data.procurement?.signal);
    const freightDecision = getFreightDecisionText(data.charter_decision);

    // Update the two large decision text elements
    if (recCargoEl) {
      recCargoEl.textContent = cargoDecision;
      recCargoEl.className = "text-2xl sm:text-3xl font-extrabold tracking-tight " + 
        (cargoDecision === "BUY CARGO" ? "text-emerald-700" : cargoDecision === "WAIT TO BUY" ? "text-rose-700" : "text-amber-700");
    }

    if (recFreightEl) {
      recFreightEl.textContent = freightDecision;
      recFreightEl.className = "text-2xl sm:text-3xl font-extrabold tracking-tight " +
        (freightDecision === "CHARTER NOW" ? "text-brand-700" : freightDecision === "WAIT TO CHARTER" ? "text-amber-700" : "text-gray-700");
    }

    // Natural single explanation sentence combining both decisions
    if (cargoDecision === "WAIT TO BUY" && freightDecision === "CHARTER NOW") {
      explEl.textContent = "Cargo prices are high, but freight rates are expected to rise. Wait to buy the cargo, but charter the vessel now.";
    } else if (cargoDecision === "WAIT TO BUY" && freightDecision === "WAIT TO CHARTER") {
      explEl.textContent = "Cargo prices are high and freight rates are expected to decrease. Wait to buy the cargo and wait before chartering.";
    } else if (cargoDecision === "WAIT TO BUY" && freightDecision === "MONITOR FREIGHT") {
      explEl.textContent = "Cargo prices are high while freight rates are steady. Wait to buy the cargo and monitor freight rates.";
    } else if (cargoDecision === "BUY CARGO" && freightDecision === "CHARTER NOW") {
      explEl.textContent = "Cargo prices are attractive and freight rates are expected to rise. Buy the cargo and charter the vessel now.";
    } else if (cargoDecision === "BUY CARGO" && freightDecision === "WAIT TO CHARTER") {
      explEl.textContent = "Cargo prices are attractive, but freight rates are expected to decrease. Buy the cargo now, but wait before chartering.";
    } else if (cargoDecision === "BUY CARGO" && freightDecision === "MONITOR FREIGHT") {
      explEl.textContent = "Cargo prices are attractive while freight rates are steady. Buy the cargo now and monitor freight rates.";
    } else if (cargoDecision === "MONITOR CARGO" && freightDecision === "CHARTER NOW") {
      explEl.textContent = "Cargo prices are steady, but freight rates are expected to rise. Monitor cargo prices, but charter the vessel now.";
    } else if (cargoDecision === "MONITOR CARGO" && freightDecision === "WAIT TO CHARTER") {
      explEl.textContent = "Cargo prices are steady and freight rates are expected to decrease. Monitor cargo prices and wait before chartering.";
    } else {
      explEl.textContent = "Cargo and freight markets are steady. Monitor cargo prices and freight rates for better entry windows.";
    }
  }

  // Timestamp
  const now = new Date();
  tsEl.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/**
 * 3. Decision Summary Cards (Cargo / Freight / Vessel)
 */
function renderDecisionSummary(data) {
  const vRec = data.vessel || {};
  const isNoVessel = vRec.status === "NO_SUITABLE_VESSEL" || (data.overall_strategy || "").includes("NO SUITABLE VESSEL");

  // 3.1 Cargo Card
  const proc = data.procurement || {};
  const cargoSignalBadge = document.getElementById("summary-cargo-signal-badge");
  const cargoPrice = document.getElementById("summary-cargo-price");
  const cargoUnit = document.getElementById("summary-cargo-unit");
  const cargoPercentile = document.getElementById("summary-cargo-percentile");
  const cargoMomentum = document.getElementById("summary-cargo-momentum");

  const cargoDecision = getCargoDecisionText(proc.signal);
  cargoSignalBadge.textContent = cargoDecision;
  cargoSignalBadge.className = "badge";
  if (cargoDecision === "BUY CARGO") cargoSignalBadge.classList.add("badge-success");
  else if (cargoDecision === "WAIT TO BUY") cargoSignalBadge.classList.add("badge-danger");
  else cargoSignalBadge.classList.add("badge-warning");

  cargoPrice.textContent = proc.benchmark_price_usd_per_mt !== undefined ? `$${proc.benchmark_price_usd_per_mt.toFixed(2)}` : "—";
  cargoUnit.textContent = proc.unit || "USD/mt";
  
  // Simple Price vs. History text
  if (proc.percentile !== undefined) {
    if (proc.percentile >= 75) {
      cargoPercentile.textContent = `High (${Math.round(proc.percentile)}%)`;
    } else if (proc.percentile <= 35) {
      cargoPercentile.textContent = `Low (${Math.round(proc.percentile)}%)`;
    } else {
      cargoPercentile.textContent = `Moderate (${Math.round(proc.percentile)}%)`;
    }
  } else {
    cargoPercentile.textContent = "—";
  }
  
  if (proc.momentum_3m_pct !== undefined) {
    const sign = proc.momentum_3m_pct > 0 ? "+" : "";
    cargoMomentum.textContent = `${sign}${proc.momentum_3m_pct.toFixed(1)}%`;
  } else {
    cargoMomentum.textContent = "—";
  }

  // 3.2 Freight Card
  const freightRate = document.getElementById("summary-freight-rate");
  const freightUnit = document.getElementById("summary-freight-unit");
  const charterBadge = document.getElementById("summary-freight-charter-badge");
  const freightTiming = document.getElementById("summary-freight-timing");
  const freightTimingContainer = document.getElementById("summary-freight-timing-container");

  if (isNoVessel) {
    freightRate.textContent = "Not available";
    freightRate.className = "text-xl font-bold text-gray-400";
    if (freightUnit) freightUnit.textContent = "";
    charterBadge.textContent = "Not available";
    charterBadge.className = "badge badge-neutral";
    // Option A: Hide Charter Decision field when there is no suitable vessel
    if (freightTimingContainer) {
      freightTimingContainer.classList.add("hidden");
    }
  } else {
    freightRate.textContent = vRec.predicted_freight_usd_per_tonne !== null && vRec.predicted_freight_usd_per_tonne !== undefined
      ? `$${vRec.predicted_freight_usd_per_tonne.toFixed(2)}`
      : "Not available";
    freightRate.className = "text-2xl font-bold font-mono text-gray-900";
    if (freightUnit) freightUnit.textContent = "USD/t";

    const freightDecision = getFreightDecisionText(data.charter_decision);
    charterBadge.textContent = freightDecision;
    charterBadge.className = "badge";
    if (freightDecision === "CHARTER NOW") charterBadge.classList.add("badge-primary");
    else if (freightDecision === "WAIT TO CHARTER") charterBadge.classList.add("badge-warning");
    else charterBadge.classList.add("badge-neutral");

    if (freightTimingContainer) {
      freightTimingContainer.classList.remove("hidden");
    }
    if (freightTiming) {
      freightTiming.textContent = freightDecision;
    }
  }

  // 3.3 Vessel Card
  const vesselBadge = document.getElementById("summary-vessel-status-badge");
  const vesselName = document.getElementById("summary-vessel-name");
  const vesselUtil = document.getElementById("summary-vessel-utilization");
  const vesselOutlay = document.getElementById("summary-vessel-outlay");
  const vesselScore = document.getElementById("summary-vessel-score");

  if (!isNoVessel && vRec.status === "OPTIMIZED" && vRec.recommended_vessel) {
    vesselBadge.className = "badge badge-primary";
    vesselBadge.textContent = vRec.recommended_vessel.toUpperCase();
    vesselName.textContent = vRec.recommended_vessel;

    // Find utilization
    const winningEval = (vRec.evaluated_vessels || []).find(v => v.vessel_type === vRec.recommended_vessel);
    if (winningEval && winningEval.utilization_pct) {
      vesselUtil.textContent = `(${Math.round(winningEval.utilization_pct)}% used)`;
    } else {
      vesselUtil.textContent = "";
    }

    vesselOutlay.textContent = vRec.estimated_freight_outlay_usd
      ? `$${Math.round(vRec.estimated_freight_outlay_usd).toLocaleString()}`
      : "Not available";
    vesselScore.textContent = "Best fit";
  } else {
    vesselBadge.className = "badge badge-danger";
    vesselBadge.textContent = "NOT SUITABLE";
    vesselName.textContent = "NO SUITABLE VESSEL";
    vesselUtil.textContent = "";
    vesselOutlay.textContent = "Not available";
    vesselScore.textContent = "Port / Cargo Limit";
  }
}

/**
 * 4. Why? (Simple, human-readable bullets. Zero weather/ML jargon.)
 */
function renderWhyReasons(data) {
  const listEl = document.getElementById("decision-reasons-list");
  listEl.innerHTML = "";

  const proc = data.procurement || {};
  const vRec = data.vessel || {};
  const isNoVessel = vRec.status === "NO_SUITABLE_VESSEL" || (data.overall_strategy || "").includes("NO SUITABLE VESSEL");
  const charterDec = data.charter_decision || "";
  const reasons = [];

  // Reason 1: Cargo price comparison
  if (proc.percentile !== undefined) {
    if (proc.percentile >= 75) {
      reasons.push("Cargo prices are currently high compared with history.");
    } else if (proc.percentile <= 35) {
      reasons.push("Cargo prices are currently attractive compared with history.");
    } else {
      reasons.push("Cargo prices are in a moderate historical range.");
    }
  }

  if (isNoVessel) {
    reasons.push("No suitable vessel is available for this route and cargo size.");
    reasons.push("Try a smaller cargo volume or another destination port.");
  } else {
    // Reason 2: Price momentum
    if (proc.momentum_3m_pct !== undefined) {
      if (proc.momentum_3m_pct <= -2) {
        reasons.push("Cargo prices have been trending downward over the past 3 months.");
      } else if (proc.momentum_3m_pct >= 2) {
        reasons.push("Cargo prices have been trending upward over the past 3 months.");
      }
    }

    // Reason 3: Freight trajectory / timing
    if (charterDec === "WAIT TO CHARTER") {
      reasons.push("Freight is expected to decrease over the coming weeks.");
      reasons.push("Waiting before chartering may reduce your ocean freight cost.");
    } else if (charterDec === "CHARTER NOW") {
      reasons.push("Freight rates are expected to rise. Chartering now may secure lower rates.");
    } else {
      reasons.push("Freight rates are currently steady.");
    }

    // Reason 4: Vessel fit
    if (vRec.recommended_vessel && vRec.status === "OPTIMIZED") {
      reasons.push(`${vRec.recommended_vessel} is the best fit for this shipment.`);
    }
  }

  // Deduplicate and cap at 4 concise bullets
  const uniqueReasons = [...new Set(reasons)].slice(0, 4);

  uniqueReasons.forEach(reasonText => {
    const li = document.createElement("li");
    li.className = "flex items-start gap-2.5";
    li.innerHTML = `
      <span class="w-1.5 h-1.5 rounded-full bg-brand-600 mt-2 flex-shrink-0"></span>
      <span class="text-gray-700 leading-snug">${escapeHtml(reasonText)}</span>
    `;
    listEl.appendChild(li);
  });
}

/**
 * 5. Estimated Landed Cost
 */
function renderLandedCost(data) {
  const lc = data.landed_cost;
  const vRec = data.vessel || {};
  const isNoVessel = vRec.status === "NO_SUITABLE_VESSEL" || (data.overall_strategy || "").includes("NO SUITABLE VESSEL");

  const fobEl = document.getElementById("lc-fob-rate");
  const fobUnitEl = document.getElementById("lc-fob-unit");
  const freightEl = document.getElementById("lc-freight-rate");
  const freightUnitEl = document.getElementById("lc-freight-unit");
  const landedEl = document.getElementById("lc-landed-rate");
  const landedUnitEl = document.getElementById("lc-landed-unit");
  const tonnesSummary = document.getElementById("lc-tonnes-summary");
  const outlayEl = document.getElementById("lc-total-outlay");

  if (isNoVessel || !lc || lc.estimated_landed_cost_usd === null || lc.estimated_landed_cost_usd === undefined) {
    fobEl.textContent = data.procurement?.benchmark_price_usd_per_mt ? `$${data.procurement.benchmark_price_usd_per_mt.toFixed(2)}` : "—";
    fobUnitEl.textContent = data.procurement?.unit || "USD/mt";
    freightEl.textContent = "Not available";
    freightEl.className = "text-xl font-bold text-gray-400";
    if (freightUnitEl) freightUnitEl.textContent = "";
    landedEl.textContent = "Not available";
    landedEl.className = "text-xl font-bold text-gray-400";
    if (landedUnitEl) landedUnitEl.textContent = "";
    tonnesSummary.textContent = `${Math.round(data.cargo_tonnes).toLocaleString()} mt`;
    outlayEl.textContent = "Not available";
    return;
  }

  fobEl.textContent = `$${lc.commodity_fob_usd.toFixed(2)}`;
  fobUnitEl.textContent = lc.commodity_unit || "USD/mt";

  freightEl.textContent = `$${lc.ocean_freight_usd_per_tonne.toFixed(2)}`;
  freightEl.className = "text-xl font-bold font-mono text-gray-900";
  freightUnitEl.textContent = lc.freight_unit || "USD/t";

  landedEl.textContent = `$${lc.estimated_landed_cost_usd.toFixed(2)}`;
  landedEl.className = "text-2xl font-extrabold font-mono text-brand-700";
  landedUnitEl.textContent = lc.landed_cost_unit || "USD/mt";

  const cargoUnitDisplay = lc.commodity_unit.includes("dmt") ? "dmt" : "mt";
  tonnesSummary.textContent = `${Math.round(data.cargo_tonnes).toLocaleString()} ${cargoUnitDisplay}`;

  if (lc.estimated_total_landed_outlay_usd) {
    outlayEl.textContent = `$${Math.round(lc.estimated_total_landed_outlay_usd).toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })} USD`;
  } else {
    outlayEl.textContent = "Not available";
  }
}

/**
 * 6. Vessel Options Table (Simple terminology)
 */
function renderVesselOptions(data) {
  const tbody = document.getElementById("vessel-options-body");
  const destPortEl = document.getElementById("vessel-table-port");
  destPortEl.textContent = data.destination;
  tbody.innerHTML = "";

  const evaluated = data.vessel?.evaluated_vessels || [];
  const winningClass = data.vessel?.recommended_vessel;

  if (evaluated.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="6" class="text-center py-6 text-gray-500 text-xs">
          No vessel options available for this route.
        </td>
      </tr>
    `;
    return;
  }

  evaluated.forEach(item => {
    const tr = document.createElement("tr");
    const isRecommended = item.vessel_type === winningClass;
    const isEligible = item.eligible === true;

    if (isRecommended) {
      tr.className = "row-recommended font-medium";
    } else if (!isEligible) {
      tr.className = "row-ineligible";
    }

    // Recommendation Status badge
    let statusCell = "";
    if (isRecommended) {
      statusCell = `<span class="badge badge-primary text-[11px]">Recommended</span>`;
    } else if (isEligible) {
      statusCell = `<span class="badge badge-neutral text-[11px]">Suitable</span>`;
    } else {
      statusCell = `<span class="badge badge-danger text-[10px]">Not suitable</span>`;
    }

    // Estimated Freight rate + total outlay
    let freightCell = "";
    if (item.predicted_freight_usd_per_tonne && item.estimated_freight_outlay_usd) {
      freightCell = `
        <div>
          <span class="font-mono text-xs font-semibold text-gray-900">$${item.predicted_freight_usd_per_tonne.toFixed(2)}/t</span>
          <span class="text-gray-500 font-normal text-[11px]">($${Math.round(item.estimated_freight_outlay_usd).toLocaleString()})</span>
        </div>
      `;
    } else {
      freightCell = `<span class="text-gray-400 font-mono text-xs">—</span>`;
    }

    // Cargo Fit description
    let cargoFitCell = "";
    if (item.cargo_fit) {
      const fitLabel = item.cargo_fit_label || (item.utilization_pct >= 80 ? "Optimal fit" : "Suitable");
      const fitColor = isRecommended ? "text-emerald-700 font-medium" : "text-gray-700";
      cargoFitCell = `<span class="${fitColor}">${escapeHtml(fitLabel)} (${Math.round(item.utilization_pct)}%)</span>`;
    } else {
      const fitLabel = item.cargo_fit_label || "Cargo exceeds capacity";
      cargoFitCell = `<span class="text-rose-600 text-xs font-medium">${escapeHtml(fitLabel)}</span>`;
    }

    // Port / Draft Fit description
    let portFitCell = "";
    if (item.port_compatible) {
      portFitCell = `<span class="text-gray-700 font-mono text-xs">${item.draft_m.toFixed(1)}m (Compatible)</span>`;
    } else {
      portFitCell = `<span class="text-rose-600 font-mono text-xs font-medium">${item.draft_m.toFixed(1)}m (Exceeds draft)</span>`;
    }

    tr.innerHTML = `
      <td>
        <div class="font-bold text-gray-900">${escapeHtml(item.vessel_type)}</div>
      </td>
      <td class="font-mono text-xs text-gray-700">
        ${Math.round(item.standard_dwt).toLocaleString()} DWT
      </td>
      <td class="text-xs">
        ${cargoFitCell}
      </td>
      <td class="text-xs">
        ${portFitCell}
      </td>
      <td>
        ${freightCell}
      </td>
      <td>
        ${statusCell}
      </td>
    `;

    tbody.appendChild(tr);
  });
}

/**
 * 7. Freight History (Compact SVG Chart)
 */
async function renderFreightHistory(origin, commodity, vesselType) {
  const container = document.getElementById("chart-container");
  const legendLabel = document.getElementById("chart-legend-label");
  const vType = vesselType || "Capesize";

  legendLabel.textContent = `${vType} ($/t) — ${origin} to Dhamra`;

  try {
    const params = {
      origin,
      commodity,
      vessel_type: vType,
    };

    const trends = await API.getFreightTrends(params);
    const series = trends.series || [];

    if (series.length > 0) {
      Charts.renderTimeSeries(container, series, {
        xKey: "date",
        yKey: "freight_rate_usd_per_tonne",
        strokeColor: "#3B82F6",
        unit: "$/t",
        height: 200,
      });
    } else {
      container.innerHTML = `<div class="p-8 text-center text-xs text-gray-400">Historical rates unavailable for this specific route.</div>`;
    }
  } catch (err) {
    console.warn("Could not load freight trend chart:", err);
    container.innerHTML = `<div class="p-8 text-center text-xs text-gray-400">Historical chart preview unavailable.</div>`;
  }
}

/**
 * Screen 2: Audit & Evidence Synchronization
 */
function syncAuditScreen(data) {
  const tsEl = document.getElementById("audit-timestamp");
  const routeEl = document.getElementById("audit-param-route");
  const volEl = document.getElementById("audit-param-volume");
  const stratEl = document.getElementById("audit-param-strategy");

  const vRec = data.vessel || {};
  const isNoVessel = vRec.status === "NO_SUITABLE_VESSEL" || (data.overall_strategy || "").includes("NO SUITABLE VESSEL");

  tsEl.textContent = new Date().toISOString().replace("T", " ").substring(0, 19) + " UTC";
  routeEl.textContent = `${data.commodity}: ${data.origin} → ${data.destination}`;
  volEl.textContent = `${Math.round(data.cargo_tonnes).toLocaleString()} mt`;
  if (isNoVessel) {
    stratEl.textContent = "NO SUITABLE VESSEL";
  } else {
    const cargoDecision = getCargoDecisionText(data.procurement?.signal);
    const freightDecision = getFreightDecisionText(data.charter_decision);
    stratEl.textContent = `${cargoDecision} · ${freightDecision}`;
  }
}

// Utility: HTML Escaping for Security
function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// ---------------------------------------------------------------------------
// 5. Navigation Helpers (Back to Top & Results Scroll)
// ---------------------------------------------------------------------------
function initBackToTop() {
  const btnBackToTop = document.getElementById("btn-back-to-top");
  if (!btnBackToTop) return;

  window.addEventListener("scroll", () => {
    if (window.scrollY > 250) {
      btnBackToTop.classList.add("visible");
    } else {
      btnBackToTop.classList.remove("visible");
    }
  }, { passive: true });

  btnBackToTop.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}

function scrollToResults() {
  const resultsEl = document.getElementById("section-recommendation") || document.getElementById("results-landed-vessel");
  if (resultsEl) {
    resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

// ---------------------------------------------------------------------------
// 6. Analysis Loading Card Controller (Simple White Enterprise Panel)
// ---------------------------------------------------------------------------
const ANALYSIS_SENTENCES = [
  "Analyzing freight rates",
  "Reviewing cargo prices",
  "Checking weather conditions",
  "Selecting suitable vessel",
  "Calculating landed cost"
];

function showLoadingCard() {
  const overlay = document.getElementById("analysis-overlay");
  const textEl = document.getElementById("analysis-sentence-text");
  if (textEl) {
    textEl.textContent = ANALYSIS_SENTENCES[0];
  }
  if (overlay) {
    overlay.classList.remove("hidden");
  }
}

function hideLoadingCard() {
  const overlay = document.getElementById("analysis-overlay");
  if (overlay) {
    overlay.classList.add("hidden");
  }
}

function updateLoadingSentence(sentence) {
  const textEl = document.getElementById("analysis-sentence-text");
  if (textEl && textEl.textContent !== sentence) {
    textEl.textContent = sentence;
  }
}

