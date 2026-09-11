/**
 * High-Performance SVG Charting Engine for Freight Intelligence Platform
 * Zero external dependencies. Render crisp, responsive SVG visualizations.
 */

export const Charts = {
  /**
   * Render single or dual series time-series SVG chart
   */
  renderTimeSeries(container, dataPoints, options = {}) {
    if (!container) return;
    container.innerHTML = "";

    if (!dataPoints || dataPoints.length === 0) {
      container.innerHTML = `<div class="chart-empty" style="text-align:center;padding:40px;color:var(--text-muted);font-size:0.85rem;">No historical data available</div>`;
      return;
    }

    const width = options.width || container.clientWidth || 600;
    const height = options.height || 260;
    const padding = { top: 20, right: 30, bottom: 40, left: 50 };

    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    const yKey = options.yKey || "freight_rate_usd_per_tonne";
    const xKey = options.xKey || "date";
    const strokeColor = options.strokeColor || "#3B82F6";
    const unit = options.unit || "$/t";

    const yValues = dataPoints.map(d => Number(d[yKey]) || 0);
    const minY = Math.min(...yValues);
    const maxY = Math.max(...yValues);
    const yBuffer = (maxY - minY) * 0.15 || 1.0;
    const yMin = Math.max(0, minY - yBuffer);
    const yMax = maxY + yBuffer;

    const getX = (i) => padding.left + (i / (dataPoints.length - 1 || 1)) * chartW;
    const getY = (val) => padding.top + chartH - ((val - yMin) / (yMax - yMin || 1)) * chartH;

    // Build SVG Path
    let pathD = "";
    let areaD = `M ${padding.left} ${padding.top + chartH} `;

    dataPoints.forEach((d, i) => {
      const x = getX(i);
      const y = getY(d[yKey]);
      if (i === 0) {
        pathD += `M ${x} ${y} `;
        areaD += `L ${x} ${y} `;
      } else {
        pathD += `L ${x} ${y} `;
        areaD += `L ${x} ${y} `;
      }
    });

    areaD += `L ${getX(dataPoints.length - 1)} ${padding.top + chartH} Z`;

    // Horizontal Grid lines (4 ticks)
    let gridLines = "";
    let yLabels = "";
    const yTicksCount = 4;
    for (let t = 0; t <= yTicksCount; t++) {
      const val = yMin + (t / yTicksCount) * (yMax - yMin);
      const y = getY(val);
      gridLines += `<line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="#E5E7EB" stroke-dasharray="3,3" />`;
      yLabels += `<text x="${padding.left - 10}" y="${y + 4}" fill="#6B7280" font-size="10" font-family="monospace" text-anchor="end">${val.toFixed(1)}</text>`;
    }

    // X Axis labels (show ~5 points)
    let xLabels = "";
    const step = Math.max(1, Math.floor(dataPoints.length / 5));
    dataPoints.forEach((d, i) => {
      if (i % step === 0 || i === dataPoints.length - 1) {
        const x = getX(i);
        const dateStr = (d[xKey] || "").substring(0, 7);
        xLabels += `<text x="${x}" y="${height - 12}" fill="#6B7280" font-size="10" font-family="monospace" text-anchor="middle">${dateStr}</text>`;
      }
    });

    // Interactive Data Dots
    let dots = "";
    dataPoints.forEach((d, i) => {
      const x = getX(i);
      const y = getY(d[yKey]);
      const val = Number(d[yKey]).toFixed(2);
      dots += `
        <circle cx="${x}" cy="${y}" r="3.5" fill="${strokeColor}" stroke="#ffffff" stroke-width="2">
          <title>${d[xKey]}: ${val} ${unit}</title>
        </circle>
      `;
    });

    const svg = `
      <svg width="100%" height="${height}" viewBox="0 0 ${width} ${height}" class="chart-svg" style="overflow:visible;">
        <defs>
          <linearGradient id="areaGrad-${yKey}" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="${strokeColor}" stop-opacity="0.25" />
            <stop offset="100%" stop-color="${strokeColor}" stop-opacity="0.0" />
          </linearGradient>
        </defs>
        ${gridLines}
        ${yLabels}
        ${xLabels}
        <path d="${areaD}" fill="url(#areaGrad-${yKey})" />
        <path d="${pathD}" fill="none" stroke="${strokeColor}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />
        ${dots}
      </svg>
    `;

    container.innerHTML = svg;
  },

  /**
   * Render Multi-series macro trends chart
   */
  renderMultiSeries(container, seriesData, options = {}) {
    if (!container || !seriesData || seriesData.length === 0) return;
    container.innerHTML = "";

    const width = options.width || container.clientWidth || 800;
    const height = options.height || 300;
    const padding = { top: 20, right: 80, bottom: 40, left: 60 };

    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    // Series configurations: BDI (left axis/norm), VLSFO (norm), Freight (norm)
    const dates = seriesData.map(d => d.date);

    // Normalize each series 0..100 for multi-scale comparison or plot on primary/secondary
    const bdiVals = seriesData.map(d => d.bdi);
    const vlsfoVals = seriesData.map(d => d.vlsfo_usd_per_tonne);
    const freightVals = seriesData.map(d => d.average_freight_usd_per_tonne);

    const minBdi = Math.min(...bdiVals), maxBdi = Math.max(...bdiVals);
    const minVlsfo = Math.min(...vlsfoVals), maxVlsfo = Math.max(...vlsfoVals);
    const minFr = Math.min(...freightVals), maxFr = Math.max(...freightVals);

    const getX = (i) => padding.left + (i / (dates.length - 1 || 1)) * chartW;
    const getY = (val, min, max) => padding.top + chartH - ((val - min) / (max - min || 1)) * chartH;

    const buildPath = (vals, min, max) => {
      return vals.map((v, i) => `${i === 0 ? "M" : "L"} ${getX(i)} ${getY(v, min, max)}`).join(" ");
    };

    const bdiPath = buildPath(bdiVals, minBdi, maxBdi);
    const vlsfoPath = buildPath(vlsfoVals, minVlsfo, maxVlsfo);
    const frPath = buildPath(freightVals, minFr, maxFr);

    // X Axis
    let xLabels = "";
    const step = Math.max(1, Math.floor(dates.length / 6));
    dates.forEach((d, i) => {
      if (i % step === 0 || i === dates.length - 1) {
        const x = getX(i);
        xLabels += `<text x="${x}" y="${height - 10}" fill="var(--text-muted)" font-size="10" font-family="var(--font-mono)" text-anchor="middle">${d.substring(0,7)}</text>`;
      }
    });

    const svg = `
      <svg width="100%" height="${height}" viewBox="0 0 ${width} ${height}" style="overflow:visible;">
        <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${padding.top + chartH}" stroke="var(--border)" />
        <line x1="${padding.left}" y1="${padding.top + chartH}" x2="${width - padding.right}" y2="${padding.top + chartH}" stroke="var(--border)" />
        ${xLabels}

        <!-- Freight Rate Line -->
        <path d="${frPath}" fill="none" stroke="#3B82F6" stroke-width="3" stroke-linecap="round" />
        <!-- BDI Line -->
        <path d="${bdiPath}" fill="none" stroke="#38BDF8" stroke-width="2" stroke-dasharray="4,3" />
        <!-- VLSFO Line -->
        <path d="${vlsfoPath}" fill="none" stroke="#FBBF24" stroke-width="2" stroke-dasharray="2,2" />
      </svg>
    `;

    container.innerHTML = svg;
  },

  /**
   * Render Horizontal Waterfall Driver Attribution bars
   */
  renderDriverWaterfall(container, drivers) {
    if (!container) return;
    container.innerHTML = "";

    if (!drivers || drivers.length === 0) {
      container.innerHTML = `<p style="color:var(--text-muted);font-size:0.85rem;">No feature drivers computed.</p>`;
      return;
    }

    const maxMagnitude = Math.max(...drivers.map(d => Math.abs(d.contribution_usd_per_tonne))) || 1.0;

    let html = `<div class="drivers-waterfall-list">`;
    drivers.forEach(d => {
      const isPos = d.contribution_usd_per_tonne >= 0;
      const pct = Math.min(100, Math.round((Math.abs(d.contribution_usd_per_tonne) / maxMagnitude) * 100));
      const effectClass = isPos ? "positive" : "negative";
      const sign = isPos ? "+" : "";

      html += `
        <div class="driver-item">
          <div class="driver-label" title="${d.feature}">
            <span>${d.feature_label}</span>
            <small style="display:block;font-size:11px;color:var(--text-muted);">${d.value} ${d.unit}</small>
          </div>
          <div class="driver-bar-track">
            <div class="driver-bar-fill ${effectClass}" style="width:${pct}%;"></div>
          </div>
          <div class="driver-val ${effectClass}">
            ${sign}${d.contribution_usd_per_tonne.toFixed(2)} <span style="font-size:10px;font-weight:normal;color:var(--text-muted);">$/t</span>
          </div>
        </div>
      `;
    });
    html += `</div>`;

    container.innerHTML = html;
  },

  /**
   * Render Correlation ranking bars
   */
  renderCorrelations(container, correlations) {
    if (!container) return;
    container.innerHTML = "";

    if (!correlations || correlations.length === 0) {
      container.innerHTML = `<p style="color:var(--text-muted);font-size:0.85rem;">No correlations available.</p>`;
      return;
    }

    let html = `<div class="correlation-bars-list">`;
    correlations.forEach(c => {
      const isPos = c.correlation >= 0;
      const pct = Math.round(Math.abs(c.correlation) * 100);
      const color = isPos ? "var(--primary)" : "var(--info)";

      html += `
        <div style="margin-bottom:14px;">
          <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:4px;">
            <span style="font-weight:600;color:var(--text-primary);">${c.feature_label}</span>
            <span style="font-family:var(--font-mono);font-weight:700;color:${color};">
              r = ${c.correlation > 0 ? "+" : ""}${c.correlation.toFixed(3)}
            </span>
          </div>
          <div style="height:6px;background:var(--surface-lowest);border-radius:999px;overflow:hidden;">
            <div style="width:${pct}%;height:100%;background:${color};border-radius:999px;"></div>
          </div>
          <div style="font-size:0.72rem;color:var(--text-muted);margin-top:2px;">${c.interpretation}</div>
        </div>
      `;
    });
    html += `</div>`;

    container.innerHTML = html;
  },
};
