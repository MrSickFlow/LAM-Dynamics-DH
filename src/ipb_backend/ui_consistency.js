/**
 * Data Consistency Engine UI — trust panel, anomaly feed, heatmap, clusters.
 * Expects global `state`, DOM refs, and helpers from ui_placeholder.html.
 */
(function () {
  function trustBarClass(confidence) {
    if (confidence >= 0.8) return "trust-high";
    if (confidence >= 0.5) return "trust-mid";
    if (confidence >= 0.25) return "trust-low";
    return "trust-critical";
  }

  function renderTrustPanel(report) {
    const trustList = document.getElementById("trust-list");
    const ewPatternBanner = document.getElementById("ew-pattern-banner");
    if (!trustList) return;
    trustList.innerHTML = "";
    ewPatternBanner.innerHTML = "";
    if (!report?.layer_trust?.length) {
      trustList.innerHTML = '<div class="empty-state">Trust scores appear after consistency check.</div>';
      return;
    }
    report.layer_trust
      .slice()
      .sort((a, b) => a.confidence - b.confidence)
      .forEach((layer) => {
        const row = document.createElement("div");
        row.className = "trust-row";
        const pct = Math.round(layer.confidence * 100);
        row.innerHTML = `
          <div class="trust-row-header"><span>${layer.source_id}</span><span>${pct}%</span></div>
          <div class="trust-row-meta">${layer.ew_classification}${layer.gnss_dependent ? " · GNSS" : ""}</div>
          <div class="trust-bar"><div class="trust-bar-fill ${trustBarClass(layer.confidence)}" style="width:${pct}%"></div></div>
        `;
        trustList.appendChild(row);
      });
    if (report.ew_pattern_detected && ewPatternBanner) {
      ewPatternBanner.className = "ew-banner";
      ewPatternBanner.textContent =
        "Clustered anomalies detected — pattern may indicate localized EM/GNSS degradation. Analyst verification required.";
    }
  }

  function renderAnomalyFeed(report) {
    const anomalyFeed = document.getElementById("anomaly-feed");
    const consistencySummary = document.getElementById("consistency-summary");
    const consistencyDisclaimer = document.getElementById("consistency-disclaimer");
    if (!anomalyFeed) return;
    anomalyFeed.innerHTML = "";
    if (!report) {
      anomalyFeed.innerHTML = '<div class="empty-state">No consistency report yet.</div>';
      return;
    }
    if (consistencySummary) consistencySummary.textContent = report.summary || "Consistency check complete.";
    if (consistencyDisclaimer) consistencyDisclaimer.textContent = report.disclaimer || "";
    if (!report.anomalies?.length) {
      anomalyFeed.innerHTML = '<div class="empty-state">No cross-source anomalies detected.</div>';
      return;
    }
    report.anomalies.forEach((anomaly) => {
      const card = document.createElement("article");
      card.className = "anomaly-card";
      card.dataset.anomalyId = anomaly.anomaly_id;
      const demoTag = anomaly.synthetic_demo ? '<span class="agent-chip is-warning">demo</span>' : "";
      const sources = [...(anomaly.vulnerable_sources || []), ...(anomaly.immune_sources || [])]
        .filter((v, i, a) => a.indexOf(v) === i)
        .map((s) => `<span class="agent-chip">${s}</span>`)
        .join("");
      card.innerHTML = `
        <div class="anomaly-card-header">
          <div class="anomaly-title">${anomaly.title}</div>
          <span class="severity-badge severity-${anomaly.severity}">${anomaly.severity}</span>
        </div>
        <div class="anomaly-copy">${anomaly.description}</div>
        <div class="anomaly-tags">${demoTag}${sources}</div>
      `;
      card.addEventListener("click", () => window.focusConsistencyAnomaly(anomaly.anomaly_id));
      anomalyFeed.appendChild(card);
    });
  }

  window.focusConsistencyAnomaly = function focusAnomaly(anomalyId) {
    state.activeAnomalyId = anomalyId;
    document.querySelectorAll(".anomaly-card").forEach((card) => {
      card.classList.toggle("is-active", card.dataset.anomalyId === anomalyId);
    });
    const marker = state.anomalyMarkers[anomalyId];
    if (marker && state.map) {
      state.map.setView(marker.getLatLng(), Math.max(state.map.getZoom(), 10));
      marker.openPopup();
    }
  };

  function clearConsistencyLayers() {
    state.collectionLayers["consistency-anomaly"]?.clearLayers();
    state.collectionLayers.ais?.clearLayers();
    state.collectionLayers.sar?.clearLayers();
    if (state.heatmapLayer && state.map) {
      state.map.removeLayer(state.heatmapLayer);
      state.heatmapLayer = null;
    }
    state.clusterLayer?.clearLayers();
    state.anomalyMarkers = {};
  }

  window.renderConsistencyMap = function renderConsistencyMap(payload) {
    clearConsistencyLayers();
    if (!payload?.available || !state.map) return;
    const layer = state.collectionLayers["consistency-anomaly"];
    if (layer && payload.features?.length) {
      layer.addData({ type: "FeatureCollection", features: payload.features });
      layer.getLayers().forEach((marker) => {
        const id = marker.feature?.properties?.anomaly_id;
        if (id) state.anomalyMarkers[id] = marker;
      });
      const checkbox = document.querySelector('input[data-collection="consistency-anomaly"]');
      if (!checkbox || checkbox.checked) layer.addTo(state.map);
    }
    const heatPoints = (payload.features || [])
      .map((f) => {
        const c = f.geometry?.coordinates;
        if (!c || c.length < 2) return null;
        return [c[1], c[0], f.properties?.intensity ?? 0.5];
      })
      .filter(Boolean);
    if (heatPoints.length && typeof L.heatLayer === "function") {
      state.heatmapLayer = L.heatLayer(heatPoints, {
        radius: 28,
        blur: 22,
        maxZoom: 12,
        gradient: { 0.2: "#c4a832", 0.5: "#c46c32", 0.85: "#8a2d21", 1: "#5c1010" },
      });
      const heatCb = document.querySelector('input[data-collection="consistency-heatmap"]');
      if (!heatCb || heatCb.checked) state.heatmapLayer.addTo(state.map);
    }
    (payload.clusters || []).forEach((cluster) => {
      const { lat, lon } = cluster.centroid || {};
      if (lat == null || lon == null) return;
      const circle = L.circle([lat, lon], {
        radius: (cluster.radius_km || 5) * 1000,
        color: "#8a2d21",
        weight: 2,
        dashArray: "6,8",
        fillColor: "#c46c32",
        fillOpacity: 0.08,
      }).bindPopup(
        `<b>${cluster.cluster_id}</b><br>${cluster.anomaly_count} anomalies<br>${cluster.pattern_assessment}`
      );
      state.clusterLayer.addLayer(circle);
    });
    const clusterCb = document.querySelector('input[data-collection="consistency-clusters"]');
    if (state.clusterLayer.getLayers().length && (!clusterCb || clusterCb.checked)) {
      state.clusterLayer.addTo(state.map);
    }
  };

  window.applyConsistencyReport = function applyConsistencyReport(report) {
    state.consistencyReport = report;
    renderTrustPanel(report);
    renderAnomalyFeed(report);
  };

  window.runConsistency = async function runConsistency() {
    const consistencySummary = document.getElementById("consistency-summary");
    try {
      const params = new URLSearchParams({
        area: state.bootstrapArea,
        timeframe: state.timeframe,
      });
      const [reportResp, mapResp] = await Promise.all([
        fetch(`/api/consistency/run?${params}`, { method: "POST" }),
        fetch(`/api/map-data/consistency?${params}`),
      ]);
      if (!reportResp.ok) throw new Error(`consistency ${reportResp.status}`);
      const report = await reportResp.json();
      applyConsistencyReport(report);
      if (mapResp.ok) renderConsistencyMap(await mapResp.json());
      return report;
    } catch (error) {
      console.error("Consistency check failed:", error);
      if (consistencySummary) consistencySummary.textContent = "Data consistency check failed.";
      return null;
    }
  };

  window.loadMaritimeMapData = async function loadMaritimeMapData() {
    if (!state.bootstrapArea.toLowerCase().includes("archipelago")) {
      if (typeof setLayerAvailability === "function") {
        setLayerAvailability(["ais", "sar"], false, "Maritime demo: Archipelago Sea only.");
      }
      return;
    }
    try {
      const response = await fetch(
        `/api/map-data/maritime?area=${encodeURIComponent(state.bootstrapArea)}`
      );
      const geojson = await response.json();
      state.collectionLayers.ais?.clearLayers();
      state.collectionLayers.sar?.clearLayers();
      if (!geojson.available) {
        if (typeof setLayerAvailability === "function") {
          setLayerAvailability(["ais", "sar"], false, "Maritime data unavailable.");
        }
        return;
      }
      if (typeof setLayerAvailability === "function") setLayerAvailability(["ais", "sar"], true);
      const grouped = {};
      geojson.features.forEach((feature) => {
        const coll = feature.properties?._collection;
        if (!grouped[coll]) grouped[coll] = [];
        grouped[coll].push(feature);
      });
      Object.entries(grouped).forEach(([coll, features]) => {
        const layer = state.collectionLayers[coll];
        if (!layer) return;
        layer.addData({ type: "FeatureCollection", features });
        const checkbox = document.querySelector(`input[data-collection="${coll}"]`);
        if (!checkbox || checkbox.checked) layer.addTo(state.map);
      });
    } catch (error) {
      console.error("Maritime map load failed:", error);
    }
  };

  window.getWorkspaceSourceIds = function getWorkspaceSourceIds() {
    const base = window.WORKSPACE_SOURCE_IDS || [
      "fmi", "nls", "statistics-finland", "digiroad", "opencellid", "osm-poi", "satellites",
    ];
    const ids = [...base];
    if (state.bootstrapArea.toLowerCase().includes("archipelago")) ids.push("maritime-demo");
    return ids;
  };

  window.patchFreshnessWithTrust = function patchFreshnessWithTrust(freshness) {
    const freshnessList = document.getElementById("freshness-list");
    if (!freshnessList) return;
    const trustById = {};
    (state.consistencyReport?.layer_trust || []).forEach((t) => {
      trustById[t.source_id] = t;
    });
    const rows = freshness?.length ? freshness : state.sources || [];
    freshnessList.innerHTML = "";
    if (!rows.length) {
      freshnessList.innerHTML =
        '<div class="empty-state">Source freshness will appear after datasets load.</div>';
      return;
    }
    rows.forEach((source) => {
      const row = document.createElement("div");
      row.className = "freshness-row";
      const refreshed = source.retrieved_at || source.last_successful_refresh || "Not yet refreshed";
      const trust = trustById[source.source_id];
      const trustNote = trust
        ? `<div style="font-size:0.78rem;color:var(--muted)">Trust ${Math.round(trust.confidence * 100)}%</div>`
        : "";
      row.innerHTML = `
        <div><div class="source-name">${source.name || source.source_id}</div><div>${refreshed}</div>${trustNote}</div>
        <div class="source-status">${source.status}</div>
      `;
      freshnessList.appendChild(row);
    });
  };
})();
