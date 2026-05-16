from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from ipb_backend.consistency.classification import get_source_profile, normalize_area, resolve_area_center
from ipb_backend.consistency.clustering import cluster_anomalies
from ipb_backend.consistency.rules import DEFAULT_RULES, ConsistencyRule, _is_demo_record
from ipb_backend.ingestion.sources.fmi import FmiAdapter
from ipb_backend.models import (
    AnomalySeverity,
    ConsistencyAnomaly,
    ConsistencyReport,
    DatasetRecord,
    EwClassification,
    LayerTrustScore,
    SourceDefinition,
    SourceStatus,
)


class DataConsistencyEngine:
    """Cross-validates EW-vulnerable feeds against EW-immune baselines."""

    def __init__(
        self,
        *,
        rules: Optional[list[ConsistencyRule]] = None,
        fmi_adapter: Optional[FmiAdapter] = None,
    ) -> None:
        self._rules = rules or DEFAULT_RULES
        self._fmi_adapter = fmi_adapter

    async def evaluate(
        self,
        *,
        area: str,
        timeframe: str,
        records: dict[str, DatasetRecord],
        sources: list[SourceDefinition],
    ) -> ConsistencyReport:
        context = await self._build_context(area=area, timeframe=timeframe, records=records)
        anomalies: list[ConsistencyAnomaly] = []
        for rule in self._rules:
            anomalies.extend(
                rule.evaluate(
                    area=area,
                    timeframe=timeframe,
                    records=records,
                    sources=sources,
                    context=context,
                )
            )

        layer_trust = self._score_layers(
            records=records,
            sources=sources,
            anomalies=anomalies,
        )
        clusters = cluster_anomalies(anomalies)
        ew_pattern = any(
            cluster.anomaly_count >= 2
            and any(
                get_source_profile(source_id).ew_classification == EwClassification.VULNERABLE
                for source_id in cluster.affected_sources
            )
            for cluster in clusters
        ) or sum(
            1 for a in anomalies if a.severity in (AnomalySeverity.HIGH, AnomalySeverity.CRITICAL)
        ) >= 2

        summary = self._build_summary(area, anomalies, clusters, ew_pattern)
        return ConsistencyReport(
            area=area,
            timeframe=timeframe,
            layer_trust=layer_trust,
            anomalies=anomalies,
            clusters=clusters,
            summary=summary,
            ew_pattern_detected=ew_pattern,
        )

    async def _build_context(
        self,
        *,
        area: str,
        timeframe: str,
        records: dict[str, DatasetRecord],
    ) -> dict[str, Any]:
        context: dict[str, Any] = {}
        fmi_record = records.get("fmi")
        if not self._fmi_adapter or not fmi_record:
            return context

        station = fmi_record.data.get("station", {})
        lat = station.get("latitude")
        lon = station.get("longitude")
        if lat is not None and lon is not None:
            try:
                context["fmi_forecast"] = await self._fmi_adapter.fetch_forecast_by_latlon(
                    float(lat), float(lon)
                )
            except Exception:
                context["fmi_forecast"] = None

        center = resolve_area_center(area)
        try:
            start_time, end_time = self._fmi_adapter._resolve_time_window(timeframe)
            multi = await self._fmi_adapter.fetch_observations_by_latlon(
                center["lat"], center["lon"], start_time, end_time
            )
            context["fmi_multi_station"] = {"stations": self._extract_multi_stations(multi)}
        except Exception:
            context["fmi_multi_station"] = None

        return context

    def _extract_multi_stations(self, parsed: dict[str, Any]) -> list[dict[str, Any]]:
        """FMI WFS returns one station per member group in practice; preserve structure for outlier rule."""
        station = parsed.get("station", {})
        if station:
            return [{"name": station.get("name"), **station, "observations": parsed.get("observations", {})}]
        return []

    def _score_layers(
        self,
        *,
        records: dict[str, DatasetRecord],
        sources: list[SourceDefinition],
        anomalies: list[ConsistencyAnomaly],
    ) -> list[LayerTrustScore]:
        now = datetime.now(timezone.utc)
        scores: list[LayerTrustScore] = []
        anomaly_by_source: dict[str, int] = {}
        for anomaly in anomalies:
            for source_id in anomaly.vulnerable_sources + anomaly.immune_sources:
                anomaly_by_source[source_id] = anomaly_by_source.get(source_id, 0) + 1

        for source in sources:
            profile = get_source_profile(source.source_id)
            record = records.get(source.source_id)
            factors: list[str] = []
            confidence = 1.0

            if not source.enabled:
                confidence = 0.0
                factors.append("source disabled")
            elif source.status == SourceStatus.ERROR:
                confidence = 0.15
                factors.append(f"ingestion error: {source.last_error or 'unknown'}")
            elif source.status == SourceStatus.DISABLED:
                confidence = 0.0
                factors.append("credentials not configured")
            elif record is None:
                confidence = 0.25 if profile.ew_classification == EwClassification.VULNERABLE else 0.5
                factors.append("no ingested record for area")
            else:
                if _is_demo_record(record):
                    confidence -= 0.35
                    factors.append("demo or fallback payload")
                staleness = int((now - record.retrieved_at).total_seconds())
                if staleness > source.refresh_interval_seconds * 2:
                    confidence -= 0.15
                    factors.append(f"stale ({staleness}s since retrieval)")
                if profile.gnss_dependent:
                    confidence -= 0.05
                    factors.append("GNSS-dependent collection")

            hits = anomaly_by_source.get(source.source_id, 0)
            if hits:
                confidence -= min(0.4, 0.1 * hits)
                factors.append(f"{hits} consistency anomaly(s)")

            confidence = max(0.0, min(1.0, confidence))
            staleness_seconds = None
            if record is not None:
                staleness_seconds = int((now - record.retrieved_at).total_seconds())

            scores.append(
                LayerTrustScore(
                    source_id=source.source_id,
                    ew_classification=EwClassification(profile.ew_classification.value),
                    gnss_dependent=profile.gnss_dependent,
                    confidence=round(confidence, 2),
                    status=source.status,
                    staleness_seconds=staleness_seconds,
                    factors=factors or ["no issues detected"],
                )
            )
        return scores

    def _build_summary(
        self,
        area: str,
        anomalies: list[ConsistencyAnomaly],
        clusters: list,
        ew_pattern: bool,
    ) -> str:
        if not anomalies:
            return f"Data consistency check for {area}: no cross-source anomalies detected."
        cluster_note = (
            f" {len(clusters)} spatial cluster(s) suggest localized degradation."
            if clusters
            else ""
        )
        pattern_note = " Pattern consistent with contested EM environment — analyst review recommended." if ew_pattern else ""
        return (
            f"Data consistency check for {area}: {len(anomalies)} anomaly(s) flagged "
            f"across {len({s for a in anomalies for s in a.vulnerable_sources})} vulnerable source(s)."
            f"{cluster_note}{pattern_note}"
        )
