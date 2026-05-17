from __future__ import annotations

from typing import Any

from ipb_backend.analysis.aoi import build_aoi_metrics, build_evidence_bundle
from ipb_backend.analysis.contracts import (
    CountsSummary,
    DataPackage,
    DataScope,
    DerivedIndicator,
    EvidenceDataRef,
    EvidenceItem,
    QualitySummary,
    SelectionContext,
    SourceFreshnessRecord,
    SourceProvenance,
    SourceSummary,
)


def _detect_confidence(payload: dict[str, Any], freshness_item: dict[str, Any] | None) -> str:
    if payload.get("provenance", {}).get("fallback_used"):
        return "low"
    if freshness_item and freshness_item.get("status") == "error":
        return "low"
    if payload.get("feature_count") or payload.get("station_count") or payload.get("population_total"):
        return "high"
    return "medium"


def _freshness_model(item: dict[str, Any]) -> SourceFreshnessRecord:
    return SourceFreshnessRecord(
        source_id=item["source_id"],
        name=item["name"],
        category=item.get("category"),
        status=str(item.get("status", "unknown")),
        last_successful_refresh=item.get("last_successful_refresh"),
        last_error=item.get("last_error"),
        retrieved_at=item.get("retrieved_at"),
        refresh_interval_seconds=item.get("refresh_interval_seconds"),
        freshness_label=item.get("freshness_label"),
    )


def _source_summary(source_id: str, payload: dict[str, Any], freshness_item: dict[str, Any] | None) -> SourceSummary:
    provenance = payload.get("provenance", {})
    raw_summary: dict[str, Any] = {}
    for key in (
        "feature_count",
        "station_count",
        "population_total",
        "population_source_total",
        "population_coverage_ratio",
        "collections",
        "observations",
        "forecast",
        "query",
        "satellites",
        "total_tracked",
    ):
        value = payload.get(key)
        if value:
            raw_summary[key] = value

    return SourceSummary(
        source_id=source_id,
        category=str(payload.get("category", "other")),
        title=str(payload.get("title") or source_id),
        summary=str(payload.get("summary") or "No source summary available."),
        raw_summary=raw_summary,
        confidence=_detect_confidence(payload, freshness_item),
        provenance=SourceProvenance(
            provider=str(provenance.get("provider") or payload.get("title") or source_id),
            adapter=provenance.get("adapter"),
            retrieved_at=provenance.get("retrieved_at"),
            fallback_used=bool(provenance.get("fallback_used")),
            fallback_reason=provenance.get("fallback_reason"),
            deterministic=bool(provenance.get("deterministic", True)),
            note=provenance.get("note"),
        ),
    )


def _build_indicators(selection_area_sqkm: float, metrics: dict[str, Any], raw_data: dict[str, Any]) -> list[DerivedIndicator]:
    indicators = [
        DerivedIndicator(
            indicator_id="selection_area_sqkm",
            name="Selection Area",
            value=round(selection_area_sqkm, 3),
            unit="sqkm",
            method="polygon_area",
            source_ids=[],
            confidence="high",
        ),
        DerivedIndicator(
            indicator_id="population_total",
            name="Population Total",
            value=int(metrics.get("population_total", 0) or 0),
            unit="persons",
            method="area_weighted_overlap",
            source_ids=["statistics-finland"] if "statistics-finland" in raw_data else [],
            confidence="medium",
        ),
        DerivedIndicator(
            indicator_id="weather_station_count",
            name="Weather Station Count",
            value=int(metrics.get("weather_station_count", 0) or 0),
            unit="count",
            method="aoi_intersection",
            source_ids=[source_id for source_id, payload in raw_data.items() if payload.get("station_count")],
            confidence="high",
        ),
    ]

    if selection_area_sqkm > 0:
        for source_id, count in sorted(metrics.get("feature_counts_by_source", {}).items()):
            indicators.append(
                DerivedIndicator(
                    indicator_id=f"{source_id}_feature_density",
                    name=f"{source_id} Feature Density",
                    value=round((count or 0) / selection_area_sqkm, 3),
                    unit="features_per_sqkm",
                    method="feature_count_divided_by_area",
                    source_ids=[source_id],
                    confidence="medium",
                )
            )

    return indicators


def _build_evidence_items(evidence_bundle: list[dict[str, Any]]) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    source_indexes: dict[str, int] = {}
    for item in evidence_bundle:
        source_id = str(item.get("source_id") or "unknown")
        source_indexes[source_id] = source_indexes.get(source_id, 0) + 1
        index = source_indexes[source_id]
        items.append(
            EvidenceItem(
                evidence_id=f"ev-{source_id}-{index:03d}",
                source_id=source_id,
                kind="count-summary",
                title=f"{source_id} evidence {index}",
                detail=str(item.get("detail") or ""),
                support=str(item.get("support") or ""),
                data_ref=EvidenceDataRef(
                    section="source_summaries",
                    path=f"source_summaries[{len(items)}].raw_summary",
                ),
            )
        )
    return items


def _build_quality(source_summaries: list[SourceSummary], source_freshness: list[SourceFreshnessRecord]) -> QualitySummary:
    fallback_sources = [item.source_id for item in source_summaries if item.provenance.fallback_used]
    error_sources = [item.source_id for item in source_freshness if item.status == "error"]

    coverage_gaps: list[str] = []
    if not source_summaries:
        coverage_gaps.append("No source data intersected the selection.")

    if fallback_sources or error_sources:
        overall_confidence = "low"
    elif source_summaries:
        overall_confidence = "high"
    else:
        overall_confidence = "medium"

    return QualitySummary(
        fallback_sources=fallback_sources,
        error_sources=error_sources,
        coverage_gaps=coverage_gaps,
        overall_confidence=overall_confidence,
    )


def build_data_package(
    *,
    selection: dict[str, Any],
    timeframe: str,
    raw_data: dict[str, Any],
    freshness: list[dict[str, Any]],
    requested_sources: list[str] | None = None,
) -> DataPackage:
    selection_area_sqkm = float(selection.get("area_sqkm", 0.0) or 0.0)
    metrics = build_aoi_metrics(selection_area_sqkm, raw_data)
    evidence_bundle = build_evidence_bundle(metrics, raw_data)
    freshness_by_source = {item["source_id"]: item for item in freshness}

    source_freshness = [_freshness_model(item) for item in freshness]
    source_summaries = [
        _source_summary(source_id, payload, freshness_by_source.get(source_id))
        for source_id, payload in sorted(raw_data.items())
    ]

    return DataPackage(
        selection=SelectionContext(
            geometry=selection["geometry"],
            bounds_wgs84=list(selection.get("bounds", [])),
            area_sqkm=selection_area_sqkm,
            label=selection.get("label"),
            area_id=selection.get("area_id"),
        ),
        scope=DataScope(
            timeframe=timeframe,
            requested_sources=sorted(requested_sources or list(raw_data.keys())),
            resolved_sources=sorted(raw_data.keys()),
        ),
        source_freshness=source_freshness,
        source_summaries=source_summaries,
        counts=CountsSummary(
            by_source=dict(metrics.get("feature_counts_by_source", {})),
            by_category=dict(metrics.get("feature_counts_by_category", {})),
            geometry_types=dict(metrics.get("geometry_counts", {})),
        ),
        derived_indicators=_build_indicators(selection_area_sqkm, metrics, raw_data),
        evidence_items=_build_evidence_items(evidence_bundle),
        quality=_build_quality(source_summaries, source_freshness),
    )