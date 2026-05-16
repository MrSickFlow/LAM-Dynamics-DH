from __future__ import annotations

from typing import Any

from ipb_backend.analysis.contracts import DataPackage
from ipb_backend.llm.contracts import (
    AnalysisProfile,
    ConversationTurn,
    LlmEvidenceDigest,
    LlmIndicatorDigest,
    LlmSelectionDigest,
    LlmSourceDigest,
    LlmWrapperGuardrails,
    LlmWrapperInput,
)
from ipb_backend.llm.profiles import PROFILE_SPECS


def _summarize_source_details(raw_summary: dict[str, Any]) -> dict[str, Any]:
    details: dict[str, Any] = {}

    for key in ("feature_count", "station_count", "population_total", "population_coverage_ratio"):
        value = raw_summary.get(key)
        if value not in (None, "", [], {}):
            details[key] = value

    collections = raw_summary.get("collections") or []
    if collections:
        details["top_collections"] = collections[:4]

    observations = raw_summary.get("observations") or {}
    if observations:
        details["observation_keys"] = sorted(observations.keys())[:8]

    return details


def _conversation_tail(history: list[ConversationTurn], limit: int = 6) -> list[ConversationTurn]:
    return history[-limit:]


def build_llm_wrapper_input(
    *,
    data_package: DataPackage,
    profile: AnalysisProfile = AnalysisProfile.GENERAL,
    question: str | None = None,
    conversation_history: list[ConversationTurn] | None = None,
) -> LlmWrapperInput:
    source_digests = [
        LlmSourceDigest(
            source_id=source.source_id,
            category=source.category,
            title=source.title,
            summary=source.summary,
            confidence=source.confidence,
            fallback_used=source.provenance.fallback_used,
            fallback_reason=source.provenance.fallback_reason,
            details=_summarize_source_details(source.raw_summary),
        )
        for source in data_package.source_summaries
    ]

    indicator_digests = [
        LlmIndicatorDigest(
            indicator_id=indicator.indicator_id,
            name=indicator.name,
            value=indicator.value,
            unit=indicator.unit,
            method=indicator.method,
            source_ids=indicator.source_ids,
            confidence=indicator.confidence,
            notes=indicator.notes,
        )
        for indicator in data_package.derived_indicators
    ]

    evidence_catalog = [
        LlmEvidenceDigest(
            evidence_id=evidence.evidence_id,
            source_id=evidence.source_id,
            kind=evidence.kind,
            title=evidence.title,
            detail=evidence.detail,
            support=evidence.support,
        )
        for evidence in data_package.evidence_items
    ]

    requested_sources = data_package.scope.requested_sources
    resolved_sources = data_package.scope.resolved_sources
    coverage_gaps = list(data_package.quality.coverage_gaps)
    if len(resolved_sources) < len(requested_sources):
        missing = sorted(set(requested_sources) - set(resolved_sources))
        if missing:
            coverage_gaps.append(f"No resolved AOI data for requested sources: {', '.join(missing)}")

    return LlmWrapperInput(
        package_id=data_package.package_id,
        profile=profile,
        profile_focus=PROFILE_SPECS.get(profile, PROFILE_SPECS[AnalysisProfile.GENERAL])["focus"],
        question=question,
        selection=LlmSelectionDigest(
            selection_type=data_package.selection.selection_type,
            bounds_wgs84=data_package.selection.bounds_wgs84,
            area_sqkm=data_package.selection.area_sqkm,
            timeframe=data_package.scope.timeframe,
        ),
        counts={
            "by_source": data_package.counts.by_source,
            "by_category": data_package.counts.by_category,
            "geometry_types": data_package.counts.geometry_types,
        },
        source_digests=source_digests,
        indicator_digests=indicator_digests,
        evidence_catalog=evidence_catalog,
        guardrails=LlmWrapperGuardrails(
            overall_confidence=data_package.quality.overall_confidence,
            fallback_sources=data_package.quality.fallback_sources,
            error_sources=data_package.quality.error_sources,
            coverage_gaps=coverage_gaps,
        ),
        conversation_history=_conversation_tail(conversation_history or []),
    )