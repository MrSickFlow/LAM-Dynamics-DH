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


def _format_brief_scalar(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return str(value)


def _source_brief_line(source: LlmSourceDigest) -> str:
    detail_parts: list[str] = []
    details = source.details or {}

    for key in ("feature_count", "station_count", "population_total"):
        value = details.get(key)
        if value not in (None, "", [], {}):
            label = key.replace("_", " ")
            detail_parts.append(f"{label}={_format_brief_scalar(value)}")

    collections = details.get("top_collections") or []
    if collections:
        top_labels = []
        for item in collections[:2]:
            label = item.get("label") or item.get("collection") or "unknown"
            count = item.get("count")
            if count is None:
                top_labels.append(str(label))
            else:
                top_labels.append(f"{label} ({count})")
        if top_labels:
            detail_parts.append("collections: " + ", ".join(top_labels))

    observations = details.get("current_observations") or {}
    if observations:
        obs_parts = []
        for item in list(observations.values())[:3]:
            value = item.get("value")
            if value is None:
                continue
            label = item.get("label") or "value"
            unit = item.get("unit") or ""
            obs_parts.append(f"{label} {_format_brief_scalar(value)}{unit}")
        if obs_parts:
            detail_parts.append("obs: " + ", ".join(obs_parts))

    sat_counts = details.get("sat_pass_counts") or {}
    if sat_counts.get("total_passes"):
        detail_parts.append(
            "passes: "
            f"{sat_counts.get('total_passes')} total / "
            f"{sat_counts.get('satellites_observing_aoi', 0)} observing"
        )

    suffix = f"; {'; '.join(detail_parts[:3])}" if detail_parts else ""
    fallback_note = " [fallback]" if source.fallback_used else ""
    return (
        f"- {source.title} ({source.source_id}, {source.category}, conf={source.confidence}{fallback_note}): "
        f"{source.summary}{suffix}"
    )


def _indicator_brief_line(indicator: LlmIndicatorDigest) -> str:
    value = _format_brief_scalar(indicator.value)
    unit = f" {indicator.unit}" if indicator.unit else ""
    return f"- {indicator.name}: {value}{unit} ({indicator.confidence})"


def _evidence_brief_line(evidence: LlmEvidenceDigest) -> str:
    return f"- {evidence.title}: {evidence.detail}"


def _build_schematic_brief(
    *,
    selection: LlmSelectionDigest,
    profile: AnalysisProfile,
    profile_focus: str,
    source_digests: list[LlmSourceDigest],
    indicator_digests: list[LlmIndicatorDigest],
    evidence_catalog: list[LlmEvidenceDigest],
    guardrails: LlmWrapperGuardrails,
) -> str:
    lines = [
        (
            f"AOI {selection.area_sqkm:.1f} km^2 | timeframe {selection.timeframe} | "
            f"profile {profile.value} ({profile_focus})"
        ),
        f"Selection type: {selection.selection_type}",
    ]

    if selection.bounds_wgs84:
        bounds = ", ".join(f"{value:.4f}" for value in selection.bounds_wgs84)
        lines.append(f"Bounds W,S,E,N: {bounds}")

    if source_digests:
        lines.append("Sources:")
        lines.extend(_source_brief_line(source) for source in source_digests)
    else:
        lines.append("Sources: none resolved")

    if indicator_digests:
        lines.append("Indicators:")
        lines.extend(_indicator_brief_line(indicator) for indicator in indicator_digests[:8])

    if evidence_catalog:
        lines.append("Evidence anchors:")
        lines.extend(_evidence_brief_line(evidence) for evidence in evidence_catalog[:5])

    lines.append(
        "Guardrails: "
        f"confidence={guardrails.overall_confidence}; "
        f"fallback={', '.join(guardrails.fallback_sources) or 'none'}; "
        f"errors={', '.join(guardrails.error_sources) or 'none'}"
    )
    if guardrails.coverage_gaps:
        lines.append("Coverage gaps:")
        lines.extend(f"- {gap}" for gap in guardrails.coverage_gaps[:5])

    return "\n".join(lines)


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
        # Latest values per parameter so the LLM sees actual numbers.
        details["current_observations"] = {
            key: {
                "label": param.get("label", key),
                "value": (param.get("latest") or {}).get("value"),
                "unit": param.get("unit", ""),
            }
            for key, param in observations.items()
            if (param.get("latest") or {}).get("value") is not None
        }

    # Forecast time-series — keep all hourly points so the LLM can reason about
    # diurnal trends, frontal passages, and operating windows. Forecast horizon
    # is already capped at FMI HARMONIE's 48h ceiling.
    forecast = raw_summary.get("forecast") or {}
    fc_obs = forecast.get("observations") if isinstance(forecast, dict) else None
    if fc_obs:
        series: dict[str, dict[str, Any]] = {}
        for key, param in fc_obs.items():
            values = param.get("values") or []
            cleaned = [
                {"t": pt.get("time"), "v": pt.get("value")}
                for pt in values
                if pt.get("value") is not None
            ]
            if not cleaned:
                continue
            series[key] = {
                "label": param.get("label", key),
                "unit": param.get("unit", ""),
                "points": cleaned,
            }
        if series:
            details["forecast_series"] = series

    # Satellite overpass schedule — emit a compact, chronologically sorted list
    # of pass windows so the LLM can answer "when is the next gap?" / "is the
    # AOI imaged tonight?".
    sats_with_passes = raw_summary.get("satellites_with_passes") or []
    if sats_with_passes:
        all_passes: list[dict[str, Any]] = []
        for sat in sats_with_passes:
            for p in sat.get("passes", []):
                all_passes.append({
                    "name": sat.get("name"),
                    "type": sat.get("type"),
                    "is_sar": sat.get("is_sar", False),
                    "start_utc": p.get("start_utc"),
                    "end_utc": p.get("end_utc"),
                    "closest_km": p.get("closest_km"),
                    "duration_seconds": p.get("duration_seconds"),
                })
        all_passes.sort(key=lambda p: p.get("start_utc") or "")
        details["pass_schedule"] = all_passes
        details["sat_pass_counts"] = {
            "total_passes": raw_summary.get("total_passes_in_window", 0),
            "satellites_observing_aoi": len(sats_with_passes),
            "satellites_tracked": raw_summary.get("satellites_total_tracked", 0),
        }
        window = raw_summary.get("window") or {}
        if window:
            details["window"] = window

    # Include query window so the LLM knows the temporal scope.
    query = raw_summary.get("query") or {}
    if query.get("start_time") or query.get("window_start"):
        details["time_window"] = {
            "start": query.get("start_time") or query.get("window_start"),
            "end": query.get("end_time") or query.get("window_end"),
            "forecast_hours": query.get("forecast_hours"),
        }

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

    guardrails = LlmWrapperGuardrails(
        overall_confidence=data_package.quality.overall_confidence,
        fallback_sources=data_package.quality.fallback_sources,
        error_sources=data_package.quality.error_sources,
        coverage_gaps=coverage_gaps,
    )

    selection_digest = LlmSelectionDigest(
        selection_type=data_package.selection.selection_type,
        bounds_wgs84=data_package.selection.bounds_wgs84,
        area_sqkm=data_package.selection.area_sqkm,
        timeframe=data_package.scope.timeframe,
    )

    schematic_brief = _build_schematic_brief(
        selection=selection_digest,
        profile=profile,
        profile_focus=PROFILE_SPECS.get(profile, PROFILE_SPECS[AnalysisProfile.GENERAL])["focus"],
        source_digests=source_digests,
        indicator_digests=indicator_digests,
        evidence_catalog=evidence_catalog,
        guardrails=guardrails,
    )

    return LlmWrapperInput(
        package_id=data_package.package_id,
        profile=profile,
        profile_focus=PROFILE_SPECS.get(profile, PROFILE_SPECS[AnalysisProfile.GENERAL])["focus"],
        question=question,
        schematic_brief=schematic_brief,
        selection=selection_digest,
        counts={
            "by_source": data_package.counts.by_source,
            "by_category": data_package.counts.by_category,
            "geometry_types": data_package.counts.geometry_types,
        },
        source_digests=source_digests,
        indicator_digests=indicator_digests,
        evidence_catalog=evidence_catalog,
        guardrails=guardrails,
        conversation_history=_conversation_tail(conversation_history or []),
    )