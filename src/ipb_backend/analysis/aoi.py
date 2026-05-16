from __future__ import annotations

from collections import Counter
from typing import Any


def _count_geometries(features: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for feature in features:
        geometry_type = feature.get("geometry", {}).get("type")
        if geometry_type:
            counts[geometry_type] += 1
    return dict(sorted(counts.items()))


def _metric_value(payload: dict[str, Any], key: str) -> int:
    return int(payload.get(key, 0) or 0)


def _source_title(source_id: str, payload: dict[str, Any]) -> str:
    return str(
        payload.get("title")
        or {
            "nls": "NLS Terrain",
            "digiroad": "Digiroad",
            "statistics-finland": "Statistics Finland",
            "fmi": "FMI Weather",
        }.get(source_id, source_id)
    )


def build_aoi_metrics(selection_area_sqkm: float, raw_data: dict[str, Any]) -> dict[str, Any]:
    source_feature_counts = {
        source_id: payload.get("feature_count", payload.get("station_count", 0))
        for source_id, payload in raw_data.items()
    }
    feature_counts_by_category = Counter()
    all_feature_samples = [
        feature
        for payload in raw_data.values()
        for feature in payload.get("features", [])
    ]
    collection_counts = {
        item["collection"]: item["count"]
        for item in raw_data.get("nls", {}).get("collections", [])
    }
    for payload in raw_data.values():
        category = str(payload.get("category", "other") or "other")
        feature_counts_by_category[category] += _metric_value(payload, "feature_count")

    weather = raw_data.get("fmi", {})
    parameter_count = sum(len(payload.get("observations", {})) for payload in raw_data.values())
    population_total = sum(_metric_value(payload, "population_total") for payload in raw_data.values())
    weather_station_count = sum(_metric_value(payload, "station_count") for payload in raw_data.values())

    return {
        "selection_area_sqkm": selection_area_sqkm,
        "nls_feature_count": raw_data.get("nls", {}).get("feature_count", 0),
        "digiroad_feature_count": raw_data.get("digiroad", {}).get("feature_count", 0),
        "population_total": population_total,
        "weather_station_count": weather_station_count,
        "feature_counts_by_source": source_feature_counts,
        "feature_counts_by_category": dict(sorted(feature_counts_by_category.items())),
        "geometry_counts": _count_geometries(all_feature_samples),
        "nls_collection_counts": collection_counts,
        "weather_parameter_count": parameter_count,
        "active_sources": sorted(raw_data.keys()),
    }


def build_evidence_bundle(metrics: dict[str, Any], raw_data: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []

    for source_id, payload in raw_data.items():
        if payload.get("collections") and payload.get("feature_count"):
            top = ", ".join(
                f"{item['label']} ({item['count']})"
                for item in payload.get("collections", [])[:3]
            )
            evidence.append(
                {
                    "source_id": source_id,
                    "detail": f"{payload['feature_count']} intersecting features",
                    "support": top,
                }
            )
            continue

        if payload.get("population_total"):
            evidence.append(
                {
                    "source_id": source_id,
                    "detail": f"Population total {payload['population_total']}",
                    "support": f"{payload.get('feature_count', 0)} intersecting population cells",
                }
            )
            continue

        if payload.get("station_count"):
            evidence.append(
                {
                    "source_id": source_id,
                    "detail": f"{payload['station_count']} station intersecting the AOI",
                    "support": f"{len(payload.get('observations', {}))} observed weather parameters available",
                }
            )
            continue

        if payload.get("feature_count"):
            title = _source_title(source_id, payload)
            evidence.append(
                {
                    "source_id": source_id,
                    "detail": f"{payload['feature_count']} intersecting features from {title}",
                    "support": payload.get("summary", ""),
                }
            )

    return evidence


def build_raw_sections(raw_data: dict[str, Any]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []

    for source_id, payload in raw_data.items():
        source_title = _source_title(source_id, payload)

        subsections: list[dict[str, Any]] = []

        counts = {
            "feature_count": payload.get("feature_count", 0),
            "station_count": payload.get("station_count", 0),
            "population_total": payload.get("population_total", 0),
        }
        counts = {key: value for key, value in counts.items() if value}
        if counts:
            subsections.append(
                {
                    "id": "counts",
                    "title": "Counts",
                    "kind": "json",
                    "data": counts,
                }
            )

        if payload.get("collections"):
            subsections.append(
                {
                    "id": "collections",
                    "title": "Collections",
                    "kind": "json",
                    "data": payload.get("collections", []),
                }
            )

        if payload.get("features"):
            subsections.append(
                {
                    "id": "samples",
                    "title": "Feature Samples",
                    "kind": "json",
                    "data": [feature.get("properties", {}) for feature in payload.get("features", [])[:5]],
                }
            )

        if payload.get("stations"):
            subsections.append(
                {
                    "id": "stations",
                    "title": "Stations",
                    "kind": "json",
                    "data": [station.get("properties", {}) for station in payload.get("stations", [])],
                }
            )

        if payload.get("observations"):
            subsections.append(
                {
                    "id": "observations",
                    "title": "Observations",
                    "kind": "json",
                    "data": payload.get("observations", {}),
                }
            )

        subsections.append(
            {
                "id": "full-payload",
                "title": "Full Payload",
                "kind": "json",
                "data": payload,
            }
        )

        sections.append(
            {
                "source_id": source_id,
                "title": source_title,
                "summary": payload.get("summary", "No source summary available."),
                "subsections": subsections,
            }
        )

    return sections