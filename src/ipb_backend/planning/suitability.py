from __future__ import annotations

import logging
import math
import time
from typing import Any, Optional

from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry

from ipb_backend.models import DatasetRecord
from ipb_backend.planning.constraints import (
    CellFeatures,
    SourceIndex,
    check_constraints,
    extract_cell_features_indexed,
    is_feasible,
)
from ipb_backend.planning.force_model import (
    ForceComposition,
    Operation,
    PlanningRequest,
    PlanningResponse,
    RecommendedSite,
)
from ipb_backend.planning.operations import SCORING_CRITERIA, get_operation_profile
from ipb_backend.spatial import geojson_to_shape, resolve_area_bbox

logger = logging.getLogger(__name__)


_DEG_LAT_KM = 110.57

# Hard ceiling on cells per planning run. With the indexed extractor each cell
# costs ~milliseconds, so 20k cells caps a run at roughly a minute on commodity
# hardware. Anything above this is almost certainly a misconfigured grid res.
PLANNING_MAX_CELLS = 20_000


def _deg_lon_km(lat: float) -> float:
    return 111.32 * math.cos(math.radians(lat))


def _build_cell_grid(
    mask: BaseGeometry, resolution_m: int
) -> list[tuple[BaseGeometry, tuple[float, float]]]:
    minx, miny, maxx, maxy = mask.bounds
    mean_lat = (miny + maxy) / 2.0

    cell_deg_lat = resolution_m / 1000.0 / _DEG_LAT_KM
    deg_lon_per_km = _deg_lon_km(mean_lat)
    cell_deg_lon = resolution_m / 1000.0 / max(deg_lon_per_km, 0.0001)

    cells: list[tuple[BaseGeometry, tuple[float, float]]] = []
    y = miny
    while y < maxy:
        x = minx
        while x < maxx:
            cell = box(x, y, min(x + cell_deg_lon, maxx), min(y + cell_deg_lat, maxy))
            if cell.intersects(mask):
                clipped = cell.intersection(mask)
                if not clipped.is_empty and clipped.area > 0:
                    centroid = clipped.centroid
                    cells.append((clipped, (centroid.x, centroid.y)))
            x += cell_deg_lon
        y += cell_deg_lat
    return cells


def _normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _score_components(features: CellFeatures, operation: Operation) -> dict[str, float]:
    concealment_count = features.high_concealment_count + 0.5 * features.medium_concealment_count
    concealment = _normalize(concealment_count, 0, 8)

    cloud_bonus = _normalize(features.weather_cloud_pct or 0, 30, 90)
    observation = max(
        0.0,
        min(
            1.0,
            0.6 * (1.0 - concealment) + 0.4 * cloud_bonus,
        ),
    )

    if features.max_road_width_m is None and features.road_segment_count == 0:
        road_access = 0.0
    else:
        width_score = _normalize(features.max_road_width_m or 0, 3, 10)
        density_score = _normalize(features.road_segment_count, 0, 5)
        road_access = 0.6 * width_score + 0.4 * density_score

    civilian_avoidance = 1.0 - _normalize(features.population_estimate, 0, 500)

    if features.nearest_cell_tower_km is None:
        comms_coverage = 0.0
    else:
        comms_coverage = 1.0 - _normalize(features.nearest_cell_tower_km, 0, 15)

    if features.nearest_road_km is None:
        logistics_proximity = 0.0
    else:
        logistics_proximity = 1.0 - _normalize(features.nearest_road_km, 0, 20)

    return {
        "concealment": concealment,
        "observation": observation,
        "road_access": road_access,
        "civilian_avoidance": civilian_avoidance,
        "comms_coverage": comms_coverage,
        "logistics_proximity": logistics_proximity,
    }


def _weighted_score(components: dict[str, float], weights: dict[str, float]) -> float:
    return round(sum(components.get(key, 0.0) * weights.get(key, 0.0) for key in SCORING_CRITERIA), 4)


def _rationale_lines(features: CellFeatures, components: dict[str, float]) -> list[str]:
    lines: list[str] = []
    if features.high_concealment_count or features.medium_concealment_count:
        lines.append(
            f"Concealment: {features.high_concealment_count} high + {features.medium_concealment_count} medium forest features (score {components['concealment']:.2f})"
        )
    if features.max_road_width_m is not None:
        lines.append(
            f"Road access: max width {features.max_road_width_m:.1f} m across {features.road_segment_count} segments (score {components['road_access']:.2f})"
        )
    if features.weather_wind_ms is not None:
        gust = f", gust {features.weather_gust_ms:.1f} m/s" if features.weather_gust_ms else ""
        lines.append(f"Wind: {features.weather_wind_ms:.1f} m/s{gust}")
    if features.weather_precip_mm is not None and features.weather_precip_mm > 0:
        lines.append(f"Precipitation: {features.weather_precip_mm:.1f} mm")
    if features.population_estimate:
        lines.append(
            f"Civilian footprint: ~{features.population_estimate} residents (avoidance score {components['civilian_avoidance']:.2f})"
        )
    if features.nearest_cell_tower_km is not None:
        lines.append(
            f"Comms: nearest cell tower {features.nearest_cell_tower_km:.1f} km (score {components['comms_coverage']:.2f})"
        )
    return lines


def _resolve_mask(request: PlanningRequest) -> tuple[BaseGeometry, list[str]]:
    notes: list[str] = []
    if request.geometry is not None:
        mask = geojson_to_shape(request.geometry)
        if mask.is_empty:
            raise ValueError("Planning geometry is empty")
        if mask.geom_type == "MultiPolygon":
            parts = list(getattr(mask, "geoms", [mask]))
            if len(parts) > 1:
                notes.append(
                    f"Geometry has {len(parts)} disjoint polygons; planning only the largest."
                )
            mask = max(parts, key=lambda geom: geom.area)
        elif mask.geom_type not in ("Polygon",):
            raise ValueError(f"Planning geometry must be a Polygon or MultiPolygon, got {mask.geom_type}")
        return mask, notes

    try:
        minx, miny, maxx, maxy = resolve_area_bbox(request.area)
    except Exception as exc:
        raise ValueError(f"Unknown planning area '{request.area}': {exc}") from exc
    return box(minx, miny, maxx, maxy), notes


def _records_by_source(records: list[DatasetRecord]) -> dict[str, DatasetRecord]:
    latest: dict[str, DatasetRecord] = {}
    for record in records:
        existing = latest.get(record.source_id)
        if existing is None or record.retrieved_at > existing.retrieved_at:
            latest[record.source_id] = record
    return latest


def _scored_cells(
    request: PlanningRequest,
    mask: BaseGeometry,
    records: dict[str, DatasetRecord],
    weights: dict[str, float],
) -> tuple[list[dict[str, Any]], int, int, list[str]]:
    cells = _build_cell_grid(mask, request.grid_resolution_m)
    notes: list[str] = []

    if len(cells) > PLANNING_MAX_CELLS:
        raise ValueError(
            f"Grid produces {len(cells)} cells (cap {PLANNING_MAX_CELLS}). "
            "Increase grid_resolution_m or draw a smaller AOI."
        )

    # Build all spatial indexes ONCE for the whole run. Previously each cell
    # re-parsed every GeoJSON feature, which dominated runtime on Finland-sized
    # bboxes.
    t_idx = time.perf_counter()
    index = SourceIndex.build(records)
    idx_ms = (time.perf_counter() - t_idx) * 1000

    feasible_count = 0
    scored: list[dict[str, Any]] = []
    error_count = 0
    t_loop = time.perf_counter()

    for cell, centroid in cells:
        try:
            features = extract_cell_features_indexed(cell, centroid, index)
            matches = check_constraints(features, request.force)
            feasible = is_feasible(matches)
            if feasible:
                feasible_count += 1
            components = _score_components(features, request.operation)
            score = _weighted_score(components, weights)
        except Exception as exc:
            # A single bad feature in any source shouldn't kill the entire
            # planning run. Skip the cell and keep going.
            error_count += 1
            if error_count <= 3:
                logger.warning("planning: cell scoring failed at %s: %s", centroid, exc)
            continue

        scored.append(
            {
                "cell": cell,
                "centroid": centroid,
                "features": features,
                "matches": matches,
                "feasible": feasible,
                "components": components,
                "score": score,
            }
        )

    loop_ms = (time.perf_counter() - t_loop) * 1000
    logger.info(
        "planning: scored %d/%d cells in %.0f ms (index build %.0f ms, %d skipped)",
        len(scored), len(cells), loop_ms, idx_ms, error_count,
    )
    if error_count:
        notes.append(f"Skipped {error_count} cells due to feature errors (see server logs).")

    return scored, len(cells), feasible_count, notes


def recommend_sites(
    request: PlanningRequest,
    records: list[DatasetRecord],
    freshness: Optional[list[dict[str, Any]]] = None,
) -> PlanningResponse:
    t_total = time.perf_counter()
    mask, mask_notes = _resolve_mask(request)
    weights = get_operation_profile(request.operation)
    records_by_source = _records_by_source(records)

    scored, total_cells, feasible_count, score_notes = _scored_cells(
        request, mask, records_by_source, weights
    )

    feasible_cells = [c for c in scored if c["feasible"]]
    ranked_source = feasible_cells if feasible_cells else scored
    ranked = sorted(ranked_source, key=lambda c: c["score"], reverse=True)[: request.top_n]

    top_sites: list[RecommendedSite] = []
    for idx, entry in enumerate(ranked, start=1):
        cell = entry["cell"]
        features = entry["features"]
        rationale = _rationale_lines(features, entry["components"])
        if not entry["feasible"]:
            failed = [m for m in entry["matches"] if not m.passed]
            if failed:
                rationale.insert(
                    0,
                    "FAILED constraints: " + "; ".join(m.detail or m.name for m in failed),
                )

        top_sites.append(
            RecommendedSite(
                rank=idx,
                score=min(1.0, max(0.0, entry["score"])),
                centroid=[round(entry["centroid"][0], 6), round(entry["centroid"][1], 6)],
                geometry=mapping(cell),
                feasible=entry["feasible"],
                constraint_matches=entry["matches"],
                score_breakdown={k: round(v, 4) for k, v in entry["components"].items()},
                rationale=rationale,
            )
        )

    notes: list[str] = []
    notes.extend(mask_notes)
    notes.extend(score_notes)
    if not feasible_cells:
        notes.append(
            "No cell satisfies all hard constraints. Returning highest-scoring cells with failed constraints flagged."
        )
    if not records_by_source:
        notes.append(
            "No ingested data found for this area. Recommendations are based on grid geometry only — call /api/ingest first."
        )
    else:
        missing = [
            source
            for source in ("digiroad", "osm-poi", "fmi", "opencellid", "statistics-finland")
            if source not in records_by_source
        ]
        if missing:
            notes.append("Missing data sources for richer scoring: " + ", ".join(missing))

    total_ms = (time.perf_counter() - t_total) * 1000
    logger.info(
        "planning: complete area=%s cells=%d feasible=%d top_n=%d in %.0f ms",
        request.area, total_cells, feasible_count, len(top_sites), total_ms,
    )

    return PlanningResponse(
        area=request.area,
        timeframe=request.timeframe,
        operation_type=request.operation.type,
        grid_resolution_m=request.grid_resolution_m,
        cells_evaluated=total_cells,
        feasible_cells=feasible_count,
        top_sites=top_sites,
        weights=weights,
        data_freshness=freshness or [],
        notes=notes,
    )
