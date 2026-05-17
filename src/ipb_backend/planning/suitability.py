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
    OperationType,
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


def _grid_dimensions(mask: BaseGeometry, resolution_m: int) -> tuple[int, int]:
    minx, miny, maxx, maxy = mask.bounds
    mean_lat = (miny + maxy) / 2.0
    height_km = max(0.0, maxy - miny) * _DEG_LAT_KM
    width_km = max(0.0, maxx - minx) * max(_deg_lon_km(mean_lat), 0.0001)
    resolution_km = max(resolution_m, 1) / 1000.0
    rows = max(1, math.ceil(height_km / resolution_km))
    cols = max(1, math.ceil(width_km / resolution_km))
    return rows, cols


def _effective_grid_resolution(mask: BaseGeometry, requested_resolution_m: int) -> tuple[int, int]:
    resolution_m = max(1, requested_resolution_m)
    rows, cols = _grid_dimensions(mask, resolution_m)
    estimated_cells = rows * cols
    if estimated_cells <= PLANNING_MAX_CELLS:
        return resolution_m, estimated_cells

    # Scale the request to land under the hard cell budget using the AOI bbox as
    # an upper bound. This avoids building an oversized grid only to reject it.
    resolution_m = int(math.ceil(resolution_m * math.sqrt(estimated_cells / PLANNING_MAX_CELLS)))
    while True:
        rows, cols = _grid_dimensions(mask, resolution_m)
        estimated_cells = rows * cols
        if estimated_cells <= PLANNING_MAX_CELLS:
            return resolution_m, estimated_cells
        resolution_m += 1


def _normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


_RELIEF_TARGET = {
    OperationType.OFFENSIVE: 0.15,
    OperationType.DEFENSIVE: 0.85,
    OperationType.RECON: 0.65,
    OperationType.SCREEN: 0.70,
    OperationType.LOGISTICS_HUB: 0.20,
    OperationType.WITHDRAWAL: 0.30,
    OperationType.FIRE_SUPPORT: 0.75,
}

_OPEN_GROUND_TARGET = {
    OperationType.OFFENSIVE: 0.75,
    OperationType.DEFENSIVE: 0.25,
    OperationType.RECON: 0.40,
    OperationType.SCREEN: 0.45,
    OperationType.LOGISTICS_HUB: 0.65,
    OperationType.WITHDRAWAL: 0.60,
    OperationType.FIRE_SUPPORT: 0.50,
}

_ROCKY_TARGET = {
    OperationType.OFFENSIVE: 0.15,
    OperationType.DEFENSIVE: 0.65,
    OperationType.RECON: 0.55,
    OperationType.SCREEN: 0.50,
    OperationType.LOGISTICS_HUB: 0.20,
    OperationType.WITHDRAWAL: 0.25,
    OperationType.FIRE_SUPPORT: 0.55,
}


def _terrain_relief(features: CellFeatures) -> float:
    contour_density = _normalize(features.contour_count, 0, 6)
    contour_span = _normalize(features.contour_span_m or 0.0, 0, 40)
    return _clamp01(0.6 * contour_density + 0.4 * contour_span)


def _force_mobility_burden(force: ForceComposition) -> float:
    total_mounted = sum(vehicle.count for vehicle in force.vehicles) + sum(
        artillery.count for artillery in force.artillery
    )
    heaviest_load = max(force.heaviest_vehicle_t, force.heaviest_artillery_t)
    burden = (
        0.6 * _normalize(heaviest_load, 10, 70)
        + 0.25 * _normalize(total_mounted, 0, 20)
        + 0.15 * _normalize(force.logistics_demand_t_per_day, 0, 40)
    )
    if force.column_movement:
        burden += 0.08
    return _clamp01(burden)


def _score_components(
    features: CellFeatures,
    operation: Operation,
    force: ForceComposition,
) -> dict[str, float]:
    water_ratio = features.water_coverage_ratio or 0.0
    bog_ratio = features.bog_coverage_ratio or 0.0
    rocky_ratio = features.rocky_coverage_ratio or 0.0
    vegetation_ratio = features.forest_vegetation_coverage_ratio or 0.0
    agriculture_ratio = features.agricultural_coverage_ratio or 0.0

    concealment_count = features.high_concealment_count * 1.3 + features.medium_concealment_count * 0.7
    forest_density = _normalize(concealment_count, 0, 6)
    vegetation_cover = _normalize(vegetation_ratio, 0.05, 0.75)
    open_ground = _clamp01(0.7 * agriculture_ratio + 0.3 * (1.0 - vegetation_cover))
    concealment = _clamp01(
        0.55 * forest_density
        + 0.35 * vegetation_cover
        + 0.10 * (1.0 - _normalize(agriculture_ratio, 0.05, 0.6))
    )

    relief = _terrain_relief(features)
    relief_fit = _clamp01(1.0 - abs(relief - _RELIEF_TARGET[operation.type]))
    open_fit = _clamp01(1.0 - abs(open_ground - _OPEN_GROUND_TARGET[operation.type]))
    rocky_fit = _clamp01(
        1.0 - abs(_normalize(rocky_ratio, 0.05, 0.5) - _ROCKY_TARGET[operation.type])
    )
    dry_land = _clamp01(1.0 - min(1.0, 1.3 * water_ratio + 0.8 * bog_ratio))
    terrain_fit = _clamp01(
        0.40 * relief_fit + 0.25 * dry_land + 0.20 * open_fit + 0.15 * rocky_fit
    )

    observation = _clamp01(0.70 * relief + 0.30 * (1.0 - concealment))

    cloud_mask = _normalize(features.weather_cloud_pct or 0, 40, 100)
    precip_mask = _normalize(features.weather_precip_mm or 0, 0.3, 6.0)
    terrain_mask = _clamp01(0.60 * relief + 0.40 * _normalize(rocky_ratio, 0.05, 0.5))
    drone_cover = _clamp01(
        0.45 * concealment + 0.25 * terrain_mask + 0.20 * cloud_mask + 0.10 * precip_mask
    )

    road_width_score = _normalize(features.max_road_width_m or 0, 3.5, 10)
    road_density_score = _normalize(features.road_segment_count, 0, 5)
    road_proximity_score = (
        0.0 if features.nearest_road_km is None else 1.0 - _normalize(features.nearest_road_km, 0, 10)
    )
    heaviest_load = max(force.heaviest_vehicle_t, force.heaviest_artillery_t)
    if heaviest_load > 0 and features.min_bridge_capacity_t is not None:
        bridge_score = _clamp01(features.min_bridge_capacity_t / max(heaviest_load, 1.0))
    elif heaviest_load > 0 and features.road_segment_count:
        bridge_score = 0.55
    else:
        bridge_score = 0.7 if features.road_segment_count else 0.0

    burden = _force_mobility_burden(force)
    precip_stress = _normalize(features.weather_precip_mm or 0, 0.5, 8.0)
    thaw_stress = (
        1.0
        if features.weather_temp_c is not None
        and -2.0 <= features.weather_temp_c <= 2.0
        and (features.weather_precip_mm or 0.0) > 0.2
        else 0.0
    )
    weather_penalty = (0.7 * precip_stress + 0.3 * thaw_stress) * (0.35 + 0.65 * burden)
    narrow_route_penalty = (1.0 - road_width_score) * (0.35 + 0.55 * burden)
    route_resilience = _clamp01(
        0.30 * road_width_score
        + 0.20 * road_density_score
        + 0.15 * road_proximity_score
        + 0.20 * bridge_score
        + 0.15 * (1.0 - weather_penalty)
        - 0.10 * _normalize(bog_ratio, 0.05, 0.4)
        - 0.10 * _normalize(water_ratio, 0.05, 0.35)
        - 0.15 * narrow_route_penalty
    )

    civilian_avoidance = 1.0 - _normalize(features.population_estimate, 0, 500)

    if features.nearest_cell_tower_km is None:
        comms_coverage = 0.0
    else:
        comms_coverage = 1.0 - _normalize(features.nearest_cell_tower_km, 0, 15)

    logistics_proximity = _clamp01(
        0.55 * road_proximity_score + 0.25 * road_density_score + 0.20 * dry_land
    )

    return {
        "terrain_fit": terrain_fit,
        "concealment": concealment,
        "drone_cover": drone_cover,
        "observation": observation,
        "route_resilience": route_resilience,
        "civilian_avoidance": civilian_avoidance,
        "comms_coverage": comms_coverage,
        "logistics_proximity": logistics_proximity,
    }


def _weighted_score(components: dict[str, float], weights: dict[str, float]) -> float:
    return round(sum(components.get(key, 0.0) * weights.get(key, 0.0) for key in SCORING_CRITERIA), 4)


def _rationale_lines(features: CellFeatures, components: dict[str, float]) -> list[str]:
    lines: list[str] = []
    if features.contour_count or features.contour_span_m:
        lines.append(
            f"Terrain fit: {features.contour_count} contour lines, relief span {features.contour_span_m or 0:.1f} m (score {components['terrain_fit']:.2f})"
        )
    if features.water_coverage_ratio:
        lines.append(
            f"Dry ground: {(1.0 - features.water_coverage_ratio):.0%} land after water obstacles"
        )
    if features.high_concealment_count or features.medium_concealment_count:
        lines.append(
            f"Concealment: {features.high_concealment_count} high + {features.medium_concealment_count} medium forest features (score {components['concealment']:.2f})"
        )
    if features.max_road_width_m is not None or features.road_segment_count:
        lines.append(
            f"Route resilience: max width {(features.max_road_width_m or 0):.1f} m across {features.road_segment_count} segments (score {components['route_resilience']:.2f})"
        )
    if features.weather_wind_ms is not None:
        gust = f", gust {features.weather_gust_ms:.1f} m/s" if features.weather_gust_ms else ""
        lines.append(f"Wind: {features.weather_wind_ms:.1f} m/s{gust}")
    if features.weather_precip_mm is not None and features.weather_precip_mm > 0:
        lines.append(f"Precipitation: {features.weather_precip_mm:.1f} mm")
    if features.forest_vegetation_coverage_ratio:
        lines.append(
            f"Drone cover: vegetation on {features.forest_vegetation_coverage_ratio:.0%} of the cell (score {components['drone_cover']:.2f})"
        )
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
) -> tuple[list[dict[str, Any]], int, int, int, list[str]]:
    notes: list[str] = []
    effective_resolution_m, estimated_cells = _effective_grid_resolution(mask, request.grid_resolution_m)
    if effective_resolution_m != request.grid_resolution_m:
        notes.append(
            "Grid auto-coarsened from "
            f"{request.grid_resolution_m} m to {effective_resolution_m} m "
            f"to keep the search under {PLANNING_MAX_CELLS} cells "
            f"(estimated {estimated_cells})."
        )

    cells = _build_cell_grid(mask, effective_resolution_m)

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
            components = _score_components(features, request.operation, request.force)
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
        "planning: scored %d/%d cells at %dm in %.0f ms (index build %.0f ms, %d skipped)",
        len(scored), len(cells), effective_resolution_m, loop_ms, idx_ms, error_count,
    )
    if error_count:
        notes.append(f"Skipped {error_count} cells due to feature errors (see server logs).")

    return scored, len(cells), feasible_count, effective_resolution_m, notes


def recommend_sites(
    request: PlanningRequest,
    records: list[DatasetRecord],
    freshness: Optional[list[dict[str, Any]]] = None,
) -> PlanningResponse:
    t_total = time.perf_counter()
    mask, mask_notes = _resolve_mask(request)
    weights = get_operation_profile(request.operation)
    records_by_source = _records_by_source(records)

    scored, total_cells, feasible_count, effective_resolution_m, score_notes = _scored_cells(
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
        "planning: complete area=%s cells=%d feasible=%d top_n=%d resolution=%dm in %.0f ms",
        request.area, total_cells, feasible_count, len(top_sites), effective_resolution_m, total_ms,
    )

    return PlanningResponse(
        area=request.area,
        timeframe=request.timeframe,
        operation_type=request.operation.type,
        grid_resolution_m=effective_resolution_m,
        cells_evaluated=total_cells,
        feasible_cells=feasible_count,
        top_sites=top_sites,
        weights=weights,
        data_freshness=freshness or [],
        notes=notes,
    )
