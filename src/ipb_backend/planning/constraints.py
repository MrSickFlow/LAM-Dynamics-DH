from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Iterable, Optional

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points
from shapely.strtree import STRtree

from ipb_backend.models import DatasetRecord
from ipb_backend.planning.force_model import ConstraintMatch, ForceComposition
from ipb_backend.spatial import geojson_to_shape

logger = logging.getLogger(__name__)


_DEG_LAT_KM = 110.57


def _deg_lon_km(lat: float) -> float:
    return 111.32 * math.cos(math.radians(lat))


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lon1, lat1 = a
    lon2, lat2 = b
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


class CellFeatures:
    """Numeric feature vector extracted for a single grid cell."""

    __slots__ = (
        "min_bridge_capacity_t",
        "min_bridge_height_m",
        "min_bridge_width_m",
        "max_road_width_m",
        "road_segment_count",
        "forest_feature_count",
        "high_concealment_count",
        "medium_concealment_count",
        "weather_wind_ms",
        "weather_gust_ms",
        "weather_precip_mm",
        "weather_temp_c",
        "weather_cloud_pct",
        "weather_humidity_pct",
        "nearest_cell_tower_km",
        "cell_tower_count",
        "population_estimate",
        "nearest_poi_km",
        "nearest_road_km",
        "water_coverage_ratio",
        "bog_coverage_ratio",
        "rocky_coverage_ratio",
        "forest_vegetation_coverage_ratio",
        "agricultural_coverage_ratio",
        "contour_count",
        "contour_span_m",
    )

    def __init__(self) -> None:
        self.min_bridge_capacity_t: Optional[float] = None
        self.min_bridge_height_m: Optional[float] = None
        self.min_bridge_width_m: Optional[float] = None
        self.max_road_width_m: Optional[float] = None
        self.road_segment_count: int = 0
        self.forest_feature_count: int = 0
        self.high_concealment_count: int = 0
        self.medium_concealment_count: int = 0
        self.weather_wind_ms: Optional[float] = None
        self.weather_gust_ms: Optional[float] = None
        self.weather_precip_mm: Optional[float] = None
        self.weather_temp_c: Optional[float] = None
        self.weather_cloud_pct: Optional[float] = None
        self.weather_humidity_pct: Optional[float] = None
        self.nearest_cell_tower_km: Optional[float] = None
        self.cell_tower_count: int = 0
        self.population_estimate: int = 0
        self.nearest_poi_km: Optional[float] = None
        self.nearest_road_km: Optional[float] = None
        self.water_coverage_ratio: Optional[float] = None
        self.bog_coverage_ratio: Optional[float] = None
        self.rocky_coverage_ratio: Optional[float] = None
        self.forest_vegetation_coverage_ratio: Optional[float] = None
        self.agricultural_coverage_ratio: Optional[float] = None
        self.contour_count: int = 0
        self.contour_span_m: Optional[float] = None


_HIGH_CONCEALMENT = {("needleleaved", "evergreen"), ("mixed", "evergreen"), ("mixed", None)}
_MEDIUM_CONCEALMENT = {
    ("needleleaved", "deciduous"),
    ("broadleaved", "evergreen"),
    ("mixed", "deciduous"),
}


def _classify_forest(tags: dict[str, Any]) -> str:
    leaf_type = tags.get("leaf_type")
    leaf_cycle = tags.get("leaf_cycle")
    key = (leaf_type, leaf_cycle)
    if key in _HIGH_CONCEALMENT or (leaf_type == "needleleaved" and leaf_cycle is None):
        return "high"
    if key in _MEDIUM_CONCEALMENT or leaf_type == "needleleaved":
        return "medium"
    return "low"


def _iter_digiroad_features(record: DatasetRecord, coll_id: str) -> Iterable[dict[str, Any]]:
    return record.data.get("collections", {}).get(coll_id, {}).get("features", []) or []


def _iter_nls_features(record: DatasetRecord, coll_id: str) -> Iterable[dict[str, Any]]:
    return record.data.get("collections", {}).get(coll_id, {}).get("features", []) or []


def _build_link_limits(record: DatasetRecord) -> dict[str, dict[str, Optional[float]]]:
    by_link: dict[str, dict[str, Optional[float]]] = {}
    for coll_id, key in [
        ("dr_max_massa", "max_weight_kg"),
        ("dr_max_korkeus", "max_height_cm"),
        ("dr_max_leveys", "max_width_cm"),
    ]:
        for feature in _iter_digiroad_features(record, coll_id):
            properties = feature.get("properties") or {}
            link_id = properties.get("link_id")
            value = properties.get("arvo")
            if link_id is None or value is None:
                continue
            entry = by_link.setdefault(link_id, {})
            existing = entry.get(key)
            if existing is None or value < existing:
                entry[key] = value
    return by_link


def _safe_shape(geometry: dict[str, Any]) -> Optional[BaseGeometry]:
    try:
        shape = geojson_to_shape(geometry)
    except Exception:
        return None
    if shape.is_empty:
        return None
    return shape


def _safe_centroid(geometry: dict[str, Any]) -> Optional[tuple[float, float]]:
    shape = _safe_shape(geometry)
    if shape is None:
        return None
    centroid = shape.centroid
    return (centroid.x, centroid.y)


# ---------------------------------------------------------------------------
# Per-source spatial index built once per planning call.
# ---------------------------------------------------------------------------


@dataclass
class _IndexedFeature:
    geom: BaseGeometry
    props: dict[str, Any]


@dataclass
class _IndexedLayer:
    """Pre-shapified features + STRtree for O(log N) cell intersection."""

    features: list[_IndexedFeature] = field(default_factory=list)
    tree: Optional[STRtree] = None

    @classmethod
    def from_geojson_features(cls, items: Iterable[dict[str, Any]]) -> "_IndexedLayer":
        feats: list[_IndexedFeature] = []
        for item in items:
            geometry = item.get("geometry")
            if not geometry:
                continue
            shape = _safe_shape(geometry)
            if shape is None:
                continue
            feats.append(_IndexedFeature(geom=shape, props=item.get("properties") or {}))
        tree = STRtree([f.geom for f in feats]) if feats else None
        return cls(features=feats, tree=tree)

    @classmethod
    def from_points(cls, items: Iterable[dict[str, Any]], lon_key: str = "lon", lat_key: str = "lat") -> "_IndexedLayer":
        feats: list[_IndexedFeature] = []
        for item in items:
            lon, lat = item.get(lon_key), item.get(lat_key)
            if lon is None or lat is None:
                continue
            feats.append(_IndexedFeature(geom=Point(lon, lat), props=item))
        tree = STRtree([f.geom for f in feats]) if feats else None
        return cls(features=feats, tree=tree)

    def query_intersecting(self, cell: BaseGeometry) -> list[_IndexedFeature]:
        if self.tree is None or not self.features:
            return []
        idxs = self.tree.query(cell, predicate="intersects")
        return [self.features[i] for i in idxs]

    def query_within(self, cell: BaseGeometry) -> list[_IndexedFeature]:
        if self.tree is None or not self.features:
            return []
        idxs = self.tree.query(cell, predicate="within")
        return [self.features[i] for i in idxs]

    def nearest_distance_km(self, lon: float, lat: float) -> Optional[float]:
        """Great-circle km from (lon, lat) to nearest feature's geometry."""
        if self.tree is None or not self.features:
            return None
        query_point = Point(lon, lat)
        nearest = self.tree.nearest(query_point)
        if nearest is None:
            return None

        # Shapely 2.x returns an integer index here; older releases returned
        # the geometry object itself. Support both so planning stays stable
        # across local environments.
        nearest_geom = (
            self.features[int(nearest)].geom if isinstance(nearest, Integral) else nearest
        )
        if nearest_geom.is_empty:
            return None

        _, nearest_geom_pt = nearest_points(query_point, nearest_geom)
        return haversine_km((lon, lat), (nearest_geom_pt.x, nearest_geom_pt.y))


@dataclass
class SourceIndex:
    """All spatial indexes + scalar weather snapshot used during a planning run.

    Built ONCE in recommend_sites() and reused across every cell. Avoids the
    previous behaviour of re-parsing every GeoJSON feature per cell, which
    drove 30–90s planning runs into the minutes for large areas.
    """

    bridges: _IndexedLayer = field(default_factory=_IndexedLayer)
    bridge_link_limits: dict[str, dict[str, Optional[float]]] = field(default_factory=dict)
    roads: _IndexedLayer = field(default_factory=_IndexedLayer)
    waters: _IndexedLayer = field(default_factory=_IndexedLayer)
    bogs: _IndexedLayer = field(default_factory=_IndexedLayer)
    rocky_areas: _IndexedLayer = field(default_factory=_IndexedLayer)
    forest_vegetation: _IndexedLayer = field(default_factory=_IndexedLayer)
    agricultural_land: _IndexedLayer = field(default_factory=_IndexedLayer)
    contours: _IndexedLayer = field(default_factory=_IndexedLayer)
    forests: _IndexedLayer = field(default_factory=_IndexedLayer)
    other_pois: _IndexedLayer = field(default_factory=_IndexedLayer)
    cell_towers: _IndexedLayer = field(default_factory=_IndexedLayer)
    population: _IndexedLayer = field(default_factory=_IndexedLayer)
    nls_available: bool = False

    # Weather is global (one station per ingest), so cache the latest readings.
    weather_temp_c: Optional[float] = None
    weather_wind_ms: Optional[float] = None
    weather_gust_ms: Optional[float] = None
    weather_precip_mm: Optional[float] = None
    weather_cloud_pct: Optional[float] = None
    weather_humidity_pct: Optional[float] = None

    @classmethod
    def build(cls, records: dict[str, DatasetRecord]) -> "SourceIndex":
        idx = cls()

        digiroad = records.get("digiroad")
        if digiroad is not None:
            idx.bridge_link_limits = _build_link_limits(digiroad)
            idx.bridges = _IndexedLayer.from_geojson_features(
                _iter_digiroad_features(digiroad, "dr_tielinkki_silta_alikulku_tunneli")
            )
            idx.roads = _IndexedLayer.from_geojson_features(
                _iter_digiroad_features(digiroad, "dr_leveys")
            )

        nls = records.get("nls")
        if nls is not None:
            idx.nls_available = True
            water_features: list[dict[str, Any]] = []
            for coll_id in ("jarvi", "meri", "virtavesialue"):
                water_features.extend(_iter_nls_features(nls, coll_id))
            idx.waters = _IndexedLayer.from_geojson_features(water_features)
            idx.bogs = _IndexedLayer.from_geojson_features(_iter_nls_features(nls, "suo"))
            idx.rocky_areas = _IndexedLayer.from_geojson_features(
                _iter_nls_features(nls, "kallioalue")
            )
            idx.forest_vegetation = _IndexedLayer.from_geojson_features(
                _iter_nls_features(nls, "metsamaankasvillisuus")
            )
            idx.agricultural_land = _IndexedLayer.from_geojson_features(
                _iter_nls_features(nls, "maatalousmaa")
            )
            idx.contours = _IndexedLayer.from_geojson_features(
                _iter_nls_features(nls, "korkeuskayra")
            )
            if not idx.roads.features:
                idx.roads = _IndexedLayer.from_geojson_features(
                    _iter_nls_features(nls, "tieviiva")
                )

        osm = records.get("osm-poi")
        if osm is not None:
            categories = osm.data.get("categories", {}) or {}
            idx.forests = _IndexedLayer.from_points(categories.get("forest", []) or [])
            other_items: list[dict[str, Any]] = []
            for category_id, items in categories.items():
                if category_id == "forest":
                    continue
                other_items.extend(items or [])
            idx.other_pois = _IndexedLayer.from_points(other_items)

        opencellid = records.get("opencellid")
        if opencellid is not None:
            idx.cell_towers = _IndexedLayer.from_points(opencellid.data.get("cells", []) or [])

        population = records.get("statistics-finland")
        if population is not None:
            idx.population = _IndexedLayer.from_geojson_features(
                population.data.get("features", []) or []
            )

        fmi = records.get("fmi")
        if fmi is not None:
            observations = fmi.data.get("observations") or {}

            def _latest(name: str) -> Optional[float]:
                entry = observations.get(name) or {}
                value = (entry.get("latest") or {}).get("value")
                try:
                    return float(value) if value is not None else None
                except (TypeError, ValueError):
                    return None

            idx.weather_temp_c = _latest("temperature")
            idx.weather_wind_ms = _latest("wind_speed")
            idx.weather_gust_ms = _latest("wind_gust")
            idx.weather_precip_mm = _latest("precipitation")
            idx.weather_cloud_pct = _latest("cloud_cover")
            idx.weather_humidity_pct = _latest("humidity")

        logger.debug(
            "planning index built: bridges=%d roads=%d waters=%d contours=%d vegetation=%d forests=%d pois=%d towers=%d pop=%d",
            len(idx.bridges.features),
            len(idx.roads.features),
            len(idx.waters.features),
            len(idx.contours.features),
            len(idx.forest_vegetation.features),
            len(idx.forests.features),
            len(idx.other_pois.features),
            len(idx.cell_towers.features),
            len(idx.population.features),
        )
        return idx


# ---------------------------------------------------------------------------
# Per-cell extraction (uses pre-built indexes; no per-cell re-shapification)
# ---------------------------------------------------------------------------


def _layer_coverage_ratio(layer: _IndexedLayer, cell: BaseGeometry) -> float:
    if not layer.features or cell.area <= 0:
        return 0.0
    covered = 0.0
    for feat in layer.query_intersecting(cell):
        if feat.geom.area <= 0:
            continue
        try:
            covered += feat.geom.intersection(cell).area / cell.area
        except Exception:
            continue
    return min(1.0, covered)


def extract_cell_features_indexed(
    cell: BaseGeometry,
    cell_centroid: tuple[float, float],
    index: SourceIndex,
) -> CellFeatures:
    features = CellFeatures()
    centroid_lon, centroid_lat = cell_centroid

    # Bridges within the cell
    for bridge in index.bridges.query_intersecting(cell):
        link_id = bridge.props.get("link_id")
        limits = index.bridge_link_limits.get(link_id, {})

        weight_kg = limits.get("max_weight_kg")
        if weight_kg is not None:
            weight_t = weight_kg / 1000.0
            if features.min_bridge_capacity_t is None or weight_t < features.min_bridge_capacity_t:
                features.min_bridge_capacity_t = weight_t
        height_cm = limits.get("max_height_cm")
        if height_cm is not None:
            height_m = height_cm / 100.0
            if features.min_bridge_height_m is None or height_m < features.min_bridge_height_m:
                features.min_bridge_height_m = height_m
        width_cm = limits.get("max_width_cm")
        if width_cm is not None:
            width_m = width_cm / 100.0
            if features.min_bridge_width_m is None or width_m < features.min_bridge_width_m:
                features.min_bridge_width_m = width_m

    # Roads intersecting the cell
    for road in index.roads.query_intersecting(cell):
        features.road_segment_count += 1
        width_cm = road.props.get("arvo")
        if width_cm is not None:
            width_m = width_cm / 100.0
            if features.max_road_width_m is None or width_m > features.max_road_width_m:
                features.max_road_width_m = width_m

    # Nearest road (true point-on-geometry distance, not centroid-to-centroid)
    features.nearest_road_km = index.roads.nearest_distance_km(centroid_lon, centroid_lat)

    if index.nls_available:
        features.water_coverage_ratio = _layer_coverage_ratio(index.waters, cell)
        features.bog_coverage_ratio = _layer_coverage_ratio(index.bogs, cell)
        features.rocky_coverage_ratio = _layer_coverage_ratio(index.rocky_areas, cell)
        features.forest_vegetation_coverage_ratio = _layer_coverage_ratio(index.forest_vegetation, cell)
        features.agricultural_coverage_ratio = _layer_coverage_ratio(index.agricultural_land, cell)
        contour_elevations: list[float] = []
        for contour in index.contours.query_intersecting(cell):
            features.contour_count += 1
            elevation = contour.props.get("korkeus")
            try:
                contour_elevations.append(float(elevation))
            except (TypeError, ValueError):
                continue
        features.contour_span_m = (
            max(contour_elevations) - min(contour_elevations) if contour_elevations else 0.0
        )

    # Forest points inside the cell → concealment counts
    for forest in index.forests.query_within(cell):
        features.forest_feature_count += 1
        rating = _classify_forest(forest.props.get("tags") or {})
        if rating == "high":
            features.high_concealment_count += 1
        elif rating == "medium":
            features.medium_concealment_count += 1

    # Nearest non-forest POI
    features.nearest_poi_km = index.other_pois.nearest_distance_km(centroid_lon, centroid_lat)

    # Weather is global for this run — same for every cell
    features.weather_temp_c = index.weather_temp_c
    features.weather_wind_ms = index.weather_wind_ms
    features.weather_gust_ms = index.weather_gust_ms
    features.weather_precip_mm = index.weather_precip_mm
    features.weather_cloud_pct = index.weather_cloud_pct
    features.weather_humidity_pct = index.weather_humidity_pct

    # Cell towers
    for tower in index.cell_towers.query_within(cell):
        features.cell_tower_count += 1
    features.nearest_cell_tower_km = index.cell_towers.nearest_distance_km(centroid_lon, centroid_lat)

    # Population — area-weighted overlap on the indexed candidates only
    total_pop = 0
    for pop in index.population.query_intersecting(cell):
        source_pop_raw = pop.props.get("population", 0)
        try:
            source_pop = int(source_pop_raw or 0)
        except (TypeError, ValueError):
            continue
        if source_pop <= 0:
            continue
        geom = pop.geom
        if geom.area <= 0:
            continue
        overlap = geom.intersection(cell).area / geom.area
        total_pop += int(round(source_pop * overlap))
    features.population_estimate = total_pop

    return features


def extract_cell_features(
    cell: BaseGeometry,
    cell_centroid: tuple[float, float],
    records: dict[str, DatasetRecord],
) -> CellFeatures:
    """Backwards-compatible wrapper — builds the index on every call.

    Prefer `extract_cell_features_indexed` with a pre-built `SourceIndex` for
    planning runs that touch many cells (the per-call build dominates for a
    single-cell extraction, so this stays correct for unit-test usage).
    """
    index = SourceIndex.build(records)
    return extract_cell_features_indexed(cell, cell_centroid, index)


# ---------------------------------------------------------------------------
# Hard-constraint checks
# ---------------------------------------------------------------------------


def check_constraints(
    cell_features: CellFeatures,
    force: ForceComposition,
) -> list[ConstraintMatch]:
    matches: list[ConstraintMatch] = []

    if cell_features.water_coverage_ratio is not None:
        land_ratio = 1.0 - cell_features.water_coverage_ratio
        passed = land_ratio >= 0.55
        matches.append(
            ConstraintMatch(
                name="dry_land",
                passed=passed,
                observed=round(land_ratio, 2),
                required=0.55,
                detail=f"Dry land coverage {land_ratio:.2f} vs required 0.55",
            )
        )

    heaviest = force.heaviest_vehicle_t
    if heaviest > 0 and cell_features.min_bridge_capacity_t is not None:
        passed = cell_features.min_bridge_capacity_t >= heaviest
        matches.append(
            ConstraintMatch(
                name="bridge_weight",
                passed=passed,
                observed=round(cell_features.min_bridge_capacity_t, 1),
                required=round(heaviest, 1),
                detail=(
                    f"Bridge capacity {cell_features.min_bridge_capacity_t:.1f} t vs heaviest vehicle {heaviest:.1f} t"
                ),
            )
        )

    heaviest_arty = force.heaviest_artillery_t
    if heaviest_arty > 0 and cell_features.min_bridge_capacity_t is not None:
        passed = cell_features.min_bridge_capacity_t >= heaviest_arty
        matches.append(
            ConstraintMatch(
                name="bridge_weight_artillery",
                passed=passed,
                observed=round(cell_features.min_bridge_capacity_t, 1),
                required=round(heaviest_arty, 1),
                detail=(
                    f"Bridge capacity {cell_features.min_bridge_capacity_t:.1f} t vs heaviest artillery {heaviest_arty:.1f} t"
                ),
            )
        )

    widest = force.widest_vehicle_m
    if widest > 0 and cell_features.max_road_width_m is not None:
        required = widest * (2.2 if force.column_movement else 1.2)
        passed = cell_features.max_road_width_m >= required
        matches.append(
            ConstraintMatch(
                name="road_width",
                passed=passed,
                observed=round(cell_features.max_road_width_m, 1),
                required=round(required, 1),
                detail=(
                    f"Road width {cell_features.max_road_width_m:.1f} m vs required "
                    f"{required:.1f} m "
                    f"({'column' if force.column_movement else 'single file'})"
                ),
            )
        )

    tallest = force.tallest_vehicle_m
    if tallest > 0 and cell_features.min_bridge_height_m is not None:
        passed = cell_features.min_bridge_height_m >= tallest
        matches.append(
            ConstraintMatch(
                name="bridge_clearance",
                passed=passed,
                observed=round(cell_features.min_bridge_height_m, 1),
                required=round(tallest, 1),
                detail=(
                    f"Bridge clearance {cell_features.min_bridge_height_m:.1f} m vs tallest vehicle {tallest:.1f} m"
                ),
            )
        )

    wind_tol = force.min_drone_wind_tolerance_ms
    if wind_tol is not None and cell_features.weather_wind_ms is not None:
        wind = cell_features.weather_gust_ms or cell_features.weather_wind_ms
        passed = wind <= wind_tol
        matches.append(
            ConstraintMatch(
                name="drone_wind",
                passed=passed,
                observed=round(wind, 1),
                required=round(wind_tol, 1),
                detail=(
                    f"Wind {wind:.1f} m/s vs lowest drone tolerance {wind_tol:.1f} m/s"
                ),
            )
        )

    if force.drones and cell_features.weather_precip_mm is not None:
        precip_limits = [d.max_precip_mm_h for d in force.drones]
        min_precip = min(precip_limits)
        passed = cell_features.weather_precip_mm <= min_precip
        matches.append(
            ConstraintMatch(
                name="drone_precipitation",
                passed=passed,
                observed=round(cell_features.weather_precip_mm, 1),
                required=round(min_precip, 1),
                detail=f"Precipitation {cell_features.weather_precip_mm:.1f} mm vs drone tolerance {min_precip:.1f} mm",
            )
        )

    return matches


def is_feasible(matches: list[ConstraintMatch]) -> bool:
    if not matches:
        return True
    return all(match.passed for match in matches)
