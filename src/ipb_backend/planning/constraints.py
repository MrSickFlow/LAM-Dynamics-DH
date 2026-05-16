from __future__ import annotations

import math
from typing import Any, Iterable, Optional

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from ipb_backend.models import DatasetRecord
from ipb_backend.planning.force_model import ConstraintMatch, ForceComposition
from ipb_backend.spatial import geojson_to_shape


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


def _safe_centroid(geometry: dict[str, Any]) -> Optional[tuple[float, float]]:
    try:
        shape = geojson_to_shape(geometry)
    except Exception:
        return None
    if shape.is_empty:
        return None
    centroid = shape.centroid
    return (centroid.x, centroid.y)


def extract_cell_features(
    cell: BaseGeometry,
    cell_centroid: tuple[float, float],
    records: dict[str, DatasetRecord],
) -> CellFeatures:
    features = CellFeatures()

    digiroad = records.get("digiroad")
    if digiroad is not None:
        link_limits = _build_link_limits(digiroad)

        for bridge in _iter_digiroad_features(digiroad, "dr_tielinkki_silta_alikulku_tunneli"):
            geometry = bridge.get("geometry")
            if not geometry:
                continue
            try:
                geom = geojson_to_shape(geometry)
            except Exception:
                continue
            if geom.is_empty or not geom.intersects(cell):
                continue

            props = bridge.get("properties") or {}
            link_id = props.get("link_id")
            limits = link_limits.get(link_id, {})

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

        nearest_road_km: Optional[float] = None
        for road in _iter_digiroad_features(digiroad, "dr_leveys"):
            geometry = road.get("geometry")
            if not geometry:
                continue
            try:
                geom = geojson_to_shape(geometry)
            except Exception:
                continue
            if geom.is_empty:
                continue

            if geom.intersects(cell):
                features.road_segment_count += 1
                width_cm = (road.get("properties") or {}).get("arvo")
                if width_cm is not None:
                    width_m = width_cm / 100.0
                    if features.max_road_width_m is None or width_m > features.max_road_width_m:
                        features.max_road_width_m = width_m

            road_centroid = geom.centroid
            if not road_centroid.is_empty:
                distance_km = haversine_km(cell_centroid, (road_centroid.x, road_centroid.y))
                if nearest_road_km is None or distance_km < nearest_road_km:
                    nearest_road_km = distance_km

        features.nearest_road_km = nearest_road_km

    osm = records.get("osm-poi")
    if osm is not None:
        categories = osm.data.get("categories", {})
        forests = categories.get("forest", []) or []
        nearest_poi_km: Optional[float] = None

        for forest in forests:
            lon, lat = forest.get("lon"), forest.get("lat")
            if lon is None or lat is None:
                continue
            point = Point(lon, lat)
            if not point.within(cell):
                continue
            features.forest_feature_count += 1
            rating = _classify_forest(forest.get("tags") or {})
            if rating == "high":
                features.high_concealment_count += 1
            elif rating == "medium":
                features.medium_concealment_count += 1

        for category_id, items in categories.items():
            if category_id == "forest":
                continue
            for item in items:
                lon, lat = item.get("lon"), item.get("lat")
                if lon is None or lat is None:
                    continue
                distance_km = haversine_km(cell_centroid, (lon, lat))
                if nearest_poi_km is None or distance_km < nearest_poi_km:
                    nearest_poi_km = distance_km
        features.nearest_poi_km = nearest_poi_km

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

        features.weather_temp_c = _latest("temperature")
        features.weather_wind_ms = _latest("wind_speed")
        features.weather_gust_ms = _latest("wind_gust")
        features.weather_precip_mm = _latest("precipitation")
        features.weather_cloud_pct = _latest("cloud_cover")
        features.weather_humidity_pct = _latest("humidity")

    opencellid = records.get("opencellid")
    if opencellid is not None:
        nearest: Optional[float] = None
        for cell_record in opencellid.data.get("cells", []) or []:
            lon, lat = cell_record.get("lon"), cell_record.get("lat")
            if lon is None or lat is None:
                continue
            distance_km = haversine_km(cell_centroid, (lon, lat))
            point = Point(lon, lat)
            if point.within(cell):
                features.cell_tower_count += 1
            if nearest is None or distance_km < nearest:
                nearest = distance_km
        features.nearest_cell_tower_km = nearest

    population = records.get("statistics-finland")
    if population is not None:
        total = 0
        for feature in population.data.get("features", []) or []:
            geometry = feature.get("geometry")
            if not geometry:
                continue
            try:
                geom = geojson_to_shape(geometry)
            except Exception:
                continue
            if geom.is_empty or not geom.intersects(cell):
                continue

            source_pop = int((feature.get("properties") or {}).get("population", 0) or 0)
            if source_pop <= 0:
                continue
            overlap = geom.intersection(cell).area / geom.area if geom.area > 0 else 0
            total += int(round(source_pop * overlap))
        features.population_estimate = total

    return features


def check_constraints(
    cell_features: CellFeatures,
    force: ForceComposition,
) -> list[ConstraintMatch]:
    matches: list[ConstraintMatch] = []

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
