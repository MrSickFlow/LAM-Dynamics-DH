from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Optional

from shapely import affinity
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry

from ipb_backend.models import LoadTarget, LoadTargetKind


AREA_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "archipelago sea": (21.0, 59.7, 23.0, 60.6),
    "north karelia": (29.0, 62.0, 31.5, 63.5),
    "lapland": (20.5, 68.5, 22.5, 69.4),
    "lapland (kasivarren lappi)": (20.5, 68.5, 22.5, 69.4),
    "kasivarren lappi": (20.5, 68.5, 22.5, 69.4),
}


def normalize_area_name(area: str) -> str:
    ascii_area = unicodedata.normalize("NFKD", area).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_area).strip().lower()


def resolve_area_bbox(area: str) -> tuple[float, float, float, float]:
    return AREA_BBOXES.get(normalize_area_name(area), AREA_BBOXES["north karelia"])


def geometry_bounds(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    geom = shape(geometry)
    min_x, min_y, max_x, max_y = geom.bounds
    return float(min_x), float(min_y), float(max_x), float(max_y)


def bbox_to_polygon(bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    min_x, min_y, max_x, max_y = bbox
    return mapping(box(min_x, min_y, max_x, max_y))


def resolve_load_target_bbox(area: str, load_target: Optional[LoadTarget] = None) -> tuple[float, float, float, float]:
    if load_target is None:
        return resolve_area_bbox(area)

    if load_target.kind == LoadTargetKind.BBOX and load_target.bbox_wgs84:
        min_x, min_y, max_x, max_y = load_target.bbox_wgs84
        return float(min_x), float(min_y), float(max_x), float(max_y)

    if load_target.kind == LoadTargetKind.GEOMETRY and load_target.geometry:
        return geometry_bounds(load_target.geometry)

    if load_target.kind == LoadTargetKind.NAMED_AREA:
        return resolve_area_bbox(load_target.label or area)

    return resolve_area_bbox(area)


def resolve_load_target_label(area: str, load_target: Optional[LoadTarget] = None) -> str:
    if load_target is None:
        return area
    if load_target.label:
        return load_target.label
    if load_target.kind == LoadTargetKind.BBOX:
        return "Custom BBox"
    if load_target.kind == LoadTargetKind.GEOMETRY:
        return "Custom Geometry"
    return area


def bbox_centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = bbox
    return (min_y + max_y) / 2.0, (min_x + max_x) / 2.0


def resolve_load_target_centroid(area: str, load_target: Optional[LoadTarget] = None) -> tuple[float, float]:
    return bbox_centroid(resolve_load_target_bbox(area, load_target))


def format_bbox(bbox: tuple[float, float, float, float]) -> str:
    return f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"


def geojson_to_shape(geometry: dict[str, Any]) -> BaseGeometry:
    return shape(geometry)


def clip_geojson_feature(feature: dict[str, Any], mask: BaseGeometry) -> Optional[dict[str, Any]]:
    geometry = feature.get("geometry")
    if not geometry:
        return None

    try:
        source_geometry = shape(geometry)
    except Exception:
        return None

    if source_geometry.is_empty or not source_geometry.intersects(mask):
        return None

    clipped_geometry = source_geometry.intersection(mask)
    if clipped_geometry.is_empty:
        return None

    return {
        "type": "Feature",
        "geometry": mapping(clipped_geometry),
        "properties": dict(feature.get("properties", {})),
    }


def polygon_area_sqkm(mask: BaseGeometry) -> float:
    mean_latitude = mask.centroid.y if not mask.is_empty else 0.0
    scaled = affinity.scale(
        mask,
        xfact=111.32 * math.cos(math.radians(mean_latitude)),
        yfact=110.57,
        origin=(0, 0),
    )
    return round(float(scaled.area), 2)