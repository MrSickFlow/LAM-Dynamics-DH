from __future__ import annotations

import asyncio
import hashlib
import json
import time
from functools import partial
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, Response

from ipb_backend.agents.bridge_load import BridgeLoadAgent
from ipb_backend.agents.celltower import CellTowerAgent
from ipb_backend.agents.demographics import DemographicsAgent
from ipb_backend.agents.forest_concealment import ForestConcealmentAgent
from ipb_backend.agents.placeholders import SummaryAgent
from ipb_backend.agents.power_grid import PowerGridAgent
from ipb_backend.agents.satellite import SatelliteAgent
from ipb_backend.agents.weather_impact import WeatherImpactAgent
from ipb_backend.analysis import (
    ClaudeAnalyzer,
    RulesAnalyzer,
    _build_raw_data_from_package,
    _rules_intsum_sections,
    build_analyzer,
    build_aoi_metrics,
    build_data_package,
    build_evidence_bundle,
    build_raw_sections,
    get_analyzer_health,
    render_intsum_html,
)
from ipb_backend.analysis.contracts import DataPackage
from ipb_backend.config import settings
from ipb_backend.ingestion.sources.fmi import FmiAdapter
from ipb_backend.llm import LlmAnalysisOutput, LlmInterpretRequest, build_llm_wrapper_input, list_profile_specs
from ipb_backend.models import (
    AoiInspectionRequest,
    AoiInspectionResponse,
    DatasetRecord,
    AgentDefinition,
    IngestionRequest,
    PointInspectionRequest,
    PointInspectionResponse,
    SourceCategory,
    TerrainSnapshot,
    UiLayer,
    UiPlaceholderResponse,
)
from ipb_backend.planning import (
    OPERATION_PROFILES,
    PlanningRequest,
    PlanningResponse,
    recommend_sites,
)
from ipb_backend.planning.explainer import enrich_with_narratives
from ipb_backend.spatial import (
    bbox_to_mask,
    clip_geojson_feature,
    filter_features_by_bbox,
    geojson_to_shape,
    parse_bbox_param,
    polygon_area_sqkm,
)

router = APIRouter()
UI_PLACEHOLDER_PATH = Path(__file__).resolve().parents[1] / "ui_placeholder.html"

NLS_TILE_LAYERS = {
    "taustakartta": "Background map",
    "maastokartta": "Topographic map",
    "ortokuva": "Orthophoto",
    "selkokartta": "Plain map",
}
NLS_TILE_URL = (
    "https://avoin-karttakuva.maanmittauslaitos.fi/avoin/wmts/1.0.0"
    "/{layer}/default/WGS84_Pseudo-Mercator/{z}/{y}/{x}.png"
)


def get_services():
    from ipb_backend.main import state

    return state


def _is_bbox_load_target(load_target) -> bool:
    """True when the load target is a user-drawn bounding box. The satellite
    overlay only renders for bbox-scoped loads so it doesn't flood the screen
    when the user is just browsing a named area."""
    if load_target is None:
        return False
    kind = getattr(load_target, "kind", None)
    kind_value = kind.value if hasattr(kind, "value") else kind
    return kind_value == "bbox" and bool(getattr(load_target, "bbox_wgs84", None))


def _latest_records_by_source(records):
    latest = {}
    for record in records:
        current = latest.get(record.source_id)
        if current is None or record.retrieved_at > current.retrieved_at:
            latest[record.source_id] = record
    return latest


def _record_for_area_or_latest(records, source_id: str, area: Optional[str] = None):
    matching = [record for record in records if record.source_id == source_id]
    if not matching:
        return None
    if area:
        for record in reversed(matching):
            if record.area == area:
                return record
        return None
    return max(matching, key=lambda record: record.retrieved_at)


def _collection_summary(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = {}
    for feature in features:
        props = feature.get("properties", {})
        coll_id = props.get("_collection", "unknown")
        label = props.get("_label", coll_id)
        if coll_id not in summary:
            summary[coll_id] = {"label": label, "count": 0}
        summary[coll_id]["count"] += 1
    return [
        {"collection": collection, **details}
        for collection, details in sorted(summary.items(), key=lambda item: item[1]["count"], reverse=True)
    ]


def _clip_nls_record(record, mask):
    features = []
    collections = record.data.get("collections", {})
    for coll_id, coll_data in collections.items():
        label = coll_data.get("label", coll_id)
        for sample in coll_data.get("features", []):
            clipped = clip_geojson_feature(sample, mask)
            if clipped is None:
                continue
            props = clipped.setdefault("properties", {})
            props["_collection"] = coll_id
            props["_label"] = label
            features.append(clipped)
    return {
        "summary": record.summary,
        "feature_count": len(features),
        "collections": _collection_summary(features),
        "features": features[:60],
    }


def _clip_feature_dataset(record, mask, feature_limit: int = 30):
    features = [
        clipped
        for feature in record.data.get("features", [])
        if (clipped := clip_geojson_feature(feature, mask)) is not None
    ]
    return {
        "summary": record.summary,
        "feature_count": len(features),
        "features": features[:feature_limit],
    }


def _clip_osm_poi_record(record, mask, feature_limit: int = 30):
    features = []
    category_features = []

    for category_id, items in record.data.get("categories", {}).items():
        label = category_id.replace("_", " ").title()
        category_count = 0
        for item in items:
            lat = item.get("lat")
            lon = item.get("lon")
            if lat is None or lon is None:
                continue

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat],
                },
                "properties": {
                    "id": item.get("id"),
                    "osm_type": item.get("type"),
                    **item.get("tags", {}),
                },
            }
            clipped = clip_geojson_feature(feature, mask)
            if clipped is None:
                continue

            props = clipped.setdefault("properties", {})
            props["_collection"] = category_id
            props["_label"] = label
            category_count += 1
            features.append(clipped)

        if category_count:
            category_features.append(
                {
                    "collection": category_id,
                    "label": label,
                    "count": category_count,
                }
            )

    return {
        "summary": record.summary,
        "feature_count": len(features),
        "collections": category_features,
        "features": features[:feature_limit],
    }


def _clip_cell_tower_record(record, mask, feature_limit: int = 30):
    features = []

    for cell in record.data.get("cells", []):
        lat = cell.get("lat")
        lon = cell.get("lon")
        if lat is None or lon is None:
            continue

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
            "properties": {
                "cell_id": cell.get("cellid") or cell.get("cellId"),
                "radio": cell.get("radio"),
                "mcc": cell.get("mcc"),
                "mnc": cell.get("mnc"),
                "lac": cell.get("lac"),
                "samples": cell.get("samples"),
                "range": cell.get("range"),
            },
        }
        clipped = clip_geojson_feature(feature, mask)
        if clipped is None:
            continue
        features.append(clipped)

    return {
        "summary": record.summary,
        "feature_count": len(features),
        "features": features[:feature_limit],
    }


def _clip_population_dataset(record, mask, feature_limit: int = 30):
    features = []
    population_total = 0
    source_population_total = int(record.data.get("population_total", 0) or 0)

    for feature in record.data.get("features", []):
        clipped = clip_geojson_feature(feature, mask)
        if clipped is None:
            continue

        try:
            source_geometry = geojson_to_shape(feature.get("geometry", {}))
            clipped_geometry = geojson_to_shape(clipped.get("geometry", {}))
        except Exception:
            continue

        if (
            source_geometry.is_empty
            or clipped_geometry.is_empty
            or source_geometry.area <= 0
            or clipped_geometry.area <= 0
        ):
            continue

        overlap_ratio = clipped_geometry.area / source_geometry.area
        if overlap_ratio <= 0:
            continue

        source_population = int(feature.get("properties", {}).get("population", 0) or 0)
        estimated_population = round(source_population * overlap_ratio)
        if source_population > 0 and estimated_population == 0:
            estimated_population = 1

        properties = clipped.setdefault("properties", {})
        properties["population_source"] = source_population
        properties["population"] = estimated_population
        properties["overlap_ratio"] = round(overlap_ratio, 4)
        population_total += estimated_population
        features.append(clipped)

    return {
        "summary": record.summary,
        "feature_count": len(features),
        "population_total": population_total,
        "population_source_total": source_population_total,
        "population_coverage_ratio": round(population_total / source_population_total, 4) if source_population_total else 0,
        "features": features[:feature_limit],
    }


# Sensor swath half-widths (km) keyed by substring of the satellite type string.
# Used by _clip_satellite_record to decide whether a ground-track point's
# footprint sweeps the AOI. Values are conservative — actual swath varies with
# off-nadir angle, but this gives the LLM and UI a faithful first-cut.
_SAT_SWATH_HALFWIDTH_KM = {
    "sar": 40.0,          # Sentinel-1, TerraSAR-X, Russian SAR (X/C-band)
    "kh-11": 30.0,        # Hi-res optical reconnaissance
    "worldview": 15.0,    # commercial hi-res
    "kompsat": 20.0,
    "geoeye": 15.0,
    "pleiades": 20.0,
    "sentinel-2": 145.0,  # wide multispectral swath
    "landsat": 92.5,      # 185 km swath
    "resurs": 30.0,
    "kosmos": 30.0,       # Russian military imaging (default)
}
_SAT_SWATH_DEFAULT_KM = 50.0


def _sat_swath_halfwidth_km(sat_type: str) -> float:
    t = (sat_type or "").lower()
    for key, hw in _SAT_SWATH_HALFWIDTH_KM.items():
        if key in t:
            return hw
    return _SAT_SWATH_DEFAULT_KM


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    r_lat1, r_lat2 = math.radians(lat1), math.radians(lat2)
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(r_lat1) * math.cos(r_lat2) * math.sin(d_lon / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(a)))


def _clip_satellite_record(record, mask) -> dict[str, Any]:
    """Compute satellite overpass windows that intersect the AOI via the
    sensor's modeled swath. Emits a structured per-satellite pass schedule the
    LLM and UI can render directly."""
    from datetime import datetime, timezone, timedelta
    from ipb_backend.ingestion.sources.satellites import SatelliteTleAdapter
    from ipb_backend.ingestion.timeframe import forecast_horizon_hours

    # AOI centroid + extent (degrees → rough km radius via the bbox diagonal)
    minx, miny, maxx, maxy = mask.bounds
    aoi_lat = (miny + maxy) / 2.0
    aoi_lon = (minx + maxx) / 2.0
    aoi_radius_km = max(
        _haversine_km(aoi_lat, aoi_lon, miny, minx),
        _haversine_km(aoi_lat, aoi_lon, maxy, maxx),
    )

    satellites = record.data.get("satellites", {}) or {}
    query = record.data.get("query", {}) or {}
    hours = forecast_horizon_hours(
        record.timeframe or "24h", default=24.0, cap=48.0
    )
    now = datetime.now(timezone.utc)
    window_end = now + timedelta(hours=hours)

    # We need an adapter instance to call compute_ground_track. Build a
    # lightweight one — it has no per-instance state we depend on.
    adapter = SatelliteTleAdapter.__new__(SatelliteTleAdapter)

    sat_summaries: list[dict[str, Any]] = []
    total_passes = 0

    for name, info in satellites.items():
        tle1 = info.get("tle_line_1", "")
        tle2 = info.get("tle_line_2", "")
        if not tle1 or not tle2:
            continue

        sat_type = info.get("type", "")
        is_sar = "sar" in sat_type.lower()
        swath_km = _sat_swath_halfwidth_km(sat_type) + aoi_radius_km

        track = adapter.compute_ground_track(tle1, tle2, now, hours=hours, step_seconds=60)
        if not track:
            continue

        # Walk the track, group consecutive in-swath samples into passes.
        passes: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for pt in track:
            d = _haversine_km(aoi_lat, aoi_lon, pt["lat"], pt["lon"])
            inside = d <= swath_km
            if inside:
                if current is None:
                    current = {
                        "start_utc": pt["t_iso"],
                        "end_utc": pt["t_iso"],
                        "closest_km": d,
                        "closest_utc": pt["t_iso"],
                    }
                else:
                    current["end_utc"] = pt["t_iso"]
                    if d < current["closest_km"]:
                        current["closest_km"] = d
                        current["closest_utc"] = pt["t_iso"]
            elif current is not None:
                passes.append(current)
                current = None
        if current is not None:
            passes.append(current)

        if not passes:
            continue

        # Annotate each pass with duration and round closest distance
        for p in passes:
            from datetime import datetime as _dt
            try:
                dur = (_dt.fromisoformat(p["end_utc"]) - _dt.fromisoformat(p["start_utc"])).total_seconds()
            except ValueError:
                dur = 0
            p["duration_seconds"] = int(dur)
            p["closest_km"] = round(p["closest_km"], 1)

        sat_summaries.append({
            "name": name,
            "type": sat_type,
            "is_sar": is_sar,
            "origin": info.get("origin", ""),
            "swath_halfwidth_km": _sat_swath_halfwidth_km(sat_type),
            "passes": passes,
            "pass_count": len(passes),
            "next_pass_utc": passes[0]["start_utc"],
        })
        total_passes += len(passes)

    # Sort satellites by their next pass time
    sat_summaries.sort(key=lambda s: s["next_pass_utc"])

    return {
        "summary": record.summary,
        "provider": record.data.get("provider"),
        "window": {
            "start_utc": now.isoformat().replace("+00:00", "Z"),
            "end_utc": window_end.isoformat().replace("+00:00", "Z"),
            "hours": hours,
        },
        "aoi": {
            "centroid": [aoi_lat, aoi_lon],
            "radius_km": round(aoi_radius_km, 1),
        },
        "satellites_with_passes": sat_summaries,
        "total_passes_in_window": total_passes,
        "satellites_total_tracked": len(satellites),
        # Also forward the raw catalog for downstream tooling that wants TLE
        # info, but trim it (no need to ship full TLE bodies to the LLM).
        "satellites_catalog": {
            n: {"type": info.get("type"), "origin": info.get("origin"), "norad_id": info.get("norad_id")}
            for n, info in satellites.items()
        },
    }


def _clip_weather_record(record, mask):
    station = record.data.get("station", {})
    latitude = station.get("latitude")
    longitude = station.get("longitude")
    # Forecast + query are always forwarded so the analyzer and LLM context can
    # reason about the full planning horizon, not just the latest observation.
    forecast = record.data.get("forecast", {})
    query = record.data.get("query", {})

    if latitude is None or longitude is None:
        return {
            "summary": record.summary,
            "station_count": 0,
            "stations": [],
            "station": station,
            "observations": record.data.get("observations", {}),
            "forecast": forecast,
            "query": query,
        }

    station_feature = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [longitude, latitude],
        },
        "properties": {
            "name": station.get("name"),
            "region": station.get("region"),
            "distance_from_bbox_km": station.get("distance_from_bbox_km"),
            "fallback": station.get("fallback", False),
        },
    }
    clipped = clip_geojson_feature(station_feature, mask)
    return {
        "summary": record.summary,
        "station_count": 1 if clipped else 0,
        "stations": [clipped] if clipped else [station_feature],
        "station": station,
        "observations": record.data.get("observations", {}),
        "forecast": forecast,
        "query": query,
    }


def _is_population_dataset(record: DatasetRecord) -> bool:
    if record.category != SourceCategory.DEMOGRAPHICS:
        return False

    for feature in record.data.get("features", []):
        if "population" in feature.get("properties", {}):
            return True

    return False


def _clip_record_for_aoi(record: DatasetRecord, mask, source_name: Optional[str] = None) -> dict[str, Any]:
    if record.data.get("collections"):
        payload = _clip_nls_record(record, mask)
    elif record.data.get("station") or record.data.get("observations"):
        payload = _clip_weather_record(record, mask)
    elif record.data.get("categories"):
        payload = _clip_osm_poi_record(record, mask)
    elif record.data.get("cells"):
        payload = _clip_cell_tower_record(record, mask)
    elif record.data.get("satellites"):
        payload = _clip_satellite_record(record, mask)
    elif record.data.get("features") and _is_population_dataset(record):
        payload = _clip_population_dataset(record, mask)
    elif record.data.get("features"):
        payload = _clip_feature_dataset(record, mask)
    else:
        payload = {"summary": record.summary}

    payload["source_id"] = record.source_id
    payload["category"] = record.category.value
    if source_name:
        payload["title"] = source_name
    note = str(record.data.get("note", "") or "")
    fallback_used = "demo" in note.lower() or "fallback" in note.lower() or "demo data" in str(record.data.get("provider", "")).lower()
    payload["provenance"] = {
        "provider": str(record.data.get("provider") or source_name or record.source_id),
        "adapter": type(record).__name__,
        "retrieved_at": record.retrieved_at,
        "fallback_used": fallback_used,
        "fallback_reason": "demo-fallback" if fallback_used else None,
        "deterministic": True,
        "note": note or None,
    }
    return payload


def _build_freshness(services, latest_records):
    freshness = []
    seen_source_ids = set()
    for source in services["registry"].list_sources():
        seen_source_ids.add(source.source_id)
        record = latest_records.get(source.source_id)
        freshness.append(
            {
                "source_id": source.source_id,
                "name": source.name,
                "status": source.status,
                "category": source.category.value,
                "last_successful_refresh": source.last_successful_refresh,
                "last_error": source.last_error,
                "retrieved_at": record.retrieved_at if record else None,
                "refresh_interval_seconds": source.refresh_interval_seconds,
                "freshness_label": "fresh" if record is not None and source.last_error is None else "stale",
            }
        )

    for source_id, record in sorted(latest_records.items()):
        if source_id in seen_source_ids:
            continue
        freshness.append(
            {
                "source_id": source_id,
                "name": str(record.data.get("provider") or source_id),
                "status": "ready",
                "category": record.category.value,
                "last_successful_refresh": record.retrieved_at,
                "last_error": None,
                "retrieved_at": record.retrieved_at,
                "refresh_interval_seconds": None,
                "freshness_label": "fresh",
            }
        )
    return freshness


# AOI snapshot cache — keyed by (geometry_hash, timeframe).
# TTL matches the fastest-refreshing source (FMI = 900 s) so cached results
# are never staler than a fresh ingest would produce.
_AOI_CACHE_TTL_S = 120.0
_AOI_CACHE_MAX_ENTRIES = 8
_aoi_cache: dict[str, tuple[float, Any]] = {}  # key → (expires_at, result)


def _aoi_cache_key(request: AoiInspectionRequest) -> str:
    payload = json.dumps(
        {"geometry": request.geometry, "timeframe": request.timeframe or "latest", "area": request.area or ""},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _prune_aoi_cache(now: float) -> None:
    expired_keys = [key for key, (expires_at, _) in _aoi_cache.items() if expires_at <= now]
    for key in expired_keys:
        _aoi_cache.pop(key, None)

    while len(_aoi_cache) > _AOI_CACHE_MAX_ENTRIES:
        oldest_key = next(iter(_aoi_cache))
        _aoi_cache.pop(oldest_key, None)


def _build_aoi_snapshot(request: AoiInspectionRequest, services) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], DataPackage]:
    cache_key = _aoi_cache_key(request)
    now = time.monotonic()
    _prune_aoi_cache(now)
    cached = _aoi_cache.get(cache_key)
    if cached is not None and cached[0] > now:
        return cached[1]

    mask = geojson_to_shape(request.geometry)
    if mask.is_empty:
        raise HTTPException(status_code=400, detail="AOI geometry is empty")
    if mask.geom_type == "MultiPolygon":
        mask = max(getattr(mask, "geoms", [mask]), key=lambda geom: geom.area)
    elif mask.geom_type != "Polygon":
        raise HTTPException(status_code=400, detail="AOI geometry must be a Polygon or MultiPolygon")

    all_records = services["ingestion_service"].records
    source_names = {
        definition.source_id: definition.name
        for definition in services["registry"].list_sources()
    }
    latest_records = _latest_records_by_source(all_records)
    candidate_source_ids = list(source_names)
    candidate_source_ids.extend(
        source_id for source_id in sorted(latest_records) if source_id not in source_names
    )

    raw_data: dict[str, Any] = {}
    for source_id in candidate_source_ids:
        record = _record_for_area_or_latest(all_records, source_id, request.area or None)
        if record is None:
            record = _record_for_area_or_latest(all_records, source_id)
        if record is None:
            continue
        raw_data[source_id] = _clip_record_for_aoi(
            record,
            mask,
            source_name=source_names.get(source_id) or str(record.data.get("provider") or source_id),
        )

    selection = {
        "geometry": request.geometry,
        "bounds": list(mask.bounds),
        "area_sqkm": polygon_area_sqkm(mask),
    }
    freshness = _build_freshness(services, latest_records)
    metrics = build_aoi_metrics(selection["area_sqkm"], raw_data)
    evidence_bundle = build_evidence_bundle(metrics, raw_data)
    data_package = build_data_package(
        selection=selection,
        timeframe=request.timeframe or "latest",
        raw_data=raw_data,
        freshness=freshness,
        requested_sources=list(latest_records.keys()),
    )
    result = (selection, raw_data, freshness, metrics, evidence_bundle, data_package)
    _aoi_cache[cache_key] = (now + _AOI_CACHE_TTL_S, result)
    _prune_aoi_cache(now)
    return result


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/analysis/health")
async def analysis_health():
    return await get_analyzer_health()


@router.get("/sources")
async def list_sources(services=Depends(get_services)):
    return services["registry"].list_sources()


@router.post("/ingest", status_code=202)
async def ingest(request: IngestionRequest, background_tasks: BackgroundTasks, services=Depends(get_services)):
    try:
        source_ids = request.source_ids or services["registry"].enabled_source_ids()
        missing = services["registry"].missing_source_ids(source_ids)
        if missing:
            raise HTTPException(status_code=400, detail=f"Unknown source ids: {', '.join(sorted(missing))}")
        background_tasks.add_task(services["ingestion_service"].ingest, request)
        return {"status": "started", "source_ids": source_ids}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/datasets")
async def list_datasets(services=Depends(get_services)):
    return services["ingestion_service"].records


@router.get("/weather/current")
async def current_weather(area: str = Query("North Karelia"), services=Depends(get_services)):
    record = _record_for_area_or_latest(services["ingestion_service"].records, "fmi", area)
    return {"record": record}


@router.get("/ui-placeholder", response_model=UiPlaceholderResponse)
async def ui_placeholder(area: str, timeframe: str):
    return UiPlaceholderResponse(
        area=area,
        timeframe=timeframe,
        map_layers=[
            UiLayer(
                layer_id="terrain",
                title="Terrain and Topography",
                category=SourceCategory.TERRAIN,
                description="Elevation, land cover, routes, and water bodies.",
            ),
            UiLayer(
                layer_id="infrastructure",
                title="Infrastructure",
                category=SourceCategory.INFRASTRUCTURE,
                description="Roads, bridges, power, healthcare, and communications.",
            ),
            UiLayer(
                layer_id="population",
                title="Population",
                category=SourceCategory.DEMOGRAPHICS,
                description="Population density and civic context.",
            ),
        ],
        dashboard_cards=[
            "Source availability",
            "Recent refresh status",
            "Terrain summary",
            "Weather impact summary",
        ],
    )


@router.get("/map-tiles/{layer}/{z}/{x}/{y}.png")
async def proxy_tile(layer: str, z: int, x: int, y: int):
    if layer not in NLS_TILE_LAYERS:
        return Response(status_code=404)
    if not settings.nls_api_key:
        return Response(status_code=502, content="NLS API key not configured")
    nls_url = f"{NLS_TILE_URL.format(layer=layer, z=z, y=y, x=x)}?api-key={settings.nls_api_key}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(nls_url)
        return Response(content=resp.content, media_type=resp.headers.get("content-type", "image/png"))


@router.get("/map-layers")
async def map_layers():
    return {
        "tile_layers": [
            {
                "id": layer_id,
                "name": name,
                "proxy_url": f"/api/map-tiles/{layer_id}/{{z}}/{{x}}/{{y}}.png",
                "type": "wmts",
            }
            for layer_id, name in NLS_TILE_LAYERS.items()
        ],
        "default_tiles": {
            "id": "osm",
            "name": "OpenStreetMap",
            "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "attribution": "&copy; OpenStreetMap contributors",
            "type": "xyz",
        },
    }


@router.get("/map-data/nls")
async def map_data_nls(
    area: str = Query("North Karelia"),
    bbox: str | None = Query(None, description="WGS84 viewport filter: west,south,east,north"),
    services=Depends(get_services),
):
    records = services["ingestion_service"].records
    nls_record = _record_for_area_or_latest(records, "nls", area)
    if nls_record is None:
        return {"type": "FeatureCollection", "features": [], "available": False, "reason": "missing"}

    note = str(nls_record.data.get("note", "") or "")
    is_demo_fallback = "Demo spatial fallback" in note

    # Build a clip mask from the record's own load bbox so that municipality polygons
    # (which the NLS API returns clipped only at its tile/feature boundary, not at our bbox)
    # are confined to the requested rectangle.
    CLIP_TO_BBOX_COLLECTIONS = {"kunta", "kunnanhallintoraja"}
    raw_bbox_str = nls_record.data.get("query", {}).get("bbox_wgs84")
    load_bbox = parse_bbox_param(raw_bbox_str) if raw_bbox_str else None
    load_mask = bbox_to_mask(load_bbox) if load_bbox else None

    collections = nls_record.data.get("collections", {})
    features = []
    for coll_id, coll_data in collections.items():
        label = coll_data.get("label", coll_id)
        for sample in coll_data.get("features", []):
            if "geometry" in sample and sample["geometry"]:
                props = sample.get("properties", {})
                props["_collection"] = coll_id
                props["_label"] = label
                feature = {
                    "type": "Feature",
                    "geometry": sample["geometry"],
                    "properties": props,
                }
                if coll_id in CLIP_TO_BBOX_COLLECTIONS and load_mask is not None:
                    feature = clip_geojson_feature(feature, load_mask)
                    if feature is None:
                        continue
                features.append(feature)
    features = filter_features_by_bbox(features, parse_bbox_param(bbox))
    payload = {"type": "FeatureCollection", "features": features, "available": True}
    if is_demo_fallback:
        payload["reason"] = "demo-fallback"
        payload["message"] = "Showing demo NLS vector overlays because NLS_API_KEY is not configured."
    return payload


@router.get("/map-data/digiroad")
async def map_data_digiroad(
    area: str = Query("North Karelia"),
    bbox: str | None = Query(None, description="WGS84 viewport filter: west,south,east,north"),
    services=Depends(get_services),
):
    records = services["ingestion_service"].records
    record = _record_for_area_or_latest(records, "digiroad", area)
    if record is None:
        return {"type": "FeatureCollection", "features": [], "available": False}
    collections = record.data.get("collections", {})

    # Build lookup tables: tielinkki_id -> restriction value for each limit collection
    def _build_limit_index(coll_id: str, value_key: str) -> dict[str, Any]:
        index: dict[str, Any] = {}
        for feat in collections.get(coll_id, {}).get("features", []):
            props = feat.get("properties") or {}
            link_id = props.get("tielinkki_id")
            if link_id is not None:
                index[str(link_id)] = props.get(value_key)
        return index

    mass_index = _build_limit_index("dr_max_massa", "massarajoitus")
    height_index = _build_limit_index("dr_max_korkeus", "korkeus")
    width_index = _build_limit_index("dr_max_leveys", "leveys")
    axle_index = _build_limit_index("dr_max_akselimassa", "akselimassarajoitus")

    SILTA_ALIK_LABEL = {1: "Bridge", 2: "Underpass", 3: "Tunnel"}
    SILTA_ALIK_COLLECTION = {1: "bridges", 2: "bridges", 3: "tunnels"}
    bridge_data = collections.get("dr_tielinkki_silta_alikulku_tunneli", {})
    features = []
    for sample in bridge_data.get("features", []):
        props = sample.get("properties") or {}
        silta_alik = props.get("silta_alik")
        if silta_alik not in (1, 2, 3):
            continue
        if "geometry" not in sample or not sample["geometry"]:
            continue
        link_id = str(props.get("tielinkki_id", ""))
        coll = SILTA_ALIK_COLLECTION[silta_alik]
        enriched = dict(props)
        enriched["_collection"] = coll
        enriched["_label"] = SILTA_ALIK_LABEL[silta_alik]
        enriched["structure_type"] = SILTA_ALIK_LABEL[silta_alik]
        enriched["max_mass_t"] = mass_index.get(link_id)
        enriched["max_height_m"] = height_index.get(link_id)
        enriched["max_width_m"] = width_index.get(link_id)
        enriched["max_axle_mass_t"] = axle_index.get(link_id)
        features.append({
            "type": "Feature",
            "geometry": sample["geometry"],
            "properties": enriched,
        })
    n_bridges = sum(1 for f in features if f["properties"]["_collection"] == "bridges")
    n_tunnels = len(features) - n_bridges
    return {"type": "FeatureCollection", "features": features, "available": True, "message": f"{n_bridges} bridges/underpasses, {n_tunnels} tunnels"}


@router.get("/map-data/opencellid")
async def map_data_opencellid(
    area: str = Query("North Karelia"),
    bbox: str | None = Query(None, description="WGS84 viewport filter: west,south,east,north"),
    services=Depends(get_services),
):
    records = services["ingestion_service"].records
    record = _record_for_area_or_latest(records, "opencellid", area)
    if record is None:
        return {"type": "FeatureCollection", "features": [], "available": False}
    cells = record.data.get("cells", [])
    features = []
    for cell in cells:
        lat = cell.get("lat")
        lon = cell.get("lon")
        if lat is None or lon is None:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "_collection": "celltowers",
                "_label": "Cell Tower",
                "cell_id": str(cell.get("cellid") or cell.get("cellId", "")),
                "radio": cell.get("radio", ""),
                "mcc": cell.get("mcc", ""),
                "mnc": cell.get("mnc", ""),
                "lac": cell.get("lac", ""),
                "samples": cell.get("samples", ""),
                "range": cell.get("range", ""),
            },
        })
    features = filter_features_by_bbox(features, parse_bbox_param(bbox))
    return {"type": "FeatureCollection", "features": features, "available": True}


@router.get("/map-data/osm-poi")
async def map_data_osm_poi(
    area: str = Query("North Karelia"),
    bbox: str | None = Query(None, description="WGS84 viewport filter: west,south,east,north"),
    services=Depends(get_services),
):
    records = services["ingestion_service"].records
    record = _record_for_area_or_latest(records, "osm-poi", area)
    categories = record.data.get("categories", {}) if record is not None else None
    if not categories:
        return {"type": "FeatureCollection", "features": [], "available": False}
    features = []
    for cat_id, items in categories.items():
        label = cat_id.replace("_", " ").title()
        for item in items:
            geom = item.get("geometry")
            if geom is None:
                lat = item.get("lat")
                lon = item.get("lon")
                if lat is None or lon is None:
                    continue
                geom = {"type": "Point", "coordinates": [lon, lat]}
            tags = item.get("tags", {})
            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "_collection": f"poi-{cat_id}",
                    "_label": label,
                    **tags,
                },
            })
    features = filter_features_by_bbox(features, parse_bbox_param(bbox))
    return {"type": "FeatureCollection", "features": features, "available": True}


@router.get("/map-data/satellites")
async def map_data_satellites(area: str = Query("North Karelia"), services=Depends(get_services)):
    records = services["ingestion_service"].records
    record = _record_for_area_or_latest(records, "satellites", area)
    if record is None or not _is_bbox_load_target(record.load_target):
        # Satellite overlay only renders for bbox-scoped loads — TLE data is
        # global, but the visual is too cluttered without an AOI focus.
        return {
            "type": "FeatureCollection",
            "features": [],
            "available": False,
            "message": "Satellite overlay activates after a load rectangle is drawn.",
        }
    satellites = record.data.get("satellites", {})
    query = record.data.get("query", {})
    center_lat = query.get("lat", 62.8)
    center_lon = query.get("lon", 30.2)
    features = []
    for name, info in satellites.items():
        passes = info.get("predicted_passes", [])
        next_pass = next((p for p in passes if p.get("pass_time_unix", 0) > 0), passes[0] if passes else {})
        lat = info.get("current_lat", center_lat)
        lon = info.get("current_lon", center_lon)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "_collection": "satellites",
                "_label": "Satellite",
                "name": name,
                "type": info.get("type", ""),
                "norad_id": info.get("norad_id"),
                "current_alt_km": info.get("current_alt_km"),
                "next_pass": next_pass.get("pass_time_utc", ""),
                "altitude_km": next_pass.get("altitude_km", ""),
                "has_position": "current_lat" in info,
                "origin": info.get("origin", "other"),
            },
        })
    return {
        "type": "FeatureCollection",
        "features": features,
        "available": True,
        "message": f"Tracking {len(features)} reconnaissance/imaging satellites",
    }


@router.get("/map-data/satellite-tracks")
async def map_data_satellite_tracks(
    area: str = Query("North Karelia"),
    hours: float | None = Query(None, description="Hours to project ground tracks forward"),
    timeframe: str | None = Query(None, description="Timeframe string (overridden by hours)"),
    services=Depends(get_services),
):
    """Ground track corridors for Russian satellites — the core concealment planning layer."""
    from ipb_backend.ingestion.sources.satellites import SatelliteTleAdapter
    from datetime import datetime, timezone
    from ipb_backend.ingestion.timeframe import forecast_horizon_hours

    # Resolve hours: explicit > timeframe-derived > 8h default. Cap to 24h —
    # longer tracks alias around the globe and clutter the map.
    if hours is None:
        if timeframe:
            hours = forecast_horizon_hours(timeframe, default=8.0, cap=24.0)
        else:
            hours = 8.0
    hours = max(1.0, min(float(hours), 24.0))

    records = services["ingestion_service"].records
    record = _record_for_area_or_latest(records, "satellites", area)
    if record is None or not _is_bbox_load_target(record.load_target):
        return {
            "type": "FeatureCollection",
            "features": [],
            "available": False,
            "message": "Satellite ground tracks activate after a load rectangle is drawn.",
        }

    satellites = record.data.get("satellites", {})
    adapter = services["adapters"].get("satellites")
    if not isinstance(adapter, SatelliteTleAdapter):
        return {"type": "FeatureCollection", "features": [], "available": False}

    now = datetime.now(timezone.utc)
    features = []

    for name, info in satellites.items():
        if info.get("origin") != "russian":
            continue
        tle1 = info.get("tle_line_1", "")
        tle2 = info.get("tle_line_2", "")
        if not tle1 or not tle2:
            continue

        # 300s steps: ~5× fewer points, visually identical lines at map scale.
        points = adapter.compute_ground_track(tle1, tle2, now, hours=hours, step_seconds=300)
        if not points:
            continue

        # Split into segments at anti-meridian crossings
        segments: list[list[list[float]]] = []
        current: list[list[float]] = []
        for pt in points:
            if pt.get("crossing") and current:
                segments.append(current)
                current = []
            current.append([pt["lon"], pt["lat"]])
        if current:
            segments.append(current)

        sat_type = info.get("type", "")
        is_sar = "sar" in sat_type.lower()
        # Color encoded in properties so the frontend style function can read it
        line_color = "#e67e22" if is_sar else "#c0392b"

        # One feature per continuous segment
        for seg_coords in segments:
            if len(seg_coords) < 2:
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": seg_coords},
                "properties": {
                    "_collection": "satellite-tracks",
                    "name": name,
                    "type": sat_type,
                    "norad_id": info.get("norad_id"),
                    "track_start_utc": points[0]["t_iso"],
                    "track_end_utc": points[-1]["t_iso"],
                    "track_hours": hours,
                    "is_sar": is_sar,
                    "_color": line_color,
                },
            })

        # Time-label waypoints — frequency scales with horizon so labels don't
        # blanket the map at long timeframes. Points list is in 300s steps.
        if hours <= 3:
            label_step = 3   # every 15 min
        elif hours <= 8:
            label_step = 6   # every 30 min
        else:
            label_step = 12  # every 60 min
        for i, pt in enumerate(points):
            if i % label_step == 0 and not pt.get("crossing"):
                t_label = pt["t_iso"][11:16] + "Z"
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [pt["lon"], pt["lat"]]},
                    "properties": {
                        "_collection": "satellite-tracks",
                        "feature_type": "time_label",
                        "t_label": t_label,
                        "t_iso": pt["t_iso"],
                        "name": name,
                        "is_sar": is_sar,
                        "_color": line_color,
                    },
                })

    return {
        "type": "FeatureCollection",
        "features": features,
        "available": True,
        "russian_count": sum(1 for f in features if f["properties"].get("name")),
    }


@router.get("/map-data/road-surface")
async def map_data_road_surface(
    area: str = Query("North Karelia"),
    bbox: str | None = Query(None, description="WGS84 viewport filter: west,south,east,north"),
    services=Depends(get_services),
):
    records = services["ingestion_service"].records
    record = _record_for_area_or_latest(records, "digitraffic-road-surface", area)
    if record is None:
        return {"type": "FeatureCollection", "features": [], "available": False}
    raw_features = record.data.get("features", [])
    features = []
    for f in raw_features:
        props = dict(f.get("properties", {}))
        props["_collection"] = "road-surface"
        props["_label"] = "Road Surface Station"
        features.append({
            "type": "Feature",
            "geometry": f.get("geometry"),
            "properties": props,
        })
    features = filter_features_by_bbox(features, parse_bbox_param(bbox))
    return {"type": "FeatureCollection", "features": features, "available": True,
            "message": f"{len(features)} road surface stations"}


@router.get("/weather/point")
async def weather_point(lat: float = Query(...), lon: float = Query(...), timeframe: str = Query("24h"), services=Depends(get_services)):
    adapter: FmiAdapter = services["adapters"].get("fmi")
    if not adapter:
        return {"error": "FMI adapter not available"}
    result = await adapter.fetch_point_weather(lat, lon, timeframe)
    return result


def _nearby_context(lat: float, lon: float, records) -> dict[str, Any]:
    from ipb_backend.spatial import nearby_index
    RADIUS_DEG = 0.018  # ~2 km at 60°N
    return nearby_index.query_radius(lat, lon, RADIUS_DEG)


async def _build_point_snapshot(lat: float, lon: float, timeframe: str, services) -> dict[str, Any]:
    from ipb_backend.terrain.elevation import build_elevation_provider, compute_radial_los

    elevation_provider = build_elevation_provider(settings.nls_api_key)
    fmi: FmiAdapter = services["adapters"].get("fmi")

    async def _get_weather() -> dict[str, Any]:
        if not fmi:
            return {}
        try:
            return await fmi.fetch_point_weather(lat, lon, timeframe)
        except Exception:
            return {}

    # All three external calls are independent — run concurrently.
    elevation_m, weather, los = await asyncio.gather(
        elevation_provider.get_elevation(lat, lon),
        _get_weather(),
        compute_radial_los(lat, lon, settings.nls_api_key),
    )

    terrain = TerrainSnapshot(
        elevation_m=elevation_m,
        elevation_source="nls_dem_2m",
        available=elevation_m is not None,
    )

    # _nearby_context now uses the in-memory STRtree index — no I/O.
    nearby = _nearby_context(lat, lon, services["ingestion_service"].records)

    parts: list[str] = []
    if elevation_m is not None:
        parts.append(f"elevation {elevation_m} m")
    obs = weather.get("observations", {}).get("observations", {})
    temp = (obs.get("temperature") or {}).get("latest", {}).get("value")
    if temp is not None:
        parts.append(f"temp {temp}°C")
    wind = (obs.get("wind_speed") or {}).get("latest", {}).get("value")
    if wind is not None:
        parts.append(f"wind {wind} m/s")
    poi_total = sum(nearby.get("poi_counts", {}).values())
    if poi_total:
        parts.append(f"{poi_total} nearby POIs")
    if los.get("available"):
        los_summary = los.get("summary", "")
        if los_summary:
            parts.append(f"LOS: {los_summary}")
    summary = f"Point ({lat:.4f}, {lon:.4f}): {', '.join(parts) if parts else 'no data available'}"

    return {
        "lat": lat,
        "lon": lon,
        "terrain": terrain,
        "weather": weather,
        "nearby_context": nearby,
        "los": los,
        "summary": summary,
    }


@router.post("/point/inspect", response_model=PointInspectionResponse)
async def inspect_point(request: PointInspectionRequest, services=Depends(get_services)):
    try:
        snapshot = await _build_point_snapshot(request.lat, request.lon, request.timeframe, services)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Point inspection failed: {exc}") from exc
    return snapshot


@router.post("/aoi/inspect", response_model=AoiInspectionResponse)
async def inspect_aoi(request: AoiInspectionRequest, services=Depends(get_services)):
    try:
        selection, raw_data, freshness, metrics, evidence_bundle, data_package = _build_aoi_snapshot(request, services)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid AOI geometry: {exc}") from exc

    analyzer = build_analyzer()
    llm_input = build_llm_wrapper_input(
        data_package=data_package,
        profile=request.profile,
    )
    try:
        agent = await analyzer.analyze(
            data_package=data_package.model_dump(mode="json"),
            llm_input=llm_input.model_dump(mode="json"),
            profile=request.profile,
            selection=selection,
            metrics=metrics,
            raw_data=raw_data,
            freshness=freshness,
            evidence_bundle=evidence_bundle,
        )
    except Exception as exc:
        fallback = RulesAnalyzer()
        agent = await fallback.analyze(
            data_package=data_package.model_dump(mode="json"),
            llm_input=llm_input.model_dump(mode="json"),
            profile=request.profile,
            selection=selection,
            metrics=metrics,
            raw_data=raw_data,
            freshness=freshness,
            evidence_bundle=evidence_bundle,
        )
        agent["status"] = "fallback"
        agent["error"] = str(exc)

    return AoiInspectionResponse(
        selection=selection,
        metrics=metrics,
        raw_data=raw_data,
        raw_sections=build_raw_sections(raw_data),
        freshness=freshness,
        data_package=data_package,
        llm_input=llm_input,
        llm_output=LlmAnalysisOutput.model_validate(agent.get("output", {})),
        agent=agent,
    )


@router.post("/aoi/data-package", response_model=DataPackage)
async def aoi_data_package(request: AoiInspectionRequest, services=Depends(get_services)):
    try:
        _, _, _, _, _, data_package = _build_aoi_snapshot(request, services)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid AOI geometry: {exc}") from exc
    return data_package


@router.post("/aoi/interpret", response_model=LlmAnalysisOutput)
async def aoi_interpret(request: LlmInterpretRequest):
    analyzer = build_analyzer()
    llm_input = build_llm_wrapper_input(
        data_package=request.data_package,
        profile=request.profile,
        question=request.question,
        conversation_history=request.conversation_history,
    )
    try:
        result = await analyzer.analyze(
            data_package=request.data_package.model_dump(mode="json"),
            llm_input=llm_input.model_dump(mode="json"),
            profile=request.profile,
            question=request.question,
            conversation_history=[item.model_dump(mode="json") for item in request.conversation_history],
        )
    except Exception as exc:
        fallback = RulesAnalyzer()
        result = await fallback.analyze(
            data_package=request.data_package.model_dump(mode="json"),
            llm_input=llm_input.model_dump(mode="json"),
            profile=request.profile,
            question=request.question,
            conversation_history=[item.model_dump(mode="json") for item in request.conversation_history],
        )
        result["status"] = "fallback"
        result["error"] = str(exc)
    return LlmAnalysisOutput.model_validate(result.get("output", {}))


@router.post("/aoi/intsum", response_class=HTMLResponse)
async def aoi_intsum(request: LlmInterpretRequest):
    """Generate a NATO-style INTSUM document (print-ready HTML) for the AOI."""
    data_package_dict = request.data_package.model_dump(mode="json")
    raw_data = _build_raw_data_from_package(data_package_dict)

    provider = "rules"
    model: Optional[str] = None
    sections: dict[str, str]

    analyzer = build_analyzer()
    if isinstance(analyzer, ClaudeAnalyzer):
        try:
            result = await analyzer.generate_intsum(
                data_package=data_package_dict,
                raw_data=raw_data,
            )
            sections = result["sections"]
            provider = "claude"
            model = result.get("model")
        except Exception:
            sections = _rules_intsum_sections(data_package_dict, raw_data)
    else:
        sections = _rules_intsum_sections(data_package_dict, raw_data)

    selection = data_package_dict.get("selection") or {}
    html = render_intsum_html(
        sections=sections,
        selection=selection,
        provider=provider,
        model=model,
    )
    return HTMLResponse(content=html)


@router.get("/analysis/profiles")
async def analysis_profiles():
    return list_profile_specs()


@router.get("/ui-demo", response_class=HTMLResponse)
async def ui_demo():
    return UI_PLACEHOLDER_PATH.read_text(encoding="utf-8")


@router.get("/agents")
async def list_agents():
    return [
        AgentDefinition(
            agent_id="summary-agent",
            name="Summary Agent",
            purpose="Placeholder derived analysis over normalized datasets.",
            status="placeholder",
        ),
        AgentDefinition(
            agent_id="celltower-agent",
            name="Cell Tower Agent",
            purpose="Analyzes cell tower coverage, operators, and technologies from OpenCellID data.",
            status="active",
        ),
        AgentDefinition(
            agent_id="satellite-agent",
            name="Satellite Agent",
            purpose="Tracks reconnaissance and imaging satellite overpass schedules for surveillance windows.",
            status="active",
        ),
        AgentDefinition(
            agent_id="bridge-load-agent",
            name="Bridge Load Capacity Agent",
            purpose="Analyzes bridge/tunnel weight, height, and width limits from Digiroad data for military route viability.",
            status="active",
        ),
        AgentDefinition(
            agent_id="demographics-agent",
            name="Demographics Agent",
            purpose="Analyzes population, age distribution, sex distribution, and urban/rural classification per municipality.",
            status="active",
        ),
        AgentDefinition(
            agent_id="forest-concealment-agent",
            name="Forest Concealment Agent",
            purpose="Assesses concealment potential from forest/woodland cover using OSM leaf_type and leaf_cycle data.",
            status="active",
        ),
        AgentDefinition(
            agent_id="weather-impact-agent",
            name="Weather Impact Agent",
            purpose="Analyzes how current weather conditions affect drone ops, surveillance, mobility, and visibility.",
            status="active",
        ),
        AgentDefinition(
            agent_id="power-grid-agent",
            name="Power Grid Agent",
            purpose="Analyzes power line infrastructure density and identifies chokepoints for logistics assessment.",
            status="active",
        ),
    ]


@router.post("/agents/{agent_id}/run")
async def run_agent(agent_id: str, area: str, timeframe: str, services=Depends(get_services)):
    if agent_id == "summary-agent":
        return await SummaryAgent().run(area=area, timeframe=timeframe)
    if agent_id == "celltower-agent":
        adapter = services["adapters"].get("opencellid")
        if not adapter:
            return {"error": "Cell tower adapter not available"}
        return await CellTowerAgent(adapter).run(area=area, timeframe=timeframe)
    if agent_id == "satellite-agent":
        adapter = services["adapters"].get("satellites")
        if not adapter:
            return {"error": "Satellite adapter not available"}
        return await SatelliteAgent(adapter).run(area=area, timeframe=timeframe)
    if agent_id == "bridge-load-agent":
        adapter = services["adapters"].get("digiroad")
        if not adapter:
            return {"error": "Digiroad adapter not available"}
        return await BridgeLoadAgent(adapter).run(area=area, timeframe=timeframe)
    if agent_id == "demographics-agent":
        adapter = services["adapters"].get("statistics-finland")
        if not adapter:
            return {"error": "Statistics Finland adapter not available"}
        return await DemographicsAgent(adapter).run(area=area, timeframe=timeframe)
    if agent_id == "forest-concealment-agent":
        adapter = services["adapters"].get("osm-poi")
        if not adapter:
            return {"error": "OSM POI adapter not available"}
        return await ForestConcealmentAgent(adapter).run(area=area, timeframe=timeframe)
    if agent_id == "weather-impact-agent":
        adapter = services["adapters"].get("fmi")
        if not adapter:
            return {"error": "FMI adapter not available"}
        return await WeatherImpactAgent(adapter).run(area=area, timeframe=timeframe)
    if agent_id == "power-grid-agent":
        adapter = services["adapters"].get("nls")
        if not adapter:
            return {"error": "NLS adapter not available"}
        return await PowerGridAgent(adapter).run(area=area, timeframe=timeframe)
    return {"error": f"Unknown agent: {agent_id}"}


@router.get("/planning/profiles")
async def planning_profiles():
    return {
        "operation_types": [
            {"id": op_type.value, "weights": weights}
            for op_type, weights in OPERATION_PROFILES.items()
        ]
    }


# Planning is CPU-bound and runs synchronously in the thread executor — two
# concurrent runs starve each other and the event loop. Serialise them at the
# route level so the second caller queues instead of doubling up.
_planning_lock = asyncio.Lock()
_PLANNING_TIMEOUT_S = 180.0


@router.post("/planning/recommend", response_model=PlanningResponse)
async def planning_recommend(
    request: PlanningRequest, services=Depends(get_services)
) -> PlanningResponse:
    import logging
    import time as _time
    logger = logging.getLogger("ipb_backend.api.planning")
    try:
        records = services["ingestion_service"].records
        latest = _latest_records_by_source(records)
        freshness = _build_freshness(services, latest)

        async with _planning_lock:
            t_start = _time.perf_counter()
            logger.info(
                "planning recommend: area=%s grid_res=%s top_n=%s explain=%s",
                request.area, request.grid_resolution_m, request.top_n, request.explain,
            )
            try:
                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        partial(recommend_sites, request, records=list(latest.values()), freshness=freshness),
                    ),
                    timeout=_PLANNING_TIMEOUT_S,
                )
            except asyncio.TimeoutError as exc:
                raise HTTPException(
                    status_code=504,
                    detail=(
                        f"Planning exceeded {_PLANNING_TIMEOUT_S:.0f}s. "
                        "Try a coarser grid (grid_resolution_m) or a smaller AOI."
                    ),
                ) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            if request.explain:
                try:
                    response = await enrich_with_narratives(
                        response, request.force, request.operation
                    )
                except Exception as exc:
                    logger.warning("planning explainer failed: %s", exc)
                    response = response.model_copy(
                        update={
                            "notes": response.notes
                            + [f"Explainer disabled due to error: {exc}"]
                        }
                    )

            logger.info(
                "planning recommend done: cells=%d feasible=%d top_n=%d in %.1fs",
                response.cells_evaluated, response.feasible_cells, len(response.top_sites),
                _time.perf_counter() - t_start,
            )
            return response
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("planning recommend failed")
        raise HTTPException(status_code=500, detail=f"Planning failed: {exc}") from exc
