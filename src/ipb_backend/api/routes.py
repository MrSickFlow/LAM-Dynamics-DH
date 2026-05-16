from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
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
    RulesAnalyzer,
    build_analyzer,
    build_aoi_metrics,
    build_evidence_bundle,
    build_raw_sections,
    get_analyzer_health,
)
from ipb_backend.config import settings
from ipb_backend.ingestion.sources.fmi import FmiAdapter
from ipb_backend.models import (
    AoiInspectionRequest,
    AoiInspectionResponse,
    DatasetRecord,
    AgentDefinition,
    IngestionRequest,
    SourceCategory,
    UiLayer,
    UiPlaceholderResponse,
)
from ipb_backend.spatial import clip_geojson_feature, geojson_to_shape, polygon_area_sqkm

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


def _latest_records_by_source(records):
    latest = {}
    for record in records:
        current = latest.get(record.source_id)
        if current is None or record.retrieved_at > current.retrieved_at:
            latest[record.source_id] = record
    return latest


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


def _clip_weather_record(record, mask):
    station = record.data.get("station", {})
    latitude = station.get("latitude")
    longitude = station.get("longitude")
    if latitude is None or longitude is None:
        return {
            "summary": record.summary,
            "station_count": 0,
            "stations": [],
            "observations": record.data.get("observations", {}),
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
        },
    }
    clipped = clip_geojson_feature(station_feature, mask)
    return {
        "summary": record.summary,
        "station_count": 1 if clipped else 0,
        "stations": [clipped] if clipped else [],
        "observations": record.data.get("observations", {}),
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
    return payload


def _build_freshness(services, latest_records):
    freshness = []
    for source in services["registry"].list_sources():
        record = latest_records.get(source.source_id)
        freshness.append(
            {
                "source_id": source.source_id,
                "name": source.name,
                "status": source.status,
                "last_successful_refresh": source.last_successful_refresh,
                "last_error": source.last_error,
                "retrieved_at": record.retrieved_at if record else None,
            }
        )
    return freshness


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/analysis/health")
async def analysis_health():
    return await get_analyzer_health()


@router.get("/sources")
async def list_sources(services=Depends(get_services)):
    return services["registry"].list_sources()


@router.post("/ingest")
async def ingest(request: IngestionRequest, services=Depends(get_services)):
    try:
        return await services["ingestion_service"].ingest(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/datasets")
async def list_datasets(services=Depends(get_services)):
    return services["ingestion_service"].records


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
async def map_data_nls(area: str = Query("North Karelia"), services=Depends(get_services)):
    records = services["ingestion_service"].records
    nls_record = next((r for r in records if r.source_id == "nls" and r.area == area), None)
    if nls_record is None:
        return {"type": "FeatureCollection", "features": [], "available": False, "reason": "missing"}

    note = str(nls_record.data.get("note", "") or "")
    is_demo_fallback = "Demo spatial fallback" in note

    collections = nls_record.data.get("collections", {})
    features = []
    for coll_id, coll_data in collections.items():
        label = coll_data.get("label", coll_id)
        for sample in coll_data.get("features", []):
            if "geometry" in sample and sample["geometry"]:
                props = sample.get("properties", {})
                props["_collection"] = coll_id
                props["_label"] = label
                features.append({
                    "type": "Feature",
                    "geometry": sample["geometry"],
                    "properties": props,
                })
    payload = {"type": "FeatureCollection", "features": features, "available": True}
    if is_demo_fallback:
        payload["reason"] = "demo-fallback"
        payload["message"] = "Showing demo NLS vector overlays because NLS_API_KEY is not configured."
    return payload


@router.get("/weather/point")
async def weather_point(lat: float = Query(...), lon: float = Query(...), timeframe: str = Query("24h"), services=Depends(get_services)):
    adapter: FmiAdapter = services["adapters"].get("fmi")
    if not adapter:
        return {"error": "FMI adapter not available"}
    result = await adapter.fetch_point_weather(lat, lon, timeframe)
    return result


@router.post("/aoi/inspect", response_model=AoiInspectionResponse)
async def inspect_aoi(request: AoiInspectionRequest, services=Depends(get_services)):
    try:
        mask = geojson_to_shape(request.geometry)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid AOI geometry: {exc}") from exc

    if mask.is_empty:
        raise HTTPException(status_code=400, detail="AOI geometry is empty")
    if mask.geom_type == "MultiPolygon":
        mask = max(getattr(mask, "geoms", [mask]), key=lambda geom: geom.area)
    elif mask.geom_type != "Polygon":
        raise HTTPException(status_code=400, detail="AOI geometry must be a Polygon or MultiPolygon")

    latest_records = _latest_records_by_source(services["ingestion_service"].records)
    raw_data: dict[str, Any] = {}
    source_names = {
        definition.source_id: definition.name
        for definition in services["registry"].list_sources()
    }

    for record in latest_records.values():
        raw_data[record.source_id] = _clip_record_for_aoi(
            record,
            mask,
            source_name=source_names.get(record.source_id),
        )

    selection = {
        "geometry": request.geometry,
        "bounds": list(mask.bounds),
    }
    freshness = _build_freshness(services, latest_records)
    metrics = build_aoi_metrics(polygon_area_sqkm(mask), raw_data)
    evidence_bundle = build_evidence_bundle(metrics, raw_data)
    analyzer = build_analyzer()
    try:
        agent = await analyzer.analyze(
            selection=selection,
            metrics=metrics,
            raw_data=raw_data,
            freshness=freshness,
            evidence_bundle=evidence_bundle,
        )
    except Exception as exc:
        fallback = RulesAnalyzer()
        agent = await fallback.analyze(
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
        agent=agent,
    )


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
