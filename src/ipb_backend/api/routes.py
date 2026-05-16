from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, Response

from ipb_backend.agents.placeholders import SummaryAgent
from ipb_backend.config import settings
from ipb_backend.models import (
    AgentDefinition,
    IngestionRequest,
    UiLayer,
    UiPlaceholderResponse,
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


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/sources")
async def list_sources(services=Depends(get_services)):
    return services["registry"].list_sources()


@router.post("/ingest")
async def ingest(request: IngestionRequest, services=Depends(get_services)):
    return await services["ingestion_service"].ingest(request)


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
                category="terrain",
                description="Elevation, land cover, routes, and water bodies.",
            ),
            UiLayer(
                layer_id="infrastructure",
                title="Infrastructure",
                category="infrastructure",
                description="Roads, bridges, power, healthcare, and communications.",
            ),
            UiLayer(
                layer_id="population",
                title="Population",
                category="demographics",
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
        return {"type": "FeatureCollection", "features": []}
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
    return {"type": "FeatureCollection", "features": features}


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
        )
    ]


@router.post("/agents/{agent_id}/run")
async def run_agent(agent_id: str, area: str, timeframe: str):
    if agent_id != "summary-agent":
        return {"error": f"Unknown agent: {agent_id}"}
    return await SummaryAgent().run(area=area, timeframe=timeframe)
