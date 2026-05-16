from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from ipb_backend.agents.placeholders import SummaryAgent
from ipb_backend.models import (
    AgentDefinition,
    IngestionRequest,
    UiLayer,
    UiPlaceholderResponse,
)

router = APIRouter()
UI_PLACEHOLDER_PATH = Path(__file__).resolve().parents[1] / "ui_placeholder.html"


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
