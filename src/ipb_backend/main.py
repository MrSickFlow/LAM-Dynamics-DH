from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from ipb_backend.api.routes import router
from ipb_backend.config import settings
from ipb_backend.ingestion.registry import SourceRegistry
from ipb_backend.ingestion.scheduler import RefreshScheduler
from ipb_backend.ingestion.service import IngestionService
from ipb_backend.ingestion.sources.digiroad import DigiroadAdapter
from ipb_backend.ingestion.sources.fmi import FmiAdapter
from ipb_backend.ingestion.sources.nls import NationalLandSurveyAdapter
from ipb_backend.ingestion.sources.statistics_finland import StatisticsFinlandAdapter
from ipb_backend.models import SourceCategory, SourceDefinition


def build_registry() -> SourceRegistry:
    return SourceRegistry(
        definitions=[
            SourceDefinition(
                source_id="fmi",
                name="Finnish Meteorological Institute",
                category=SourceCategory.WEATHER,
                description="Current conditions, forecasts, and historical weather patterns.",
                refresh_interval_seconds=900,
            ),
            SourceDefinition(
                source_id="nls",
                name="National Land Survey of Finland",
                category=SourceCategory.TERRAIN,
                description="Topographic, elevation, and land cover datasets.",
                refresh_interval_seconds=86400,
            ),
            SourceDefinition(
                source_id="statistics-finland",
                name="Statistics Finland",
                category=SourceCategory.DEMOGRAPHICS,
                description="Population and demographic datasets.",
                refresh_interval_seconds=86400,
            ),
            SourceDefinition(
                source_id="digiroad",
                name="Digiroad",
                category=SourceCategory.INFRASTRUCTURE,
                description="Road and transport infrastructure datasets.",
                refresh_interval_seconds=43200,
            ),
        ]
    )


def build_services():
    registry = build_registry()
    adapters = {
        "fmi": FmiAdapter(registry.get("fmi")),
        "nls": NationalLandSurveyAdapter(registry.get("nls")),
        "statistics-finland": StatisticsFinlandAdapter(registry.get("statistics-finland")),
        "digiroad": DigiroadAdapter(registry.get("digiroad")),
    }
    ingestion_service = IngestionService(registry=registry, adapters=adapters)
    scheduler = RefreshScheduler(ingestion_service=ingestion_service)
    return {
        "registry": registry,
        "ingestion_service": ingestion_service,
        "scheduler": scheduler,
    }


state = build_services()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await state["scheduler"].start()
    yield
    await state["scheduler"].stop()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(router, prefix="/api")
