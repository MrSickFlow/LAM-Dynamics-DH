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
from ipb_backend.ingestion.sources.opencellid import OpenCellIdAdapter
from ipb_backend.ingestion.sources.osm_poi import OsmPoiAdapter
from ipb_backend.ingestion.sources.satellites import SatelliteTleAdapter
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
            SourceDefinition(
                source_id="opencellid",
                name="OpenCellID",
                category=SourceCategory.INFRASTRUCTURE,
                description="Cell tower locations, operators, and technologies.",
                refresh_interval_seconds=86400,
            ),
            SourceDefinition(
                source_id="osm-poi",
                name="OpenStreetMap POIs",
                category=SourceCategory.OTHER,
                description="Schools, hospitals, water sources, places of worship, government buildings, and other key institutions.",
                refresh_interval_seconds=86400,
            ),
            SourceDefinition(
                source_id="satellites",
                name="Satellite TLE Data",
                category=SourceCategory.SATELLITE,
                description="TLE orbital data for reconnaissance and imaging satellites.",
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
        "opencellid": OpenCellIdAdapter(registry.get("opencellid")),
        "osm-poi": OsmPoiAdapter(registry.get("osm-poi")),
        "satellites": SatelliteTleAdapter(registry.get("satellites")),
    }
    ingestion_service = IngestionService(registry=registry, adapters=adapters)
    scheduler = RefreshScheduler(ingestion_service=ingestion_service)
    return {
        "registry": registry,
        "ingestion_service": ingestion_service,
        "scheduler": scheduler,
        "adapters": adapters,
    }


state = build_services()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await state["scheduler"].start()
    yield
    await state["scheduler"].stop()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(router, prefix="/api")
