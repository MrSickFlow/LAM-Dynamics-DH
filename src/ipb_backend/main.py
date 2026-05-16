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
from ipb_backend.consistency.classification import get_source_profile
from ipb_backend.consistency.engine import DataConsistencyEngine
from ipb_backend.models import EwClassification, SourceCategory, SourceDefinition, SourceStatus
from ipb_backend.ingestion.sources.maritime_demo import MaritimeDemoAdapter


def _credential_gated_source(*, configured: bool, error_message: str) -> dict:
    if configured:
        return {}
    return {
        "enabled": False,
        "status": SourceStatus.DISABLED,
        "last_error": error_message,
    }


def _apply_ew_profile(definition: SourceDefinition) -> SourceDefinition:
    profile = get_source_profile(definition.source_id)
    return definition.model_copy(
        update={
            "ew_classification": EwClassification(profile.ew_classification.value),
            "gnss_dependent": profile.gnss_dependent,
            "ew_rationale": profile.rationale,
        }
    )


def build_registry() -> SourceRegistry:
    definitions = [
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
                **_credential_gated_source(
                    configured=bool(settings.nls_api_key),
                    error_message="Disabled until NLS_API_KEY is configured.",
                ),
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
                **_credential_gated_source(
                    configured=bool(settings.opencellid_api_key),
                    error_message="Disabled until OPENCELLID_API_KEY is configured.",
                ),
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
    if settings.consistency_maritime_demo:
        definitions.append(
            SourceDefinition(
                source_id="maritime-demo",
                name="Maritime AIS/SAR Demo",
                category=SourceCategory.OTHER,
                description="Demonstration AIS tracks and SAR returns for vessel cross-validation.",
                refresh_interval_seconds=3600,
            )
        )
    return SourceRegistry(definitions=[_apply_ew_profile(defn) for defn in definitions])


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
    if settings.consistency_maritime_demo:
        adapters["maritime-demo"] = MaritimeDemoAdapter(registry.get("maritime-demo"))
    ingestion_service = IngestionService(registry=registry, adapters=adapters)
    scheduler = RefreshScheduler(ingestion_service=ingestion_service)
    consistency_engine = DataConsistencyEngine(fmi_adapter=adapters.get("fmi"))
    return {
        "registry": registry,
        "ingestion_service": ingestion_service,
        "scheduler": scheduler,
        "adapters": adapters,
        "consistency_engine": consistency_engine,
        "last_consistency_report": None,
    }


state = build_services()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await state["scheduler"].start()
    yield
    await state["scheduler"].stop()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(router, prefix="/api")
