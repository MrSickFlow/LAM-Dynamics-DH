import httpx
import pytest
from shapely.geometry import box, shape

from ipb_backend.config import settings
from ipb_backend.ingestion.sources.digiroad import DigiroadAdapter
from ipb_backend.ingestion.sources.nls import NationalLandSurveyAdapter
from ipb_backend.ingestion.sources.osm_poi import OsmPoiAdapter
from ipb_backend.ingestion.sources.satellites import SatelliteTleAdapter
from ipb_backend.ingestion.sources.statistics_finland import StatisticsFinlandAdapter
from ipb_backend.models import SourceCategory, SourceDefinition


class FailingAsyncClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, params=None):
        request = httpx.Request("GET", url, params=params)
        raise httpx.ConnectError("network down", request=request)

    async def post(self, url, data=None):
        request = httpx.Request("POST", url, data=data)
        raise httpx.ConnectError("network down", request=request)


@pytest.mark.anyio
async def test_digiroad_fetch_raises_when_all_collections_fail(monkeypatch):
    definition = SourceDefinition(
        source_id="digiroad",
        name="Digiroad",
        category=SourceCategory.INFRASTRUCTURE,
        description="Road and transport infrastructure datasets.",
        refresh_interval_seconds=43200,
    )
    adapter = DigiroadAdapter(definition)

    monkeypatch.setattr("ipb_backend.ingestion.sources.digiroad.httpx.AsyncClient", lambda *args, **kwargs: FailingAsyncClient())

    with pytest.raises(ValueError, match="Digiroad fetch failed for all collections"):
        await adapter.fetch("North Karelia", "24h")


@pytest.mark.anyio
async def test_nls_fetch_raises_when_all_collections_fail(monkeypatch):
    definition = SourceDefinition(
        source_id="nls",
        name="National Land Survey of Finland",
        category=SourceCategory.TERRAIN,
        description="Topographic, elevation, and land cover datasets.",
        refresh_interval_seconds=86400,
    )
    adapter = NationalLandSurveyAdapter(definition)

    monkeypatch.setattr(settings, "nls_api_key", "configured")
    monkeypatch.setattr("ipb_backend.ingestion.sources.nls.httpx.AsyncClient", lambda *args, **kwargs: FailingAsyncClient())

    with pytest.raises(ValueError, match="NLS fetch failed for all collections"):
        await adapter.fetch("North Karelia", "24h")


@pytest.mark.anyio
async def test_osm_poi_fetch_falls_back_to_static_when_all_categories_fail(monkeypatch):
    definition = SourceDefinition(
        source_id="osm-poi",
        name="OpenStreetMap POIs",
        category=SourceCategory.OTHER,
        description="Key civilian and institutional locations.",
        refresh_interval_seconds=86400,
    )
    adapter = OsmPoiAdapter(definition)

    monkeypatch.setattr("ipb_backend.ingestion.sources.osm_poi.httpx.AsyncClient", lambda *args, **kwargs: FailingAsyncClient())

    record = await adapter.fetch("North Karelia", "24h")
    assert record.data.get("provider", "").startswith("OpenStreetMap")
    categories = record.data.get("categories", {})
    total = record.data.get("total_features", 0)
    assert total > 0


@pytest.mark.anyio
async def test_satellite_fetch_falls_back_to_demo_when_all_feeds_fail(monkeypatch):
    definition = SourceDefinition(
        source_id="satellites",
        name="Satellite TLE Data",
        category=SourceCategory.SATELLITE,
        description="TLE orbital data for reconnaissance and imaging satellites.",
        refresh_interval_seconds=43200,
    )
    adapter = SatelliteTleAdapter(definition)

    monkeypatch.setattr("ipb_backend.ingestion.sources.satellites.httpx.AsyncClient", lambda *args, **kwargs: FailingAsyncClient())

    record = await adapter.fetch("North Karelia", "24h")
    satellites = record.data.get("satellites", {})
    assert len(satellites) > 0
    assert "USA 224" in satellites or "Sentinel-2A" in satellites


def test_statistics_finland_grid_concentrates_joensuu_population():
    definition = SourceDefinition(
        source_id="statistics-finland",
        name="Statistics Finland",
        category=SourceCategory.DEMOGRAPHICS,
        description="Population and demographic datasets.",
        refresh_interval_seconds=86400,
    )
    adapter = StatisticsFinlandAdapter(definition)

    pop_data = {
        "total": 129500,
        "per_muni": {
            "KU167": {"total": 77000},
            "KU176": {"total": 4500},
            "KU260": {"total": 9800},
            "KU422": {"total": 10700},
            "KU426": {"total": 12200},
            "KU276": {"total": 15300},
        },
    }

    features = adapter._build_features("North Karelia", (29.0, 62.0, 31.5, 63.5), pop_data)

    assert len(features) == 576
    assert features[0]["geometry"] != features[1]["geometry"]

    joensuu_mask = box(29.68, 62.56, 29.98, 62.70)
    joensuu_population = 0
    for feature in features:
        geometry = shape(feature["geometry"])
        if geometry.intersects(joensuu_mask):
            joensuu_population += feature["properties"]["population"]

    assert joensuu_population >= 70000