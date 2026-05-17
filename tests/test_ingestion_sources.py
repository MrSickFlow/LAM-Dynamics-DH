import httpx
import pytest
from shapely.geometry import box, shape

from ipb_backend.config import settings
from ipb_backend.ingestion.sources.digiroad import DigiroadAdapter
from ipb_backend.ingestion.sources.nls import NationalLandSurveyAdapter
from ipb_backend.ingestion.sources.osm_poi import OsmPoiAdapter
from ipb_backend.ingestion.sources.satellites import SatelliteTleAdapter
from ipb_backend.ingestion.sources.statistics_finland import StatisticsFinlandAdapter
from ipb_backend.models import LoadTarget, LoadTargetKind, SourceCategory, SourceDefinition


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


class DigiroadCaptureClient:
    def __init__(self):
        self.get_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, params=None):
        self.get_calls.append((url, params))
        collection_id = url.rstrip("/").split("/")[-2]
        return httpx.Response(
            200,
            json={"numberMatched": 1, "features": [{"type": "Feature", "geometry": None, "properties": {"collection": collection_id}}]},
            request=httpx.Request("GET", url, params=params),
        )


@pytest.mark.anyio
async def test_digiroad_fetch_trims_custom_bbox_scope(monkeypatch):
    definition = SourceDefinition(
        source_id="digiroad",
        name="Digiroad",
        category=SourceCategory.INFRASTRUCTURE,
        description="Road and transport infrastructure datasets.",
        refresh_interval_seconds=43200,
    )
    adapter = DigiroadAdapter(definition)
    client = DigiroadCaptureClient()

    monkeypatch.setattr(
        "ipb_backend.ingestion.sources.digiroad.httpx.AsyncClient",
        lambda *args, **kwargs: client,
    )

    record = await adapter.fetch(
        "North Karelia",
        "24h",
        load_target=LoadTarget(
            kind=LoadTargetKind.BBOX,
            label="Custom Load Area",
            bbox_wgs84=[29.9, 62.4, 30.2, 62.7],
        ),
    )

    assert set(record.data["collections"]) == set(adapter.ESSENTIAL_COLLECTION_NAMES)
    assert record.data["query"]["feature_limit"] == adapter.BBOX_LIMIT
    assert all(call[1]["limit"] == adapter.BBOX_LIMIT for call in client.get_calls)
    assert len(client.get_calls) == len(adapter.ESSENTIAL_COLLECTION_NAMES)


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
async def test_osm_poi_fetch_raises_when_all_categories_fail(monkeypatch):
    definition = SourceDefinition(
        source_id="osm-poi",
        name="OpenStreetMap POIs",
        category=SourceCategory.OTHER,
        description="Key civilian and institutional locations.",
        refresh_interval_seconds=86400,
    )
    adapter = OsmPoiAdapter(definition)

    monkeypatch.setattr("ipb_backend.ingestion.sources.osm_poi.httpx.AsyncClient", lambda *args, **kwargs: FailingAsyncClient())

    with pytest.raises(ValueError, match="OSM POI fetch failed for all categories"):
        await adapter.fetch("North Karelia", "24h")


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

    assert len(features) == 2304
    assert features[0]["geometry"] != features[1]["geometry"]

    joensuu_mask = box(29.68, 62.56, 29.98, 62.70)
    joensuu_population = 0
    for feature in features:
        geometry = shape(feature["geometry"])
        if geometry.intersects(joensuu_mask):
            joensuu_population += feature["properties"]["population"]

    assert joensuu_population >= 70000


class StatisticsFinlandBBoxClient:
    def __init__(self):
        self.posts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, params=None):
        assert url.endswith("/collections/kunta/items")
        return httpx.Response(
            200,
            json={
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[24.0, 60.0], [24.4, 60.0], [24.4, 60.2], [24.0, 60.2], [24.0, 60.0]]],
                        },
                        "properties": {"kuntatunnus": 91},
                    }
                ]
            },
            request=httpx.Request("GET", url, params=params),
        )

    async def post(self, url, json=None):
        self.posts.append((url, json))
        query_codes = [entry["code"] for entry in json["query"]]
        if "Kaupunki-maaseutu-luokitus" in query_codes:
            return httpx.Response(
                200,
                json={"value": [100, 80, 0, 0, 0, 20, 0, 0, 0, 0]},
                request=httpx.Request("POST", url),
            )
        age_values = [0] * 102
        age_values[1] = 10
        age_values[21] = 40
        age_values[71] = 50
        if len(json["query"][1]["selection"]["values"]) == 102:
            return httpx.Response(
                200,
                json={"value": age_values},
                request=httpx.Request("POST", url),
            )
        return httpx.Response(
            200,
            json={"size": [1, 1, 3, 1, 1], "value": [100, 48, 52]},
            request=httpx.Request("POST", url),
        )


@pytest.mark.anyio
async def test_statistics_finland_fetch_scales_population_for_custom_bbox(monkeypatch):
    definition = SourceDefinition(
        source_id="statistics-finland",
        name="Statistics Finland",
        category=SourceCategory.DEMOGRAPHICS,
        description="Population and demographic datasets.",
        refresh_interval_seconds=86400,
    )
    adapter = StatisticsFinlandAdapter(definition)

    monkeypatch.setattr(settings, "nls_api_key", "configured")
    monkeypatch.setattr(
        "ipb_backend.ingestion.sources.statistics_finland.httpx.AsyncClient",
        lambda *args, **kwargs: StatisticsFinlandBBoxClient(),
    )

    record = await adapter.fetch(
        "North Karelia",
        "24h",
        load_target=LoadTarget(kind=LoadTargetKind.BBOX, label="Helsinki Slice", bbox_wgs84=[24.0, 60.0, 24.2, 60.2]),
    )

    assert record.area == "Helsinki Slice"
    assert record.load_target is not None
    assert record.data["query"]["municipalities"] == ["KU091"]
    assert record.data["population_total"] == 50
    assert record.data["male"] == 24
    assert record.data["female"] == 26
    assert record.data["age_distribution"]["groups"] == {"0-14": 5, "15-64": 20, "65+": 25}
    assert record.data["urban_rural"]["total_by_class"]["Total"] == 50