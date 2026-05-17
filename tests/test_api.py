from fastapi.testclient import TestClient

from ipb_backend.analysis.analyzers import OllamaAnalyzer
from ipb_backend.ingestion.sources.digiroad import DigiroadAdapter
from ipb_backend.ingestion.sources.fmi import FmiAdapter
from ipb_backend.ingestion.sources.nls import NationalLandSurveyAdapter
from ipb_backend.ingestion.sources.opencellid import OpenCellIdAdapter
from ipb_backend.ingestion.sources.osm_poi import OsmPoiAdapter
from ipb_backend.ingestion.sources.satellites import SatelliteTleAdapter
from ipb_backend.ingestion.sources.statistics_finland import StatisticsFinlandAdapter
from ipb_backend.main import app
from ipb_backend.main import state
from ipb_backend.models import DatasetRecord, SourceCategory
from ipb_backend.config import settings


client = TestClient(app)


SAMPLE_FMI_XML = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<wfs:FeatureCollection xmlns:wfs=\"http://www.opengis.net/wfs/2.0\" xmlns:om=\"http://www.opengis.net/om/2.0\" xmlns:omso=\"http://inspire.ec.europa.eu/schemas/omso/3.0\" xmlns:gml=\"http://www.opengis.net/gml/3.2\" xmlns:sams=\"http://www.opengis.net/samplingSpatial/2.0\" xmlns:sam=\"http://www.opengis.net/sampling/2.0\" xmlns:wml2=\"http://www.opengis.net/waterml/2.0\" xmlns:target=\"http://xml.fmi.fi/namespace/om/atmosphericfeatures/1.1\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">
    <wfs:member>
        <omso:PointTimeSeriesObservation>
            <om:observedProperty xlink:href=\"https://opendata.fmi.fi/meta?observableProperty=observation&amp;param=t2m&amp;language=eng\"/>
            <om:featureOfInterest>
                <sams:SF_SpatialSamplingFeature>
                    <sam:sampledFeature>
                        <target:LocationCollection>
                            <target:member>
                                <target:Location>
                                    <gml:name codeSpace=\"http://xml.fmi.fi/namespace/locationcode/name\">Joensuu Linnunlahti</gml:name>
                                    <target:region>Joensuu</target:region>
                                </target:Location>
                            </target:member>
                        </target:LocationCollection>
                    </sam:sampledFeature>
                    <sams:shape>
                        <gml:Point>
                            <gml:pos>62.60179 29.72713</gml:pos>
                        </gml:Point>
                    </sams:shape>
                </sams:SF_SpatialSamplingFeature>
            </om:featureOfInterest>
            <om:result>
                <wml2:MeasurementTimeseries>
                    <wml2:point>
                        <wml2:MeasurementTVP>
                            <wml2:time>2026-05-16T05:00:00Z</wml2:time>
                            <wml2:value>10.2</wml2:value>
                        </wml2:MeasurementTVP>
                    </wml2:point>
                    <wml2:point>
                        <wml2:MeasurementTVP>
                            <wml2:time>2026-05-16T06:00:00Z</wml2:time>
                            <wml2:value>10.8</wml2:value>
                        </wml2:MeasurementTVP>
                    </wml2:point>
                </wml2:MeasurementTimeseries>
            </om:result>
        </omso:PointTimeSeriesObservation>
    </wfs:member>
    <wfs:member>
        <omso:PointTimeSeriesObservation>
            <om:observedProperty xlink:href=\"https://opendata.fmi.fi/meta?observableProperty=observation&amp;param=ws_10min&amp;language=eng\"/>
            <om:featureOfInterest>
                <sams:SF_SpatialSamplingFeature>
                    <sam:sampledFeature>
                        <target:LocationCollection>
                            <target:member>
                                <target:Location>
                                    <gml:name codeSpace=\"http://xml.fmi.fi/namespace/locationcode/name\">Joensuu Linnunlahti</gml:name>
                                    <target:region>Joensuu</target:region>
                                </target:Location>
                            </target:member>
                        </target:LocationCollection>
                    </sam:sampledFeature>
                    <sams:shape>
                        <gml:Point>
                            <gml:pos>62.60179 29.72713</gml:pos>
                        </gml:Point>
                    </sams:shape>
                </sams:SF_SpatialSamplingFeature>
            </om:featureOfInterest>
            <om:result>
                <wml2:MeasurementTimeseries>
                    <wml2:point>
                        <wml2:MeasurementTVP>
                            <wml2:time>2026-05-16T05:00:00Z</wml2:time>
                            <wml2:value>3.1</wml2:value>
                        </wml2:MeasurementTVP>
                    </wml2:point>
                    <wml2:point>
                        <wml2:MeasurementTVP>
                            <wml2:time>2026-05-16T06:00:00Z</wml2:time>
                            <wml2:value>3.4</wml2:value>
                        </wml2:MeasurementTVP>
                    </wml2:point>
                </wml2:MeasurementTimeseries>
            </om:result>
        </omso:PointTimeSeriesObservation>
    </wfs:member>
</wfs:FeatureCollection>
"""


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_map_data_nls_returns_demo_fallback_vectors():
    state["ingestion_service"]._records.clear()
    state["ingestion_service"]._records.append(
        DatasetRecord(
            source_id="nls",
            category=SourceCategory.TERRAIN,
            area="North Karelia",
            timeframe="72h",
            summary="NLS demo record",
            data={
                "collections": {
                    "rakennus": {
                        "label": "Buildings",
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [[[30.0, 62.4], [30.1, 62.4], [30.1, 62.5], [30.0, 62.5], [30.0, 62.4]]],
                                },
                                "properties": {"nimi": "Warehouse cluster"},
                            }
                        ],
                    }
                },
                "note": "Demo spatial fallback is used because NLS_API_KEY is not configured.",
            },
        )
    )

    response = client.get("/api/map-data/nls", params={"area": "North Karelia"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["reason"] == "demo-fallback"
    assert len(payload["features"]) == 1
    assert payload["features"][0]["properties"]["_collection"] == "rakennus"


def test_analysis_health_endpoint_reports_ollama_model(monkeypatch):
    async def fake_fetch_tags(self):
        return {"models": [{"name": settings.ollama_model}, {"name": "other-model"}]}

    monkeypatch.setattr(settings, "analysis_provider", "ollama")
    monkeypatch.setattr(OllamaAnalyzer, "_fetch_tags", fake_fetch_tags)

    response = client.get("/api/analysis/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["provider"] == "ollama"
    assert payload["status"] == "ready"
    assert payload["model"] == settings.ollama_model
    assert payload["model_available"] is True


def test_ui_demo_contains_workspace_shell():
    response = client.get("/api/ui-demo")
    assert response.status_code == 200
    assert "Map Overlays" in response.text
    assert "AOI Metrics" in response.text
    assert "AI Analyst Chat" in response.text
    assert "analysis-profile" in response.text
    assert "Raw Data" in response.text


def test_ingestion_flow_for_placeholder_sources(monkeypatch):
    state["ingestion_service"]._records.clear()

    async def fake_statfin_fetch(self, area, timeframe, load_target=None):
        return DatasetRecord(
            source_id="statistics-finland",
            category=SourceCategory.DEMOGRAPHICS,
            area="North Karelia",
            timeframe="72h",
            summary="Statistics Finland population data for North Karelia",
            data={"total": 170000, "population_total": 170000, "features": []},
        )

    async def fake_nls_fetch_fail(self, area, timeframe, load_target=None):
        raise ValueError("NLS_API_KEY not configured in .env")

    monkeypatch.setattr(StatisticsFinlandAdapter, "fetch", fake_statfin_fetch)
    monkeypatch.setattr(NationalLandSurveyAdapter, "fetch", fake_nls_fetch_fail)

    ingest_response = client.post(
        "/api/ingest",
        json={
            "area": "North Karelia",
            "timeframe": "72h",
            "source_ids": ["nls", "statistics-finland", "digiroad"],
        },
    )
    assert ingest_response.status_code == 200
    payload = ingest_response.json()
    assert set(payload["requested_sources"]) == {"nls", "statistics-finland", "digiroad"}
    assert len(payload["produced_records"]) == 2
    assert {record["source_id"] for record in payload["produced_records"]} == {"statistics-finland", "digiroad"}
    population_record = next(record for record in payload["produced_records"] if record["source_id"] == "statistics-finland")
    assert population_record["data"]["population_total"] > 50000

    datasets_response = client.get("/api/datasets")
    assert datasets_response.status_code == 200
    assert len(datasets_response.json()) == 2


def test_ingest_rejects_unknown_source_ids():
    response = client.post(
        "/api/ingest",
        json={
            "area": "North Karelia",
            "timeframe": "24h",
            "source_ids": ["missing-source"],
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unknown source ids: missing-source"


def test_ingest_accepts_bbox_load_target(monkeypatch):
    state["ingestion_service"]._records.clear()

    async def fake_fetch(self, area, timeframe, load_target=None):
        assert area == "Custom Load Area"
        assert timeframe == "24h"
        assert load_target is not None
        assert load_target.kind.value == "bbox"
        assert load_target.label == "Custom Load Area"
        assert load_target.bbox_wgs84 == [29.9, 62.4, 30.2, 62.7]
        return DatasetRecord(
            source_id="osm-poi",
            category=SourceCategory.OTHER,
            area="Custom Load Area",
            timeframe="24h",
            load_target=load_target,
            summary="OSM POIs for Custom Load Area",
            data={"categories": {}, "total_features": 0},
        )

    monkeypatch.setattr(OsmPoiAdapter, "fetch", fake_fetch)

    response = client.post(
        "/api/ingest",
        json={
            "area": "Custom Load Area",
            "timeframe": "24h",
            "load_target": {
                "kind": "bbox",
                "label": "Custom Load Area",
                "bbox_wgs84": [29.9, 62.4, 30.2, 62.7],
            },
            "source_ids": ["osm-poi"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["requested_sources"] == ["osm-poi"]
    assert payload["produced_records"][0]["area"] == "Custom Load Area"
    assert payload["produced_records"][0]["load_target"]["kind"] == "bbox"


def test_fmi_ingestion_flow(monkeypatch):
    state["ingestion_service"]._records.clear()

    async def fake_fetch_xml(self, place, start_time, end_time):
        assert place == "joensuu"
        return SAMPLE_FMI_XML

    monkeypatch.setattr(FmiAdapter, "_fetch_xml", fake_fetch_xml)

    ingest_response = client.post(
        "/api/ingest",
        json={"area": "North Karelia", "timeframe": "6h", "source_ids": ["fmi"]},
    )
    assert ingest_response.status_code == 200

    payload = ingest_response.json()
    assert payload["requested_sources"] == ["fmi"]
    assert len(payload["produced_records"]) == 1

    record = payload["produced_records"][0]
    assert record["source_id"] == "fmi"
    assert record["category"] == "weather"
    assert record["data"]["station"]["name"] == "Joensuu Linnunlahti"
    assert record["data"]["station"]["region"] == "Joensuu"
    assert record["data"]["observations"]["temperature"]["latest"]["value"] == 10.8
    assert record["data"]["observations"]["wind_speed"]["latest"]["value"] == 3.4

    datasets_response = client.get("/api/datasets")
    assert datasets_response.status_code == 200
    datasets = datasets_response.json()
    assert len(datasets) == 1
    assert datasets[0]["summary"].startswith("FMI weather observations for North Karelia")


def test_aoi_inspection_returns_clipped_source_data():
    state["ingestion_service"]._records.clear()
    state["ingestion_service"]._records.extend(
        [
            DatasetRecord(
                source_id="nls",
                category=SourceCategory.TERRAIN,
                area="North Karelia",
                timeframe="72h",
                summary="NLS test record",
                data={
                    "collections": {
                        "tieviiva": {
                            "label": "Road network",
                            "features": [
                                {
                                    "type": "Feature",
                                    "geometry": {
                                        "type": "LineString",
                                        "coordinates": [[30.1, 62.5], [30.5, 62.8]],
                                    },
                                    "properties": {"name": "Inside road"},
                                },
                                {
                                    "type": "Feature",
                                    "geometry": {
                                        "type": "LineString",
                                        "coordinates": [[29.1, 62.1], [29.2, 62.2]],
                                    },
                                    "properties": {"name": "Outside road"},
                                },
                            ],
                        }
                    }
                },
            ),
            DatasetRecord(
                source_id="digiroad",
                category=SourceCategory.INFRASTRUCTURE,
                area="North Karelia",
                timeframe="72h",
                summary="Digiroad test record",
                data={
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [30.35, 62.62]},
                            "properties": {"name": "Bridge"},
                        },
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [29.2, 62.15]},
                            "properties": {"name": "Outside asset"},
                        },
                    ]
                },
            ),
            DatasetRecord(
                source_id="statistics-finland",
                category=SourceCategory.DEMOGRAPHICS,
                area="North Karelia",
                timeframe="72h",
                summary="Population test record",
                data={
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[[30.0, 62.4], [30.4, 62.4], [30.4, 62.7], [30.0, 62.7], [30.0, 62.4]]],
                            },
                            "properties": {"cell_id": "a", "population": 120},
                        },
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[[30.3, 62.55], [30.7, 62.55], [30.7, 62.9], [30.3, 62.9], [30.3, 62.55]]],
                            },
                            "properties": {"cell_id": "b", "population": 80},
                        },
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[[29.0, 62.0], [29.2, 62.0], [29.2, 62.2], [29.0, 62.2], [29.0, 62.0]]],
                            },
                            "properties": {"cell_id": "c", "population": 50},
                        },
                    ]
                },
            ),
            DatasetRecord(
                source_id="fmi",
                category=SourceCategory.WEATHER,
                area="North Karelia",
                timeframe="72h",
                summary="FMI test record",
                data={
                    "station": {
                        "name": "Joensuu Linnunlahti",
                        "region": "Joensuu",
                        "latitude": 62.6,
                        "longitude": 30.3,
                    },
                    "observations": {
                        "temperature": {"latest": {"value": 10.8}},
                        "wind_speed": {"latest": {"value": 3.4}},
                    },
                },
            ),
        ]
    )

    response = client.post(
        "/api/aoi/inspect",
        json={
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[30.0, 62.4], [30.6, 62.4], [30.6, 62.85], [30.0, 62.85], [30.0, 62.4]]],
            },
            "timeframe": "72h",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"]["nls_feature_count"] == 1
    assert payload["metrics"]["digiroad_feature_count"] == 1
    assert payload["metrics"]["population_total"] == 171
    assert payload["metrics"]["weather_station_count"] == 1
    assert payload["metrics"]["feature_counts_by_source"]["nls"] == 1
    assert payload["metrics"]["geometry_counts"]["LineString"] == 1
    assert payload["raw_data"]["nls"]["collections"][0]["collection"] == "tieviiva"
    assert payload["raw_data"]["statistics-finland"]["features"][1]["properties"]["population_source"] == 80
    assert payload["raw_data"]["statistics-finland"]["features"][1]["properties"]["overlap_ratio"] == 0.6429
    assert payload["raw_data"]["fmi"]["stations"][0]["properties"]["name"] == "Joensuu Linnunlahti"
    assert payload["raw_sections"][0]["subsections"]
    assert payload["agent"]["provider"] == "rules"
    assert payload["agent"]["evidence_bundle"]
    assert payload["data_package"]["selection"]["selection_type"] == "geometry"
    assert payload["llm_input"]["profile_focus"]
    assert payload["llm_input"]["source_digests"]
    assert payload["data_package"]["quality"]["overall_confidence"] in {"high", "low", "medium"}
    assert payload["llm_output"]["profile"] == "general"
    assert payload["llm_output"]["implications"]


def test_aoi_inspection_includes_additional_feature_sources():
    state["ingestion_service"]._records.clear()
    state["ingestion_service"]._records.extend(
        [
            DatasetRecord(
                source_id="custom-infra",
                category=SourceCategory.INFRASTRUCTURE,
                area="North Karelia",
                timeframe="72h",
                summary="Custom infrastructure record",
                data={
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [30.35, 62.62],
                            },
                            "properties": {"name": "Tower site"},
                        },
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [29.1, 62.1],
                            },
                            "properties": {"name": "Outside tower"},
                        },
                    ]
                },
            )
        ]
    )

    response = client.post(
        "/api/aoi/inspect",
        json={
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[30.0, 62.4], [30.6, 62.4], [30.6, 62.85], [30.0, 62.85], [30.0, 62.4]]],
            },
            "timeframe": "72h",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["raw_data"]["custom-infra"]["feature_count"] == 1
    assert payload["metrics"]["feature_counts_by_source"]["custom-infra"] == 1
    assert payload["metrics"]["feature_counts_by_category"]["infrastructure"] == 1
    assert "custom-infra" in payload["metrics"]["active_sources"]
    assert any(section["source_id"] == "custom-infra" for section in payload["raw_sections"])
    assert any(item["source_id"] == "custom-infra" for item in payload["freshness"])
    assert any(item["source_id"] == "custom-infra" for item in payload["agent"]["evidence_bundle"])
    assert any(item["source_id"] == "custom-infra" for item in payload["data_package"]["source_summaries"])

SAMPLE_OPENCELLID_RECORD = DatasetRecord(
    source_id="opencellid",
    category=SourceCategory.INFRASTRUCTURE,
    area="North Karelia",
    timeframe="72h",
    summary="OpenCellID: 3 cell towers near North Karelia",
    data={
        "provider": "OpenCellID (by Unwired Labs)",
        "api": "cell/getInArea",
        "cells": [
            {"lat": 62.8, "lon": 30.2, "mcc": 244, "mnc": 5, "radio": "LTE", "range": 3000, "samples": 10},
            {"lat": 62.81, "lon": 30.21, "mcc": 244, "mnc": 5, "radio": "LTE", "range": 2500, "samples": 8},
            {"lat": 62.79, "lon": 30.19, "mcc": 244, "mnc": 91, "radio": "UMTS", "range": 5000, "samples": 5},
        ],
        "total_cells": 3,
    },
)


def test_opencellid_ingestion(monkeypatch):
    state["ingestion_service"]._records.clear()
    registry = state["registry"]
    original_definition = registry.get("opencellid")

    async def fake_fetch(self, area, timeframe, load_target=None):
        return SAMPLE_OPENCELLID_RECORD

    monkeypatch.setattr(OpenCellIdAdapter, "fetch", fake_fetch)
    registry.update(
        original_definition.model_copy(
            update={"enabled": True, "status": "idle", "last_error": None}
        )
    )

    try:
        ingest_response = client.post(
            "/api/ingest",
            json={"area": "North Karelia", "timeframe": "72h", "source_ids": ["opencellid"]},
        )
        assert ingest_response.status_code == 200
        payload = ingest_response.json()
        assert payload["requested_sources"] == ["opencellid"]
        assert len(payload["produced_records"]) == 1
        assert payload["produced_records"][0]["source_id"] == "opencellid"
        assert payload["produced_records"][0]["data"]["total_cells"] == 3
    finally:
        registry.update(original_definition)


SAMPLE_OSM_RECORD = DatasetRecord(
    source_id="osm-poi",
    category=SourceCategory.OTHER,
    area="North Karelia",
    timeframe="72h",
    summary="OSM POIs for North Karelia: 4 features across 2 categories",
    data={
        "provider": "OpenStreetMap contributors (ODbL)",
        "api": "Overpass API",
        "categories": {
            "education": [
                {"id": "node/1", "type": "node", "lat": 62.8, "lon": 30.2, "tags": {"amenity": "school", "name": "Koulu"}},
                {"id": "node/2", "type": "node", "lat": 62.81, "lon": 30.21, "tags": {"amenity": "library", "name": "Kirjasto"}},
            ],
            "healthcare": [
                {"id": "node/3", "type": "node", "lat": 62.79, "lon": 30.19, "tags": {"amenity": "hospital", "name": "Sairaala"}},
                {"id": "node/4", "type": "node", "lat": 62.78, "lon": 30.18, "tags": {"amenity": "pharmacy"}},
            ],
        },
        "total_features": 4,
    },
)


def test_osm_poi_ingestion(monkeypatch):
    state["ingestion_service"]._records.clear()

    async def fake_fetch(self, area, timeframe, load_target=None):
        return SAMPLE_OSM_RECORD

    monkeypatch.setattr(OsmPoiAdapter, "fetch", fake_fetch)

    ingest_response = client.post(
        "/api/ingest",
        json={"area": "North Karelia", "timeframe": "72h", "source_ids": ["osm-poi"]},
    )
    assert ingest_response.status_code == 200
    payload = ingest_response.json()
    assert payload["requested_sources"] == ["osm-poi"]
    assert len(payload["produced_records"]) == 1
    record = payload["produced_records"][0]
    assert record["source_id"] == "osm-poi"
    assert record["data"]["total_features"] == 4
    assert "education" in record["data"]["categories"]
    assert "healthcare" in record["data"]["categories"]


SAMPLE_TLE_RECORD = DatasetRecord(
    source_id="satellites",
    category=SourceCategory.SATELLITE,
    area="North Karelia",
    timeframe="72h",
    summary="Satellite TLE data for North Karelia: tracking 3 reconnaissance/imaging satellites",
    data={
        "provider": "Celestrak",
        "satellites": {
            "Sentinel-1A": {
                "norad_id": 39634,
                "type": "SAR imaging (C-band)",
                "predicted_passes": [
                    {"pass_time_utc": "2026-05-16T14:00:00", "duration_min": 90.0, "altitude_km": 693},
                ],
            },
            "Sentinel-2A": {
                "norad_id": 40697,
                "type": "Multispectral imaging (10m)",
                "predicted_passes": [
                    {"pass_time_utc": "2026-05-16T15:00:00", "duration_min": 100.0, "altitude_km": 786},
                ],
            },
            "WorldView-3": {
                "norad_id": 40115,
                "type": "Commercial imaging (31cm)",
                "predicted_passes": [
                    {"pass_time_utc": "2026-05-16T16:00:00", "duration_min": 95.0, "altitude_km": 617},
                ],
            },
        },
        "total_tracked": 3,
    },
)


def test_satellite_ingestion(monkeypatch):
    state["ingestion_service"]._records.clear()

    async def fake_fetch(self, area, timeframe, load_target=None):
        return SAMPLE_TLE_RECORD

    monkeypatch.setattr(SatelliteTleAdapter, "fetch", fake_fetch)

    ingest_response = client.post(
        "/api/ingest",
        json={"area": "North Karelia", "timeframe": "72h", "source_ids": ["satellites"]},
    )
    assert ingest_response.status_code == 200
    payload = ingest_response.json()
    assert payload["requested_sources"] == ["satellites"]
    assert len(payload["produced_records"]) == 1
    record = payload["produced_records"][0]
    assert record["source_id"] == "satellites"
    assert record["data"]["total_tracked"] == 3


def test_agents_listing():
    response = client.get("/api/agents")
    assert response.status_code == 200
    agents = response.json()
    agent_ids = [a["agent_id"] for a in agents]
    assert "summary-agent" in agent_ids
    assert "celltower-agent" in agent_ids
    assert "satellite-agent" in agent_ids
    assert "bridge-load-agent" in agent_ids


def test_celltower_agent_run(monkeypatch):
    async def fake_fetch(self, area, timeframe, load_target=None):
        return SAMPLE_OPENCELLID_RECORD

    monkeypatch.setattr(OpenCellIdAdapter, "fetch", fake_fetch)

    response = client.post("/api/agents/celltower-agent/run?area=North+Karelia&timeframe=72h")
    assert response.status_code == 200
    result = response.json()
    assert result["agent_id"] == "celltower-agent"
    assert result["summary"].startswith("Cell tower analysis")
    assert any("Total cell towers: 3" in f for f in result["findings"])
    assert any("244-5" in f for f in result["findings"])
    assert any("LTE" in f for f in result["findings"])
    assert any("UMTS" in f for f in result["findings"])


def test_satellite_agent_run(monkeypatch):
    async def fake_fetch(self, area, timeframe, load_target=None):
        return SAMPLE_TLE_RECORD

    monkeypatch.setattr(SatelliteTleAdapter, "fetch", fake_fetch)

    response = client.post("/api/agents/satellite-agent/run?area=North+Karelia&timeframe=72h")
    assert response.status_code == 200
    result = response.json()
    assert result["agent_id"] == "satellite-agent"
    assert "3" in result["summary"]
    assert any("SAR" in f for f in result["findings"])
    assert any("multispectral" in f or "Multispectral" in f for f in result["findings"])


SAMPLE_BRIDGE_COLLECTIONS = {
    "dr_tielinkki_silta_alikulku_tunneli": {
        "label": "Bridges, underpasses, tunnels",
        "number_matched": 5,
        "number_returned": 3,
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[30.0, 62.5], [30.1, 62.5]]},
                "properties": {"link_id": "bridge-a-1111", "silta_alik": 0},
            },
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[30.2, 62.6], [30.3, 62.6]]},
                "properties": {"link_id": "bridge-b-2222", "silta_alik": 0},
            },
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[30.4, 62.7], [30.5, 62.7]]},
                "properties": {"link_id": "tunnel-c-3333", "silta_alik": -1},
            },
        ],
    },
    "dr_max_massa": {
        "label": "Max weight limit",
        "features": [
            {"properties": {"link_id": "bridge-a-1111", "arvo": 60000}},
            {"properties": {"link_id": "bridge-b-2222", "arvo": 12000}},
        ],
    },
    "dr_max_korkeus": {
        "label": "Max height limit",
        "features": [
            {"properties": {"link_id": "bridge-a-1111", "arvo": 450}},
            {"properties": {"link_id": "tunnel-c-3333", "arvo": 350}},
        ],
    },
    "dr_max_leveys": {"label": "Max width limit", "features": []},
    "dr_max_akselimassa": {"label": "Max axle mass", "features": []},
    "dr_yhdistelman_max_massa": {"label": "Combined max mass", "features": []},
}

SAMPLE_DIGIROAD_RECORD = DatasetRecord(
    source_id="digiroad",
    category=SourceCategory.INFRASTRUCTURE,
    area="North Karelia",
    timeframe="72h",
    summary="Digiroad road data for North Karelia",
    data={
        "provider": "Finnish Transport Infrastructure Agency",
        "collections": SAMPLE_BRIDGE_COLLECTIONS,
        "total_features": 5,
    },
)


def test_bridge_load_agent_run(monkeypatch):
    async def fake_fetch(self, area, timeframe, load_target=None):
        return SAMPLE_DIGIROAD_RECORD

    monkeypatch.setattr(DigiroadAdapter, "fetch", fake_fetch)

    response = client.post("/api/agents/bridge-load-agent/run?area=North+Karelia&timeframe=72h")
    assert response.status_code == 200
    result = response.json()
    assert result["agent_id"] == "bridge-load-agent"
    assert result["summary"].startswith("Bridge load capacity analysis")
    assert "2 bridges" in result["summary"]
    assert "1 tunnels" in result["summary"]
    assert any("Main battle tank" in f for f in result["findings"])
    assert any("Light vehicles" in f for f in result["findings"])
    assert any("height < 4.0m" in f for f in result["findings"])

    enriched = result["data"]["enriched_features"]
    assert len(enriched) == 3

    bridge_a = next(e for e in enriched if e["properties"]["link_id"] == "bridge-a-1111")
    assert bridge_a["properties"]["max_weight_tonnes"] == 60.0
    assert bridge_a["properties"]["max_height_m"] == 4.5
    assert bridge_a["properties"]["vehicle_class"] == "Main battle tank"

    bridge_b = next(e for e in enriched if e["properties"]["link_id"] == "bridge-b-2222")
    assert bridge_b["properties"]["max_weight_tonnes"] == 12.0
    assert bridge_b["properties"]["vehicle_class"] == "Light vehicles only"

    tunnel_c = next(e for e in enriched if e["properties"]["link_id"] == "tunnel-c-3333")
    assert tunnel_c["properties"]["type"] == "tunnel"
    assert tunnel_c["properties"]["max_height_m"] == 3.5
    assert tunnel_c["properties"]["max_weight_tonnes"] is None


def test_bridge_load_agent_empty(monkeypatch):
    empty_collections = {
        "dr_tielinkki_silta_alikulku_tunneli": {"label": "Bridges, underpasses, tunnels", "number_matched": 0, "features": []},
    }
    empty_record = DatasetRecord(
        source_id="digiroad",
        category=SourceCategory.INFRASTRUCTURE,
        area="North Karelia",
        timeframe="72h",
        summary="Digiroad road data",
        data={"collections": empty_collections, "total_features": 0},
    )

    async def fake_fetch(self, area, timeframe, load_target=None):
        return empty_record

    monkeypatch.setattr(DigiroadAdapter, "fetch", fake_fetch)

    response = client.post("/api/agents/bridge-load-agent/run?area=North+Karelia&timeframe=72h")
    assert response.status_code == 200
    result = response.json()
    assert result["agent_id"] == "bridge-load-agent"
    assert "0 bridges" in result["summary"]
    assert result["data"]["enriched_features"] == []


SAMPLE_DEMOGRAPHICS_RECORD = DatasetRecord(
    source_id="statistics-finland",
    category=SourceCategory.DEMOGRAPHICS,
    area="North Karelia",
    timeframe="72h",
    summary="Statistics Finland population data for North Karelia",
    data={
        "total": 170000,
        "male": 85000,
        "female": 85000,
        "per_municipality": {},
        "age_distribution": {"groups": {"0-14": 25000, "15-64": 105000, "65+": 40000}},
        "urban_rural": {
            "per_municipality": {
                "KU167": {
                    "name": "Joensuu",
                    "classes": {"Total": 78000, "Urban areas": 66000, "Rural areas": 11000},
                },
                "KU176": {
                    "name": "Juuka",
                    "classes": {"Total": 4000, "Urban areas": 500, "Rural areas": 3500},
                },
            },
            "total_by_class": {
                "Total": 82000,
                "Urban areas": 66500,
                "Rural areas": 14500,
                "Inner urban area": 33000,
                "Outer urban area": 26000,
            },
        },
    },
)


def test_demographics_agent_run(monkeypatch):
    async def fake_fetch(self, area, timeframe, load_target=None):
        return SAMPLE_DEMOGRAPHICS_RECORD

    monkeypatch.setattr(StatisticsFinlandAdapter, "fetch", fake_fetch)

    response = client.post("/api/agents/demographics-agent/run?area=North+Karelia&timeframe=72h")
    assert response.status_code == 200
    result = response.json()
    assert result["agent_id"] == "demographics-agent"
    assert "170,000" in result["summary"] or "170000" in result["summary"]
    assert "urban" in result["summary"].lower()
    assert any("Sex distribution" in f for f in result["findings"])
    assert any("Age distribution" in f for f in result["findings"])
    assert any("Urban/rural split" in f for f in result["findings"])
    assert any("Joensuu" in f for f in result["findings"])
    assert any("Juuka" in f for f in result["findings"])


SAMPLE_FOREST_RECORD = DatasetRecord(
    source_id="osm-poi",
    category=SourceCategory.OTHER,
    area="North Karelia",
    timeframe="72h",
    summary="OSM POIs for North Karelia",
    data={
        "categories": {
            "forest": [
                {"tags": {"leaf_type": "needleleaved", "leaf_cycle": "evergreen"}, "lat": 62.8, "lon": 30.2},
                {"tags": {"leaf_type": "needleleaved", "leaf_cycle": "evergreen"}, "lat": 62.81, "lon": 30.21},
                {"tags": {"leaf_type": "broadleaved", "leaf_cycle": "deciduous"}, "lat": 62.79, "lon": 30.19},
                {"tags": {"leaf_type": "mixed", "leaf_cycle": "mixed"}, "lat": 62.78, "lon": 30.18},
            ],
        },
        "total_features": 4,
    },
)


def test_forest_concealment_agent_run(monkeypatch):
    async def fake_fetch(self, area, timeframe, load_target=None):
        return SAMPLE_FOREST_RECORD

    monkeypatch.setattr(OsmPoiAdapter, "fetch", fake_fetch)

    response = client.post("/api/agents/forest-concealment-agent/run?area=North+Karelia&timeframe=72h")
    assert response.status_code == 200
    result = response.json()
    assert result["agent_id"] == "forest-concealment-agent"
    assert "4 features" in result["summary"] or "4" in result["findings"][0]
    assert any("coniferous" in f for f in result["findings"])
    assert any("high" in f for f in result["findings"])


SAMPLE_WEATHER_RECORD = DatasetRecord(
    source_id="fmi",
    category=SourceCategory.WEATHER,
    area="North Karelia",
    timeframe="72h",
    summary="FMI weather observations for North Karelia",
    data={
        "station": {"name": "Joensuu Linnunlahti", "region": "Joensuu"},
        "observations": {
            "temperature": {"latest": {"value": -5.0}},
            "wind_speed": {"latest": {"value": 12.5}},
            "wind_gust": {"latest": {"value": 18.0}},
            "humidity": {"latest": {"value": 85}},
            "precipitation": {"latest": {"value": 3.2}},
            "cloud_cover": {"latest": {"value": 90}},
        },
    },
)


def test_weather_impact_agent_run(monkeypatch):
    async def fake_fetch(self, area, timeframe, load_target=None):
        return SAMPLE_WEATHER_RECORD

    monkeypatch.setattr(FmiAdapter, "fetch", fake_fetch)

    response = client.post("/api/agents/weather-impact-agent/run?area=North+Karelia&timeframe=72h")
    assert response.status_code == 200
    result = response.json()
    assert result["agent_id"] == "weather-impact-agent"
    assert any("UAV" in f or "drone" in f.lower() for f in result["findings"])
    assert any("satellite" in f.lower() for f in result["findings"])
    assert any("Icy" in f or "icy" in f or "ice" in f or "sub-zero" in f or "Sub-zero" in f or "frost" in f for f in result["findings"])


SAMPLE_POWER_GRID_RECORD = DatasetRecord(
    source_id="nls",
    category=SourceCategory.TERRAIN,
    area="North Karelia",
    timeframe="72h",
    summary="NLS test record with power lines",
    data={
        "collections": {
            "sahkolinja": {
                "label": "Power lines",
                "number_matched": 5,
                "features": [
                    {"geometry": {"type": "LineString", "coordinates": [[30.0, 62.5, 0], [30.1, 62.55, 0], [30.2, 62.6, 0]]}},
                    {"geometry": {"type": "LineString", "coordinates": [[30.3, 62.7, 0], [30.4, 62.75, 0]]}},
                ],
            }
        },
    },
)


def test_power_grid_agent_run(monkeypatch):
    async def fake_fetch(self, area, timeframe, load_target=None):
        return SAMPLE_POWER_GRID_RECORD

    monkeypatch.setattr(NationalLandSurveyAdapter, "fetch", fake_fetch)

    response = client.post("/api/agents/power-grid-agent/run?area=North+Karelia&timeframe=72h")
    assert response.status_code == 200
    result = response.json()
    assert result["agent_id"] == "power-grid-agent"
    assert "2 line segments" in result["summary"] or "2" in result["findings"][0]
    assert "km" in result["summary"]
    assert any("chokepoint" in f.lower() for f in result["findings"])


def test_forest_concealment_agent_empty(monkeypatch):
    empty_record = DatasetRecord(
        source_id="osm-poi",
        category=SourceCategory.OTHER,
        area="North Karelia",
        timeframe="72h",
        summary="OSM POIs",
        data={"categories": {"forest": []}, "total_features": 0},
    )

    async def fake_fetch(self, area, timeframe, load_target=None):
        return empty_record

    monkeypatch.setattr(OsmPoiAdapter, "fetch", fake_fetch)

    response = client.post("/api/agents/forest-concealment-agent/run?area=North+Karelia&timeframe=72h")
    assert response.status_code == 200
    result = response.json()
    assert "no forest data" in result["summary"].lower()


def test_aoi_population_ignores_zero_area_boundary_touches():
    state["ingestion_service"]._records.clear()
    state["ingestion_service"]._records.append(
        DatasetRecord(
            source_id="statistics-finland",
            category=SourceCategory.DEMOGRAPHICS,
            area="North Karelia",
            timeframe="72h",
            summary="Population boundary test record",
            data={
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[29.8, 62.4], [30.0, 62.4], [30.0, 62.7], [29.8, 62.7], [29.8, 62.4]]],
                        },
                        "properties": {"cell_id": "boundary-touch", "population": 100},
                    },
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[30.0, 62.4], [30.4, 62.4], [30.4, 62.7], [30.0, 62.7], [30.0, 62.4]]],
                        },
                        "properties": {"cell_id": "inside", "population": 120},
                    },
                ],
                "population_total": 220,
            },
        )
    )

    response = client.post(
        "/api/aoi/inspect",
        json={
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[30.0, 62.4], [30.6, 62.4], [30.6, 62.85], [30.0, 62.85], [30.0, 62.4]]],
            },
            "timeframe": "72h",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    stats_payload = payload["raw_data"]["statistics-finland"]
    assert stats_payload["population_total"] == 120
    assert stats_payload["feature_count"] == 1
    assert [feature["properties"]["cell_id"] for feature in stats_payload["features"]] == ["inside"]


def test_aoi_inspection_includes_osm_and_cell_sources():
    state["ingestion_service"]._records.clear()
    state["ingestion_service"]._records.extend(
        [
            DatasetRecord(
                source_id="osm-poi",
                category=SourceCategory.OTHER,
                area="North Karelia",
                timeframe="24h",
                summary="OSM POI test record",
                data={
                    "categories": {
                        "education": [
                            {
                                "id": "node/1",
                                "type": "node",
                                "lat": 62.62,
                                "lon": 30.35,
                                "tags": {"name": "School"},
                            },
                            {
                                "id": "node/2",
                                "type": "node",
                                "lat": 62.1,
                                "lon": 29.1,
                                "tags": {"name": "Outside School"},
                            },
                        ]
                    },
                    "total_features": 2,
                },
            ),
            DatasetRecord(
                source_id="opencellid",
                category=SourceCategory.INFRASTRUCTURE,
                area="North Karelia",
                timeframe="24h",
                summary="OpenCellID test record",
                data={
                    "cells": [
                        {
                            "cellid": 1001,
                            "lat": 62.63,
                            "lon": 30.32,
                            "radio": "LTE",
                            "samples": 12,
                        },
                        {
                            "cellid": 1002,
                            "lat": 62.12,
                            "lon": 29.15,
                            "radio": "LTE",
                            "samples": 4,
                        },
                    ],
                    "total_cells": 2,
                },
            ),
        ]
    )

    response = client.post(
        "/api/aoi/inspect",
        json={
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[30.0, 62.4], [30.6, 62.4], [30.6, 62.85], [30.0, 62.85], [30.0, 62.4]]],
            },
            "timeframe": "24h",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["raw_data"]["osm-poi"]["feature_count"] == 1
    assert payload["raw_data"]["osm-poi"]["collections"][0]["collection"] == "education"
    assert payload["raw_data"]["opencellid"]["feature_count"] == 1
    assert payload["metrics"]["feature_counts_by_source"]["osm-poi"] == 1
    assert payload["metrics"]["feature_counts_by_source"]["opencellid"] == 1
    assert payload["metrics"]["feature_counts_by_category"]["infrastructure"] == 1
    assert payload["metrics"]["feature_counts_by_category"]["other"] == 1
    assert payload["metrics"]["geometry_counts"]["Point"] == 2
    assert any(section["source_id"] == "osm-poi" for section in payload["raw_sections"])
    assert any(section["source_id"] == "opencellid" for section in payload["raw_sections"])


def test_aoi_data_package_endpoint_returns_normalized_contract():
    state["ingestion_service"]._records.clear()
    state["ingestion_service"]._records.extend(
        [
            DatasetRecord(
                source_id="custom-infra",
                category=SourceCategory.INFRASTRUCTURE,
                area="North Karelia",
                timeframe="72h",
                summary="Custom infrastructure record",
                data={
                    "provider": "Custom Infrastructure Feed",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [30.35, 62.62],
                            },
                            "properties": {"name": "Tower site"},
                        }
                    ],
                },
            )
        ]
    )

    response = client.post(
        "/api/aoi/data-package",
        json={
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[30.0, 62.4], [30.6, 62.4], [30.6, 62.85], [30.0, 62.85], [30.0, 62.4]]],
            },
            "timeframe": "72h",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "1.0"
    assert payload["selection"]["selection_type"] == "geometry"
    assert payload["counts"]["by_source"]["custom-infra"] == 1
    assert any(item["source_id"] == "custom-infra" for item in payload["source_freshness"])
    assert payload["source_summaries"][0]["provenance"]["deterministic"] is True
    assert payload["evidence_items"]


def test_analysis_profiles_endpoint_lists_wrapper_profiles():
    response = client.get("/api/analysis/profiles")

    assert response.status_code == 200
    payload = response.json()
    profiles = {item["profile"] for item in payload}
    assert "general" in profiles
    assert "mobility" in profiles
    assert "communications" in profiles


def test_aoi_interpret_accepts_question_and_history():
    response = client.post(
        "/api/aoi/interpret",
        json={
            "profile": "mobility",
            "question": "What matters most for movement?",
            "conversation_history": [
                {"role": "user", "content": "Give me a movement-focused read."}
            ],
            "data_package": {
                "schema_version": "1.0",
                "package_id": "pkg-1",
                "generated_at": "2026-05-16T12:00:00Z",
                "selection": {
                    "selection_type": "geometry",
                    "area_id": None,
                    "label": None,
                    "geometry": {"type": "Polygon", "coordinates": []},
                    "bounds_wgs84": [30.0, 62.4, 30.6, 62.85],
                    "area_sqkm": 10.0,
                },
                "scope": {
                    "timeframe": "72h",
                    "requested_sources": ["digiroad"],
                    "resolved_sources": ["digiroad"],
                },
                "source_freshness": [],
                "source_summaries": [
                    {
                        "source_id": "digiroad",
                        "category": "infrastructure",
                        "title": "Digiroad",
                        "summary": "1 transport feature intersects the AOI",
                        "raw_summary": {"feature_count": 1},
                        "confidence": "high",
                        "provenance": {
                            "provider": "Digiroad",
                            "adapter": "DatasetRecord",
                            "retrieved_at": None,
                            "fallback_used": False,
                            "fallback_reason": None,
                            "deterministic": True,
                            "note": None,
                        },
                    }
                ],
                "counts": {
                    "by_source": {"digiroad": 1},
                    "by_category": {"infrastructure": 1},
                    "geometry_types": {"LineString": 1},
                },
                "derived_indicators": [
                    {
                        "indicator_id": "selection_area_sqkm",
                        "name": "Selection Area",
                        "value": 10.0,
                        "unit": "sqkm",
                        "method": "polygon_area",
                        "source_ids": [],
                        "confidence": "high",
                        "notes": [],
                    }
                ],
                "evidence_items": [
                    {
                        "evidence_id": "ev-digiroad-001",
                        "source_id": "digiroad",
                        "kind": "count-summary",
                        "title": "digiroad evidence 1",
                        "detail": "1 intersecting feature",
                        "support": "Transport link",
                        "data_ref": {"section": "source_summaries", "path": "source_summaries[0].raw_summary"},
                    }
                ],
                "quality": {
                    "fallback_sources": [],
                    "error_sources": [],
                    "coverage_gaps": [],
                    "overall_confidence": "high",
                },
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"] == "mobility"
    assert payload["summary"]  # non-empty summary returned


# ───────────────────────── point inspect ─────────────────────────

def test_point_inspect_returns_required_fields():
    response = client.post(
        "/api/point/inspect",
        json={"lat": 62.6, "lon": 29.8, "timeframe": "24h"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["lat"] == 62.6
    assert payload["lon"] == 29.8
    assert "terrain" in payload
    assert "weather" in payload
    assert "nearby_context" in payload
    assert "los" in payload
    assert "summary" in payload
    assert isinstance(payload["summary"], str)


def test_point_inspect_terrain_schema():
    response = client.post(
        "/api/point/inspect",
        json={"lat": 62.6, "lon": 29.8},
    )
    assert response.status_code == 200
    terrain = response.json()["terrain"]
    assert "elevation_m" in terrain
    assert "elevation_source" in terrain
    assert isinstance(terrain["available"], bool)


def test_point_inspect_los_has_available_field():
    response = client.post(
        "/api/point/inspect",
        json={"lat": 62.6, "lon": 29.8},
    )
    assert response.status_code == 200
    los = response.json()["los"]
    assert isinstance(los.get("available"), bool)


def test_point_inspect_nearby_context_schema():
    response = client.post(
        "/api/point/inspect",
        json={"lat": 62.6, "lon": 29.8},
    )
    assert response.status_code == 200
    nearby = response.json()["nearby_context"]
    assert "search_radius_km" in nearby
    assert "poi_counts" in nearby
    assert "cell_towers_within_radius" in nearby


# ───────────────────────── elevation provider ─────────────────────────

def test_unavailable_elevation_provider_returns_none():
    import asyncio
    from ipb_backend.terrain.elevation import UnavailableElevationProvider
    provider = UnavailableElevationProvider()
    result = asyncio.run(provider.get_elevation(62.6, 29.8))
    assert result is None


def test_build_elevation_provider_returns_open_topo():
    from ipb_backend.terrain.elevation import build_elevation_provider, OpenTopoElevationProvider
    provider = build_elevation_provider("")
    assert isinstance(provider, OpenTopoElevationProvider)
