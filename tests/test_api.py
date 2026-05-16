from fastapi.testclient import TestClient

from ipb_backend.analysis.analyzers import OllamaAnalyzer
from ipb_backend.ingestion.sources.fmi import FmiAdapter
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
    assert "Agent Interpreter" in response.text
    assert "Raw Data" in response.text


def test_ingestion_flow_for_placeholder_sources():
    state["ingestion_service"]._records.clear()
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
    assert len(payload["produced_records"]) == 3
    population_record = next(record for record in payload["produced_records"] if record["source_id"] == "statistics-finland")
    assert population_record["data"]["population_total"] > 50000

    datasets_response = client.get("/api/datasets")
    assert datasets_response.status_code == 200
    assert len(datasets_response.json()) == 3


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
    assert any(item["source_id"] == "custom-infra" for item in payload["agent"]["evidence_bundle"])
