from fastapi.testclient import TestClient

from ipb_backend.ingestion.sources.fmi import FmiAdapter
from ipb_backend.main import app
from ipb_backend.main import state


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
    # NLS requires API key, statistics-finland and digiroad are still placeholders
    assert len(payload["produced_records"]) == 2

    datasets_response = client.get("/api/datasets")
    assert datasets_response.status_code == 200
    assert len(datasets_response.json()) == 2


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
