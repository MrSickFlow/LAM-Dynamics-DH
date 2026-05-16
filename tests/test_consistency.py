from datetime import datetime, timezone

import pytest

from ipb_backend.consistency.clustering import cluster_anomalies
from ipb_backend.consistency.engine import DataConsistencyEngine
from ipb_backend.consistency.fixtures import ARCHIPELAGO_MARITIME_FIXTURE
from ipb_backend.consistency.rules import MaritimeAisSarRule
from ipb_backend.main import build_registry
from ipb_backend.models import (
    AnomalySeverity,
    ConsistencyAnomaly,
    DatasetRecord,
    EwClassification,
    SourceCategory,
)


def _record(source_id: str, data: dict, *, area: str = "Archipelago Sea") -> DatasetRecord:
    return DatasetRecord(
        source_id=source_id,
        category=SourceCategory.OTHER,
        area=area,
        timeframe="72h",
        retrieved_at=datetime.now(timezone.utc),
        summary=f"{source_id} test record",
        data=data,
    )


@pytest.mark.asyncio
async def test_maritime_ais_sar_rule_flags_position_mismatch():
    rule = MaritimeAisSarRule()
    records = {
        "maritime-demo": _record(
            "maritime-demo",
            {
                "ais_vessels": ARCHIPELAGO_MARITIME_FIXTURE["ais_vessels"],
                "sar_returns": ARCHIPELAGO_MARITIME_FIXTURE["sar_returns"],
            },
        )
    }
    anomalies = rule.evaluate(
        area="Archipelago Sea",
        timeframe="72h",
        records=records,
        sources=[],
        context={},
    )
    assert anomalies
    assert any(a.rule_id == "ais-sar-vessel" for a in anomalies)
    assert any(a.severity in (AnomalySeverity.HIGH, AnomalySeverity.MEDIUM) for a in anomalies)


@pytest.mark.asyncio
async def test_engine_archipelago_detects_maritime_and_demo_trust():
    engine = DataConsistencyEngine(fmi_adapter=None)
    registry = build_registry()
    records = {
        "maritime-demo": _record(
            "maritime-demo",
            {
                "ais_vessels": ARCHIPELAGO_MARITIME_FIXTURE["ais_vessels"],
                "sar_returns": ARCHIPELAGO_MARITIME_FIXTURE["sar_returns"],
            },
        ),
        "opencellid": _record(
            "opencellid",
            {
                "provider": "Demo data (OPENCELLID_API_KEY not configured)",
                "cells": [{"lat": 60.2, "lon": 22.0, "range": 3000}] * 10,
            },
        ),
        "osm-poi": _record("osm-poi", {"categories": {"healthcare": [], "government": []}}),
    }
    report = await engine.evaluate(
        area="Archipelago Sea",
        timeframe="72h",
        records=records,
        sources=registry.list_sources(),
    )
    assert report.anomalies
    opencellid_trust = next(t for t in report.layer_trust if t.source_id == "opencellid")
    assert opencellid_trust.confidence < 0.85
    assert any("demo" in factor.lower() for factor in opencellid_trust.factors)


def test_spatial_clustering_groups_nearby_anomalies():
    anomalies = [
        ConsistencyAnomaly(
            anomaly_id="a1",
            rule_id="test",
            title="one",
            description="d",
            severity=AnomalySeverity.HIGH,
            location={"type": "Point", "coordinates": [22.0, 60.2]},
            vulnerable_sources=["fmi"],
        ),
        ConsistencyAnomaly(
            anomaly_id="a2",
            rule_id="test",
            title="two",
            description="d",
            severity=AnomalySeverity.HIGH,
            location={"type": "Point", "coordinates": [22.01, 60.21]},
            vulnerable_sources=["opencellid"],
        ),
    ]
    clusters = cluster_anomalies(anomalies, cluster_radius_km=30.0)
    assert len(clusters) == 1
    assert clusters[0].anomaly_count == 2


def test_registry_sources_include_ew_metadata():
    registry = build_registry()
    fmi = registry.get("fmi")
    assert fmi.ew_classification == EwClassification.VULNERABLE
    assert fmi.ew_rationale


@pytest.mark.asyncio
async def test_consistency_api_endpoint():
    from fastapi.testclient import TestClient
    from ipb_backend.main import app, state

    state["ingestion_service"]._records.clear()
    state["ingestion_service"]._records.append(
        _record("maritime-demo", {"ais_vessels": ARCHIPELAGO_MARITIME_FIXTURE["ais_vessels"], "sar_returns": ARCHIPELAGO_MARITIME_FIXTURE["sar_returns"]})
    )
    client = TestClient(app)
    response = client.post("/api/consistency/run", params={"area": "Archipelago Sea", "timeframe": "72h"})
    assert response.status_code == 200
    body = response.json()
    assert body["area"] == "Archipelago Sea"
    assert body["anomalies"]
    assert body["layer_trust"]
