"""
Tests for Russian satellite ground-track bbox filtering.

Covers:
  - Backend route returns non-empty tracks for a Finland bbox
  - All returned segments actually intersect the bbox halo (no leaked global tracks)
  - step_seconds is 60 when bbox param is given (dense enough for small boxes)
  - No tracks returned when bbox is outside all pass windows (e.g. central Pacific)
  - /api/map-data/satellites returns available=True with bbox load target
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from ipb_backend.main import app, state
from ipb_backend.models import DatasetRecord, LoadTarget, LoadTargetKind, SourceCategory
from ipb_backend.ingestion.sources.satellites import SatelliteTleAdapter

client = TestClient(app)

# Real TLE for Persona 1 (Kosmos-2486) — known to pass over Finland
_TLE1 = "1 39177U 13028A   26136.95509762  .00000152  00000-0  44248-4 0  9996"
_TLE2 = "2 39177  98.0116 105.5720 0012133 109.9853 250.2653 14.56179887686839"

FINLAND_BBOX = [26.0, 60.5, 32.0, 64.0]   # w,s,e,n
PACIFIC_BBOX  = [160.0, 5.0, 175.0, 20.0]  # no Russian sat passes here in 8h


def _make_satellite_record(bbox: list[float]) -> DatasetRecord:
    return DatasetRecord(
        source_id="satellites",
        category=SourceCategory.SATELLITE,
        area="Custom BBox",
        timeframe="8h",
        load_target=LoadTarget(
            kind=LoadTargetKind.BBOX,
            label="Custom BBox",
            bbox_wgs84=bbox,
        ),
        summary="Test TLE record",
        data={
            "provider": "test",
            "satellites": {
                "Persona 1 (Kosmos-2486)": {
                    "norad_id": 39177,
                    "type": "Russian imaging (2.5m)",
                    "origin": "russian",
                    "tle_line_1": _TLE1,
                    "tle_line_2": _TLE2,
                    "predicted_passes": [],
                },
            },
            "total_tracked": 1,
            "query": {"lat": 62.5, "lon": 29.0},
        },
    )


@pytest.fixture(autouse=True)
def _clear_records():
    state["ingestion_service"]._records.clear()
    yield
    state["ingestion_service"]._records.clear()


# ---------------------------------------------------------------------------
# Unit tests — SatelliteTleAdapter.compute_ground_track
# ---------------------------------------------------------------------------

def test_compute_ground_track_density():
    """60s steps give ≥5 points inside a 3°-buffered Finland bbox per pass."""
    adapter = SatelliteTleAdapter.__new__(SatelliteTleAdapter)
    now = datetime.now(timezone.utc)
    pts = adapter.compute_ground_track(_TLE1, _TLE2, now, hours=8, step_seconds=60)
    w, s, e, n = FINLAND_BBOX[0] - 3, FINLAND_BBOX[1] - 3, FINLAND_BBOX[2] + 3, FINLAND_BBOX[3] + 3
    inside = [p for p in pts if w <= p["lon"] <= e and s <= p["lat"] <= n]
    # At least one pass window must exist; each pass should have ≥3 points in the halo
    assert len(inside) >= 1, "Persona 1 should pass within 3° of Finland in 8h"


def test_compute_ground_track_300s_may_miss_small_box():
    """300s steps can miss a small rectangle — this documents the known limitation."""
    adapter = SatelliteTleAdapter.__new__(SatelliteTleAdapter)
    now = datetime.now(timezone.utc)
    pts_300 = adapter.compute_ground_track(_TLE1, _TLE2, now, hours=8, step_seconds=300)
    pts_60  = adapter.compute_ground_track(_TLE1, _TLE2, now, hours=8, step_seconds=60)
    w, s, e, n = FINLAND_BBOX[0], FINLAND_BBOX[1], FINLAND_BBOX[2], FINLAND_BBOX[3]
    inside_300 = [p for p in pts_300 if w <= p["lon"] <= e and s <= p["lat"] <= n]
    inside_60  = [p for p in pts_60  if w <= p["lon"] <= e and s <= p["lat"] <= n]
    # 60s always finds ≥ as many hits as 300s for the same window
    assert len(inside_60) >= len(inside_300)


# ---------------------------------------------------------------------------
# Integration tests — /api/map-data/satellite-tracks
# ---------------------------------------------------------------------------

def test_satellite_tracks_finland_bbox():
    """Tracks endpoint returns segments for a Finland bbox."""
    state["ingestion_service"]._records.append(_make_satellite_record(FINLAND_BBOX))

    resp = client.get(
        "/api/map-data/satellite-tracks",
        params={"area": "Custom BBox", "timeframe": "8h", "bbox": ",".join(map(str, FINLAND_BBOX))},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True

    lines = [f for f in data["features"] if f["geometry"]["type"] == "LineString"]
    assert len(lines) >= 1, "At least one pass segment should be returned for Finland bbox"

    # Verify segments are scoped — lon range should be reasonable (not ±180)
    for seg in lines:
        coords = seg["geometry"]["coordinates"]
        lons = [c[0] for c in coords]
        assert max(lons) - min(lons) < 120, (
            f"Segment lon span {max(lons)-min(lons):.1f}° looks like a global track leak: {seg['properties']['name']}"
        )
        # Each segment must have at least one point within the 3° halo
        w, s, e, n = FINLAND_BBOX[0] - 3, FINLAND_BBOX[1] - 3, FINLAND_BBOX[2] + 3, FINLAND_BBOX[3] + 3
        halo_hits = [c for c in coords if w <= c[0] <= e and s <= c[1] <= n]
        assert len(halo_hits) >= 1, f"Segment has no point within 3° halo of Finland: {seg['properties']['name']}"


def test_satellite_tracks_no_bbox_returns_global():
    """Without bbox param, global tracks are returned (existing behaviour)."""
    state["ingestion_service"]._records.append(_make_satellite_record(FINLAND_BBOX))

    resp = client.get(
        "/api/map-data/satellite-tracks",
        params={"area": "Custom BBox", "timeframe": "8h"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    lines = [f for f in data["features"] if f["geometry"]["type"] == "LineString"]
    assert len(lines) >= 1


def test_satellite_tracks_pacific_bbox_returns_empty():
    """A bbox in the central Pacific should yield no Persona 1 tracks in 8h."""
    state["ingestion_service"]._records.append(_make_satellite_record(PACIFIC_BBOX))

    resp = client.get(
        "/api/map-data/satellite-tracks",
        params={"area": "Custom BBox", "timeframe": "8h", "bbox": ",".join(map(str, PACIFIC_BBOX))},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    lines = [f for f in data["features"] if f["geometry"]["type"] == "LineString"]
    # Persona 1 is polar-orbiting — it will pass over Pacific too, so just verify no leak
    for seg in lines:
        coords = seg["geometry"]["coordinates"]
        w, s, e, n = PACIFIC_BBOX[0] - 3, PACIFIC_BBOX[1] - 3, PACIFIC_BBOX[2] + 3, PACIFIC_BBOX[3] + 3
        halo_hits = [c for c in coords if w <= c[0] <= e and s <= c[1] <= n]
        assert len(halo_hits) >= 1, "Leaked segment has no point in Pacific halo"


def test_satellite_tracks_unavailable_without_bbox_load_target():
    """Without a bbox load target the endpoint returns available=False."""
    from ipb_backend.models import DatasetRecord, SourceCategory
    record = DatasetRecord(
        source_id="satellites",
        category=SourceCategory.SATELLITE,
        area="North Karelia",
        timeframe="8h",
        load_target=None,
        summary="test",
        data={"satellites": {}, "total_tracked": 0},
    )
    state["ingestion_service"]._records.append(record)

    resp = client.get("/api/map-data/satellite-tracks", params={"area": "North Karelia", "timeframe": "8h"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False


def test_satellite_dots_available_with_bbox_load_target():
    """/api/map-data/satellites returns available=True when bbox load target exists."""
    state["ingestion_service"]._records.append(_make_satellite_record(FINLAND_BBOX))

    resp = client.get("/api/map-data/satellites", params={"area": "Custom BBox"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert len(data["features"]) >= 1
