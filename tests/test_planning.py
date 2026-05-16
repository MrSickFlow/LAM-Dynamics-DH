from datetime import datetime, timezone

from fastapi.testclient import TestClient

from ipb_backend.main import app, state
from ipb_backend.models import DatasetRecord, SourceCategory
from ipb_backend.planning import (
    Artillery,
    OPERATION_PROFILES,
    Drone,
    ForceComposition,
    Operation,
    OperationType,
    PlanningRequest,
    Vehicle,
    recommend_sites,
)
from ipb_backend.planning.constraints import CellFeatures, check_constraints
from ipb_backend.planning.operations import SCORING_CRITERIA, get_operation_profile


client = TestClient(app)


def _sample_force() -> ForceComposition:
    return ForceComposition(
        infantry=120,
        vehicles=[
            Vehicle(
                designation="BMP-2",
                count=4,
                weight_t=14.0,
                width_m=3.15,
                height_m=2.45,
                length_m=6.74,
            ),
        ],
        drones=[
            Drone(
                designation="Bayraktar TB2",
                count=2,
                max_wind_ms=12.0,
                max_precip_mm_h=2.0,
                min_visibility_m=2000.0,
                range_km=150.0,
            )
        ],
        logistics_demand_t_per_day=8.0,
        comms_range_required_km=20.0,
    )


def _sample_operation(op_type: OperationType = OperationType.DEFENSIVE) -> Operation:
    return Operation(type=op_type, duration_hours=24)


def _digiroad_record() -> DatasetRecord:
    return DatasetRecord(
        source_id="digiroad",
        category=SourceCategory.INFRASTRUCTURE,
        area="North Karelia",
        timeframe="24h",
        retrieved_at=datetime.now(timezone.utc),
        summary="digiroad",
        data={
            "collections": {
                "dr_tielinkki_silta_alikulku_tunneli": {
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [30.2, 62.5]},
                            "properties": {"link_id": "L1", "silta_alik": 0},
                        }
                    ]
                },
                "dr_max_massa": {
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [30.2, 62.5]},
                            "properties": {"link_id": "L1", "arvo": 60000},
                        }
                    ]
                },
                "dr_max_korkeus": {"features": []},
                "dr_max_leveys": {"features": []},
                "dr_leveys": {
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "LineString",
                                "coordinates": [[30.15, 62.45], [30.25, 62.55]],
                            },
                            "properties": {"link_id": "L1", "arvo": 800},
                        }
                    ]
                },
            }
        },
    )


def _fmi_record(wind: float = 5.0, precip: float = 0.0) -> DatasetRecord:
    return DatasetRecord(
        source_id="fmi",
        category=SourceCategory.WEATHER,
        area="North Karelia",
        timeframe="24h",
        retrieved_at=datetime.now(timezone.utc),
        summary="fmi",
        data={
            "station": {"name": "Joensuu", "region": "Joensuu"},
            "observations": {
                "temperature": {"latest": {"value": 8.0}},
                "wind_speed": {"latest": {"value": wind}},
                "wind_gust": {"latest": {"value": wind + 1.5}},
                "precipitation": {"latest": {"value": precip}},
                "cloud_cover": {"latest": {"value": 60.0}},
                "humidity": {"latest": {"value": 70.0}},
            },
        },
    )


def _osm_record() -> DatasetRecord:
    return DatasetRecord(
        source_id="osm-poi",
        category=SourceCategory.OTHER,
        area="North Karelia",
        timeframe="24h",
        retrieved_at=datetime.now(timezone.utc),
        summary="osm",
        data={
            "categories": {
                "forest": [
                    {
                        "lat": 62.5,
                        "lon": 30.2,
                        "tags": {"leaf_type": "needleleaved", "leaf_cycle": "evergreen"},
                    }
                ],
                "education": [
                    {"lat": 62.4, "lon": 30.1, "tags": {"name": "School"}}
                ],
            }
        },
    )


def test_operation_weights_sum_to_one():
    for op_type, weights in OPERATION_PROFILES.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, f"{op_type} weights total {total}"
        for criterion in SCORING_CRITERIA:
            assert criterion in weights


def test_priority_adjustment_renormalizes():
    operation = Operation(
        type=OperationType.DEFENSIVE,
        concealment_priority="high",
        speed_priority="low",
        civilian_avoidance="medium",
        comms_priority="medium",
    )
    weights = get_operation_profile(operation)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    base_defensive = OPERATION_PROFILES[OperationType.DEFENSIVE]
    assert weights["concealment"] > base_defensive["concealment"]
    assert weights["road_access"] < base_defensive["road_access"]


def test_constraint_match_bridge_weight_passes_and_fails():
    features = CellFeatures()
    features.min_bridge_capacity_t = 60.0
    matches = check_constraints(features, _sample_force())
    bridge_match = next(m for m in matches if m.name == "bridge_weight")
    assert bridge_match.passed is True

    heavy_force = ForceComposition(
        vehicles=[
            Vehicle(
                designation="Leopard 2",
                count=1,
                weight_t=62.0,
                width_m=3.75,
                height_m=3.0,
                length_m=9.97,
            )
        ]
    )
    matches = check_constraints(features, heavy_force)
    bridge_match = next(m for m in matches if m.name == "bridge_weight")
    assert bridge_match.passed is False
    assert bridge_match.required == 62.0
    assert bridge_match.observed == 60.0


def test_constraint_match_drone_wind_uses_gust_when_available():
    features = CellFeatures()
    features.weather_wind_ms = 8.0
    features.weather_gust_ms = 14.0
    matches = check_constraints(features, _sample_force())
    wind_match = next(m for m in matches if m.name == "drone_wind")
    assert wind_match.passed is False
    assert wind_match.observed == 14.0


def test_recommend_sites_runs_without_ingested_data():
    request = PlanningRequest(
        area="North Karelia",
        force=_sample_force(),
        operation=_sample_operation(),
        grid_resolution_m=5000,
        top_n=3,
    )
    response = recommend_sites(request, records=[])
    assert response.cells_evaluated > 0
    assert len(response.top_sites) <= 3
    assert any("No ingested data" in note for note in response.notes)


def test_recommend_sites_with_records_scores_and_ranks():
    request = PlanningRequest(
        area="North Karelia",
        force=_sample_force(),
        operation=_sample_operation(OperationType.DEFENSIVE),
        grid_resolution_m=2500,
        top_n=5,
    )
    records = [_digiroad_record(), _fmi_record(), _osm_record()]
    response = recommend_sites(request, records=records)

    assert response.cells_evaluated > 0
    assert response.top_sites
    scores = [site.score for site in response.top_sites]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)

    top = response.top_sites[0]
    assert "concealment" in top.score_breakdown
    assert set(response.weights) == set(SCORING_CRITERIA)


def test_recommend_sites_marks_unfeasible_when_wind_exceeds_drone_limit():
    request = PlanningRequest(
        area="North Karelia",
        force=_sample_force(),
        operation=_sample_operation(),
        grid_resolution_m=5000,
        top_n=2,
    )
    records = [_fmi_record(wind=20.0)]
    response = recommend_sites(request, records=records)
    assert response.feasible_cells == 0
    top = response.top_sites[0]
    assert top.feasible is False
    failed_names = [m.name for m in top.constraint_matches if not m.passed]
    assert "drone_wind" in failed_names


def test_planning_profiles_endpoint():
    response = client.get("/api/planning/profiles")
    assert response.status_code == 200
    payload = response.json()
    op_ids = {entry["id"] for entry in payload["operation_types"]}
    assert op_ids == {op.value for op in OperationType}


def test_planning_recommend_endpoint_round_trip():
    state["ingestion_service"]._records.clear()
    state["ingestion_service"]._records.extend(
        [_digiroad_record(), _fmi_record(), _osm_record()]
    )

    body = {
        "area": "North Karelia",
        "timeframe": "24h",
        "grid_resolution_m": 5000,
        "top_n": 2,
        "force": {
            "infantry": 50,
            "vehicles": [
                {
                    "designation": "BMP-2",
                    "count": 2,
                    "weight_t": 14.0,
                    "width_m": 3.15,
                    "height_m": 2.45,
                    "length_m": 6.74,
                }
            ],
            "drones": [
                {
                    "designation": "TB2",
                    "count": 1,
                    "max_wind_ms": 12.0,
                    "range_km": 100,
                }
            ],
        },
        "operation": {"type": "defensive", "duration_hours": 24},
    }

    response = client.post("/api/planning/recommend", json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["area"] == "North Karelia"
    assert payload["operation_type"] == "defensive"
    assert payload["top_sites"]
    assert payload["top_sites"][0]["rank"] == 1
    assert "concealment" in payload["weights"]


def test_artillery_model_and_force_properties():
    arty = Artillery(designation="K9 Thunder", count=2, weight_t=48.5, caliber_mm=155, max_range_km=40.0, is_self_propelled=True)
    assert arty.weight_t == 48.5
    assert arty.is_self_propelled is True

    force = ForceComposition(
        artillery=[arty, Artillery(designation="D-30", count=4, weight_t=3.2, caliber_mm=122, max_range_km=22.0, is_self_propelled=False)],
    )
    assert force.heaviest_artillery_t == 48.5


def test_constraint_artillery_bridge_weight():
    features = CellFeatures()
    features.min_bridge_capacity_t = 40.0

    force = ForceComposition(
        artillery=[Artillery(designation="K9 Thunder", count=1, weight_t=48.5, caliber_mm=155, max_range_km=40.0, is_self_propelled=True)],
    )
    matches = check_constraints(features, force)
    arty_match = next(m for m in matches if m.name == "bridge_weight_artillery")
    assert arty_match.passed is False
    assert arty_match.required == 48.5

    light_force = ForceComposition(
        artillery=[Artillery(designation="D-30", count=2, weight_t=3.2, caliber_mm=122, max_range_km=22.0, is_self_propelled=False)],
    )
    matches = check_constraints(features, light_force)
    arty_match = next(m for m in matches if m.name == "bridge_weight_artillery")
    assert arty_match.passed is True


def test_fire_support_profile_in_operation_profiles():
    assert OperationType.FIRE_SUPPORT in OPERATION_PROFILES
    weights = OPERATION_PROFILES[OperationType.FIRE_SUPPORT]
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert weights["concealment"] >= 0.25


def test_planning_recommend_endpoint_rejects_bad_geometry():
    body = {
        "area": "North Karelia",
        "geometry": {"type": "BadType"},
        "force": {"infantry": 1},
        "operation": {"type": "defensive"},
    }
    response = client.post("/api/planning/recommend", json=body)
    assert response.status_code == 422
