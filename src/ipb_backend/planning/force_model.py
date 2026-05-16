from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class OperationType(str, Enum):
    DEFENSIVE = "defensive"
    OFFENSIVE = "offensive"
    RECON = "recon"
    SCREEN = "screen"
    LOGISTICS_HUB = "logistics_hub"
    WITHDRAWAL = "withdrawal"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Vehicle(BaseModel):
    designation: str = Field(..., description="Model/type, e.g. 'BMP-2', 'Leopard 2A6'")
    count: int = Field(1, ge=1)
    weight_t: float = Field(..., ge=0, description="Combat weight in tonnes")
    width_m: float = Field(..., ge=0)
    height_m: float = Field(..., ge=0)
    length_m: float = Field(..., ge=0)
    ground_pressure_kpa: Optional[float] = Field(None, ge=0)
    fording_depth_m: Optional[float] = Field(None, ge=0)
    max_road_speed_kmh: Optional[float] = Field(None, ge=0)
    off_road_capable: bool = True


class Drone(BaseModel):
    designation: str
    count: int = Field(1, ge=1)
    max_wind_ms: float = Field(..., ge=0, description="Maximum operable wind speed")
    max_precip_mm_h: float = Field(5.0, ge=0)
    min_visibility_m: float = Field(1000.0, ge=0)
    operating_temp_c: tuple[float, float] = Field((-20.0, 45.0))
    range_km: float = Field(..., ge=0)
    endurance_min: Optional[float] = Field(None, ge=0)


class ForceComposition(BaseModel):
    infantry: int = Field(0, ge=0)
    vehicles: list[Vehicle] = Field(default_factory=list)
    drones: list[Drone] = Field(default_factory=list)
    logistics_demand_t_per_day: float = Field(0.0, ge=0)
    comms_range_required_km: float = Field(0.0, ge=0)
    column_movement: bool = Field(
        True,
        description="Whether vehicles need to move two-abreast on roads. False = single file.",
    )

    @property
    def heaviest_vehicle_t(self) -> float:
        return max((v.weight_t for v in self.vehicles), default=0.0)

    @property
    def widest_vehicle_m(self) -> float:
        return max((v.width_m for v in self.vehicles), default=0.0)

    @property
    def tallest_vehicle_m(self) -> float:
        return max((v.height_m for v in self.vehicles), default=0.0)

    @property
    def min_drone_wind_tolerance_ms(self) -> Optional[float]:
        if not self.drones:
            return None
        return min(d.max_wind_ms for d in self.drones)


class Operation(BaseModel):
    type: OperationType
    duration_hours: int = Field(24, ge=1)
    concealment_priority: Priority = Priority.MEDIUM
    speed_priority: Priority = Priority.MEDIUM
    civilian_avoidance: Priority = Priority.MEDIUM
    comms_priority: Priority = Priority.MEDIUM
    notes: Optional[str] = None


class PlanningRequest(BaseModel):
    area: str = Field(..., description="Named area, e.g. 'North Karelia'")
    timeframe: str = Field("24h")
    geometry: Optional[dict[str, Any]] = Field(
        None,
        description="Optional AOI polygon (GeoJSON). If omitted, the named area bbox is used.",
    )
    grid_resolution_m: int = Field(
        1000,
        ge=200,
        le=5000,
        description="Cell edge length in meters for the scoring grid.",
    )
    top_n: int = Field(5, ge=1, le=20)
    force: ForceComposition
    operation: Operation
    explain: bool = Field(
        False,
        description="If true, run the configured analyzer (Ollama/Rules) to narrate the top sites.",
    )

    @field_validator("geometry")
    @classmethod
    def _validate_geometry(cls, value: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if value is None:
            return value
        if "type" not in value or "coordinates" not in value:
            raise ValueError("geometry must be a GeoJSON Geometry with 'type' and 'coordinates'")
        return value


class ConstraintMatch(BaseModel):
    name: str
    passed: bool
    observed: Optional[float] = None
    required: Optional[float] = None
    detail: Optional[str] = None


class RecommendedSite(BaseModel):
    rank: int
    score: float = Field(..., ge=0, le=1)
    centroid: list[float] = Field(..., description="[lon, lat]")
    geometry: dict[str, Any]
    feasible: bool
    constraint_matches: list[ConstraintMatch] = Field(default_factory=list)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    rationale: list[str] = Field(default_factory=list)
    narrative: Optional[str] = None


class PlanningResponse(BaseModel):
    area: str
    timeframe: str
    operation_type: OperationType
    grid_resolution_m: int
    cells_evaluated: int
    feasible_cells: int
    top_sites: list[RecommendedSite]
    weights: dict[str, float]
    data_freshness: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
