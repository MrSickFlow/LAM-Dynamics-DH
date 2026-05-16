from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from ipb_backend.analysis.contracts import DataPackage
from ipb_backend.llm.contracts import AnalysisProfile, LlmAnalysisOutput, LlmInterpretRequest, LlmWrapperInput


class SourceCategory(str, Enum):
    TERRAIN = "terrain"
    WEATHER = "weather"
    INFRASTRUCTURE = "infrastructure"
    DEMOGRAPHICS = "demographics"
    SATELLITE = "satellite"
    OTHER = "other"


class SourceStatus(str, Enum):
    IDLE = "idle"
    READY = "ready"
    ERROR = "error"
    DISABLED = "disabled"


class LoadTargetKind(str, Enum):
    NAMED_AREA = "named_area"
    BBOX = "bbox"
    GEOMETRY = "geometry"


class LoadTarget(BaseModel):
    kind: LoadTargetKind
    label: Optional[str] = None
    bbox_wgs84: Optional[list[float]] = None
    geometry: Optional[dict[str, Any]] = None


class DatasetRecord(BaseModel):
    source_id: str
    category: SourceCategory
    area: str
    timeframe: str
    load_target: Optional[LoadTarget] = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    summary: str
    data: dict[str, Any]


class SourceDefinition(BaseModel):
    source_id: str
    name: str
    category: SourceCategory
    description: str
    refresh_interval_seconds: int
    enabled: bool = True
    status: SourceStatus = SourceStatus.IDLE
    last_successful_refresh: Optional[datetime] = None
    last_error: Optional[str] = None


class IngestionRequest(BaseModel):
    area: str = "North Karelia"
    timeframe: str
    load_target: Optional[LoadTarget] = None
    source_ids: Optional[list[str]] = None


class IngestionResult(BaseModel):
    requested_sources: list[str]
    produced_records: list[DatasetRecord]


class AoiInspectionRequest(BaseModel):
    geometry: dict[str, Any]
    timeframe: Optional[str] = None
    profile: AnalysisProfile = AnalysisProfile.GENERAL


class AoiInspectionResponse(BaseModel):
    selection: dict[str, Any]
    metrics: dict[str, Any]
    raw_data: dict[str, Any]
    raw_sections: list[dict[str, Any]]
    freshness: list[dict[str, Any]]
    data_package: DataPackage
    llm_input: LlmWrapperInput
    llm_output: LlmAnalysisOutput
    agent: dict[str, Any]


class UiLayer(BaseModel):
    layer_id: str
    title: str
    category: SourceCategory
    enabled_by_default: bool = True
    description: str


class UiPlaceholderResponse(BaseModel):
    area: str
    timeframe: str
    map_layers: list[UiLayer]
    dashboard_cards: list[str]


class AgentDefinition(BaseModel):
    agent_id: str
    name: str
    purpose: str
    status: str


class AgentRunResult(BaseModel):
    agent_id: str
    area: str
    timeframe: str
    summary: str
    findings: list[str]
    data: Optional[dict[str, Any]] = None
