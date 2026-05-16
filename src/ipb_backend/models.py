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


class EwClassification(str, Enum):
    IMMUNE = "immune"
    VULNERABLE = "vulnerable"
    MIXED = "mixed"


class AnomalySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


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
    ew_classification: EwClassification = EwClassification.MIXED
    gnss_dependent: bool = False
    ew_rationale: Optional[str] = None


class LikelyExplanation(BaseModel):
    cause: str
    likelihood: float = Field(ge=0.0, le=1.0)
    note: str


class ConsistencyAnomaly(BaseModel):
    anomaly_id: str
    rule_id: str
    title: str
    description: str
    severity: AnomalySeverity
    location: Optional[dict[str, Any]] = None
    vulnerable_sources: list[str] = Field(default_factory=list)
    immune_sources: list[str] = Field(default_factory=list)
    measured: dict[str, Any] = Field(default_factory=dict)
    expected: dict[str, Any] = Field(default_factory=dict)
    likely_explanations: list[LikelyExplanation] = Field(default_factory=list)
    synthetic_demo: bool = False


class LayerTrustScore(BaseModel):
    source_id: str
    ew_classification: EwClassification
    gnss_dependent: bool
    confidence: float = Field(ge=0.0, le=1.0)
    status: SourceStatus
    staleness_seconds: Optional[int] = None
    factors: list[str] = Field(default_factory=list)


class SpatialCluster(BaseModel):
    cluster_id: str
    centroid: dict[str, float]
    radius_km: float
    anomaly_count: int
    anomaly_ids: list[str]
    affected_sources: list[str]
    pattern_assessment: str


class ConsistencyReport(BaseModel):
    area: str
    timeframe: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    layer_trust: list[LayerTrustScore]
    anomalies: list[ConsistencyAnomaly]
    clusters: list[SpatialCluster]
    summary: str
    ew_pattern_detected: bool
    disclaimer: str = (
        "Anomalies indicate cross-source disagreement, not confirmed EW. "
        "Analysts assess whether degradation reflects jamming, spoofing, staleness, or equipment failure."
    )


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
