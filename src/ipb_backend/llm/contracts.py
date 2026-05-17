from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from ipb_backend.analysis.contracts import DataPackage


class AnalysisProfile(str, Enum):
    GENERAL = "general"
    MOBILITY = "mobility"
    CONCEALMENT = "concealment"
    SURVEILLANCE = "surveillance"
    INFRASTRUCTURE_RESILIENCE = "infrastructure_resilience"
    LOGISTICS = "logistics"
    COMMUNICATIONS = "communications"


class ConversationTurn(BaseModel):
    role: str
    content: str


class LlmSelectionDigest(BaseModel):
    selection_type: str
    bounds_wgs84: list[float] = Field(default_factory=list)
    area_sqkm: float
    timeframe: str


class LlmSourceDigest(BaseModel):
    source_id: str
    category: str
    title: str
    summary: str
    confidence: str
    fallback_used: bool = False
    fallback_reason: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class LlmIndicatorDigest(BaseModel):
    indicator_id: str
    name: str
    value: Any
    unit: str
    method: str
    source_ids: list[str] = Field(default_factory=list)
    confidence: str = "medium"
    notes: list[str] = Field(default_factory=list)


class LlmEvidenceDigest(BaseModel):
    evidence_id: str
    source_id: str
    kind: str
    title: str
    detail: str
    support: str


class LlmWrapperGuardrails(BaseModel):
    overall_confidence: str = "medium"
    fallback_sources: list[str] = Field(default_factory=list)
    error_sources: list[str] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)


class LlmWrapperInput(BaseModel):
    schema_version: str = "1.0"
    package_id: str
    profile: AnalysisProfile = AnalysisProfile.GENERAL
    profile_focus: str
    question: str | None = None
    schematic_brief: str = ""
    selection: LlmSelectionDigest
    counts: dict[str, Any] = Field(default_factory=dict)
    source_digests: list[LlmSourceDigest] = Field(default_factory=list)
    indicator_digests: list[LlmIndicatorDigest] = Field(default_factory=list)
    evidence_catalog: list[LlmEvidenceDigest] = Field(default_factory=list)
    guardrails: LlmWrapperGuardrails
    conversation_history: list[ConversationTurn] = Field(default_factory=list)


class AnalysisStatement(BaseModel):
    id: str
    topic: str
    assessment: str
    statement: str
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: str = "medium"


class LimitationStatement(BaseModel):
    id: str
    statement: str
    evidence_refs: list[str] = Field(default_factory=list)
    severity: str = "medium"


class LlmAnalysisOutput(BaseModel):
    schema_version: str = "1.0"
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    profile: AnalysisProfile = AnalysisProfile.GENERAL
    status: str
    summary: str
    implications: list[AnalysisStatement] = Field(default_factory=list)
    defensive_relevance: list[AnalysisStatement] = Field(default_factory=list)
    limitations: list[LimitationStatement] = Field(default_factory=list)
    uncertainties: list[LimitationStatement] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LlmInterpretRequest(BaseModel):
    data_package: DataPackage
    profile: AnalysisProfile = AnalysisProfile.GENERAL
    question: str | None = None
    conversation_history: list[ConversationTurn] = Field(default_factory=list)