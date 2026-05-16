from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class SelectionContext(BaseModel):
    selection_type: str = "geometry"
    area_id: Optional[str] = None
    label: Optional[str] = None
    geometry: dict[str, Any]
    bounds_wgs84: list[float]
    area_sqkm: float


class DataScope(BaseModel):
    timeframe: str
    requested_sources: list[str] = Field(default_factory=list)
    resolved_sources: list[str] = Field(default_factory=list)


class SourceFreshnessRecord(BaseModel):
    source_id: str
    name: str
    category: Optional[str] = None
    status: str
    last_successful_refresh: Optional[datetime] = None
    last_error: Optional[str] = None
    retrieved_at: Optional[datetime] = None
    refresh_interval_seconds: Optional[int] = None
    freshness_label: Optional[str] = None


class SourceProvenance(BaseModel):
    provider: str
    adapter: Optional[str] = None
    retrieved_at: Optional[datetime] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    deterministic: bool = True
    note: Optional[str] = None


class SourceSummary(BaseModel):
    source_id: str
    category: str
    title: str
    summary: str
    raw_summary: dict[str, Any] = Field(default_factory=dict)
    confidence: str = "high"
    provenance: SourceProvenance


class CountsSummary(BaseModel):
    by_source: dict[str, int] = Field(default_factory=dict)
    by_category: dict[str, int] = Field(default_factory=dict)
    geometry_types: dict[str, int] = Field(default_factory=dict)


class DerivedIndicator(BaseModel):
    indicator_id: str
    name: str
    value: Any
    unit: str
    method: str
    source_ids: list[str] = Field(default_factory=list)
    confidence: str = "medium"
    notes: list[str] = Field(default_factory=list)


class EvidenceDataRef(BaseModel):
    section: str
    path: str


class EvidenceItem(BaseModel):
    evidence_id: str
    source_id: str
    kind: str
    title: str
    detail: str
    support: str
    data_ref: Optional[EvidenceDataRef] = None


class QualitySummary(BaseModel):
    fallback_sources: list[str] = Field(default_factory=list)
    error_sources: list[str] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    overall_confidence: str = "medium"


class DataPackage(BaseModel):
    schema_version: str = "1.0"
    package_id: str = Field(default_factory=lambda: str(uuid4()))
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    selection: SelectionContext
    scope: DataScope
    source_freshness: list[SourceFreshnessRecord] = Field(default_factory=list)
    source_summaries: list[SourceSummary] = Field(default_factory=list)
    counts: CountsSummary
    derived_indicators: list[DerivedIndicator] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    quality: QualitySummary