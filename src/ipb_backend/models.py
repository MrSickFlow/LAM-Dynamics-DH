from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


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


class DatasetRecord(BaseModel):
    source_id: str
    category: SourceCategory
    area: str
    timeframe: str
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
    last_successful_refresh: datetime | None = None
    last_error: str | None = None


class IngestionRequest(BaseModel):
    area: str
    timeframe: str
    source_ids: list[str] | None = None


class IngestionResult(BaseModel):
    requested_sources: list[str]
    produced_records: list[DatasetRecord]


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
