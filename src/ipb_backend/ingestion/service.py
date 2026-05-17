from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Optional

from ipb_backend.ingestion.base import SourceAdapter
from ipb_backend.ingestion.registry import SourceRegistry
from ipb_backend.models import DatasetRecord, IngestionRequest, IngestionResult, LoadTarget, SourceStatus


def _load_target_key(load_target: Optional[LoadTarget]) -> str:
    """Stable signature for deduplication. Same target → same key."""
    if load_target is None:
        return "named"
    payload = {
        "kind": load_target.kind.value if hasattr(load_target.kind, "value") else load_target.kind,
        "label": load_target.label,
        "bbox": load_target.bbox_wgs84,
        "geom": load_target.geometry,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class IngestionService:
    def __init__(self, registry: SourceRegistry, adapters: dict[str, SourceAdapter]) -> None:
        self._registry = registry
        self._adapters = adapters
        # Keyed by (source_id, area, load_target_signature) so repeated ingests
        # for the same scope overwrite rather than accumulate. Timeframe is
        # intentionally NOT part of the key — the latest record per scope wins,
        # which matches "Load Data" UX.
        self._records: dict[tuple[str, str, str], DatasetRecord] = {}

    @property
    def records(self) -> list[DatasetRecord]:
        return list(self._records.values())

    async def ingest(self, request: IngestionRequest) -> IngestionResult:
        source_ids = request.source_ids or self._registry.enabled_source_ids()
        missing_source_ids = self._registry.missing_source_ids(source_ids)
        if missing_source_ids:
            missing_list = ", ".join(sorted(missing_source_ids))
            raise ValueError(f"Unknown source ids: {missing_list}")

        tasks = [
            self._fetch_one_timed(source_id, request.area, request.timeframe, request.load_target)
            for source_id in source_ids
        ]
        records = [r for r in await asyncio.gather(*tasks) if r is not None]
        lt_key = _load_target_key(request.load_target)
        for record in records:
            self._records[(record.source_id, record.area, lt_key)] = record
        return IngestionResult(requested_sources=source_ids, produced_records=records)

    async def _fetch_one_timed(self, source_id: str, area: str, timeframe: str, load_target=None):
        try:
            return await asyncio.wait_for(
                self._fetch_one(source_id, area, timeframe, load_target),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            definition = self._registry.get(source_id)
            updated = definition.model_copy(update={"status": SourceStatus.ERROR, "last_error": "Ingestion timed out after 120s"})
            self._registry.update(updated)
            return None

    async def _fetch_one(self, source_id: str, area: str, timeframe: str, load_target=None):
        definition = self._registry.get(source_id)
        if not definition.enabled:
            return None

        adapter = self._adapters.get(source_id)
        if adapter is None:
            raise ValueError(f"No adapter configured for source id: {source_id}")

        self._registry.update(definition.model_copy(update={"status": SourceStatus.RUNNING}))
        try:
            record = await adapter.fetch(area, timeframe, load_target)
            # Attach the load_target so downstream readers (gate satellite
            # overlays to bbox loads, distinguish per-bbox vs per-area records)
            # don't have to re-derive it from the request payload.
            if record is not None and record.load_target is None and load_target is not None:
                record = record.model_copy(update={"load_target": load_target})
            updated_definition = definition.model_copy(
                update={
                    "status": SourceStatus.READY,
                    "last_successful_refresh": datetime.now(timezone.utc),
                    "last_error": None,
                }
            )
            self._registry.update(updated_definition)
            return record
        except Exception as exc:
            updated_definition = definition.model_copy(
                update={
                    "status": SourceStatus.ERROR,
                    "last_error": str(exc),
                }
            )
            self._registry.update(updated_definition)
            return None
