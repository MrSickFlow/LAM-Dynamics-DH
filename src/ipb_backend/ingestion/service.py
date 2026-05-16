from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from ipb_backend.ingestion.base import SourceAdapter
from ipb_backend.ingestion.registry import SourceRegistry
from ipb_backend.models import IngestionRequest, IngestionResult, SourceStatus


class IngestionService:
    def __init__(self, registry: SourceRegistry, adapters: dict[str, SourceAdapter]) -> None:
        self._registry = registry
        self._adapters = adapters
        self._records = []

    @property
    def records(self):
        return list(self._records)

    async def ingest(self, request: IngestionRequest) -> IngestionResult:
        source_ids = request.source_ids or self._registry.enabled_source_ids()
        missing_source_ids = self._registry.missing_source_ids(source_ids)
        if missing_source_ids:
            missing_list = ", ".join(sorted(missing_source_ids))
            raise ValueError(f"Unknown source ids: {missing_list}")

        tasks = [self._fetch_one(source_id, request.area, request.timeframe, request.load_target) for source_id in source_ids]
        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=50.0)
        except asyncio.TimeoutError:
            results = []
        records = [record for record in results if record is not None]
        self._records.extend(records)
        return IngestionResult(requested_sources=source_ids, produced_records=records)

    async def _fetch_one(self, source_id: str, area: str, timeframe: str, load_target=None):
        definition = self._registry.get(source_id)
        if not definition.enabled:
            return None

        adapter = self._adapters.get(source_id)
        if adapter is None:
            raise ValueError(f"No adapter configured for source id: {source_id}")

        try:
            record = await adapter.fetch(area, timeframe, load_target)
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
