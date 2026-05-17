from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timezone
from typing import Optional

from ipb_backend.ingestion.base import SourceAdapter
from ipb_backend.ingestion.registry import SourceRegistry
from ipb_backend.models import DatasetRecord, IngestionRequest, IngestionResult, LoadTarget, LoadTargetKind, SourceStatus


def _load_target_key(load_target: Optional[LoadTarget]) -> str:
    """Storage signature for deduplication.

    Named-area loads are stable and can be retained per area. Volatile custom
    bbox/geometry loads are collapsed to the latest record for that source and
    area so repeated viewport changes do not grow the in-memory cache without
    bound.
    """
    if load_target is None:
        return "named"
    if load_target.kind in {LoadTargetKind.BBOX, LoadTargetKind.GEOMETRY}:
        return load_target.kind.value
    payload = {
        "kind": load_target.kind.value if hasattr(load_target.kind, "value") else load_target.kind,
        "label": load_target.label,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _fetch_cache_key(area: str, timeframe: str, load_target: Optional[LoadTarget]) -> str:
    """Exact request signature for short-lived fetch caching.

    Unlike storage dedupe, fetch caching must distinguish different AOIs and
    timeframes so a fresh request never reuses a mismatched prior response.
    """
    if load_target is None:
        payload = {"area": area, "timeframe": timeframe, "target": None}
    else:
        payload = {
            "area": area,
            "timeframe": timeframe,
            "target": {
                "kind": load_target.kind.value if hasattr(load_target.kind, "value") else load_target.kind,
                "label": load_target.label,
                "bbox": load_target.bbox_wgs84,
                "geom": load_target.geometry,
            },
        }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _record_storage_key(record: DatasetRecord) -> tuple[str, str, str]:
    return (record.source_id, record.area, _load_target_key(record.load_target))


class _RecordStore:
    """Deduplicated record storage with list-like helpers for tests/admin use."""

    def __init__(self, on_clear: Callable[[], None] | None = None) -> None:
        self._by_key: dict[tuple[str, str, str], DatasetRecord] = {}
        self._on_clear = on_clear

    def __setitem__(self, key: tuple[str, str, str], record: DatasetRecord) -> None:
        self._by_key[key] = record

    def values(self):
        return self._by_key.values()

    def clear(self) -> None:
        self._by_key.clear()
        if self._on_clear is not None:
            self._on_clear()

    def append(self, record: DatasetRecord) -> None:
        key = _record_storage_key(record)
        # Updating an existing key should count as the newest value so callers
        # that walk records in insertion order still see the latest volatile AOI.
        self._by_key.pop(key, None)
        self._by_key[key] = record

    def extend(self, records: Iterable[DatasetRecord]) -> None:
        for record in records:
            self.append(record)

    def __iter__(self) -> Iterator[DatasetRecord]:
        return iter(self._by_key.values())

    def __len__(self) -> int:
        return len(self._by_key)

class IngestionService:
    def __init__(self, registry: SourceRegistry, adapters: dict[str, SourceAdapter]) -> None:
        self._registry = registry
        self._adapters = adapters
        self._fetch_cache: dict[tuple[str, str], tuple[float, DatasetRecord]] = {}
        # Keyed by (source_id, area, load_target_signature) so repeated ingests
        # for the same scope overwrite rather than accumulate. Timeframe is
        # intentionally NOT part of the key — the latest record per scope wins,
        # which matches "Load Data" UX.
        self._records = _RecordStore(on_clear=self._fetch_cache.clear)
        # Fetch cache: (source_id, lt_key) → (expires_at_monotonic, DatasetRecord)
        # TTL comes from each source's refresh_interval_seconds so we never serve
        # data staler than a background refresh would produce.

    @property
    def records(self) -> list[DatasetRecord]:
        return list(self._records.values())

    def _cache_get(self, source_id: str, lt_key: str) -> DatasetRecord | None:
        entry = self._fetch_cache.get((source_id, lt_key))
        if entry is None:
            return None
        expires_at, record = entry
        if time.monotonic() > expires_at:
            del self._fetch_cache[(source_id, lt_key)]
            return None
        return record

    def _cache_put(self, source_id: str, lt_key: str, record: DatasetRecord) -> None:
        definition = self._registry.get(source_id)
        ttl = float(getattr(definition, "refresh_interval_seconds", 300) or 300)
        self._fetch_cache[(source_id, lt_key)] = (time.monotonic() + ttl, record)

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
        for record in records:
            self._records.append(record)

        # Rebuild spatial index so point-inspection queries stay O(log n).
        from ipb_backend.spatial import nearby_index
        nearby_index.rebuild(list(self._records.values()))
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

        cache_key = _fetch_cache_key(area, timeframe, load_target)
        cached = self._cache_get(source_id, cache_key)
        if cached is not None:
            self._registry.update(definition.model_copy(
                update={"status": SourceStatus.READY, "last_error": None}
            ))
            return cached

        self._registry.update(definition.model_copy(update={"status": SourceStatus.RUNNING}))
        try:
            record = await adapter.fetch(area, timeframe, load_target)
            if record is not None and record.load_target is None and load_target is not None:
                record = record.model_copy(update={"load_target": load_target})
            self._registry.update(definition.model_copy(
                update={
                    "status": SourceStatus.READY,
                    "last_successful_refresh": datetime.now(timezone.utc),
                    "last_error": None,
                }
            ))
            if record is not None:
                self._cache_put(source_id, cache_key, record)
            return record
        except Exception as exc:
            self._registry.update(definition.model_copy(
                update={"status": SourceStatus.ERROR, "last_error": str(exc)}
            ))
            return None
