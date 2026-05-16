from __future__ import annotations

from ipb_backend.ingestion.base import SourceAdapter
from ipb_backend.models import DatasetRecord


class DemoSourceAdapter(SourceAdapter):
    async def fetch(self, area: str, timeframe: str) -> DatasetRecord:
        return DatasetRecord(
            source_id=self.definition.source_id,
            category=self.definition.category,
            area=area,
            timeframe=timeframe,
            summary=f"Placeholder dataset from {self.definition.name} for {area}",
            data={
                "refresh_interval_seconds": self.definition.refresh_interval_seconds,
                "note": "Replace this payload with real API integration.",
            },
        )
