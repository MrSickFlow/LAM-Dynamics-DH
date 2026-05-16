from __future__ import annotations

from ipb_backend.consistency.fixtures import ARCHIPELAGO_MARITIME_FIXTURE
from ipb_backend.ingestion.base import SourceAdapter
from ipb_backend.models import DatasetRecord


class MaritimeDemoAdapter(SourceAdapter):
    """Demonstration AIS + SAR cross-check for Archipelago Sea maritime IPB."""

    async def fetch(self, area: str, timeframe: str) -> DatasetRecord:
        fixture = ARCHIPELAGO_MARITIME_FIXTURE
        return DatasetRecord(
            source_id=self.definition.source_id,
            category=self.definition.category,
            area=area,
            timeframe=timeframe,
            summary=(
                f"Maritime demo: {len(fixture['ais_vessels'])} AIS tracks, "
                f"{len(fixture['sar_returns'])} SAR returns for cross-validation"
            ),
            data={
                "provider": "IPB maritime demonstration fixture",
                "ais_vessels": fixture["ais_vessels"],
                "sar_returns": fixture["sar_returns"],
                "note": "Synthetic AIS/SAR mismatch scenario for Data Consistency Engine demos.",
            },
        )
