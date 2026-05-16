from __future__ import annotations

import httpx

from ipb_backend.config import settings
from ipb_backend.ingestion.base import SourceAdapter
from ipb_backend.models import DatasetRecord

AREA_CENTERS: dict[str, dict[str, float]] = {
    "north karelia": {"lat": 62.8, "lon": 30.2},
    "archipelago sea": {"lat": 60.2, "lon": 22.0},
    "lapland": {"lat": 68.9, "lon": 21.5},
    "lapland (kasivarren lappi)": {"lat": 68.9, "lon": 21.5},
    "kasivarren lappi": {"lat": 68.9, "lon": 21.5},
}


class OpenCellIdAdapter(SourceAdapter):
    BASE_URL = "https://opencellid.org"

    def _get_api_key(self) -> str:
        if not settings.opencellid_api_key:
            raise ValueError("OPENCELLID_API_KEY not configured in .env")
        return settings.opencellid_api_key

    def _resolve_center(self, area: str) -> dict[str, float]:
        normalized = area.lower().strip()
        return AREA_CENTERS.get(normalized, AREA_CENTERS["north karelia"])

    async def fetch(self, area: str, timeframe: str) -> DatasetRecord:
        api_key = self._get_api_key()
        center = self._resolve_center(area)

        d = 0.01
        bbox = f"{center['lat'] - d},{center['lon'] - d},{center['lat'] + d},{center['lon'] + d}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            params: dict[str, str | int] = {
                "key": api_key,
                "BBOX": bbox,
                "format": "json",
                "limit": 50,
            }
            response = await client.get(f"{self.BASE_URL}/cell/getInArea", params=params)
            response.raise_for_status()
            data = response.json()

        cells = data.get("cells", [])
        return DatasetRecord(
            source_id=self.definition.source_id,
            category=self.definition.category,
            area=area,
            timeframe=timeframe,
            summary=f"OpenCellID: {len(cells)} cell towers near {area}",
            data={
                "provider": "OpenCellID (by Unwired Labs)",
                "api": "cell/getInArea",
                "query": {
                    "area": area,
                    "lat": center["lat"],
                    "lon": center["lon"],
                    "search_area_sq_km": "~2.4",
                },
                "cells": cells,
                "total_cells": len(cells),
            },
        )
