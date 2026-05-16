from __future__ import annotations

from typing import Any

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

    def _build_demo_cells(self, center: dict[str, float]) -> list[dict[str, Any]]:
        import random
        rng = random.Random(42)
        cells = []
        for i in range(12):
            lat = center["lat"] + rng.uniform(-0.05, 0.05)
            lon = center["lon"] + rng.uniform(-0.05, 0.05)
            radio = rng.choice(["LTE", "UMTS", "GSM"])
            cells.append({
                "cellid": 1000000 + i,
                "radio": radio,
                "mcc": 244,
                "mnc": rng.choice([5, 10, 12, 14, 21]),
                "lac": rng.randint(1000, 9999),
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "samples": rng.randint(10, 500),
                "range": rng.randint(500, 5000),
            })
        return cells

    async def fetch(self, area: str, timeframe: str) -> DatasetRecord:
        center = self._resolve_center(area)

        if not settings.opencellid_api_key:
            cells = self._build_demo_cells(center)
            return DatasetRecord(
                source_id=self.definition.source_id,
                category=self.definition.category,
                area=area,
                timeframe=timeframe,
                summary=f"OpenCellID: {len(cells)} demo cell towers near {area}",
                data={
                    "provider": "Demo data (OPENCELLID_API_KEY not configured)",
                    "api": "demo fallback",
                    "query": {
                        "area": area,
                        "lat": center["lat"],
                        "lon": center["lon"],
                    },
                    "cells": cells,
                    "total_cells": len(cells),
                    "note": "Demo fallback is used because OPENCELLID_API_KEY is not configured.",
                },
            )

        d = 0.5
        bbox = f"{center['lat'] - d},{center['lon'] - d},{center['lat'] + d},{center['lon'] + d}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            params: dict[str, Any] = {
                "key": self._get_api_key(),
                "BBOX": bbox,
                "format": "json",
                "limit": 100,
            }
            response = await client.get(f"{self.BASE_URL}/cell/getInArea", params=params)
            response.raise_for_status()
            data = response.json()

        cells = data.get("cells", [])
        total_cells = len(cells)
        if total_cells == 0:
            cells = self._build_demo_cells(center)
            total_cells = len(cells)
            provider = "Demo fallback (OpenCellID API returned empty)"
        else:
            provider = "OpenCellID (by Unwired Labs)"
        return DatasetRecord(
            source_id=self.definition.source_id,
            category=self.definition.category,
            area=area,
            timeframe=timeframe,
            summary=f"OpenCellID: {total_cells} cell towers near {area}",
            data={
                "provider": provider,
                "api": "cell/getInArea",
                "query": {
                    "area": area,
                    "lat": center["lat"],
                    "lon": center["lon"],
                    "search_area_sq_km": "~5000",
                },
                "cells": cells,
                "total_cells": total_cells,
            },
        )
