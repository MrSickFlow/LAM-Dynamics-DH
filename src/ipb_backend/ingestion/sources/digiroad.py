from __future__ import annotations

import asyncio
import re
import unicodedata
from typing import Any

import httpx

from ipb_backend.ingestion.base import SourceAdapter
from ipb_backend.models import DatasetRecord


class DigiroadAdapter(SourceAdapter):
    BASE_URL = "https://avoinapi.vaylapilvi.fi/vaylatiedot/digiroad/ogc/features/v1"

    AREA_BBOXES: dict[str, tuple[float, float, float, float]] = {
        "archipelago sea": (21.0, 59.7, 23.0, 60.6),
        "north karelia": (29.0, 62.0, 31.5, 63.5),
        "lapland": (20.5, 68.5, 22.5, 69.4),
        "lapland (kasivarren lappi)": (20.5, 68.5, 22.5, 69.4),
        "kasivarren lappi": (20.5, 68.5, 22.5, 69.4),
    }

    COLLECTIONS: dict[str, str] = {
        "dr_nopeusrajoitus": "Speed limits",
        "dr_tielinkki_silta_alikulku_tunneli": "Bridges, underpasses, tunnels",
        "dr_max_massa": "Max weight limit",
        "dr_max_korkeus": "Max height limit",
        "dr_max_leveys": "Max width limit",
        "dr_max_akselimassa": "Max axle mass",
        "dr_yhdistelman_max_massa": "Combined max mass",
        "dr_tielinkki_tielinkin_tyyppi": "Road link type",
        "dr_tielinkki_toim_lk": "Functional class",
        "dr_paallystetty_tie": "Paved road",
        "dr_leveys": "Road width",
        "dr_liikennemaara": "Traffic volume",
        "dr_palvelu": "Service points",
        "dr_valaistu_tie": "Lit road",
        "dr_kelirikko": "Frost damage zones",
        "dr_kaistojen_lukumaara": "Number of lanes",
        "dr_vak_rajoitus": "Dangerous goods restriction",
        "dr_rautatien_tasoristeys": "Railway crossings",
        "dr_tietyot": "Roadworks",
        "dr_liikennevalo": "Traffic lights",
        "dr_esterakennelma": "Barrier structures",
        "dr_pysakki": "Public transport stops",
        "dr_taajama_alueet": "Urban areas",
        "dr_eurooppatienro": "European road numbers",
    }

    COLLECTION_NAMES = tuple(COLLECTIONS.keys())

    def _ensure_collection_fetch_succeeded(self, collection_data: dict[str, dict[str, Any]]) -> None:
        if any("error" not in payload for payload in collection_data.values()):
            return

        error_summary = "; ".join(
            f"{collection_id}: {payload.get('error', 'unknown error')}"
            for collection_id, payload in list(collection_data.items())[:3]
        )
        raise ValueError(f"Digiroad fetch failed for all collections ({error_summary})")

    def _resolve_bbox(self, area: str) -> tuple[float, float, float, float]:
        normalized = self._normalize_area(area)
        return self.AREA_BBOXES.get(normalized, self.AREA_BBOXES["north karelia"])

    async def fetch(self, area: str, timeframe: str) -> DatasetRecord:
        bbox = self._resolve_bbox(area)
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            async def fetch_collection(coll_id: str) -> tuple[str, dict[str, Any]]:
                url = f"{self.BASE_URL}/collections/{coll_id}/items"
                params: dict[str, Any] = {
                    "bbox": bbox_str,
                    "limit": 100,
                    "f": "json",
                }
                try:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    features = data.get("features", [])
                    number_matched = data.get("numberMatched", len(features))
                    return coll_id, {
                        "label": self.COLLECTIONS[coll_id],
                        "number_matched": number_matched,
                        "number_returned": len(features),
                        "features": features[:100],
                    }
                except Exception as e:
                    return coll_id, {
                        "label": self.COLLECTIONS[coll_id],
                        "error": str(e),
                    }

            results = await asyncio.gather(*[fetch_collection(cid) for cid in self.COLLECTION_NAMES])
            collection_data = dict(results)

        self._ensure_collection_fetch_succeeded(collection_data)

        total_features = sum(
            cd.get("number_matched", 0) for cd in collection_data.values()
        )

        return DatasetRecord(
            source_id=self.definition.source_id,
            category=self.definition.category,
            area=area,
            timeframe=timeframe,
            summary=self._build_summary(area, collection_data, total_features),
            data={
                "provider": "Finnish Transport Infrastructure Agency (Väylävirasto)",
                "api": "Digiroad OGC API Features",
                "license": "CC 4.0",
                "query": {
                    "area": area,
                    "bbox_wgs84": bbox_str,
                },
                "collections": collection_data,
                "total_features": total_features,
            },
        )

    def _build_summary(
        self, area: str, collection_data: dict[str, dict], total: int
    ) -> str:
        labels: list[str] = []
        for coll_id, data in collection_data.items():
            label = data.get("label", coll_id)
            if "error" in data:
                labels.append(f"{label}: error")
            else:
                matched = data.get("number_matched", 0)
                labels.append(f"{label}: {matched}")
        parts = ", ".join(labels)
        return f"Digiroad road data for {area}: {parts} ({total} total features)"

    def _normalize_area(self, area: str) -> str:
        ascii_area = unicodedata.normalize("NFKD", area).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", ascii_area).strip().lower()
