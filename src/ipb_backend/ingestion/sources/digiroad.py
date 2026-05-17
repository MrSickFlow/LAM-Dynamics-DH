from __future__ import annotations

import asyncio
from typing import Any

import httpx

from ipb_backend.ingestion.base import SourceAdapter
from ipb_backend.models import DatasetRecord, LoadTarget
from ipb_backend.spatial import format_bbox, resolve_load_target_bbox, resolve_load_target_label


class DigiroadAdapter(SourceAdapter):
    BASE_URL = "https://avoinapi.vaylapilvi.fi/vaylatiedot/digiroad/ogc/features/v1"

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

    async def fetch(self, area: str, timeframe: str, load_target: LoadTarget | None = None) -> DatasetRecord:
        bbox = resolve_load_target_bbox(area, load_target)
        area_label = resolve_load_target_label(area, load_target)
        bbox_str = format_bbox(bbox)

        # Per-collection timeout: a single slow endpoint must not stall the whole
        # batch. Partial results (some collections errored) are acceptable.
        _PER_COLLECTION_TIMEOUT = 20.0

        async with httpx.AsyncClient(timeout=_PER_COLLECTION_TIMEOUT, follow_redirects=True) as client:
            async def fetch_collection(coll_id: str) -> tuple[str, dict[str, Any]]:
                url = f"{self.BASE_URL}/collections/{coll_id}/items"
                params: dict[str, Any] = {
                    "bbox": bbox_str,
                    "limit": 10000,
                    "f": "json",
                }
                try:
                    response = await asyncio.wait_for(
                        client.get(url, params=params),
                        timeout=_PER_COLLECTION_TIMEOUT,
                    )
                    response.raise_for_status()
                    data = response.json()
                    features = data.get("features", [])
                    number_matched = data.get("numberMatched", len(features))
                    return coll_id, {
                        "label": self.COLLECTIONS[coll_id],
                        "number_matched": number_matched,
                        "number_returned": len(features),
                        "features": features,
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
            area=area_label,
            timeframe=timeframe,
            load_target=load_target,
            summary=self._build_summary(area_label, collection_data, total_features),
            data={
                "provider": "Finnish Transport Infrastructure Agency (Väylävirasto)",
                "api": "Digiroad OGC API Features",
                "license": "CC 4.0",
                "query": {
                    "area": area_label,
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
