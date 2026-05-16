from __future__ import annotations

import asyncio
import re
import unicodedata
from typing import Any

import httpx

from ipb_backend.config import settings
from ipb_backend.ingestion.base import SourceAdapter
from ipb_backend.models import DatasetRecord


class NationalLandSurveyAdapter(SourceAdapter):
    BASE_URL = "https://avoin-paikkatieto.maanmittauslaitos.fi/maastotiedot/features/v1"

    AREA_BBOXES: dict[str, tuple[float, float, float, float]] = {
        "archipelago sea": (21.0, 59.7, 23.0, 60.6),
        "north karelia": (29.0, 62.0, 31.5, 63.5),
        "lapland": (20.5, 68.5, 22.5, 69.4),
        "lapland (kasivarren lappi)": (20.5, 68.5, 22.5, 69.4),
        "kasivarren lappi": (20.5, 68.5, 22.5, 69.4),
    }

    COLLECTIONS: dict[str, str] = {
        "tieviiva": "Road network",
        "rakennus": "Buildings",
        "jarvi": "Lakes",
        "meri": "Sea areas",
        "virtavesialue": "Rivers and streams",
        "korkeuskayra": "Elevation contours",
        "suo": "Bogs and marshes",
        "kallioalue": "Rocky areas",
        "metsamaankasvillisuus": "Forest vegetation",
        "maatalousmaa": "Agricultural land",
        "taajaanrakennettualue": "Densely built areas",
        "rautatie": "Railways",
        "sahkolinja": "Power lines",
        "paikannimi": "Place names",
        "kunta": "Municipalities",
        "kunnanhallintoraja": "Municipal boundaries",
        "luonnonsuojelualue": "Protected areas",
        "lentokenttaalue": "Airport areas",
        "satamaalue": "Harbor areas",
        "rautatieliikennepaikka": "Railway stations",
        "osoitepiste": "Address points",
    }

    COLLECTION_NAMES = tuple(COLLECTIONS.keys())

    def _get_api_key(self) -> str:
        if not settings.nls_api_key:
            raise ValueError("NLS_API_KEY not configured in .env")
        return settings.nls_api_key

    def _resolve_bbox(self, area: str) -> tuple[float, float, float, float]:
        normalized = self._normalize_area(area)
        return self.AREA_BBOXES.get(normalized, self.AREA_BBOXES["north karelia"])

    async def fetch(self, area: str, timeframe: str) -> DatasetRecord:
        bbox = self._resolve_bbox(area)
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

        if not settings.nls_api_key:
            collection_data = self._build_demo_collections(bbox)
            total_features = sum(cd.get("number_matched", 0) for cd in collection_data.values())
            return DatasetRecord(
                source_id=self.definition.source_id,
                category=self.definition.category,
                area=area,
                timeframe=timeframe,
                summary=self._build_summary(area, collection_data, total_features),
                data={
                    "provider": "National Land Survey of Finland (demo scaffold)",
                    "api": "Topographic Database OGC API Features",
                    "license": "CC 4.0 (NLS open data)",
                    "query": {
                        "area": area,
                        "bbox_wgs84": bbox_str,
                    },
                    "collections": collection_data,
                    "note": "Demo spatial fallback is used because NLS_API_KEY is not configured.",
                },
            )

        api_key = self._get_api_key()

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            async def fetch_collection(coll_id: str) -> tuple[str, dict[str, Any]]:
                url = f"{self.BASE_URL}/collections/{coll_id}/items"
                params: dict[str, Any] = {
                    "bbox": bbox_str,
                    "limit": 100,
                    "api-key": api_key,
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
                "provider": "National Land Survey of Finland (Maanmittauslaitos)",
                "api": "Topographic Database OGC API Features",
                "license": "CC 4.0 (NLS open data)",
                "query": {
                    "area": area,
                    "bbox_wgs84": bbox_str,
                },
                "collections": collection_data,
            },
        )

    def _build_demo_collections(self, bbox: tuple[float, float, float, float]) -> dict[str, dict[str, Any]]:
        min_x, min_y, max_x, max_y = bbox
        width = max_x - min_x
        height = max_y - min_y

        def point(x_ratio: float, y_ratio: float) -> list[float]:
            return [min_x + width * x_ratio, min_y + height * y_ratio]

        def polygon(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
            return [
                point(x0, y0),
                point(x1, y0),
                point(x1, y1),
                point(x0, y1),
                point(x0, y0),
            ]

        return {
            "tieviiva": {
                "label": self.COLLECTIONS["tieviiva"],
                "number_matched": 2,
                "number_returned": 2,
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [point(0.08, 0.18), point(0.82, 0.78)],
                        },
                        "properties": {"nimi": "Primary road corridor"},
                    },
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [point(0.14, 0.72), point(0.74, 0.24)],
                        },
                        "properties": {"nimi": "Secondary forest road"},
                    },
                ],
            },
            "rakennus": {
                "label": self.COLLECTIONS["rakennus"],
                "number_matched": 2,
                "number_returned": 2,
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [polygon(0.36, 0.44, 0.42, 0.5)],
                        },
                        "properties": {"nimi": "Operations building"},
                    },
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [polygon(0.5, 0.56, 0.58, 0.64)],
                        },
                        "properties": {"nimi": "Warehouse cluster"},
                    },
                ],
            },
            "jarvi": {
                "label": self.COLLECTIONS["jarvi"],
                "number_matched": 1,
                "number_returned": 1,
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [polygon(0.18, 0.32, 0.32, 0.46)],
                        },
                        "properties": {"nimi": "Lake sample"},
                    },
                ],
            },
            "korkeuskayra": {
                "label": self.COLLECTIONS["korkeuskayra"],
                "number_matched": 2,
                "number_returned": 2,
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [point(0.12, 0.58), point(0.78, 0.62)],
                        },
                        "properties": {"korkeus": 140},
                    },
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [point(0.16, 0.64), point(0.82, 0.68)],
                        },
                        "properties": {"korkeus": 160},
                    },
                ],
            },
        }

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
        return f"NLS topographic data for {area}: {parts} ({total} total features)"

    def _normalize_area(self, area: str) -> str:
        ascii_area = unicodedata.normalize("NFKD", area).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", ascii_area).strip().lower()
