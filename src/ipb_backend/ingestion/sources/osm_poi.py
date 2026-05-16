from __future__ import annotations

import re
import unicodedata
from typing import Any

import httpx

from ipb_backend.ingestion.base import SourceAdapter
from ipb_backend.models import DatasetRecord

AREA_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "archipelago sea": (59.7, 21.0, 60.6, 23.0),
    "north karelia": (62.0, 29.0, 63.5, 31.5),
    "lapland": (68.5, 20.5, 69.4, 22.5),
    "lapland (kasivarren lappi)": (68.5, 20.5, 69.4, 22.5),
    "kasivarren lappi": (68.5, 20.5, 69.4, 22.5),
}

POI_CATEGORIES: dict[str, dict[str, str]] = {
    "education": {"amenity": "school|university|college|kindergarten|childcare|library"},
    "healthcare": {"amenity": "hospital|clinic|doctors|dentist|pharmacy|veterinary"},
    "water": {"amenity": "drinking_water", "man_made": "water_tower|water_well|water_works", "natural": "spring"},
    "religion": {"amenity": "place_of_worship", "religion": "christian|muslim|jewish|buddhist|hindu|sikh"},
    "emergency": {"amenity": "police|fire_station|ambulance_station"},
    "government": {"amenity": "townhall|courthouse|prison|embassy|community_centre"},
    "transport": {"amenity": "bus_station|ferry_terminal|fuel", "highway": "bus_stop"},
    "industry": {"man_made": "water_tower|reservoir|storage_tank"},
}

CATEGORY_QUERIES: dict[str, str] = {
    "education": 'node["amenity"~"school|university|college|kindergarten|childcare|library"]({bbox});way["amenity"~"school|university|college|kindergarten|childcare|library"]({bbox});',
    "healthcare": 'node["amenity"~"hospital|clinic|doctors|dentist|pharmacy|veterinary"]({bbox});way["amenity"~"hospital|clinic|doctors|dentist|pharmacy|veterinary"]({bbox});',
    "water_sources": 'node["amenity"="drinking_water"]({bbox});node["man_made"~"water_tower|water_well|water_works"]({bbox});node["natural"="spring"]({bbox});way["man_made"~"water_tower|water_well|water_works"]({bbox});',
    "religion": 'node["amenity"="place_of_worship"]({bbox});way["amenity"="place_of_worship"]({bbox});',
    "emergency_services": 'node["amenity"~"police|fire_station|ambulance_station"]({bbox});way["amenity"~"police|fire_station|ambulance_station"]({bbox});',
    "government": 'node["amenity"~"townhall|courthouse|prison|embassy|community_centre"]({bbox});way["amenity"~"townhall|courthouse|prison|embassy|community_centre"]({bbox});',
    "transport": 'node["amenity"~"bus_station|ferry_terminal|fuel"]({bbox});node["highway"="bus_stop"]({bbox});',
    "forest": 'way["natural"="wood"]({bbox});way["landuse"="forest"]({bbox});relation["natural"="wood"]({bbox});relation["landuse"="forest"]({bbox});',
}


FOREST_TAGS = {"leaf_type", "leaf_cycle", "natural", "landuse", "name", "wood"}

def _filter_tags(tags: dict[str, str], category: str = "") -> dict[str, str]:
    if category == "forest":
        allowed = FOREST_TAGS
    else:
        allowed = {"amenity", "name", "religion", "denomination", "school", "healthcare", "operator", "capacity", "drinking_water"}
    return {k: v for k, v in tags.items() if k in allowed}


def _feature_to_poi(el: dict[str, Any], category: str = "") -> dict[str, Any]:
    tags = el.get("tags", {})
    center = el.get("center") or el
    return {
        "id": f"{el['type']}/{el['id']}",
        "type": el["type"],
        "osm_id": el["id"],
        "lat": center.get("lat"),
        "lon": center.get("lon"),
        "tags": _filter_tags(tags, category),
    }


class OsmPoiAdapter(SourceAdapter):
    BASE_URL = "https://overpass-api.de/api/interpreter"

    def _resolve_bbox(self, area: str) -> tuple[float, float, float, float]:
        normalized = self._normalize_area(area)
        return AREA_BBOXES.get(normalized, AREA_BBOXES["north karelia"])

    async def fetch(self, area: str, timeframe: str) -> DatasetRecord:
        bbox = self._resolve_bbox(area)
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

        categories: dict[str, list[dict[str, Any]]] = {}
        total = 0

        async with httpx.AsyncClient(timeout=60.0, headers={"User-Agent": "IPB-Backend/1.0"}) as client:
            for cat_name, cat_query in CATEGORY_QUERIES.items():
                query = f'[out:json];({cat_query.format(bbox=bbox_str)});out center 200;'
                try:
                    resp = await client.post(self.BASE_URL, data={"data": query})
                    resp.raise_for_status()
                    data = resp.json()
                    pois = [_feature_to_poi(el, cat_name) for el in data.get("elements", [])]
                except Exception as e:
                    pois = [{"error": str(e)}]

                categories[cat_name] = pois
                total += len(pois)

        provider = "OpenStreetMap contributors (ODbL)"
        return DatasetRecord(
            source_id=self.definition.source_id,
            category=self.definition.category,
            area=area,
            timeframe=timeframe,
            summary=f"OSM POIs for {area}: {total} features across {len(categories)} categories",
            data={
                "provider": provider,
                "api": "Overpass API",
                "license": "ODbL",
                "query": {"area": area, "bbox": bbox_str},
                "categories": {k: v for k, v in sorted(categories.items())},
                "total_features": total,
            },
        )

    def _normalize_area(self, area: str) -> str:
        ascii_area = unicodedata.normalize("NFKD", area).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", ascii_area).strip().lower()
