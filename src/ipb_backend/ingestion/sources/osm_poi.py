from __future__ import annotations

import json
import os
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

_AREA_FILE_SUFFIX: dict[str, str] = {
    "north karelia": "north_karelia",
    "archipelago sea": "archipelago_sea",
    "lapland": "lapland",
    "lapland (kasivarren lappi)": "lapland",
    "kasivarren lappi": "lapland",
}

_STATIC_POI_DIR = os.path.join(os.path.dirname(__file__))


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


_CATEGORY_MAP: dict[str, str] = {
    "school": "education", "university": "education", "college": "education",
    "kindergarten": "education", "childcare": "education", "library": "education",
    "hospital": "healthcare", "clinic": "healthcare", "doctors": "healthcare",
    "dentist": "healthcare", "pharmacy": "healthcare", "veterinary": "healthcare",
    "drinking_water": "water_sources", "spring": "water_sources",
    "water_tower": "water_sources", "water_well": "water_sources", "water_works": "water_sources",
    "place_of_worship": "religion", "christian": "religion",
    "police": "emergency_services", "fire_station": "emergency_services",
    "ambulance_station": "emergency_services",
    "townhall": "government", "courthouse": "government", "prison": "government",
    "embassy": "government", "community_centre": "government",
    "bus_station": "transport", "ferry_terminal": "transport", "fuel": "transport",
    "bus_stop": "transport",
    "reservoir": "industry", "storage_tank": "industry",
    "wood": "forest", "forest": "forest",
}


def _load_static_pois(area: str) -> dict[str, list[dict[str, Any]]]:
    normalized = area.lower().strip()
    suffix = _AREA_FILE_SUFFIX.get(normalized, "north_karelia")
    filepath = os.path.join(_STATIC_POI_DIR, f"static_osm_poi_{suffix}.json")
    if not os.path.exists(filepath):
        return {}
    with open(filepath) as f:
        return json.load(f)


class OsmPoiAdapter(SourceAdapter):
    BASE_URL = "https://overpass-api.de/api/interpreter"

    def _resolve_bbox(self, area: str) -> tuple[float, float, float, float]:
        normalized = self._normalize_area(area)
        return AREA_BBOXES.get(normalized, AREA_BBOXES["north karelia"])

    async def fetch(self, area: str, timeframe: str) -> DatasetRecord:
        bbox = self._resolve_bbox(area)
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

        categories: dict[str, list[dict[str, Any]]] = {}
        category_errors: dict[str, str] = {}
        total = 0
        provider = "OpenStreetMap contributors (ODbL)"

        async with httpx.AsyncClient(timeout=60.0, headers={"User-Agent": "IPB-Backend/1.0"}) as client:
            for cat_name, cat_query in CATEGORY_QUERIES.items():
                query = f'[out:json];({cat_query.format(bbox=bbox_str)});out center 200;'
                try:
                    resp = await client.post(self.BASE_URL, data={"data": query})
                    resp.raise_for_status()
                    data = resp.json()
                    pois = [_feature_to_poi(el, cat_name) for el in data.get("elements", [])]
                except Exception as e:
                    pois = []
                    category_errors[cat_name] = str(e)

                categories[cat_name] = pois
                total += len(pois)

        if category_errors and len(category_errors) == len(CATEGORY_QUERIES):
            static = _load_static_pois(area)
            if static:
                categories = {}
                total = 0
                for cat_name in CATEGORY_QUERIES:
                    items = static.get(cat_name, [])
                    categories[cat_name] = items
                    total += len(items)
                provider = "OpenStreetMap (static extract, Overpass API unreachable)"
            else:
                for cat_name in CATEGORY_QUERIES:
                    if cat_name not in categories:
                        categories[cat_name] = []
                provider = "OpenStreetMap contributors (static file not found)"

        return DatasetRecord(
            source_id=self.definition.source_id,
            category=self.definition.category,
            area=area,
            timeframe=timeframe,
            summary=f"OSM POIs for {area}: {total} features across {len(categories)} categories",
            data={
                "provider": provider,
                "api": "Overpass API / static extract",
                "license": "ODbL",
                "query": {"area": area, "bbox": bbox_str},
                "categories": {k: v for k, v in sorted(categories.items())},
                "errors": category_errors,
                "total_features": total,
            },
        )

    def _normalize_area(self, area: str) -> str:
        ascii_area = unicodedata.normalize("NFKD", area).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", ascii_area).strip().lower()
