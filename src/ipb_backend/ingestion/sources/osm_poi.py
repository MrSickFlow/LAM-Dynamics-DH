from __future__ import annotations

from typing import Any

import httpx

from ipb_backend.ingestion.base import SourceAdapter
from ipb_backend.models import DatasetRecord, LoadTarget
from ipb_backend.spatial import resolve_load_target_bbox, resolve_load_target_label

# Overpass bbox format: south,west,north,east (i.e. lat_min,lon_min,lat_max,lon_max)
# spatial.py stores bboxes as (west, south, east, north) so we swap here.
def _overpass_bbox(bbox: tuple[float, float, float, float]) -> str:
    west, south, east, north = bbox
    return f"{south},{west},{north},{east}"


# Single combined Overpass query — one request, no parallel rate-limit issues.
# Each element is categorized on the backend from its tags.
def _combined_query(bbox: str) -> str:
    b = bbox
    return (
        "[out:json][timeout:60];"
        "("
        # Education
        f'node["amenity"~"^(school|university|college|kindergarten|childcare|library)$"]({b});'
        f'way["amenity"~"^(school|university|college|kindergarten|childcare|library)$"]({b});'
        # Healthcare
        f'node["amenity"~"^(hospital|clinic|doctors|dentist|pharmacy|veterinary)$"]({b});'
        f'way["amenity"~"^(hospital|clinic|doctors|dentist|pharmacy|veterinary)$"]({b});'
        # Religion
        f'node["amenity"="place_of_worship"]({b});'
        f'way["amenity"="place_of_worship"]({b});'
        # Emergency services
        f'node["amenity"~"^(police|fire_station|ambulance_station)$"]({b});'
        f'way["amenity"~"^(police|fire_station|ambulance_station)$"]({b});'
        # Government
        f'node["amenity"~"^(townhall|courthouse|prison|embassy|community_centre)$"]({b});'
        f'way["amenity"~"^(townhall|courthouse|prison|embassy|community_centre)$"]({b});'
        # Transport
        f'node["amenity"~"^(bus_station|ferry_terminal)$"]({b});'
        f'node["highway"="bus_stop"]({b});'
        # Water sources
        f'node["amenity"="drinking_water"]({b});'
        f'node["man_made"~"^(water_tower|water_well|water_works)$"]({b});'
        f'node["natural"="spring"]({b});'
        f'way["man_made"~"^(water_tower|water_well|water_works)$"]({b});'
        # Forest
        f'way["natural"="wood"]({b});'
        f'way["landuse"="forest"]({b});'
        f'relation["natural"="wood"]({b});'
        f'relation["landuse"="forest"]({b});'
        # Industry
        f'node["landuse"~"^(industrial|commercial|retail)$"]({b});'
        f'way["landuse"~"^(industrial|commercial|retail)$"]({b});'
        f'node["building"~"^(industrial|warehouse|factory)$"]({b});'
        f'way["building"~"^(industrial|warehouse|factory)$"]({b});'
        # Military
        f'node["military"]({b});'
        f'way["military"]({b});'
        f'node["landuse"="military"]({b});'
        f'way["landuse"="military"]({b});'
        # Airfields
        f'node["aeroway"~"^(aerodrome|airstrip|helipad|heliport)$"]({b});'
        f'way["aeroway"~"^(aerodrome|airstrip|helipad|heliport)$"]({b});'
        # Fuel supply
        f'node["amenity"="fuel"]({b});'
        f'way["amenity"="fuel"]({b});'
        f'node["shop"="gas"]({b});'
        # Power infrastructure
        f'node["power"~"^(tower|pole|substation|plant|generator)$"]({b});'
        f'way["power"~"^(line|minor_line|cable|substation)$"]({b});'
        ");"
        "out center 5000;"
    )


def _classify(tags: dict[str, str]) -> str:
    amenity = tags.get("amenity", "")
    highway = tags.get("highway", "")
    natural = tags.get("natural", "")
    landuse = tags.get("landuse", "")
    man_made = tags.get("man_made", "")
    military = tags.get("military", "")
    aeroway = tags.get("aeroway", "")
    power = tags.get("power", "")
    building = tags.get("building", "")
    shop = tags.get("shop", "")

    if amenity in {"school", "university", "college", "kindergarten", "childcare", "library"}:
        return "education"
    if amenity in {"hospital", "clinic", "doctors", "dentist", "pharmacy", "veterinary"}:
        return "healthcare"
    if amenity == "place_of_worship":
        return "religion"
    if amenity in {"police", "fire_station", "ambulance_station"}:
        return "emergency_services"
    if amenity in {"townhall", "courthouse", "prison", "embassy", "community_centre"}:
        return "government"
    if amenity in {"bus_station", "ferry_terminal"} or highway == "bus_stop":
        return "transport"
    if amenity == "drinking_water" or man_made in {"water_tower", "water_well", "water_works"} or natural == "spring":
        return "water_sources"
    if amenity == "fuel" or shop == "gas":
        return "fuel_supply"
    if aeroway in {"aerodrome", "airstrip", "helipad", "heliport"}:
        return "airfields"
    if military or landuse == "military":
        return "military"
    if power in {"tower", "pole", "substation", "plant", "generator", "line", "minor_line", "cable"}:
        return "power_infrastructure"
    if natural == "wood" or landuse == "forest":
        return "forest"
    if landuse in {"industrial", "commercial", "retail"} or building in {"industrial", "warehouse", "factory"}:
        return "industry"
    return "other"


ALLOWED_TAGS: dict[str, set[str]] = {
    "forest": {"leaf_type", "leaf_cycle", "natural", "landuse", "name", "wood"},
    "military": {"military", "name", "landuse", "access", "description"},
    "airfields": {"aeroway", "name", "icao", "iata", "operator", "surface", "length", "width"},
    "industry": {"landuse", "building", "name", "operator", "industrial"},
    "fuel_supply": {"amenity", "name", "brand", "opening_hours", "operator"},
    "power_infrastructure": {"power", "voltage", "name", "operator", "cables", "circuits"},
    "_default": {"amenity", "name", "religion", "denomination", "school", "healthcare",
                 "operator", "capacity", "drinking_water", "shop", "highway", "natural"},
}


def _filter_tags(tags: dict[str, str], category: str = "") -> dict[str, str]:
    allowed = ALLOWED_TAGS.get(category, ALLOWED_TAGS["_default"])
    return {k: v for k, v in tags.items() if k in allowed}


def _feature_to_poi(el: dict[str, Any], category: str) -> dict[str, Any]:
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


OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


class OsmPoiAdapter(SourceAdapter):
    REQUEST_TIMEOUT = httpx.Timeout(90.0, connect=10.0)

    async def fetch(self, area: str, timeframe: str, load_target: LoadTarget | None = None) -> DatasetRecord:
        bbox = resolve_load_target_bbox(area, load_target)
        area_label = resolve_load_target_label(area, load_target)
        overpass_bbox = _overpass_bbox(bbox)
        query = _combined_query(overpass_bbox)

        last_exc: Exception | None = None
        async with httpx.AsyncClient(timeout=self.REQUEST_TIMEOUT, headers={"User-Agent": "IPB-Backend/1.0"}) as client:
            for url in OVERPASS_URLS:
                try:
                    resp = await client.post(url, data={"data": query})
                    resp.raise_for_status()
                    elements = resp.json().get("elements", [])
                    break
                except Exception as exc:
                    last_exc = exc
                    continue
            else:
                raise ValueError(f"All Overpass endpoints failed: {last_exc}") from last_exc

        categories: dict[str, list[dict[str, Any]]] = {}
        for el in elements:
            tags = el.get("tags", {})
            category = _classify(tags)
            if category == "other":
                continue
            poi = _feature_to_poi(el, category)
            if poi["lat"] is None or poi["lon"] is None:
                continue
            categories.setdefault(category, []).append(poi)

        total = sum(len(v) for v in categories.values())
        return DatasetRecord(
            source_id=self.definition.source_id,
            category=self.definition.category,
            area=area_label,
            timeframe=timeframe,
            load_target=load_target,
            summary=f"OSM POIs for {area_label}: {total} features across {len(categories)} categories",
            data={
                "provider": "OpenStreetMap contributors (ODbL)",
                "api": "Overpass API",
                "license": "ODbL",
                "query": {"area": area_label, "bbox_overpass": overpass_bbox},
                "categories": {k: v for k, v in sorted(categories.items())},
                "errors": {},
                "total_features": total,
            },
        )
