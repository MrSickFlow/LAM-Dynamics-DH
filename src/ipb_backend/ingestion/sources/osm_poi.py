from __future__ import annotations

import asyncio
from typing import Any

import httpx

from ipb_backend.ingestion.base import SourceAdapter
from ipb_backend.models import DatasetRecord, LoadTarget
from ipb_backend.spatial import resolve_load_target_bbox, resolve_load_target_label


def _overpass_bbox(bbox: tuple[float, float, float, float]) -> str:
    """Convert (west, south, east, north) to Overpass south,west,north,east."""
    west, south, east, north = bbox
    return f"{south},{west},{north},{east}"


# Each batch is {"geom": bool, "queries": {category: overpass_fragment}}.
# geom=True uses "out geom" to return full coordinates for lines/polygons.
# geom=False uses "out center" which is faster and only returns a centroid for ways.
BATCHES: list[dict[str, Any]] = [
    # Batch 1 — civilian services (point features only)
    {"geom": False, "queries": {
        "education":        ('node["amenity"~"school|university|college|kindergarten|childcare|library"]({b});'
                             'way["amenity"~"school|university|college|kindergarten|childcare|library"]({b});'),
        "healthcare":       ('node["amenity"~"hospital|clinic|doctors|dentist|pharmacy|veterinary"]({b});'
                             'way["amenity"~"hospital|clinic|doctors|dentist|pharmacy|veterinary"]({b});'),
        "religion":         ('node["amenity"="place_of_worship"]({b});'
                             'way["amenity"="place_of_worship"]({b});'),
        "emergency_services": ('node["amenity"~"police|fire_station|ambulance_station"]({b});'
                               'way["amenity"~"police|fire_station|ambulance_station"]({b});'),
        "government":       ('node["amenity"~"townhall|courthouse|prison|embassy|community_centre"]({b});'
                             'way["amenity"~"townhall|courthouse|prison|embassy|community_centre"]({b});'),
    }},
    # Batch 2 — water sources (point features only)
    {"geom": False, "queries": {
        "water_sources":    ('node["amenity"="drinking_water"]({b});'
                             'node["man_made"~"water_tower|water_well|water_works"]({b});'
                             'node["natural"="spring"]({b});'
                             'way["man_made"~"water_tower|water_well|water_works"]({b});'),
    }},
    # Batch 3 — operational / military (point features only)
    {"geom": False, "queries": {
        "military":         ('node["military"]({b});'
                             'way["military"]({b});'
                             'node["landuse"="military"]({b});'
                             'way["landuse"="military"]({b});'),
        "airfields":        ('node["aeroway"~"aerodrome|airstrip|helipad|heliport"]({b});'
                             'way["aeroway"~"aerodrome|airstrip|helipad|heliport"]({b});'),
        "fuel_supply":      ('node["amenity"="fuel"]({b});'
                             'way["amenity"="fuel"]({b});'
                             'node["shop"="gas"]({b});'),
        "industry":         ('node["landuse"~"industrial|commercial|retail"]({b});'
                             'way["landuse"~"industrial|commercial|retail"]({b});'
                             'node["building"~"industrial|warehouse|factory"]({b});'
                             'way["building"~"industrial|warehouse|factory"]({b});'
                             'node["man_made"="works"]({b});'
                             'way["man_made"="works"]({b});'),
    }},
    # Batch 4 — power infrastructure as lines + key point assets (substations/plants).
    # Towers/poles are omitted: 3000+ individual points add noise; the line geometry
    # already shows the network routing.
    {"geom": True, "queries": {
        "power_infrastructure": ('node["power"~"substation|plant|generator"]({b});'
                                 'way["power"~"line|minor_line|cable"]({b});'
                                 'way["power"="substation"]({b});'),
    }},
    # Batch 5 — forest as filled polygons (ways only; relations skipped to limit payload).
    # Coloured by leaf_type: needleleaved (dark) vs broadleaved (medium).
    {"geom": True, "queries": {
        "forest":           ('way["natural"="wood"]({b});'
                             'way["landuse"="forest"]({b});'),
    }},
    # Batch 6 — logistics & transport hubs.
    # logistics: specific high-capacity storage/processing targets a commander could commandeer
    #   (sawmills, cold storage, distribution depots, agrarian supply, farmyards).
    # ports_terminals: maritime and road transit chokepoints.
    {"geom": False, "queries": {
        "logistics":        ('node["industrial"~"warehouse|distribution|sawmill|timber|cold_storage|depot|wood_processing"]({b});'
                             'way["industrial"~"warehouse|distribution|sawmill|timber|cold_storage|depot|wood_processing"]({b});'
                             'node["building"~"storage_tank|barn"]({b});'
                             'way["building"~"storage_tank|barn"]({b});'
                             'node["shop"~"agrarian|hardware|doityourself|wholesale"]({b});'
                             'way["shop"~"agrarian|hardware|doityourself|wholesale"]({b});'
                             'node["landuse"="farmyard"]({b});'
                             'way["landuse"="farmyard"]({b});'
                             'node["amenity"="marketplace"]({b});'
                             'way["amenity"="marketplace"]({b});'),
        "ports_terminals":  ('node["amenity"="ferry_terminal"]({b});'
                             'way["amenity"="ferry_terminal"]({b});'
                             'node["waterway"~"boatyard|dock"]({b});'
                             'way["waterway"~"boatyard|dock"]({b});'
                             'node["landuse"="port"]({b});'
                             'way["landuse"="port"]({b});'
                             'node["amenity"="bus_station"]({b});'
                             'way["amenity"="bus_station"]({b});'),
    }},
]


def _build_query(batch: dict[str, Any], b: str) -> str:
    # Use a smaller limit for geometry batches (forest/power) to cap memory usage.
    limit = 300 if batch.get("geom") else 3000
    union = "".join(q.format(b=b) for q in batch["queries"].values())
    out_mode = "geom" if batch.get("geom") else "center"
    return f"[out:json][timeout:55];({union});out {out_mode} {limit};"


_LOGISTICS_INDUSTRIAL = {"warehouse", "distribution", "sawmill", "timber", "cold_storage", "depot", "wood_processing"}
_LOGISTICS_BUILDINGS  = {"storage_tank", "barn"}
_LOGISTICS_SHOPS      = {"agrarian", "hardware", "doityourself", "wholesale"}
_PORTS_AMENITIES      = {"ferry_terminal", "bus_station"}
_PORTS_WATERWAYS      = {"boatyard", "dock"}


def _classify(tags: dict[str, str]) -> str:
    amenity  = tags.get("amenity", "")
    natural  = tags.get("natural", "")
    landuse  = tags.get("landuse", "")
    man_made = tags.get("man_made", "")
    military = tags.get("military", "")
    aeroway  = tags.get("aeroway", "")
    power    = tags.get("power", "")
    building = tags.get("building", "")
    shop     = tags.get("shop", "")
    industrial = tags.get("industrial", "")
    waterway = tags.get("waterway", "")

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
    # Logistics: specific high-value targets (sawmills, depots, agrarian supply, farmyards)
    if (industrial in _LOGISTICS_INDUSTRIAL
            or building in _LOGISTICS_BUILDINGS
            or shop in _LOGISTICS_SHOPS
            or landuse == "farmyard"
            or amenity == "marketplace"):
        return "logistics"
    # Ports & transit hubs
    if amenity in _PORTS_AMENITIES or waterway in _PORTS_WATERWAYS or landuse == "port":
        return "ports_terminals"
    if landuse in {"industrial", "commercial", "retail"} or building in {"industrial", "warehouse", "factory"} or man_made == "works":
        return "industry"
    if natural == "wood" or landuse == "forest":
        return "forest"
    return "other"


ALLOWED_TAGS: dict[str, set[str]] = {
    "forest":               {"leaf_type", "leaf_cycle", "natural", "landuse", "name", "wood"},
    "military":             {"military", "name", "landuse", "access", "description"},
    "airfields":            {"aeroway", "name", "icao", "iata", "operator", "surface", "length", "width"},
    "industry":             {"landuse", "building", "name", "operator", "industrial", "man_made", "product"},
    "fuel_supply":          {"amenity", "name", "brand", "opening_hours", "operator"},
    "power_infrastructure": {"power", "voltage", "name", "operator", "cables", "circuits"},
    "logistics":            {"industrial", "building", "shop", "landuse", "amenity", "name",
                             "operator", "product", "capacity", "cold_storage", "man_made"},
    "ports_terminals":      {"amenity", "waterway", "landuse", "name", "operator",
                             "ferry", "motor_vehicle", "opening_hours"},
    "_default":             {"amenity", "name", "religion", "denomination", "school", "healthcare",
                             "operator", "capacity", "drinking_water", "shop", "natural"},
}


def _filter_tags(tags: dict[str, str], category: str = "") -> dict[str, str]:
    allowed = ALLOWED_TAGS.get(category, ALLOWED_TAGS["_default"])
    return {k: v for k, v in tags.items() if k in allowed}


def _extract_geometry(el: dict[str, Any], category: str) -> dict[str, Any] | None:
    """Return a GeoJSON geometry dict for an Overpass element, or None."""
    el_type = el.get("type")
    if el_type == "node":
        lat, lon = el.get("lat"), el.get("lon")
        if lat is None or lon is None:
            return None
        return {"type": "Point", "coordinates": [lon, lat]}
    if el_type == "way":
        raw = el.get("geometry", [])
        coords = [[g["lon"], g["lat"]] for g in raw if "lon" in g and "lat" in g]
        if not coords:
            return None
        if category == "forest":
            # Ensure the ring is closed
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            return {"type": "Polygon", "coordinates": [coords]}
        return {"type": "LineString", "coordinates": coords}
    return None


def _feature_to_poi(el: dict[str, Any], category: str, use_geom: bool = False) -> dict[str, Any]:
    tags = el.get("tags", {})
    center = el.get("center") or el
    geometry = _extract_geometry(el, category) if use_geom else None
    return {
        "id": f"{el['type']}/{el['id']}",
        "type": el["type"],
        "osm_id": el["id"],
        "lat": center.get("lat"),
        "lon": center.get("lon"),
        "geometry": geometry,
        "tags": _filter_tags(tags, category),
    }


OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


class OsmPoiAdapter(SourceAdapter):
    REQUEST_TIMEOUT = httpx.Timeout(45.0, connect=10.0)

    async def _query_batch(self, client: httpx.AsyncClient, query: str) -> list[dict[str, Any]]:
        last_exc: Exception | None = None
        for url in OVERPASS_URLS:
            try:
                resp = await client.post(url, data={"data": query})
                resp.raise_for_status()
                return resp.json().get("elements", [])
            except Exception as exc:
                last_exc = exc
        raise ValueError(f"All Overpass endpoints failed: {last_exc}") from last_exc

    async def _fetch_batch(
        self,
        client: httpx.AsyncClient,
        index: int,
        batch: dict[str, Any],
        b: str,
    ) -> tuple[int, list[dict[str, Any]], str | None]:
        use_geom = batch.get("geom", False)
        query = _build_query(batch, b)
        try:
            elements = await self._query_batch(client, query)
            return index, elements, None
        except Exception as exc:
            return index, [], f"batch{index + 1}: {exc}"

    async def fetch(self, area: str, timeframe: str, load_target: LoadTarget | None = None) -> DatasetRecord:
        bbox = resolve_load_target_bbox(area, load_target)
        area_label = resolve_load_target_label(area, load_target)
        b = _overpass_bbox(bbox)

        categories: dict[str, list[dict[str, Any]]] = {}
        batch_errors: list[str] = []

        async with httpx.AsyncClient(timeout=self.REQUEST_TIMEOUT, headers={"User-Agent": "IPB-Backend/1.0"}) as client:
            results = await asyncio.gather(
                *[self._fetch_batch(client, i, batch, b) for i, batch in enumerate(BATCHES)]
            )

        for index, elements, error in results:
            if error:
                batch_errors.append(error)
                continue
            use_geom = BATCHES[index].get("geom", False)
            for el in elements:
                tags = el.get("tags", {})
                category = _classify(tags)
                if category == "other":
                    continue
                poi = _feature_to_poi(el, category, use_geom=use_geom)
                if poi["geometry"] is None and (poi["lat"] is None or poi["lon"] is None):
                    continue
                categories.setdefault(category, []).append(poi)

        if not categories and batch_errors:
            raise ValueError(f"OSM POI fetch failed: {'; '.join(batch_errors)}")

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
                "query": {"area": area_label, "bbox_overpass": b},
                "categories": {k: v for k, v in sorted(categories.items())},
                "errors": {f"batch{i+1}": e for i, e in enumerate(batch_errors)},
                "total_features": total,
            },
        )
