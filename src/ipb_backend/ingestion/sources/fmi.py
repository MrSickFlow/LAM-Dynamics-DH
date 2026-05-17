from __future__ import annotations

import asyncio
import math
import re
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import parse_qs, urlparse

import httpx

from ipb_backend.ingestion.base import SourceAdapter
from ipb_backend.ingestion.timeframe import forecast_horizon_hours
from ipb_backend.models import DatasetRecord, LoadTarget, LoadTargetKind


# FMI HARMONIE surface forecast goes out ~48–66h. Cap a bit short of that to
# avoid empty trailing time-steps when the source horizon shifts.
FMI_FORECAST_MAX_HOURS = 48
# Observations window we always pull, regardless of user timeframe — gives the
# LLM and UI fresh "current conditions" without dragging multi-day XML payloads.
FMI_OBSERVATION_HOURS = 3


def _great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two WGS84 points. Used to surface 'nearest
    station' distance when no station lies inside a load rectangle."""
    r_lat1, r_lat2 = math.radians(lat1), math.radians(lat2)
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(r_lat1) * math.cos(r_lat2) * math.sin(d_lon / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(a)))


FORECAST_PARAMETERS = ("Temperature", "Pressure", "Humidity", "TotalCloudCover", "WindSpeedMS", "WindDirection", "Precipitation1h", "Visibility")
FORECAST_PARAMETER_METADATA = {
    "Temperature": {"key": "temperature", "label": "Temperature", "unit": "C"},
    "Pressure": {"key": "pressure", "label": "Pressure", "unit": "hPa"},
    "Humidity": {"key": "humidity", "label": "Humidity", "unit": "%"},
    "TotalCloudCover": {"key": "cloud_cover", "label": "Cloud cover", "unit": "%"},
    "WindSpeedMS": {"key": "wind_speed", "label": "Wind speed", "unit": "m/s"},
    "WindDirection": {"key": "wind_direction", "label": "Wind direction", "unit": "deg"},
    "Precipitation1h": {"key": "precipitation", "label": "Precipitation", "unit": "mm/h"},
    "Visibility": {"key": "visibility", "label": "Visibility", "unit": "m"},
}


class FmiAdapter(SourceAdapter):
    WFS_URL = "https://opendata.fmi.fi/wfs"
    QUERY_PARAMETERS = ("t2m", "ws_10min", "n_man", "vis", "rh", "p_sea", "r_1h", "wg_10min")
    PARAMETER_METADATA = {
        "t2m": {"key": "temperature", "label": "Temperature", "unit": "C"},
        "ws_10min": {"key": "wind_speed", "label": "Wind speed", "unit": "m/s"},
        "n_man": {"key": "cloud_cover", "label": "Cloud cover", "unit": "okta"},
        "vis": {"key": "visibility", "label": "Visibility", "unit": "m"},
        "rh": {"key": "humidity", "label": "Humidity", "unit": "%"},
        "p_sea": {"key": "pressure", "label": "Pressure", "unit": "hPa"},
        "r_1h": {"key": "precipitation", "label": "Precipitation", "unit": "mm/h"},
        "wg_10min": {"key": "wind_gust", "label": "Wind gust", "unit": "m/s"},
    }
    AREA_PLACE_ALIASES = {
        "archipelago sea": "turku",
        "north karelia": "joensuu",
        "lapland": "kilpisjarvi",
        "lapland (kasivarren lappi)": "kilpisjarvi",
        "kasivarren lappi": "kilpisjarvi",
    }
    NAMESPACES = {
        "gml": "http://www.opengis.net/gml/3.2",
        "om": "http://www.opengis.net/om/2.0",
        "target": "http://xml.fmi.fi/namespace/om/atmosphericfeatures/1.1",
        "wfs": "http://www.opengis.net/wfs/2.0",
        "wml2": "http://www.opengis.net/waterml/2.0",
        "xlink": "http://www.w3.org/1999/xlink",
    }

    async def fetch(self, area: str, timeframe: str, load_target: LoadTarget | None = None) -> DatasetRecord:
        obs_start, obs_end = self._resolve_observation_window()
        forecast_hours = self._resolve_forecast_hours(timeframe)
        bbox_center: tuple[float, float] | None = None

        # When a custom bbox load target is given, query by the centroid of the bbox so
        # that we find the nearest weather station regardless of whether it sits inside
        # the drawn rectangle or in a nearby municipality.
        if load_target is not None and load_target.kind == LoadTargetKind.BBOX and load_target.bbox_wgs84:
            min_x, min_y, max_x, max_y = load_target.bbox_wgs84
            center_lat = (min_y + max_y) / 2.0
            center_lon = (min_x + max_x) / 2.0
            bbox_center = (center_lat, center_lon)
            parsed = await self.fetch_observations_by_latlon(center_lat, center_lon, obs_start, obs_end)
            place = f"{center_lat:.4f},{center_lon:.4f}"
        else:
            place = self._resolve_place(area)
            xml_payload = await self._fetch_xml(place, obs_start, obs_end)
            parsed = self._parse_response(xml_payload)

        # Annotate the station with its distance from the bbox centre when one was
        # provided — the UI labels the panel ("Nearest station: X, 18 km") accordingly.
        if bbox_center is not None and parsed.get("station", {}).get("latitude") is not None:
            st = parsed["station"]
            st["distance_from_bbox_km"] = round(
                _great_circle_km(bbox_center[0], bbox_center[1], float(st["latitude"]), float(st["longitude"])),
                1,
            )
            st["fallback"] = st["distance_from_bbox_km"] > 0.1

        # Forecast horizon follows the user's planning timeframe (capped at HARMONIE's
        # ~48h ceiling). Failures are silently swallowed — observations still record.
        station = parsed.get("station", {})
        forecast: dict = {}
        if forecast_hours > 0 and station.get("latitude") and station.get("longitude"):
            try:
                forecast = await self.fetch_forecast_by_latlon(
                    float(station["latitude"]),
                    float(station["longitude"]),
                    hours=forecast_hours,
                )
            except Exception:
                pass

        return DatasetRecord(
            source_id=self.definition.source_id,
            category=self.definition.category,
            area=area,
            timeframe=timeframe,
            summary=self._build_summary(area, parsed),
            data={
                "provider": self.definition.name,
                "query": {
                    "place": place,
                    "start_time": obs_start.isoformat().replace("+00:00", "Z"),
                    "end_time": obs_end.isoformat().replace("+00:00", "Z"),
                    "timestep_minutes": 60,
                    "parameters": list(self.QUERY_PARAMETERS),
                    "forecast_hours": forecast_hours,
                },
                **parsed,
                "forecast": forecast,
            },
        )

    async def fetch_observations_by_latlon(self, lat: float, lon: float, start_time: datetime, end_time: datetime) -> dict:
        """Fetch observations around a point. Uses an expanding bbox so we always
        find the nearest reporting station — FMI's `latlon` parameter on this
        storedquery is unreliable; `bbox` is the official supported flag."""
        async def _query(s: datetime, e: datetime, half_deg: float) -> dict:
            bbox = f"{lon - half_deg},{lat - half_deg},{lon + half_deg},{lat + half_deg}"
            params = {
                "service": "WFS",
                "version": "2.0.0",
                "request": "getFeature",
                "storedquery_id": "fmi::observations::weather::timevaluepair",
                "bbox": bbox,
                "parameters": ",".join(self.QUERY_PARAMETERS),
                "starttime": s.isoformat().replace("+00:00", "Z"),
                "endtime": e.isoformat().replace("+00:00", "Z"),
                "timestep": 60,
            }
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                response = await client.get(self.WFS_URL, params=params)
                response.raise_for_status()
            return self._parse_response(response.text)

        # Fire all bbox/lookback widening attempts concurrently and return the
        # first one that has observations. Cancel the rest immediately.
        attempts = (
            (start_time, end_time, 0.5),                       # ~55 km, current
            (start_time, end_time, 1.5),                       # ~165 km, current
            (end_time - timedelta(hours=24), end_time, 1.5),   # 24h, 165 km
            (end_time - timedelta(hours=72), end_time, 3.0),   # 72h, ~330 km
        )

        tasks = [asyncio.create_task(_query(s, e, hd)) for s, e, hd in attempts]
        last_result: dict | None = None
        try:
            for coro in asyncio.as_completed(tasks):
                try:
                    parsed = await coro
                except Exception:
                    continue
                last_result = parsed
                if parsed.get("observations"):
                    return self._select_nearest_station(parsed, lat, lon)
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

        return last_result or {"station": {}, "observations": {}}

    async def fetch_forecast_by_latlon(self, lat: float, lon: float, hours: float = FMI_FORECAST_MAX_HOURS) -> dict:
        now = datetime.now(timezone.utc)
        start_time = now.replace(minute=0, second=0, microsecond=0)
        horizon = max(1, min(int(round(hours)), FMI_FORECAST_MAX_HOURS))
        end_time = start_time + timedelta(hours=horizon)
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "getFeature",
            "storedquery_id": "fmi::forecast::harmonie::surface::point::timevaluepair",
            "latlon": f"{lat},{lon}",
            "parameters": ",".join(FORECAST_PARAMETERS),
            "starttime": start_time.isoformat().replace("+00:00", "Z"),
            "endtime": end_time.isoformat().replace("+00:00", "Z"),
            "timestep": 60,
        }
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(self.WFS_URL, params=params)
            response.raise_for_status()
        return self._parse_forecast_response(response.text)

    async def fetch_point_weather(self, lat: float, lon: float, timeframe: str = "24h") -> dict:
        obs_start, obs_end = self._resolve_observation_window()
        observations = await self.fetch_observations_by_latlon(lat, lon, obs_start, obs_end)
        forecast_hours = self._resolve_forecast_hours(timeframe)
        if forecast_hours > 0:
            try:
                forecast = await self.fetch_forecast_by_latlon(lat, lon, hours=forecast_hours)
            except Exception:
                forecast = {"station": {}, "observations": {}}
        else:
            forecast = {"station": {}, "observations": {}}
        return {
            "observations": observations,
            "forecast": forecast,
            "query": {"lat": lat, "lon": lon, "timeframe": timeframe, "forecast_hours": forecast_hours},
        }

    async def _fetch_xml(self, place: str, start_time: datetime, end_time: datetime) -> str:
        async def _query(s: datetime, e: datetime) -> str:
            params = {
                "service": "WFS",
                "version": "2.0.0",
                "request": "getFeature",
                "storedquery_id": "fmi::observations::weather::timevaluepair",
                "place": place,
                "parameters": ",".join(self.QUERY_PARAMETERS),
                "starttime": s.isoformat().replace("+00:00", "Z"),
                "endtime": e.isoformat().replace("+00:00", "Z"),
                "timestep": 60,
            }
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                response = await client.get(self.WFS_URL, params=params)
                response.raise_for_status()
            return response.text

        # Fire all lookback widths concurrently; return the first with observations.
        lookbacks = [
            (start_time, end_time),
            (end_time - timedelta(hours=24), end_time),
            (end_time - timedelta(hours=72), end_time),
        ]
        tasks = [asyncio.create_task(_query(s, e)) for s, e in lookbacks]
        last_xml: str = ""
        try:
            for coro in asyncio.as_completed(tasks):
                try:
                    xml = await coro
                except Exception:
                    continue
                last_xml = xml
                if self._has_observations(xml):
                    return xml
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
        return last_xml

    def _has_observations(self, xml_payload: str) -> bool:
        try:
            root = ET.fromstring(xml_payload)
        except ET.ParseError:
            return False
        return any(
            tvp is not None
            for tvp in root.findall(".//wml2:MeasurementTVP", self.NAMESPACES)
        )

    def _resolve_place(self, area: str) -> str:
        normalized_area = self._normalize_area(area)
        return self.AREA_PLACE_ALIASES.get(normalized_area, area)

    def _resolve_observation_window(self) -> tuple[datetime, datetime]:
        """Recent observation window — always anchored to "now", independent of the
        user-selected timeframe. Forecast horizon is handled separately."""
        end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        start = end - timedelta(hours=FMI_OBSERVATION_HOURS)
        return start, end

    def _resolve_forecast_hours(self, timeframe: str) -> int:
        """Translate user timeframe → forecast horizon in hours (capped)."""
        hours = forecast_horizon_hours(
            timeframe, default=FMI_FORECAST_MAX_HOURS, cap=FMI_FORECAST_MAX_HOURS
        )
        # Snapshot ("now") still benefits from a short forecast for the UI's
        # "next few hours" preview — don't collapse it to zero.
        tf = (timeframe or "").strip().lower()
        if tf in ("now", "snapshot", "latest"):
            return min(12, FMI_FORECAST_MAX_HOURS)
        return int(round(hours))

    def _parse_forecast_response(self, xml_payload: str) -> dict:
        root = ET.fromstring(xml_payload)
        observations: dict[str, dict] = {}
        position: dict[str, object] = {}

        for member in root.findall("wfs:member", self.NAMESPACES):
            observation = next(iter(member), None)
            if observation is None:
                continue

            parameter_id = self._extract_parameter_id(observation)
            metadata = FORECAST_PARAMETER_METADATA.get(parameter_id)
            if metadata is None:
                continue

            if not position:
                pos_text = observation.findtext(".//gml:pos", default="", namespaces=self.NAMESPACES)
                if pos_text:
                    parts = pos_text.split()
                    if len(parts) >= 2:
                        position = {"latitude": self._parse_numeric(parts[0]), "longitude": self._parse_numeric(parts[1])}

            points = self._extract_points(observation)
            latest = points[-1] if points else {"time": None, "value": None}
            observations[metadata["key"]] = {
                "source_parameter": parameter_id,
                "label": metadata["label"],
                "unit": metadata["unit"],
                "latest": latest,
                "values": points,
            }

        return {
            "station": position,
            "observations": observations,
        }

    def _parse_response(self, xml_payload: str) -> dict:
        """Parse FMI WFS observations response.

        Returns the station with the most parameter coverage (and, on ties, the
        first one seen). When the response contains multiple stations (bbox
        query), the per-station breakdown is preserved under ``_stations`` so
        the nearest-station selector can pick the right one.
        """
        root = ET.fromstring(xml_payload)
        # station_key (gml id of the sampling point) → {"station": {...}, "observations": {...}}
        by_station: dict[str, dict] = {}

        for member in root.findall("wfs:member", self.NAMESPACES):
            observation = next(iter(member), None)
            if observation is None:
                continue

            parameter_id = self._extract_parameter_id(observation)
            metadata = self.PARAMETER_METADATA.get(parameter_id)
            if metadata is None:
                continue

            station = self._extract_station(observation)
            # Use pos as the dedup key — multiple parameters from the same
            # station share the same gml:pos.
            key = f"{station.get('latitude')},{station.get('longitude')}"
            bucket = by_station.setdefault(key, {"station": station, "observations": {}})

            points = self._extract_points(observation)
            real_points = [p for p in points if p.get("value") is not None]
            latest = real_points[-1] if real_points else {"time": None, "value": None}
            bucket["observations"][metadata["key"]] = {
                "source_parameter": parameter_id,
                "label": metadata["label"],
                "unit": metadata["unit"],
                "latest": latest,
                "values": points,
            }

        if not by_station:
            return {"station": {}, "observations": {}, "_stations": []}

        # Pick the station with the most parameters reporting real values as a
        # sensible default; nearest-station logic can re-pick later.
        def _coverage(bucket: dict) -> int:
            return sum(
                1
                for p in bucket["observations"].values()
                if p.get("latest", {}).get("value") is not None
            )

        primary = max(by_station.values(), key=_coverage)
        return {
            "station": primary["station"],
            "observations": primary["observations"],
            "_stations": list(by_station.values()),
        }

    def _select_nearest_station(self, parsed: dict, lat: float, lon: float) -> dict:
        """Among the stations in a bbox response, pick the one closest to (lat,lon)
        that has at least one reporting parameter."""
        stations = parsed.get("_stations") or []
        if not stations:
            return parsed

        usable = [
            b for b in stations
            if any(p.get("latest", {}).get("value") is not None for p in b["observations"].values())
        ] or stations

        def _dist(b: dict) -> float:
            s = b.get("station", {})
            slat, slon = s.get("latitude"), s.get("longitude")
            if slat is None or slon is None:
                return float("inf")
            return _great_circle_km(lat, lon, float(slat), float(slon))

        best = min(usable, key=_dist)
        return {
            "station": best["station"],
            "observations": best["observations"],
            "_stations": stations,
        }

    def _extract_parameter_id(self, observation: ET.Element) -> str:
        href = observation.find("om:observedProperty", self.NAMESPACES)
        if href is None:
            return ""
        raw_href = href.attrib.get(f"{{{self.NAMESPACES['xlink']}}}href", "")
        return parse_qs(urlparse(raw_href).query).get("param", [""])[0]

    def _extract_station(self, observation: ET.Element) -> dict[str, object]:
        name = observation.findtext(
            ".//gml:name[@codeSpace='http://xml.fmi.fi/namespace/locationcode/name']",
            default="",
            namespaces=self.NAMESPACES,
        )
        region = observation.findtext(".//target:region", default="", namespaces=self.NAMESPACES)
        pos_text = observation.findtext(".//gml:pos", default="", namespaces=self.NAMESPACES)
        latitude = None
        longitude = None
        if pos_text:
            parts = pos_text.split()
            if len(parts) >= 2:
                latitude = self._parse_numeric(parts[0])
                longitude = self._parse_numeric(parts[1])

        return {
            "name": name,
            "region": region,
            "latitude": latitude,
            "longitude": longitude,
        }

    def _extract_points(self, observation: ET.Element) -> list[dict[str, object]]:
        points: list[dict[str, object]] = []
        for tvp in observation.findall(".//wml2:MeasurementTVP", self.NAMESPACES):
            points.append(
                {
                    "time": tvp.findtext("wml2:time", default="", namespaces=self.NAMESPACES),
                    "value": self._parse_numeric(
                        tvp.findtext("wml2:value", default="", namespaces=self.NAMESPACES)
                    ),
                }
            )
        return points

    def _build_summary(self, area: str, parsed: dict) -> str:
        station_name = parsed["station"].get("name") or area
        latest_parts = []
        for key in ("temperature", "wind_speed", "humidity", "pressure", "precipitation", "wind_gust", "visibility"):
            observation = parsed["observations"].get(key)
            if not observation:
                continue
            value = observation["latest"].get("value")
            if value is None:
                continue
            latest_parts.append(f"{observation['label']} {value} {observation['unit']}")

        detail = f": {', '.join(latest_parts)}" if latest_parts else ""
        return f"FMI weather observations for {area} from {station_name}{detail}"

    def _normalize_area(self, area: str) -> str:
        ascii_area = unicodedata.normalize("NFKD", area).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", ascii_area).strip().lower()

    def _parse_numeric(self, value: str) -> Optional[float]:
        if value == "":
            return None
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed
