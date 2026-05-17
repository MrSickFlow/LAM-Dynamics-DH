from __future__ import annotations

import math
import re
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import parse_qs, urlparse

import httpx

from ipb_backend.ingestion.base import SourceAdapter
from ipb_backend.ingestion.timeframe import parse_timeframe
from ipb_backend.models import DatasetRecord, LoadTarget, LoadTargetKind


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
        start_time, end_time = self._resolve_time_window(timeframe)

        # When a custom bbox load target is given, query by the centroid of the bbox so
        # that we find the nearest weather station regardless of whether it sits inside
        # the drawn rectangle or in a nearby municipality.
        if load_target is not None and load_target.kind == LoadTargetKind.BBOX and load_target.bbox_wgs84:
            min_x, min_y, max_x, max_y = load_target.bbox_wgs84
            center_lat = (min_y + max_y) / 2.0
            center_lon = (min_x + max_x) / 2.0
            parsed = await self.fetch_observations_by_latlon(center_lat, center_lon, start_time, end_time)
            place = f"{center_lat:.4f},{center_lon:.4f}"
        else:
            place = self._resolve_place(area)
            xml_payload = await self._fetch_xml(place, start_time, end_time)
            parsed = self._parse_response(xml_payload)

        # Fetch 48-hour forecast so the LLM can reason over the full planning horizon.
        # Failures are silently swallowed — observations are still recorded.
        station = parsed.get("station", {})
        forecast: dict = {}
        if station.get("latitude") and station.get("longitude"):
            try:
                forecast = await self.fetch_forecast_by_latlon(
                    float(station["latitude"]), float(station["longitude"])
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
                    "start_time": start_time.isoformat().replace("+00:00", "Z"),
                    "end_time": end_time.isoformat().replace("+00:00", "Z"),
                    "timestep_minutes": 60,
                    "parameters": list(self.QUERY_PARAMETERS),
                },
                **parsed,
                "forecast": forecast,
            },
        )

    async def fetch_observations_by_latlon(self, lat: float, lon: float, start_time: datetime, end_time: datetime) -> dict:
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "getFeature",
            "storedquery_id": "fmi::observations::weather::timevaluepair",
            "latlon": f"{lat},{lon}",
            "parameters": ",".join(self.QUERY_PARAMETERS),
            "starttime": start_time.isoformat().replace("+00:00", "Z"),
            "endtime": end_time.isoformat().replace("+00:00", "Z"),
            "timestep": 60,
            "maxlocations": 3,
        }
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(self.WFS_URL, params=params)
            response.raise_for_status()
        return self._parse_response(response.text)

    async def fetch_forecast_by_latlon(self, lat: float, lon: float) -> dict:
        now = datetime.now(timezone.utc)
        start_time = now.replace(minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(hours=48)
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
        start_time, end_time = self._resolve_time_window(timeframe)
        observations = await self.fetch_observations_by_latlon(lat, lon, start_time, end_time)
        try:
            forecast = await self.fetch_forecast_by_latlon(lat, lon)
        except Exception:
            forecast = {"station": {}, "observations": {}}
        return {
            "observations": observations,
            "forecast": forecast,
            "query": {"lat": lat, "lon": lon, "timeframe": timeframe},
        }

    async def _fetch_xml(self, place: str, start_time: datetime, end_time: datetime) -> str:
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "getFeature",
            "storedquery_id": "fmi::observations::weather::timevaluepair",
            "place": place,
            "parameters": ",".join(self.QUERY_PARAMETERS),
            "starttime": start_time.isoformat().replace("+00:00", "Z"),
            "endtime": end_time.isoformat().replace("+00:00", "Z"),
            "timestep": 60,
        }
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(self.WFS_URL, params=params)
            response.raise_for_status()
        return response.text

    def _resolve_place(self, area: str) -> str:
        normalized_area = self._normalize_area(area)
        return self.AREA_PLACE_ALIASES.get(normalized_area, area)

    def _resolve_time_window(self, timeframe: str) -> tuple[datetime, datetime]:
        return parse_timeframe(timeframe, forward=False)

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
        root = ET.fromstring(xml_payload)
        observations: dict[str, dict] = {}
        station: dict[str, object] = {}

        for member in root.findall("wfs:member", self.NAMESPACES):
            observation = next(iter(member), None)
            if observation is None:
                continue

            parameter_id = self._extract_parameter_id(observation)
            metadata = self.PARAMETER_METADATA.get(parameter_id)
            if metadata is None:
                continue

            if not station:
                station = self._extract_station(observation)

            points = self._extract_points(observation)
            latest = points[-1] if points else {"time": None, "value": None}
            observations[metadata["key"]] = {
                "source_parameter": parameter_id,
                "label": metadata["label"],
                "unit": metadata["unit"],
                "latest": latest,
                "values": points,
            }

        if not observations:
            raise ValueError("FMI response did not contain any supported observations")

        return {
            "station": station,
            "observations": observations,
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
