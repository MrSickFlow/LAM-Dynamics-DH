from __future__ import annotations

import math
import re
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import httpx

from ipb_backend.ingestion.base import SourceAdapter
from ipb_backend.models import DatasetRecord


class FmiAdapter(SourceAdapter):
    WFS_URL = "https://opendata.fmi.fi/wfs"
    QUERY_PARAMETERS = ("t2m", "ws_10min", "n_man", "vis")
    PARAMETER_METADATA = {
        "t2m": {"key": "temperature", "label": "Temperature", "unit": "C"},
        "ws_10min": {"key": "wind_speed", "label": "Wind speed", "unit": "m/s"},
        "n_man": {"key": "cloud_cover", "label": "Cloud cover", "unit": "okta"},
        "vis": {"key": "visibility", "label": "Visibility", "unit": "m"},
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

    async def fetch(self, area: str, timeframe: str) -> DatasetRecord:
        place = self._resolve_place(area)
        start_time, end_time = self._resolve_time_window(timeframe)
        xml_payload = await self._fetch_xml(place, start_time, end_time)
        parsed = self._parse_response(xml_payload)

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
            },
        )

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
        match = re.fullmatch(r"\s*(\d+)\s*h\s*", timeframe)
        hours = int(match.group(1)) if match else 24
        end_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(hours=hours)
        return start_time, end_time

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
        for key in ("temperature", "wind_speed", "visibility"):
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

    def _parse_numeric(self, value: str) -> float | None:
        if value == "":
            return None
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed
