from __future__ import annotations

import httpx
import re

from ipb_backend.ingestion.base import SourceAdapter
from ipb_backend.models import DatasetRecord, SourceCategory, SourceDefinition
from ipb_backend.spatial import point_in_bbox, resolve_area_bbox


class RoadSurfaceAdapter(SourceAdapter):
    """
    Adapter for fetching road surface condition data from Fintraffic Digitraffic API.
    """

    # Define the unique ID for this source
    source_id = "digitraffic-road-surface"

    # Define the name and category for this source
    definition = SourceDefinition(
        source_id=source_id,
        name="Fintraffic Road Surface Conditions",
        category=SourceCategory.INFRASTRUCTURE,
        description="Real-time road surface conditions from Fintraffic's road weather stations.",
        refresh_interval_seconds=600,  # Data is updated frequently
    )

    API_BASE_URL = "https://tie.digitraffic.fi/api"

    async def fetch(self, area: str, timeframe: str) -> DatasetRecord:
        """
        Fetches road surface condition data for a given area.
        The timeframe is ignored as the API provides the latest data.
        """
        bbox = resolve_area_bbox(area)

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # 1. Fetch all station metadata to get their locations
            stations_response = await client.get(f"{self.API_BASE_URL}/weather/v1/stations")
            stations_response.raise_for_status()
            stations_metadata = stations_response.json()

            # 2. Fetch the latest sensor data for all stations
            data_response = await client.get(f"{self.API_BASE_URL}/weather/v1/stations/data")
            data_response.raise_for_status()
            stations_data = data_response.json()

        # 3. Create a lookup for station locations
        station_locations = {
            station["id"]: station["geometry"]["coordinates"]
            for station in stations_metadata.get("features", [])
            if station.get("geometry")
        }

        # 4. Process and filter data
        features = []
        station_data_map = {station["id"]: station for station in stations_data.get("stations", [])}

        for station_meta in stations_metadata.get("features", []):
            station_id = station_meta["id"]
            coords = station_locations.get(station_id)
            if not coords or not point_in_bbox(coords, bbox):
                continue

            station_data = station_data_map.get(station_id)
            if not station_data:
                continue

            properties = {
                "station_id": station_id,
                "name": station_meta["properties"]["name"],
                "road_station_id": station_meta["properties"]["roadStationId"],
                "data_updated_time": station_data.get("dataUpdatedTime"),
            }

            for sensor in station_data.get("sensors", []):
                key = self._to_snake_case(sensor.get("name", f"sensor_{sensor.get('id')}"))
                properties[key] = sensor.get("value")
                properties[f"{key}_unit"] = sensor.get("unit")

            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": coords},
                    "properties": properties,
                }
            )

        return DatasetRecord(
            source_id=self.definition.source_id,
            category=self.definition.category,
            area=area,
            timeframe=timeframe,
            summary=self._build_summary(area, len(features)),
            data={
                "provider": self.definition.name,
                "query_bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "features": features,
                "total_features": len(features),
            },
        )

    def _to_snake_case(self, name: str) -> str:
        if not name:
            return ""
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    def _build_summary(self, area: str, total_features: int) -> str:
        return f"Found {total_features} road weather stations with surface condition data in {area}."