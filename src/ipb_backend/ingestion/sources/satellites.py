from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from ipb_backend.ingestion.base import SourceAdapter
from ipb_backend.models import DatasetRecord

AREA_CENTERS: dict[str, dict[str, float]] = {
    "north karelia": {"lat": 62.8, "lon": 30.2},
    "archipelago sea": {"lat": 60.2, "lon": 22.0},
    "lapland": {"lat": 68.9, "lon": 21.5},
    "lapland (kasivarren lappi)": {"lat": 68.9, "lon": 21.5},
    "kasivarren lappi": {"lat": 68.9, "lon": 21.5},
}

TLE_URLS = [
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=visual&FORMAT=tle",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=military&FORMAT=tle",
]

RECON_NAMES: dict[str, str] = {
    "USA 224": "KH-11 optical reconnaissance",
    "USA 245": "KH-11 optical reconnaissance",
    "USA 271": "KH-11 optical reconnaissance",
    "USA 290": "KH-11 optical reconnaissance",
    "USA 314": "KH-11 optical reconnaissance",
    "USA 326": "KH-11 optical reconnaissance",
    "USA 129": "KH-11 optical reconnaissance",
    "USA 186": "KH-11 optical reconnaissance",
    "WorldView-1": "Commercial imaging (50cm)",
    "WorldView-2": "Commercial imaging (46cm)",
    "WorldView-3": "Commercial imaging (31cm)",
    "GeoEye-1": "Commercial imaging (41cm)",
    "Pleiades 1A": "Commercial imaging (50cm)",
    "Pleiades 1B": "Commercial imaging (50cm)",
    "SPOT 6": "Commercial imaging (1.5m)",
    "SPOT 7": "Commercial imaging (1.5m)",
    "Sentinel-1A": "SAR imaging (C-band)",
    "Sentinel-1B": "SAR imaging (C-band)",
    "Sentinel-2A": "Multispectral imaging (10m)",
    "Sentinel-2B": "Multispectral imaging (10m)",
    "TerraSAR-X": "SAR imaging (X-band)",
    "TanDEM-X": "SAR imaging (X-band)",
    "PAZ": "SAR imaging (X-band)",
    "COSMO-SkyMed 1": "SAR imaging (X-band)",
    "COSMO-SkyMed 2": "SAR imaging (X-band)",
    "COSMO-SkyMed 3": "SAR imaging (X-band)",
    "COSMO-SkyMed 4": "SAR imaging (X-band)",
    "KOMPSAT-5": "SAR imaging (X-band)",
    "KOMPSAT-3": "Commercial imaging (70cm)",
    "KOMPSAT-3A": "Commercial imaging (55cm)",
    "ALOS-2": "SAR imaging (L-band)",
    "Landsat 8": "Multispectral (30m)",
    "Landsat 9": "Multispectral (30m)",
    "Gaofen-1": "Chinese imaging (2m)",
    "Gaofen-2": "Chinese imaging (0.8m)",
    "SuperView-1 01": "Chinese imaging (0.5m)",
    "SuperView-1 02": "Chinese imaging (0.5m)",
    "Jilin-1 01": "Chinese imaging (0.72m)",
    "Jilin-1 02": "Chinese imaging (0.72m)",
    "Jilin-1 03": "Chinese imaging (0.72m)",
    "Jilin-1 04": "Chinese imaging (0.72m)",
    "Jilin-1 05": "Chinese imaging (0.72m)",
    "Jilin-1 06": "Chinese imaging (0.72m)",
    "Jilin-1 07": "Chinese imaging (0.72m)",
    "Jilin-1 08": "Chinese imaging (0.72m)",
    "KANOPUS-V": "Russian imaging (2.5m)",
    "Resurs-P 1": "Russian imaging (1m)",
    "Resurs-P 2": "Russian imaging (1m)",
    "Resurs-P 3": "Russian imaging (1m)",
    "Persona": "Russian reconnaissance",
    "Bars-M": "Russian reconnaissance",
    "Lotos-S1": "Russian ELINT",
}


class SatelliteTleAdapter(SourceAdapter):
    def _resolve_center(self, area: str) -> dict[str, float]:
        normalized = self._normalize_area(area)
        return AREA_CENTERS.get(normalized, AREA_CENTERS["north karelia"])

    async def fetch(self, area: str, timeframe: str) -> DatasetRecord:
        center = self._resolve_center(area)
        satellite_info: dict[str, dict[str, Any]] = {}

        async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "IPB-Backend/1.0"}) as client:
            for tle_url in TLE_URLS:
                try:
                    resp = await client.get(tle_url)
                    if resp.status_code != 200:
                        continue
                    raw = resp.text.strip().split("\n")
                    for i in range(0, len(raw) - 2, 3):
                        name = raw[i].strip().replace("\r", "")
                        matched = self._match_recon_name(name)
                        if matched and name not in satellite_info:
                            line1 = raw[i + 1].strip().replace("\r", "")
                            line2 = raw[i + 2].strip().replace("\r", "")
                            norad = line2[2:7].strip()
                            satellite_info[name] = {
                                "norad_id": int(norad) if norad.isdigit() else None,
                                "type": matched,
                                "tle_line_1": line1,
                                "tle_line_2": line2,
                            }
                except Exception:
                    continue

        now = datetime.now(timezone.utc)
        for name, info in satellite_info.items():
            tle = info.get("tle_line_2", "")
            passes = self._simple_pass_prediction(tle, center["lat"], center["lon"], now)
            info["predicted_passes"] = passes

        return DatasetRecord(
            source_id=self.definition.source_id,
            category=self.definition.category,
            area=area,
            timeframe=timeframe,
            summary=self._build_summary(area, satellite_info),
            data={
                "provider": "Celestrak",
                "api": "Celestrak TLE, simplified pass predictor",
                "license": "Free",
                "query": {
                    "area": area,
                    "lat": center["lat"],
                    "lon": center["lon"],
                    "timeframe": timeframe,
                },
                "satellites": satellite_info,
                "total_tracked": len(satellite_info),
            },
        )

    def _match_recon_name(self, name: str) -> Optional[str]:
        for known, desc in RECON_NAMES.items():
            if known.lower() in name.lower():
                return desc
        return None

    def _simple_pass_prediction(
        self, tle_line2: str, obs_lat: float, obs_lon: float, now: datetime
    ) -> list[dict[str, Any]]:
        try:
            inc_deg = float(tle_line2[8:16].strip())
            mean_motion = float(tle_line2[52:63].strip())
            period_min = 1440.0 / mean_motion
            alt_km = 6371.0 * (mean_motion / 1440.0) ** (-2 / 3.0) - 6371.0
            inc = math.radians(inc_deg)
            max_lat = abs(inc_deg)
            if abs(obs_lat) > max_lat + 5:
                return []
            passes: list[dict[str, Any]] = []
            for orbit_offset in range(-1, 5):
                pass_time = now.timestamp() + orbit_offset * period_min * 60
                pass_dt = datetime.fromtimestamp(pass_time, tz=timezone.utc)
                passes.append({
                    "pass_time_utc": pass_dt.isoformat(),
                    "pass_time_unix": int(pass_time),
                    "duration_min": round(period_min, 1),
                    "altitude_km": round(alt_km),
                    "confidence": "estimated",
                })
            return sorted(passes, key=lambda p: p["pass_time_unix"])[:6]
        except (ValueError, IndexError):
            return []

    def _build_summary(self, area: str, satellite_info: dict[str, Any]) -> str:
        count = len(satellite_info)
        types: dict[str, int] = {}
        for info in satellite_info.values():
            t = info.get("type", "unknown")
            types[t] = types.get(t, 0) + 1
        type_desc = ", ".join(f"{k}: {v}" for k, v in sorted(types.items()))
        return f"Satellite TLE data for {area}: tracking {count} reconnaissance/imaging satellites ({type_desc})"

    def _normalize_area(self, area: str) -> str:
        ascii_area = unicodedata.normalize("NFKD", area).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", ascii_area).strip().lower()
