from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from sgp4.api import Satrec, jday

from ipb_backend.ingestion.base import SourceAdapter
from ipb_backend.models import DatasetRecord, LoadTarget

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

    def _build_demo_satellites(self) -> dict[str, dict[str, Any]]:
        import random
        rng = random.Random(42)
        names = [
            ("USA 224", "KH-11 optical reconnaissance"),
            ("Sentinel-2A", "Multispectral imaging (10m)"),
            ("WorldView-3", "Commercial imaging (31cm)"),
            ("TerraSAR-X", "SAR imaging (X-band)"),
            ("Landsat 9", "Multispectral (30m)"),
            ("KOMPSAT-3A", "Commercial imaging (55cm)"),
            ("Sentinel-1A", "SAR imaging (C-band)"),
            ("Resurs-P 1", "Russian imaging (1m)"),
        ]
        now = datetime.now(timezone.utc)
        result = {}
        for name, stype in names:
            norad = rng.randint(10000, 99999)
            alt = rng.randint(300, 900)
            period = rng.uniform(90, 100)
            passes = []
            for offset in range(4):
                pt = now.timestamp() + offset * period * 60
                passes.append({
                    "pass_time_utc": datetime.fromtimestamp(pt, tz=timezone.utc).isoformat(),
                    "pass_time_unix": int(pt),
                    "duration_min": round(period, 1),
                    "altitude_km": alt,
                    "confidence": "estimated",
                })
            result[name] = {
                "norad_id": norad,
                "type": stype,
                "predicted_passes": passes,
            }
        return result

    async def _parse_tle_lines(self, raw_lines: list[str]) -> dict[str, dict[str, Any]]:
        result = {}
        for i in range(0, len(raw_lines) - 2, 3):
            name = raw_lines[i].strip().replace("\r", "")
            matched = self._match_recon_name(name)
            if matched and name not in result:
                line1 = raw_lines[i + 1].strip().replace("\r", "")
                line2 = raw_lines[i + 2].strip().replace("\r", "")
                norad = line2[2:7].strip()
                result[name] = {
                    "norad_id": int(norad) if norad.isdigit() else None,
                    "type": matched,
                    "tle_line_1": line1,
                    "tle_line_2": line2,
                }
        return result

    def _match_space_track_name(self, name: str) -> Optional[str]:
        name_lower = name.lower()
        for known, desc in RECON_NAMES.items():
            known_lower = known.lower()
            known_parts = known_lower.replace("-", " ").split()
            if all(part in name_lower for part in known_parts):
                return desc
            if known_lower in name_lower or name_lower in known_lower:
                return desc
        return None

    async def _fetch_from_space_track(self, client: httpx.AsyncClient) -> dict[str, dict[str, Any]]:
        from ipb_backend.config import settings
        login_resp = await client.post(
            "https://www.space-track.org/ajaxauth/login",
            data={"identity": settings.space_track_username, "password": settings.space_track_password},
        )
        login_resp.raise_for_status()
        gp_resp = await client.get(
            "https://www.space-track.org/basicspacedata/query/class/gp/orderby/EPOCH%20DESC/format/json/limit/2000"
        )
        gp_resp.raise_for_status()
        import json
        satellites_data = json.loads(gp_resp.text)
        result = {}
        for sat in satellites_data:
            name = (sat.get("OBJECT_NAME") or "").strip()
            matched = self._match_space_track_name(name)
            if matched and name not in result:
                tle_line1 = (sat.get("TLE_LINE1") or "").strip()
                tle_line2 = (sat.get("TLE_LINE2") or "").strip()
                norad = sat.get("NORAD_CAT_ID")
                result[name] = {
                    "norad_id": int(norad) if norad else None,
                    "type": matched,
                    "tle_line_1": tle_line1,
                    "tle_line_2": tle_line2,
                }
        return result

    async def _fetch_from_celestrak(self, client: httpx.AsyncClient) -> dict[str, dict[str, Any]]:
        result = {}
        for tle_url in TLE_URLS:
            try:
                resp = await client.get(tle_url)
                if resp.status_code != 200:
                    continue
                raw = resp.text.strip().split("\n")
                parsed = await self._parse_tle_lines(raw)
                for name, info in parsed.items():
                    if name not in result:
                        result[name] = info
            except Exception:
                break
        return result

    async def fetch(self, area: str, timeframe: str, load_target: LoadTarget | None = None) -> DatasetRecord:
        from ipb_backend.config import settings

        center = self._resolve_center(area)
        satellite_info: dict[str, dict[str, Any]] = {}
        fetch_errors: dict[str, str] = {}
        provider = ""
        used_demo = False

        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=8.0), headers={"User-Agent": "IPB-Backend/1.0"}) as client:
            if settings.space_track_username and settings.space_track_password:
                try:
                    satellite_info = await self._fetch_from_space_track(client)
                    provider = "Space-Track.org"
                except Exception as exc:
                    fetch_errors["space-track"] = str(exc)

            if not satellite_info:
                try:
                    satellite_info = await self._fetch_from_celestrak(client)
                    provider = "Celestrak"
                except Exception as exc:
                    fetch_errors["celestrak"] = str(exc)

        if not satellite_info:
            satellite_info = self._build_demo_satellites()
            provider = "Demo data (all TLE sources unreachable)"
            used_demo = True

        if not used_demo:
            now = datetime.now(timezone.utc)
            for name, info in satellite_info.items():
                tle1 = info.get("tle_line_1", "")
                tle2 = info.get("tle_line_2", "")
                passes = self._simple_pass_prediction(tle2, center["lat"], center["lon"], now)
                info["predicted_passes"] = passes
                pos = self._compute_position(tle1, tle2, now)
                if pos:
                    info["current_lat"], info["current_lon"], info["current_alt_km"] = pos

        return DatasetRecord(
            source_id=self.definition.source_id,
            category=self.definition.category,
            area=area,
            timeframe=timeframe,
            summary=self._build_summary(area, satellite_info),
            data={
                "provider": provider,
                "api": "Space-Track / Celestrak TLE, simplified pass predictor",
                "license": "Free / ODFL",
                "query": {
                    "area": area,
                    "lat": center["lat"],
                    "lon": center["lon"],
                    "timeframe": timeframe,
                },
                "errors": fetch_errors,
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

    def _compute_position(self, tle_line1: str, tle_line2: str, now: datetime) -> tuple[float, float, float] | None:
        """Return (lat, lon, alt_km) of satellite sub-point at the given time, or None on error."""
        try:
            satrec = Satrec.twoline2rv(tle_line1, tle_line2)
            jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute,
                          now.second + now.microsecond / 1e6)
            e, r, _ = satrec.sgp4(jd, fr)
            if e != 0 or r is None:
                return None
            # TEME → ECEF via GMST rotation
            T = (jd + fr - 2451545.0) / 36525.0
            gmst_sec = (67310.54841 + (876600 * 3600 + 8640184.812866) * T
                        + 0.093104 * T ** 2 - 6.2e-6 * T ** 3) % 86400.0
            gmst = math.radians(gmst_sec / 240.0)
            x = r[0] * math.cos(gmst) + r[1] * math.sin(gmst)
            y = -r[0] * math.sin(gmst) + r[1] * math.cos(gmst)
            z = r[2]
            lon = math.degrees(math.atan2(y, x))
            lat = math.degrees(math.atan2(z, math.sqrt(x ** 2 + y ** 2)))
            alt = math.sqrt(x ** 2 + y ** 2 + z ** 2) - 6371.0
            return round(lat, 4), round(lon, 4), round(alt, 1)
        except Exception:
            return None

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
