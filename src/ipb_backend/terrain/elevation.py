from __future__ import annotations

import io
import math
from abc import ABC, abstractmethod
from typing import Any, Optional

import httpx

try:
    from pyproj import Transformer
    import rasterio  # noqa: F401 — presence check only at import time
    import numpy as np
    _LIBS_AVAILABLE = True
except ImportError:
    _LIBS_AVAILABLE = False

_NLS_WCS_BASE = "https://avoin-karttakuva.maanmittauslaitos.fi/avoin/wcs"
_COVERAGE_2M = "korkeusmalli_2m"
_COVERAGE_10M = "korkeusmalli_10m"
_SPOT_BUFFER_M = 30   # 60×60 m window for single-point elevation
_LOS_DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
_LOS_AZIMUTHS   = [  0,  45,  90, 135, 180, 225, 270, 315]


class ElevationProvider(ABC):
    @abstractmethod
    async def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        ...


class UnavailableElevationProvider(ElevationProvider):
    async def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        return None


class NlsElevationProvider(ElevationProvider):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._transformer: Optional[object] = None
        if _LIBS_AVAILABLE:
            from pyproj import Transformer as T
            self._transformer = T.from_crs("EPSG:4326", "EPSG:3067", always_xy=True)

    async def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        if not _LIBS_AVAILABLE or self._transformer is None:
            return None
        x, y = self._transformer.transform(lon, lat)  # type: ignore[union-attr]
        params = [
            ("service", "WCS"),
            ("request", "GetCoverage"),
            ("version", "2.0.1"),
            ("coverageId", _COVERAGE_2M),
            ("subset", f"E({x - _SPOT_BUFFER_M:.1f},{x + _SPOT_BUFFER_M:.1f})"),
            ("subset", f"N({y - _SPOT_BUFFER_M:.1f},{y + _SPOT_BUFFER_M:.1f})"),
            ("format", "image/tiff"),
            ("api-key", self._api_key),
        ]
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(_NLS_WCS_BASE, params=params)
            if resp.status_code != 200:
                return None
            return _sample_tiff_center(resp.content)
        except Exception:
            return None


def _sample_tiff_center(tiff_bytes: bytes) -> Optional[float]:
    try:
        import rasterio
        with rasterio.open(io.BytesIO(tiff_bytes)) as ds:
            data = ds.read(1)
            h, w = data.shape
            val = float(data[h // 2, w // 2])
            nodata = ds.nodata
            if nodata is not None and val == float(nodata):
                return None
            if val < -500 or val > 10_000:
                return None
            return round(val, 1)
    except Exception:
        return None


async def compute_radial_los(
    lat: float,
    lon: float,
    api_key: str,
    radius_m: float = 2000.0,
    observer_height_m: float = 2.0,
) -> dict[str, Any]:
    """
    Fetch a 10 m DEM tile and compute radial LOS in 8 compass directions.
    Returns a dict with 'available', 'directions', and 'summary'.
    """
    if not _LIBS_AVAILABLE:
        return {"available": False, "note": "rasterio/pyproj not installed"}
    if not api_key:
        return {"available": False, "note": "NLS API key not configured"}

    from pyproj import Transformer as T
    import rasterio
    import rasterio.transform
    import numpy as np

    to_3067 = T.from_crs("EPSG:4326", "EPSG:3067", always_xy=True)
    cx, cy = to_3067.transform(lon, lat)

    buf = radius_m + 100
    params = [
        ("service", "WCS"),
        ("request", "GetCoverage"),
        ("version", "2.0.1"),
        ("coverageId", _COVERAGE_10M),
        ("subset", f"E({cx - buf:.1f},{cx + buf:.1f})"),
        ("subset", f"N({cy - buf:.1f},{cy + buf:.1f})"),
        ("format", "image/tiff"),
        ("api-key", api_key),
    ]
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.get(_NLS_WCS_BASE, params=params)
        if resp.status_code != 200:
            return {"available": False, "note": f"NLS WCS returned HTTP {resp.status_code}"}
    except Exception as exc:
        return {"available": False, "note": str(exc)}

    try:
        with rasterio.open(io.BytesIO(resp.content)) as ds:
            data = ds.read(1).astype(float)
            transform = ds.transform
            nodata = ds.nodata
            if nodata is not None:
                data[data == float(nodata)] = float("nan")

            pixel_m = abs(transform.a)  # metres per pixel (10 m)

            row_obs, col_obs = rasterio.transform.rowcol(transform, cx, cy)
            if not (0 <= row_obs < data.shape[0] and 0 <= col_obs < data.shape[1]):
                return {"available": False, "note": "Observer outside DEM tile"}

            obs_ground = float(data[row_obs, col_obs])
            if math.isnan(obs_ground):
                return {"available": False, "note": "No elevation data at observer point"}

            eye_elev = obs_ground + observer_height_m
            n_steps = max(1, int(radius_m / pixel_m))

            directions: list[dict[str, Any]] = []
            for dir_name, azimuth in zip(_LOS_DIRECTIONS, _LOS_AZIMUTHS):
                az_rad = math.radians(azimuth)
                # In EPSG:3067: x=easting (+east), y=northing (+north)
                step_x = math.sin(az_rad) * pixel_m
                step_y = math.cos(az_rad) * pixel_m

                clear_range_m = radius_m
                obstructed = False
                obstruction_dist_m: Optional[float] = None

                for step in range(1, n_steps + 1):
                    dist_m = step * pixel_m
                    sx = cx + math.sin(az_rad) * dist_m
                    sy = cy + math.cos(az_rad) * dist_m
                    row, col = rasterio.transform.rowcol(transform, sx, sy)
                    if not (0 <= row < data.shape[0] and 0 <= col < data.shape[1]):
                        clear_range_m = dist_m - pixel_m
                        break
                    terrain = float(data[row, col])
                    if math.isnan(terrain):
                        continue
                    if terrain > eye_elev:
                        obstructed = True
                        obstruction_dist_m = round(dist_m)
                        clear_range_m = dist_m
                        break

                directions.append({
                    "direction": dir_name,
                    "azimuth_deg": azimuth,
                    "clear_range_m": round(obstruction_dist_m) if obstructed else round(clear_range_m),
                    "obstructed": obstructed,
                    "obstruction_dist_m": round(obstruction_dist_m) if obstructed else None,
                })

            clear = [d["direction"] for d in directions if not d["obstructed"]]
            blocked = [f"{d['direction']} @{d['obstruction_dist_m']}m" for d in directions if d["obstructed"]]
            parts = []
            if clear:
                parts.append(f"Clear ({int(radius_m / 1000)} km): {', '.join(clear)}")
            if blocked:
                parts.append(f"Blocked: {', '.join(blocked)}")

            return {
                "available": True,
                "observer_elevation_m": round(obs_ground, 1),
                "observer_height_m": observer_height_m,
                "analysis_radius_m": radius_m,
                "directions": directions,
                "summary": "; ".join(parts) if parts else "Flat terrain — full range clear",
                "note": None,
            }
    except Exception as exc:
        return {"available": False, "note": str(exc)}


def build_elevation_provider(api_key: str) -> ElevationProvider:
    if not api_key or not _LIBS_AVAILABLE:
        return UnavailableElevationProvider()
    return NlsElevationProvider(api_key)
