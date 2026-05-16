from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import Optional

import httpx

try:
    from pyproj import Transformer
    import rasterio  # noqa: F401 — presence check only at import time
    _LIBS_AVAILABLE = True
except ImportError:
    _LIBS_AVAILABLE = False

_NLS_WCS_BASE = "https://avoin-karttakuva.maanmittauslaitos.fi/avoin/wcs"
_COVERAGE_ID = "korkeusmalli_2m"
_BUFFER_M = 30  # metres each side of the point → 60×60 m window, ~900 pixels at 2 m res


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
            ("coverageId", _COVERAGE_ID),
            ("subset", f"E({x - _BUFFER_M:.1f},{x + _BUFFER_M:.1f})"),
            ("subset", f"N({y - _BUFFER_M:.1f},{y + _BUFFER_M:.1f})"),
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


def build_elevation_provider(api_key: str) -> ElevationProvider:
    if not api_key or not _LIBS_AVAILABLE:
        return UnavailableElevationProvider()
    return NlsElevationProvider(api_key)
