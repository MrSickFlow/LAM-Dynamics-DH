from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Optional

import httpx

_OPENTOPO_BASE = "https://api.opentopodata.org/v1"
_OPENTOPO_DATASET = "eudem25m"   # EU-DEM 25 m, covers all of Finland

_LOS_DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
_LOS_AZIMUTHS   = [  0,  45,  90, 135, 180, 225, 270, 315]


# ── spot elevation ────────────────────────────────────────────────

class ElevationProvider(ABC):
    @abstractmethod
    async def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        ...


class UnavailableElevationProvider(ElevationProvider):
    async def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        return None


class OpenTopoElevationProvider(ElevationProvider):
    async def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{_OPENTOPO_BASE}/{_OPENTOPO_DATASET}",
                    params={"locations": f"{lat},{lon}"},
                )
            if resp.status_code != 200:
                return None
            results = resp.json().get("results", [])
            if not results:
                return None
            val = results[0].get("elevation")
            return round(float(val), 1) if val is not None else None
        except Exception:
            return None


# kept for API compatibility — no longer used internally
class NlsElevationProvider(UnavailableElevationProvider):
    def __init__(self, api_key: str) -> None:
        pass


def build_elevation_provider(api_key: str = "") -> ElevationProvider:
    return OpenTopoElevationProvider()


# ── radial LOS via Open Topo Data batch query ──────────────────────

async def compute_radial_los(
    lat: float,
    lon: float,
    api_key: str = "",
    radius_m: float = 2000.0,
    observer_height_m: float = 2.0,
) -> dict[str, Any]:
    """
    Sample 8 compass directions using Open Topo Data (EU-DEM 25 m) in a
    single batch request.  Returns per-direction clear/blocked range.
    """
    STEP_M = 200        # sample every 200 m → 10 steps per direction
    n_steps = max(1, int(radius_m / STEP_M))

    # Build sample list: [observer] + 8×n_steps profile points
    sample_lats: list[float] = [lat]
    sample_lons: list[float] = [lon]
    # Map (azimuth_idx, step_idx) → position in sample list (offset by 1)
    grid: list[tuple[int, int]] = []

    for az_idx, azimuth in enumerate(_LOS_AZIMUTHS):
        az_rad = math.radians(azimuth)
        cos_lat = math.cos(math.radians(lat))
        for step in range(1, n_steps + 1):
            dist_m = step * STEP_M
            dlat = math.cos(az_rad) * dist_m / 111_000
            dlon = math.sin(az_rad) * dist_m / (111_000 * cos_lat)
            sample_lats.append(lat + dlat)
            sample_lons.append(lon + dlon)
            grid.append((az_idx, step))

    locations = "|".join(f"{la},{lo}" for la, lo in zip(sample_lats, sample_lons))

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                f"{_OPENTOPO_BASE}/{_OPENTOPO_DATASET}",
                params={"locations": locations},
            )
        if resp.status_code != 200:
            return {"available": False, "note": f"Open Topo Data returned HTTP {resp.status_code}"}
        results = resp.json().get("results", [])
        if len(results) < len(sample_lats):
            return {"available": False, "note": "Incomplete elevation data returned"}
    except Exception as exc:
        return {"available": False, "note": str(exc)}

    elevations = [r.get("elevation") for r in results]
    obs_ground = elevations[0]
    if obs_ground is None:
        return {"available": False, "note": "No elevation data at observer point"}

    eye_elev = float(obs_ground) + observer_height_m

    # Group into per-direction lists
    dir_steps: list[list[Optional[float]]] = [[] for _ in _LOS_DIRECTIONS]
    for i, (az_idx, _step) in enumerate(grid):
        dir_steps[az_idx].append(elevations[i + 1])  # +1 skips observer

    directions: list[dict[str, Any]] = []
    for az_idx, (dir_name, azimuth) in enumerate(zip(_LOS_DIRECTIONS, _LOS_AZIMUTHS)):
        clear_range_m = radius_m
        obstructed = False
        obstruction_dist_m: Optional[float] = None

        for step_idx, terrain_elev in enumerate(dir_steps[az_idx]):
            dist_m = (step_idx + 1) * STEP_M
            if terrain_elev is None:
                continue
            if float(terrain_elev) > eye_elev:
                obstructed = True
                obstruction_dist_m = dist_m
                clear_range_m = dist_m
                break

        directions.append({
            "direction": dir_name,
            "azimuth_deg": azimuth,
            "clear_range_m": int(obstruction_dist_m) if obstructed else int(clear_range_m),
            "obstructed": obstructed,
            "obstruction_dist_m": int(obstruction_dist_m) if obstructed else None,
        })

    clear   = [d["direction"] for d in directions if not d["obstructed"]]
    blocked = [f"{d['direction']} @{d['obstruction_dist_m']}m" for d in directions if d["obstructed"]]
    parts: list[str] = []
    if clear:
        parts.append(f"Clear ({int(radius_m / 1000)} km): {', '.join(clear)}")
    if blocked:
        parts.append(f"Blocked: {', '.join(blocked)}")

    return {
        "available": True,
        "observer_elevation_m": round(float(obs_ground), 1),
        "observer_height_m": observer_height_m,
        "analysis_radius_m": radius_m,
        "elevation_source": "EU-DEM 25m (Open Topo Data)",
        "directions": directions,
        "summary": "; ".join(parts) if parts else "Flat terrain — full range clear",
        "note": None,
    }
