from __future__ import annotations

import math
from typing import Iterable

from ipb_backend.models import ConsistencyAnomaly, SpatialCluster


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def _anomaly_point(anomaly: ConsistencyAnomaly) -> tuple[float, float] | None:
    location = anomaly.location
    if not location:
        return None
    if location.get("type") == "Point":
        coords = location.get("coordinates", [])
        if len(coords) >= 2:
            return float(coords[1]), float(coords[0])
    lat = location.get("lat")
    lon = location.get("lon")
    if lat is not None and lon is not None:
        return float(lat), float(lon)
    return None


def cluster_anomalies(
    anomalies: Iterable[ConsistencyAnomaly],
    *,
    cluster_radius_km: float = 25.0,
    min_cluster_size: int = 2,
) -> list[SpatialCluster]:
    located: list[tuple[ConsistencyAnomaly, float, float]] = []
    for anomaly in anomalies:
        point = _anomaly_point(anomaly)
        if point is None:
            continue
        located.append((anomaly, point[0], point[1]))

    if len(located) < min_cluster_size:
        return []

    clusters: list[list[tuple[ConsistencyAnomaly, float, float]]] = []
    for item in located:
        placed = False
        for group in clusters:
            anchor = group[0]
            if _haversine_km(item[1], item[2], anchor[1], anchor[2]) <= cluster_radius_km:
                group.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])

    results: list[SpatialCluster] = []
    for index, group in enumerate(clusters, start=1):
        if len(group) < min_cluster_size:
            continue
        lat_sum = sum(lat for _, lat, _ in group)
        lon_sum = sum(lon for _, _, lon in group)
        centroid_lat = lat_sum / len(group)
        centroid_lon = lon_sum / len(group)
        max_radius = max(
            _haversine_km(centroid_lat, centroid_lon, lat, lon) for _, lat, lon in group
        )
        anomaly_ids = [anomaly.anomaly_id for anomaly, _, _ in group]
        affected_sources = sorted(
            {
                source
                for anomaly, _, _ in group
                for source in anomaly.vulnerable_sources + anomaly.immune_sources
            }
        )
        vulnerable_count = sum(1 for anomaly, _, _ in group if anomaly.vulnerable_sources)
        assessment = (
            "Spatial clustering of cross-validation failures — pattern is consistent with "
            "localized RF/GNSS degradation; analyst should verify against EW reporting."
            if vulnerable_count >= min_cluster_size
            else "Clustered inconsistencies may reflect regional infrastructure or sensor outages."
        )
        results.append(
            SpatialCluster(
                cluster_id=f"cluster-{index}",
                centroid={"lat": round(centroid_lat, 5), "lon": round(centroid_lon, 5)},
                radius_km=round(max_radius, 2),
                anomaly_count=len(group),
                anomaly_ids=anomaly_ids,
                affected_sources=affected_sources,
                pattern_assessment=assessment,
            )
        )
    return results
