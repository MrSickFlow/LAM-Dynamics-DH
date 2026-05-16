from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from ipb_backend.consistency.classification import get_source_profile, normalize_area, resolve_area_center
from ipb_backend.consistency.fixtures import ARCHIPELAGO_MARITIME_FIXTURE
from ipb_backend.models import EwClassification
from ipb_backend.models import (
    AnomalySeverity,
    ConsistencyAnomaly,
    DatasetRecord,
    LikelyExplanation,
    SourceDefinition,
    SourceStatus,
)


class ConsistencyRule(Protocol):
    rule_id: str

    def evaluate(
        self,
        *,
        area: str,
        timeframe: str,
        records: dict[str, DatasetRecord],
        sources: list[SourceDefinition],
        context: dict[str, Any],
    ) -> list[ConsistencyAnomaly]:
        ...


def _point_geojson(lon: float, lat: float) -> dict[str, Any]:
    return {"type": "Point", "coordinates": [lon, lat]}


def _new_anomaly(
    *,
    rule_id: str,
    title: str,
    description: str,
    severity: AnomalySeverity,
    vulnerable_sources: list[str],
    immune_sources: list[str],
    location: Optional[dict[str, Any]] = None,
    measured: Optional[dict[str, Any]] = None,
    expected: Optional[dict[str, Any]] = None,
    likely_explanations: Optional[list[LikelyExplanation]] = None,
    synthetic_demo: bool = False,
) -> ConsistencyAnomaly:
    return ConsistencyAnomaly(
        anomaly_id=f"{rule_id}-{uuid.uuid4().hex[:8]}",
        rule_id=rule_id,
        title=title,
        description=description,
        severity=severity,
        location=location,
        vulnerable_sources=vulnerable_sources,
        immune_sources=immune_sources,
        measured=measured or {},
        expected=expected or {},
        likely_explanations=likely_explanations or [],
        synthetic_demo=synthetic_demo,
    )


def _is_demo_record(record: DatasetRecord) -> bool:
    data = record.data
    blob = " ".join(
        str(data.get(key, ""))
        for key in ("provider", "api", "note")
    ).lower()
    return "demo" in blob or "fallback" in blob or "placeholder" in blob


def _latest_numeric(observations: dict, key: str) -> Optional[float]:
    block = observations.get(key) or {}
    latest = block.get("latest") or {}
    value = latest.get("value")
    return float(value) if value is not None else None


class SourceGapRule:
    rule_id = "source-gap"

    def evaluate(self, *, area, timeframe, records, sources, context) -> list[ConsistencyAnomaly]:
        center = resolve_area_center(area)
        anomalies: list[ConsistencyAnomaly] = []
        for source in sources:
            if not source.enabled:
                continue
            profile = get_source_profile(source.source_id)
            if profile.ew_classification != EwClassification.VULNERABLE:
                continue
            if source.source_id in records:
                continue
            anomalies.append(
                _new_anomaly(
                    rule_id=self.rule_id,
                    title=f"Missing EW-vulnerable source: {source.name}",
                    description=(
                        f"No ingested data for {source.source_id}. "
                        "Absence may be outage, denied access, or RF-contested collection."
                    ),
                    severity=AnomalySeverity.MEDIUM,
                    vulnerable_sources=[source.source_id],
                    immune_sources=[],
                    location=_point_geojson(center["lon"], center["lat"]),
                    measured={"present": False},
                    expected={"present": True},
                    likely_explanations=[
                        LikelyExplanation(cause="ingestion_failure", likelihood=0.45, note="API or credential issue"),
                        LikelyExplanation(cause="ew_degradation", likelihood=0.25, note="Live feed unavailable in contested EM environment"),
                        LikelyExplanation(cause="not_requested", likelihood=0.30, note="Source not included in ingest batch"),
                    ],
                )
            )
        return anomalies


class DemoDataTrustRule:
    rule_id = "demo-fallback"

    def evaluate(self, *, area, timeframe, records, sources, context) -> list[ConsistencyAnomaly]:
        anomalies: list[ConsistencyAnomaly] = []
        center = resolve_area_center(area)
        for source_id, record in records.items():
            if not _is_demo_record(record):
                continue
            profile = get_source_profile(source_id)
            if profile.ew_classification == EwClassification.IMMUNE:
                continue
            anomalies.append(
                _new_anomaly(
                    rule_id=self.rule_id,
                    title=f"Demo or fallback data: {source_id}",
                    description=(
                        f"{source_id} is not live open-source telemetry. "
                        "Cross-validation against immune layers is limited until real data is ingested."
                    ),
                    severity=AnomalySeverity.LOW,
                    vulnerable_sources=[source_id],
                    immune_sources=[],
                    location=_point_geojson(center["lon"], center["lat"]),
                    measured={"data_mode": "demo_or_fallback"},
                    expected={"data_mode": "live"},
                    likely_explanations=[
                        LikelyExplanation(cause="missing_api_key", likelihood=0.6, note="Configure credentials in .env"),
                        LikelyExplanation(cause="offline_development", likelihood=0.4, note="Intentional demo for hackathon UI"),
                    ],
                )
            )
        return anomalies


class FmiObservationForecastRule:
    rule_id = "fmi-obs-forecast"

    def evaluate(self, *, area, timeframe, records, sources, context) -> list[ConsistencyAnomaly]:
        fmi_record = records.get("fmi")
        forecast_bundle = context.get("fmi_forecast")
        if not fmi_record or not forecast_bundle:
            return []

        obs = fmi_record.data.get("observations", {})
        fc_obs = forecast_bundle.get("observations", {})
        station = fmi_record.data.get("station", {})
        lat = station.get("latitude")
        lon = station.get("longitude")
        if lat is None or lon is None:
            return []

        anomalies: list[ConsistencyAnomaly] = []
        checks = [
            ("wind_speed", 4.0, "m/s", AnomalySeverity.HIGH),
            ("temperature", 5.0, "C", AnomalySeverity.MEDIUM),
            ("humidity", 20.0, "%", AnomalySeverity.LOW),
        ]
        for key, threshold, unit, severity in checks:
            observed = _latest_numeric(obs, key)
            forecast = _latest_numeric(fc_obs, key)
            if observed is None or forecast is None:
                continue
            delta = abs(observed - forecast)
            if delta <= threshold:
                continue
            anomalies.append(
                _new_anomaly(
                    rule_id=self.rule_id,
                    title=f"FMI observation vs forecast divergence: {key}",
                    description=(
                        f"Live station reports {observed}{unit} while short-range model expects "
                        f"{forecast}{unit} (delta {delta:.1f}{unit})."
                    ),
                    severity=severity,
                    vulnerable_sources=["fmi"],
                    immune_sources=["fmi"],
                    location=_point_geojson(float(lon), float(lat)),
                    measured={"observation": observed, "unit": unit},
                    expected={"forecast": forecast, "max_delta": threshold},
                    likely_explanations=[
                        LikelyExplanation(cause="localized_weather", likelihood=0.5, note="Microclimate not captured by model"),
                        LikelyExplanation(cause="stale_station_feed", likelihood=0.25, note="Sensor lag or maintenance"),
                        LikelyExplanation(cause="data_interference", likelihood=0.25, note="Telemetry path degraded — verify adjacent stations"),
                    ],
                )
            )
        return anomalies


class FmiNeighboringStationsRule:
    rule_id = "fmi-station-outlier"

    def evaluate(self, *, area, timeframe, records, sources, context) -> list[ConsistencyAnomaly]:
        multi_station = context.get("fmi_multi_station")
        if not multi_station:
            return []

        stations = multi_station.get("stations", [])
        if len(stations) < 2:
            return []

        wind_values = [
            (station, _latest_numeric(station.get("observations", {}), "wind_speed"))
            for station in stations
        ]
        wind_values = [(station, value) for station, value in wind_values if value is not None]
        if len(wind_values) < 2:
            return []

        speeds = [value for _, value in wind_values]
        median = sorted(speeds)[len(speeds) // 2]
        anomalies: list[ConsistencyAnomaly] = []
        for station, speed in wind_values:
            if abs(speed - median) <= 6.0:
                continue
            lat = station.get("latitude")
            lon = station.get("longitude")
            if lat is None or lon is None:
                continue
            anomalies.append(
                _new_anomaly(
                    rule_id=self.rule_id,
                    title="FMI station wind outlier",
                    description=(
                        f"Station {station.get('name', 'unknown')} reports {speed} m/s while "
                        f"neighboring stations median is {median} m/s."
                    ),
                    severity=AnomalySeverity.MEDIUM,
                    vulnerable_sources=["fmi"],
                    immune_sources=["fmi"],
                    location=_point_geojson(float(lon), float(lat)),
                    measured={"wind_speed_m_s": speed},
                    expected={"median_wind_speed_m_s": median, "max_delta_m_s": 6.0},
                    likely_explanations=[
                        LikelyExplanation(cause="equipment_failure", likelihood=0.35, note="Single sensor fault"),
                        LikelyExplanation(cause="localized_wind", likelihood=0.3, note="Coastal or terrain channeling"),
                        LikelyExplanation(cause="data_interference", likelihood=0.35, note="Outlier feed — compare with ERA5 or adjacent sites"),
                    ],
                )
            )
        return anomalies


class CellTowerInfrastructureRule:
    rule_id = "celltower-coverage-gap"

    def evaluate(self, *, area, timeframe, records, sources, context) -> list[ConsistencyAnomaly]:
        cell_record = records.get("opencellid")
        osm_record = records.get("osm-poi")
        if not cell_record or not osm_record:
            return []

        cells = cell_record.data.get("cells", [])
        if len(cells) < 6:
            return []

        institution_count = 0
        for category in ("healthcare", "emergency_services", "government"):
            institution_count += len(osm_record.data.get("categories", {}).get(category, []))

        if institution_count >= 2:
            return []

        center = resolve_area_center(area)
        avg_range = sum(cell.get("range", 2000) for cell in cells) / len(cells)
        return [
            _new_anomaly(
                rule_id=self.rule_id,
                title="Cell infrastructure present but civic POIs sparse",
                description=(
                    f"{len(cells)} cell towers ingested with mean range {avg_range:.0f} m, "
                    f"but only {institution_count} healthcare/emergency/government POIs in OSM baseline. "
                    "Coverage mapping may disagree with expected service around institutions."
                ),
                severity=AnomalySeverity.MEDIUM,
                vulnerable_sources=["opencellid"],
                immune_sources=["osm-poi"],
                location=_point_geojson(center["lon"], center["lat"]),
                measured={"tower_count": len(cells), "institution_poi_count": institution_count},
                expected={"institution_poi_count_min": 2},
                likely_explanations=[
                    LikelyExplanation(cause="osm_incomplete", likelihood=0.4, note="POI baseline gap, not EW"),
                    LikelyExplanation(cause="jamming", likelihood=0.3, note="Handset-derived coverage suppressed"),
                    LikelyExplanation(cause="tower_outage", likelihood=0.3, note="Kinetic or cyber disruption of towers"),
                ],
            )
        ]


class PopulationInstitutionsRule:
    rule_id = "population-institutions"

    def evaluate(self, *, area, timeframe, records, sources, context) -> list[ConsistencyAnomaly]:
        stats_record = records.get("statistics-finland")
        osm_record = records.get("osm-poi")
        if not stats_record or not osm_record:
            return []

        municipalities = stats_record.data.get("municipalities", [])
        total_pop = sum(m.get("population", 0) or 0 for m in municipalities)
        urban_share = 0.0
        if municipalities:
            urban_count = sum(1 for m in municipalities if (m.get("urban_rural_class") or "").startswith("Urban"))
            urban_share = urban_count / len(municipalities)

        institution_count = sum(
            len(osm_record.data.get("categories", {}).get(category, []))
            for category in ("healthcare", "government", "education")
        )
        if total_pop < 3000 or institution_count >= 3:
            return []

        center = resolve_area_center(area)
        return [
            _new_anomaly(
                rule_id=self.rule_id,
                title="Population baseline vs institution POI density",
                description=(
                    f"Statistics Finland reports ~{total_pop:,} residents "
                    f"({urban_share:.0%} urban-classified municipalities) but only "
                    f"{institution_count} institution POIs in OSM immune layer."
                ),
                severity=AnomalySeverity.LOW,
                vulnerable_sources=[],
                immune_sources=["statistics-finland", "osm-poi"],
                location=_point_geojson(center["lon"], center["lat"]),
                measured={"population": total_pop, "institution_pois": institution_count},
                expected={"institution_pois_min": 3},
                likely_explanations=[
                    LikelyExplanation(cause="osm_incomplete", likelihood=0.7, note="Institution tagging incomplete"),
                    LikelyExplanation(cause="rural_dispersion", likelihood=0.3, note="Sparse settlement pattern"),
                ],
            )
        ]


class DigiroadWeatherRule:
    rule_id = "digiroad-weather"

    def evaluate(self, *, area, timeframe, records, sources, context) -> list[ConsistencyAnomaly]:
        digi_record = records.get("digiroad")
        fmi_record = records.get("fmi")
        if not digi_record or not fmi_record:
            return []

        collections = digi_record.data.get("collections", {})
        frost_features = 0
        frost_coll = collections.get("dr_kelirikko", {})
        frost_features = frost_coll.get("feature_count", 0) if isinstance(frost_coll, dict) else 0
        if frost_features == 0:
            return []

        temp = _latest_numeric(fmi_record.data.get("observations", {}), "temperature")
        if temp is None or temp > 2:
            return []

        center = resolve_area_center(area)
        return [
            _new_anomaly(
                rule_id=self.rule_id,
                title="Road frost zones vs above-freezing conditions",
                description=(
                    f"Digiroad reports {frost_features} frost-damage zone features while FMI "
                    f"temperature is {temp}°C. Road condition signaling may be stale."
                ),
                severity=AnomalySeverity.LOW,
                vulnerable_sources=["digiroad"],
                immune_sources=["fmi"],
                location=_point_geojson(center["lon"], center["lat"]),
                measured={"frost_zone_features": frost_features, "temperature_c": temp},
                expected={"temperature_c_max_for_active_frost": 2},
                likely_explanations=[
                    LikelyExplanation(cause="stale_road_data", likelihood=0.55, note="Seasonal layer not refreshed"),
                    LikelyExplanation(cause="sensor_lag", likelihood=0.25, note="Temperature warmed faster than road model"),
                    LikelyExplanation(cause="data_interference", likelihood=0.2, note="Verify live road sensors if available"),
                ],
            )
        ]


class MaritimeAisSarRule:
    """Cross-check GNSS-derived AIS against SAR-style vessel returns (demo-capable)."""

    rule_id = "ais-sar-vessel"

    def evaluate(self, *, area, timeframe, records, sources, context) -> list[ConsistencyAnomaly]:
        maritime = records.get("maritime-demo")
        if maritime:
            ais_vessels = maritime.data.get("ais_vessels", [])
            sar_returns = maritime.data.get("sar_returns", [])
            synthetic = False
        elif "archipelago" in normalize_area(area):
            fixture = ARCHIPELAGO_MARITIME_FIXTURE
            ais_vessels = fixture["ais_vessels"]
            sar_returns = fixture["sar_returns"]
            synthetic = True
        else:
            return []

        anomalies: list[ConsistencyAnomaly] = []
        matched_sar = set()

        for vessel in ais_vessels:
            ais_lat, ais_lon = vessel["lat"], vessel["lon"]
            best_km = None
            best_return = None
            for index, sar in enumerate(sar_returns):
                if index in matched_sar:
                    continue
                distance = _haversine_km(ais_lat, ais_lon, sar["lat"], sar["lon"])
                if best_km is None or distance < best_km:
                    best_km = distance
                    best_return = (index, sar, distance)

            if best_return is None:
                continue
            index, sar, distance_km = best_return
            if distance_km <= 0.5:
                matched_sar.add(index)
                continue

            anomalies.append(
                _new_anomaly(
                    rule_id=self.rule_id,
                    title=f"AIS vs SAR position mismatch: {vessel.get('name', vessel.get('mmsi'))}",
                    description=(
                        f"AIS reports {vessel.get('name')} at ({ais_lat:.3f}, {ais_lon:.3f}) "
                        f"but nearest SAR return is {distance_km:.1f} km away at ({sar['lat']:.3f}, {sar['lon']:.3f})."
                    ),
                    severity=AnomalySeverity.HIGH if distance_km > 2 else AnomalySeverity.MEDIUM,
                    vulnerable_sources=["maritime-demo"] if not synthetic else ["ais"],
                    immune_sources=["sar"],
                    location=_point_geojson(ais_lon, ais_lat),
                    measured={"ais_lat": ais_lat, "ais_lon": ais_lon, "distance_km": round(distance_km, 2)},
                    expected={"max_ais_sar_distance_km": 0.5},
                    likely_explanations=[
                        LikelyExplanation(cause="ais_spoofing", likelihood=0.35, note="Transponder position does not match radar return"),
                        LikelyExplanation(cause="gps_error", likelihood=0.25, note="Vessel GNSS degraded"),
                        LikelyExplanation(cause="ais_disabled", likelihood=0.2, note="Vessel operating dark; SAR sees different traffic"),
                        LikelyExplanation(cause="sar_ghost", likelihood=0.2, note="Clutter or secondary return misclassified"),
                    ],
                    synthetic_demo=synthetic,
                )
            )
            matched_sar.add(index)

        unmatched_sar = len(sar_returns) - len(matched_sar)
        ais_count = len(ais_vessels)
        if len(sar_returns) > ais_count + 1:
            center = resolve_area_center(area)
            anomalies.append(
                _new_anomaly(
                    rule_id=self.rule_id,
                    title="SAR vessel count exceeds AIS track count",
                    description=(
                        f"SAR shows {len(sar_returns)} vessel-sized returns vs {ais_count} AIS tracks "
                        f"({unmatched_sar} unmatched). Untracked traffic may be non-AIS or AIS-suppressed."
                    ),
                    severity=AnomalySeverity.MEDIUM,
                    vulnerable_sources=["ais"],
                    immune_sources=["sar"],
                    location=_point_geojson(center["lon"], center["lat"]),
                    measured={"sar_returns": len(sar_returns), "ais_tracks": ais_count},
                    expected={"count_delta_max": 1},
                    likely_explanations=[
                        LikelyExplanation(cause="ais_suppressed", likelihood=0.4, note="Grey-zone or jammed AIS"),
                        LikelyExplanation(cause="non_ais_traffic", likelihood=0.35, note="Small craft without transponder"),
                        LikelyExplanation(cause="clutter", likelihood=0.25, note="SAR false positives"),
                    ],
                    synthetic_demo=synthetic,
                )
            )
        return anomalies


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math

    radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


DEFAULT_RULES: list[ConsistencyRule] = [
    SourceGapRule(),
    DemoDataTrustRule(),
    FmiObservationForecastRule(),
    FmiNeighboringStationsRule(),
    CellTowerInfrastructureRule(),
    PopulationInstitutionsRule(),
    DigiroadWeatherRule(),
    MaritimeAisSarRule(),
]
