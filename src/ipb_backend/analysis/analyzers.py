from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from typing import Any

import httpx

from ipb_backend.config import settings
from ipb_backend.llm.contracts import AnalysisProfile, AnalysisStatement, LimitationStatement, LlmAnalysisOutput
from ipb_backend.llm.profiles import PROFILE_SPECS


class EvidenceAnalyzer(ABC):
    provider_name: str

    async def health(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "status": "ready",
        }

    @abstractmethod
    async def analyze(
        self,
        *,
        data_package: dict[str, Any] | None = None,
        llm_input: dict[str, Any] | None = None,
        profile: str | AnalysisProfile = AnalysisProfile.GENERAL,
        question: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        selection: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        raw_data: dict[str, Any] | None = None,
        freshness: list[dict[str, Any]] | None = None,
        evidence_bundle: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        pass


def _coerce_profile(profile: str | AnalysisProfile) -> AnalysisProfile:
    if isinstance(profile, AnalysisProfile):
        return profile
    try:
        return AnalysisProfile(str(profile))
    except ValueError:
        return AnalysisProfile.GENERAL


def _coerce_data_package(
    *,
    data_package: dict[str, Any] | None,
    selection: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    freshness: list[dict[str, Any]] | None,
    evidence_bundle: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if data_package is not None:
        return data_package

    return {
        "selection": selection or {},
        "counts": {
            "by_source": dict((metrics or {}).get("feature_counts_by_source", {})),
            "by_category": dict((metrics or {}).get("feature_counts_by_category", {})),
            "geometry_types": dict((metrics or {}).get("geometry_counts", {})),
        },
        "derived_indicators": [
            {"indicator_id": "selection_area_sqkm", "value": (metrics or {}).get("selection_area_sqkm", 0)},
            {"indicator_id": "population_total", "value": (metrics or {}).get("population_total", 0)},
            {"indicator_id": "weather_station_count", "value": (metrics or {}).get("weather_station_count", 0)},
        ],
        "source_freshness": freshness or [],
        "evidence_items": [
            {"evidence_id": f"ev-{item.get('source_id', 'unknown')}-{i:03d}", **item}
            for i, item in enumerate(evidence_bundle or [], start=1)
        ],
        "quality": {
            "fallback_sources": [],
            "error_sources": [item["source_id"] for item in freshness or [] if item.get("status") == "error"],
            "coverage_gaps": [],
            "overall_confidence": "medium",
        },
    }


def _indicator_value(data_package: dict[str, Any], indicator_id: str, default: Any = 0) -> Any:
    for item in data_package.get("derived_indicators", []):
        if item.get("indicator_id") == indicator_id:
            return item.get("value", default)
    return default


def _evidence_refs(data_package: dict[str, Any], limit: int = 2) -> list[str]:
    return [item.get("evidence_id", "") for item in data_package.get("evidence_items", [])[:limit] if item.get("evidence_id")]


def _coerce_llm_input(
    *,
    llm_input: dict[str, Any] | None,
    data_package: dict[str, Any],
    profile: AnalysisProfile,
    question: str | None,
    conversation_history: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if llm_input is not None:
        return llm_input

    return {
        "schema_version": "1.0",
        "package_id": data_package.get("package_id", "legacy"),
        "profile": profile.value,
        "profile_focus": PROFILE_SPECS.get(profile, PROFILE_SPECS[AnalysisProfile.GENERAL])["focus"],
        "question": question,
        "selection": data_package.get("selection", {}),
        "counts": data_package.get("counts", {}),
        "source_digests": data_package.get("source_summaries", []),
        "indicator_digests": data_package.get("derived_indicators", []),
        "evidence_catalog": data_package.get("evidence_items", []),
        "guardrails": data_package.get("quality", {}),
        "conversation_history": conversation_history or [],
    }


def _build_structured_output(
    *,
    profile: AnalysisProfile,
    summary: str,
    findings: list[str],
    limitations: list[str],
    provider: str,
    status: str,
    evidence_refs: list[str],
    quality: dict[str, Any] | None = None,
    model: str | None = None,
) -> LlmAnalysisOutput:
    implications = [
        AnalysisStatement(
            id=f"imp-{i}",
            topic=profile.value,
            assessment="grounded",
            statement=finding,
            evidence_refs=evidence_refs,
            confidence=(quality or {}).get("overall_confidence", "medium"),
        )
        for i, finding in enumerate(findings[:6], start=1)
    ]

    defensive_relevance = [
        AnalysisStatement(
            id="def-1",
            topic=PROFILE_SPECS.get(profile, PROFILE_SPECS[AnalysisProfile.GENERAL])["label"],
            assessment="bounded",
            statement=summary,
            evidence_refs=evidence_refs,
            confidence=(quality or {}).get("overall_confidence", "medium"),
        )
    ]

    limitation_items = [
        LimitationStatement(
            id=f"lim-{i}",
            statement=item,
            evidence_refs=evidence_refs[:1],
            severity="high" if "fallback" in item.lower() else "medium",
        )
        for i, item in enumerate(limitations[:4], start=1)
    ]

    metadata: dict[str, Any] = {"provider": provider, "evidence_ref_count": len(evidence_refs)}
    if model:
        metadata["model"] = model

    return LlmAnalysisOutput(
        profile=profile,
        status=status,
        summary=summary,
        implications=implications,
        defensive_relevance=defensive_relevance,
        limitations=limitation_items,
        uncertainties=[],
        metadata=metadata,
    )


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return cleaned


def _normalize_lines(value: Any) -> list[str]:
    if isinstance(value, str):
        return [line.strip("- *\t ") for line in value.splitlines() if line.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


INTSUM_SECTION_KEYS = [
    "situation_overview",
    "terrain_observation", "terrain_approach", "terrain_key", "terrain_obstacles", "terrain_cover",
    "weather_impact", "infrastructure", "civil_considerations",
    "ccir_answers", "assessment", "limitations",
]


def _parse_intsum_response(text: str) -> dict[str, str]:
    cleaned = _strip_code_fence(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback — return the whole thing as the assessment
        return {key: "" for key in INTSUM_SECTION_KEYS} | {"assessment": cleaned[:2000]}
    if not isinstance(payload, dict):
        return {key: "" for key in INTSUM_SECTION_KEYS} | {"assessment": str(payload)[:2000]}
    return {key: str(payload.get(key) or "").strip() for key in INTSUM_SECTION_KEYS}


def _parse_llm_response(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fence(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        lines = _normalize_lines(cleaned)
        return {"summary": lines[0] if lines else cleaned, "findings": lines, "limitations": []}

    if not isinstance(payload, dict):
        lines = _normalize_lines(cleaned)
        return {"summary": lines[0] if lines else cleaned, "findings": lines, "limitations": []}

    findings = _normalize_lines(payload.get("findings"))
    summary = str(payload.get("summary") or (findings[0] if findings else "")).strip()
    limitations = _normalize_lines(payload.get("limitations"))
    return {"summary": summary, "findings": findings, "limitations": limitations}


# ─── NLS collection → military significance ───────────────────────────────────

_NLS_TERRAIN_MILITARY: dict[str, tuple[str, str]] = {
    "metsamaankasvillisuus": ("Forest/woodland cover", "concealment & cover, limits observation, potential obstacle for vehicles"),
    "jarvi":                 ("Lakes",                 "significant water obstacle, limits cross-country movement"),
    "meri":                  ("Sea/coastal water",     "hard boundary, potential amphibious approach"),
    "virtavesialue":         ("Rivers/streams",        "water obstacle, requires crossing assets or bridges"),
    "suo":                   ("Bogs/marshes",          "severe mobility constraint, impassable for heavy vehicles off-road"),
    "kallioalue":            ("Rocky terrain",         "cover & concealment, limits vehicle movement, good defensive ground"),
    "tieviiva":              ("Road network",          "primary avenues of approach, logistics corridors"),
    "rautatie":              ("Railway",               "logistics route, infrastructure asset"),
    "rakennus":              ("Buildings",             "urban terrain, cover, potential strongpoints"),
    "taajaanrakennettualue": ("Built-up area",         "urban terrain, civilian presence, CIVCAS risk"),
    "korkeuskayra":          ("Elevation contours",    "relief present — assess for observation posts and defensive positions"),
    "sahkolinja":            ("Power lines",           "critical infrastructure, affects helicopter routing"),
    "maatalousmaa":          ("Agricultural/open land","exposed terrain, good fields of fire, limited concealment"),
    "luonnonsuojelualue":    ("Protected area",        "may limit operations, terrain type varies"),
    "lentokenttaalue":       ("Airfield/airport",      "key terrain for air operations and logistics"),
    "satamaalue":            ("Harbor/port",           "key terrain for maritime logistics"),
}


def _build_raw_data_from_package(data_package: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct raw_data from data_package source_summaries for follow-up chat calls."""
    return {
        s.get("source_id", ""): s.get("raw_summary", {})
        for s in data_package.get("source_summaries", [])
        if s.get("source_id")
    }


def _ts_stats(param_data: dict[str, Any]) -> dict[str, Any]:
    """Return min/max/mean/trend from a parameter's hourly values list."""
    values = [
        pt["value"] for pt in param_data.get("values", [])
        if pt.get("value") is not None
    ]
    if not values:
        return {}
    result: dict[str, Any] = {
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "mean": round(sum(values) / len(values), 1),
    }
    if len(values) >= 6:
        n = max(1, len(values) // 3)
        first_mean = sum(values[:n]) / n
        last_mean = sum(values[-n:]) / n
        diff = last_mean - first_mean
        result["trend"] = "rising" if diff > 2 else "falling" if diff < -2 else "stable"
    return result


def _format_data_for_prompt(raw_data: dict[str, Any], data_package: dict[str, Any]) -> str:
    """Build a compact, military-relevant text description of all available data."""
    lines: list[str] = []
    sel = data_package.get("selection") or {}
    area_sqkm = float(sel.get("area_sqkm") or 0)
    bounds = sel.get("bounds_wgs84", [])

    # Header: AOI summary
    if bounds and len(bounds) == 4:
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2
        extent_km_lat = (bounds[3] - bounds[1]) * 111
        extent_km_lon = (bounds[2] - bounds[0]) * 111 * math.cos(math.radians(center_lat))
        lines.append(
            f"AOI CENTER: ({center_lat:.4f}°N, {center_lon:.4f}°E) — "
            f"extent ~{extent_km_lon:.1f} km E-W × {extent_km_lat:.1f} km N-S"
        )

    timeframe = (data_package.get("scope") or {}).get("timeframe") or ""
    if timeframe:
        lines.append(f"TIMEFRAME: {timeframe}")

    quality = data_package.get("quality", {}) or {}
    conf = quality.get("overall_confidence", "")
    if conf:
        gaps = quality.get("coverage_gaps") or []
        fallbacks = quality.get("fallback_sources") or []
        confidence_note = f"DATA CONFIDENCE: {conf}"
        if fallbacks:
            confidence_note += f" — fallback sources: {', '.join(fallbacks)}"
        if gaps:
            confidence_note += f" — gaps: {'; '.join(gaps[:3])}"
        lines.append(confidence_note)

    # NLS terrain — with % composition
    nls = raw_data.get("nls", {})
    if nls.get("feature_count") or nls.get("collections"):
        cols = nls.get("collections", [])
        total_nls = sum(c.get("count", 0) for c in cols) or 1
        if cols:
            terrain_lines = []
            for col in cols:
                label = col.get("label", col.get("collection", ""))
                count = col.get("count", 0)
                cid = col.get("collection", "")
                mil = _NLS_TERRAIN_MILITARY.get(cid)
                pct = (count / total_nls * 100) if total_nls else 0
                pct_str = f", {pct:.0f}% of terrain features" if pct > 5 else ""
                if mil and count:
                    terrain_lines.append(f"  - {mil[0]} ({count} features{pct_str}): {mil[1]}")
                elif count:
                    terrain_lines.append(f"  - {label} ({count} features{pct_str})")
            if terrain_lines:
                lines.append(f"NLS TERRAIN DATA ({nls.get('feature_count', total_nls)} total features):")
                lines.extend(terrain_lines)

    # Digiroad roads/bridges — show ALL collections (bridges/tunnels/ferries are critical for IPB)
    digiroad = raw_data.get("digiroad", {})
    if digiroad.get("feature_count"):
        lines.append(f"ROAD INFRASTRUCTURE: {digiroad['feature_count']} Digiroad features")
        if digiroad.get("collections"):
            for col in digiroad["collections"]:
                cid = col.get("collection", "")
                label = col.get("label", cid)
                count = col.get("count", 0)
                note = ""
                if "silta" in cid or "bridge" in cid.lower():
                    note = " — bridge crossings (weight/clearance limits constrain heavy vehicles)"
                elif "tunneli" in cid or "tunnel" in cid.lower():
                    note = " — tunnel chokepoints (mobility constraint, defensive position)"
                elif "lautta" in cid or "ferry" in cid.lower():
                    note = " — ferry routes (subject to weather/seasonal closure)"
                lines.append(f"  - {label}: {count}{note}")

    # ── Weather ──────────────────────────────────────────────────────────────
    fmi = raw_data.get("fmi", {})
    obs = fmi.get("observations", {})
    fmi_query = fmi.get("query", {})
    if obs:
        def _val(name: str) -> Any:
            return (obs.get(name) or {}).get("latest", {}).get("value")

        temp = _val("temperature")
        wind = _val("wind_speed")
        gust = _val("wind_gust")
        precip = _val("precipitation")
        cloud = _val("cloud_cover")

        # Window header
        win_start = fmi_query.get("start_time", "")
        win_end = fmi_query.get("end_time", "")
        win_label = f" (obs window {win_start} → {win_end})" if win_start else ""

        parts = []
        if temp is not None:
            parts.append(f"temp {temp}°C")
        if wind is not None:
            parts.append(f"wind {wind} m/s" + (f" (gusts {gust} m/s)" if gust else ""))
        if precip is not None:
            parts.append(f"precip {precip} mm/h")
        if cloud is not None:
            parts.append(f"cloud {cloud}%")

        if parts:
            lines.append(f"WEATHER{win_label}:")
            lines.append(f"  CURRENT (latest obs): {', '.join(parts)}")

        # Time-series trends over the observation window
        trend_lines: list[str] = []
        temp_stats = _ts_stats(obs.get("temperature", {}))
        if temp_stats:
            trend = f" / trend: {temp_stats['trend']}" if "trend" in temp_stats else ""
            trend_lines.append(f"    Temperature: {temp_stats['min']}–{temp_stats['max']}°C (avg {temp_stats['mean']}°C{trend})")
        wind_stats = _ts_stats(obs.get("wind_speed", {}))
        gust_stats = _ts_stats(obs.get("wind_gust", {}))
        if wind_stats:
            gust_note = f", gusts up to {gust_stats.get('max', '')} m/s" if gust_stats else ""
            trend_lines.append(f"    Wind speed: avg {wind_stats['mean']} m/s, peak {wind_stats['max']} m/s{gust_note}")
        precip_vals = [pt["value"] for pt in obs.get("precipitation", {}).get("values", []) if pt.get("value") is not None]
        if precip_vals:
            total_precip = round(sum(precip_vals), 1)
            trend_lines.append(f"    Precipitation: {total_precip} mm total over window")
        cloud_stats = _ts_stats(obs.get("cloud_cover", {}))
        if cloud_stats:
            trend_lines.append(f"    Cloud cover: avg {cloud_stats['mean']}% (min {cloud_stats['min']}%, max {cloud_stats['max']}%)")
        if trend_lines:
            lines.append("  OBSERVED RANGE:")
            lines.extend(trend_lines)

        # Forecast (horizon set by the user's timeframe, capped by FMI HARMONIE)
        forecast = fmi.get("forecast", {})
        fc_obs = forecast.get("observations", {})
        if fc_obs:
            fc_parts: list[str] = []
            fc_temp = _ts_stats(fc_obs.get("temperature", {}))
            if fc_temp:
                fc_parts.append(f"temp {fc_temp['min']}–{fc_temp['max']}°C")
            fc_wind = _ts_stats(fc_obs.get("wind_speed", {}))
            if fc_wind:
                fc_parts.append(f"wind avg {fc_wind['mean']} m/s (max {fc_wind['max']} m/s)")
            fc_precip_vals = [pt["value"] for pt in fc_obs.get("precipitation", {}).get("values", []) if pt.get("value") is not None]
            if fc_precip_vals:
                fc_parts.append(f"precip {round(sum(fc_precip_vals), 1)} mm total")
            fc_cloud = _ts_stats(fc_obs.get("cloud_cover", {}))
            if fc_cloud:
                fc_parts.append(f"cloud avg {fc_cloud['mean']}%")
            if fc_parts:
                horizon = fmi_query.get("forecast_hours") or len(
                    fc_obs.get("temperature", {}).get("values", [])
                ) or 48
                lines.append(f"  FORECAST (next {int(horizon)}h): {', '.join(fc_parts)}")

        # Consolidated operational notes across observed + forecast worst-case
        peak_wind = max(
            wind_stats.get("max", 0) if wind_stats else 0,
            _ts_stats(fc_obs.get("wind_speed", {})).get("max", 0) if fc_obs else 0,
        )
        peak_precip = max(
            max((pt["value"] for pt in obs.get("precipitation", {}).get("values", []) if pt.get("value") is not None), default=0),
            max((pt["value"] for pt in (fc_obs or {}).get("precipitation", {}).get("values", []) if pt.get("value") is not None), default=0),
        )
        avg_cloud = cloud_stats.get("mean", 0) if cloud_stats else 0
        min_temp = min(
            temp_stats.get("min", 99) if temp_stats else 99,
            _ts_stats(fc_obs.get("temperature", {})).get("min", 99) if fc_obs else 99,
        )
        op_notes: list[str] = []
        if peak_wind > 10:
            op_notes.append(f"  - Peak wind {peak_wind} m/s: drone operations severely restricted during high-wind periods")
        elif peak_wind > 7:
            op_notes.append(f"  - Peak wind {peak_wind} m/s: small drone operations degraded during high-wind periods")
        if peak_precip > 2:
            op_notes.append(f"  - Precipitation peak {peak_precip} mm/h: mobility degraded on unpaved routes, drone ops restricted")
        if avg_cloud > 80:
            op_notes.append(f"  - Cloud cover avg {avg_cloud}%: optical aerial/satellite observation severely degraded throughout")
        elif avg_cloud > 50:
            op_notes.append(f"  - Cloud cover avg {avg_cloud}%: intermittent optical observation windows only")
        if min_temp < -5:
            op_notes.append(f"  - Min temperature {min_temp}°C: cold-weather equipment precautions required")
        if op_notes:
            lines.append("  OPERATIONAL WEATHER NOTES:")
            lines.extend(op_notes)

    # ── Satellite overpass schedule (AOI-scoped via swath intersection) ──────
    sat_data = raw_data.get("satellites", {})
    sats_with_passes = sat_data.get("satellites_with_passes", []) if isinstance(sat_data, dict) else []
    if sats_with_passes:
        win = sat_data.get("window", {}) or {}
        hrs = win.get("hours")
        win_label = f" (next {int(hrs)}h over AOI)" if hrs else ""
        total = sat_data.get("total_passes_in_window", 0)
        tracked = sat_data.get("satellites_total_tracked", 0)
        lines.append(
            f"SATELLITE OVERPASSES{win_label}: {total} passes from "
            f"{len(sats_with_passes)} of {tracked} tracked satellites image the AOI"
        )
        for sat in sats_with_passes[:12]:
            passes = sat.get("passes", [])
            if not passes:
                continue
            first = passes[0]
            t_label = first.get("start_utc", "")[:16].replace("T", " ")
            sensor = "SAR" if sat.get("is_sar") else "optical"
            origin = " [RUSSIAN]" if sat.get("origin") == "russian" else ""
            lines.append(
                f"  - {sat['name']} ({sat.get('type','?')}, {sensor}){origin}: "
                f"{len(passes)} passes, next {t_label}Z, closest {first.get('closest_km','?')} km"
            )
        # Warn if cloud cover will degrade optical passes
        cloud_stats_for_warn = _ts_stats(obs.get("cloud_cover", {})) if obs else {}
        if cloud_stats_for_warn.get("mean", 0) > 60:
            lines.append("  NOTE: High cloud cover forecast — optical passes likely obscured; SAR passes unaffected")
    elif isinstance(sat_data, dict) and sat_data.get("satellites_total_tracked"):
        # Tracked but none currently image the AOI in this horizon
        lines.append(
            f"SATELLITE OVERPASSES: No imaging passes over AOI in the selected window "
            f"({sat_data.get('satellites_total_tracked')} satellites tracked globally)"
        )

    # Population — with density
    pop = raw_data.get("statistics-finland", {})
    if pop.get("population_total"):
        total = pop["population_total"]
        density = total / area_sqkm if area_sqkm else 0
        if total < 100:
            civil_note = "minimal civilian presence — low CIVCAS risk"
        elif total < 1000:
            civil_note = "moderate civilian presence — CIVCAS precautions required"
        else:
            civil_note = "significant civilian population — high CIVCAS risk, urban ROE apply"
        density_descriptor = ""
        if density > 500:
            density_descriptor = " (urban density)"
        elif density > 50:
            density_descriptor = " (suburban/village density)"
        elif density > 5:
            density_descriptor = " (sparse rural)"
        else:
            density_descriptor = " (very sparse / wilderness)"
        lines.append(
            f"POPULATION: {total} residents in AOI — {density:.1f} persons/km²{density_descriptor}. {civil_note}"
        )
        coverage = pop.get("population_coverage_ratio")
        if coverage:
            lines.append(f"  - Coverage: {coverage*100:.1f}% of source population dataset intersects AOI")

    # Cell towers / comms — break down by radio technology and operator
    cell = raw_data.get("opencellid", {})
    cell_count = cell.get("feature_count") or 0
    if cell_count:
        radio_counts: dict[str, int] = {}
        operator_counts: dict[str, int] = {}
        for feat in cell.get("features", []) or []:
            props = feat.get("properties") or {}
            radio = (props.get("radio") or "").upper() or "UNKNOWN"
            radio_counts[radio] = radio_counts.get(radio, 0) + 1
            mcc = props.get("mcc")
            mnc = props.get("mnc")
            if mcc and mnc is not None:
                key = f"MCC {mcc}/MNC {mnc}"
                operator_counts[key] = operator_counts.get(key, 0) + 1
        lines.append(f"COMMUNICATIONS: {cell_count} cell towers — civilian comms baseline present")
        if radio_counts:
            mix = ", ".join(f"{r}: {n}" for r, n in sorted(radio_counts.items(), key=lambda x: -x[1]))
            lines.append(f"  - Technology mix: {mix}")
        if operator_counts:
            top_ops = sorted(operator_counts.items(), key=lambda x: -x[1])[:3]
            ops_str = ", ".join(f"{op} ({n})" for op, n in top_ops)
            lines.append(f"  - Top operator codes: {ops_str}")
        density_cell = cell_count / area_sqkm if area_sqkm else 0
        if density_cell > 2:
            lines.append(f"  - Density {density_cell:.1f}/km²: dense coverage, strong SIGINT/EW environment")
        elif density_cell > 0.5:
            lines.append(f"  - Density {density_cell:.1f}/km²: moderate coverage")
        else:
            lines.append(f"  - Density {density_cell:.2f}/km²: sparse coverage, mil comms must be self-sufficient")
    elif cell.get("feature_count") == 0:
        lines.append("COMMUNICATIONS: No cell towers detected — poor civilian comms coverage, degraded SIGINT environment")

    # OSM POIs
    osm = raw_data.get("osm-poi", {})
    if osm.get("feature_count"):
        poi_cols = osm.get("collections", [])
        col_counts = {c.get("collection", ""): c.get("count", 0) for c in poi_cols}
        lines.append(f"POINTS OF INTEREST: {osm['feature_count']} features across {len(poi_cols)} categories")

        # High-value categories with counts
        hv = ["military", "airfields", "power_infrastructure", "logistics", "ports_terminals",
              "fuel_supply", "healthcare", "emergency_services", "government", "industry"]
        for cat in hv:
            if col_counts.get(cat):
                lines.append(f"  - {cat.replace('_', ' ').title()}: {col_counts[cat]}")

        # Extract named features for tactically significant categories
        features_list = osm.get("features", [])
        named_by_cat: dict[str, list[str]] = {}
        for feat in features_list:
            props = feat.get("properties", {})
            cat = props.get("_collection", "")
            name = props.get("name", "")
            if cat in ("military", "airfields", "logistics", "ports_terminals", "power_infrastructure") and name:
                named_by_cat.setdefault(cat, []).append(name)
        for cat, names in named_by_cat.items():
            unique = list(dict.fromkeys(names))[:5]
            lines.append(f"  Named {cat}: {'; '.join(unique)}")

    # Derived indicators from package
    for ind in data_package.get("derived_indicators", []):
        if "feature_density" in ind.get("indicator_id", ""):
            sid = ind["indicator_id"].replace("_feature_density", "")
            val = ind.get("value", 0)
            if val and sid not in ("nls", "digiroad", "statistics-finland", "fmi", "opencellid", "osm-poi", "satellites"):
                lines.append(f"  {sid}: {val:.2f} features/km²")

    if not lines:
        area = (data_package.get("selection") or {}).get("area_sqkm", 0)
        lines.append(f"Limited data intersects this {area:.1f} km² AOI. Analysis based on available indicators only.")

    return "\n".join(lines)


# ─── ClaudeAnalyzer ────────────────────────────────────────────────────────────

class ClaudeAnalyzer(EvidenceAnalyzer):
    provider_name = "claude"

    async def health(self) -> dict[str, Any]:
        if not settings.anthropic_api_key:
            return {"provider": self.provider_name, "status": "error", "error": "ANTHROPIC_API_KEY not set"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://api.anthropic.com/v1/models",
                    headers={"x-api-key": settings.anthropic_api_key, "anthropic-version": "2023-06-01"},
                )
                resp.raise_for_status()
            return {"provider": self.provider_name, "status": "ready", "model": settings.anthropic_model}
        except Exception as exc:
            return {"provider": self.provider_name, "status": "error", "error": str(exc)}

    async def analyze(
        self,
        *,
        data_package: dict[str, Any] | None = None,
        llm_input: dict[str, Any] | None = None,
        profile: str | AnalysisProfile = AnalysisProfile.GENERAL,
        question: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        selection: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        raw_data: dict[str, Any] | None = None,
        freshness: list[dict[str, Any]] | None = None,
        evidence_bundle: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        profile_value = _coerce_profile(profile)
        package = _coerce_data_package(
            data_package=data_package,
            selection=selection,
            metrics=metrics,
            freshness=freshness,
            evidence_bundle=evidence_bundle,
        )
        llm_input_payload = _coerce_llm_input(
            llm_input=llm_input,
            data_package=package,
            profile=profile_value,
            question=question,
            conversation_history=conversation_history,
        )

        # Reconstruct raw_data from package summaries when not provided (follow-up chat)
        if not raw_data:
            raw_data = _build_raw_data_from_package(package)

        data_text = _format_data_for_prompt(raw_data, package)
        sel = package.get("selection") or {}
        area_sqkm = sel.get("area_sqkm", 0)
        bounds = sel.get("bounds_wgs84", [])
        profile_focus = PROFILE_SPECS.get(profile_value, PROFILE_SPECS[AnalysisProfile.GENERAL])["focus"]
        schematic_brief = str(llm_input_payload.get("schematic_brief") or "").strip()

        system_prompt = (
            "You are a senior intelligence analyst specializing in Finnish military geography. "
            "You draw on open-source data — NLS topography, Digiroad road network, FMI weather (observed + forecast), "
            "satellite overpass schedules (including Russian reconnaissance), Statistics Finland population, "
            "OpenCellID cell tower data, and OpenStreetMap POIs — to produce intelligence estimates "
            "that read like a real intel product, not a data inventory.\n\n"
            "Your discipline:\n"
            "• Synthesize across sources. The dots only matter once connected. A forested lake district means "
            "something different from open agricultural terrain bisected by roads. A military installation near "
            "a power substation and a fuel depot is a different picture than any of those alone.\n"
            "• Look for asymmetries and tensions. What's surprising or unusual about this area? "
            "Where do data points contradict or amplify each other? What's the dominant character — and what cuts against it?\n"
            "• Think from the adversary's seat. What would they target? Where would they hide? Where are the chokepoints, "
            "the soft logistics tails, the overlooked seams between sectors?\n"
            "• Translate data to operational consequence. 'Wind 11 m/s' is not an insight — 'the forecast window when "
            "small UAS can fly closes around 1400Z' is. 'N cell towers' is not an insight — 'comms coverage degrades "
            "sharply once forces move north of the river' is.\n"
            "• Use OAKOC, PMESII-PT, and CCIR thinking as analytical lenses, not checklists or section headers.\n"
            "• Never invent data not in the package. If something matters but isn't measured, flag it as a gap."
        )

        context_header = (
            f"AOI: {area_sqkm:.1f} km²"
            + (f" | Bounds: {', '.join(f'{b:.3f}' for b in bounds)}" if bounds else "")
            + f"\nAnalysis profile: {profile_value.value} — {profile_focus}"
        )
        brief_context = (
            context_header
            + "\n\nSCHEMATIC DATA BRIEF:\n"
            + (schematic_brief or "No wrapper brief available.")
        )
        detailed_context = brief_context + "\n\nDETAILED DATA ANNEX:\n" + data_text

        fresh_inspection_ask = (
            "\n\nProduce an initial intelligence brief for this area. Start from the schematic data brief so the reader can "
            "see the shape of the dataset before you interpret it. Then shift into analysis. Write it the way a senior "
            "analyst would brief a commander — substance over structure, insight over inventory.\n\n"
            "Return JSON with these keys:\n"
            "- summary: Open with a concise schematic brief of the data picture: AOI/timeframe, major source coverage, strongest "
            "signals, and key gaps/confidence. After that, characterize the area's operational profile in flowing prose. Length "
            "should fit the substance. Do not mechanically enumerate every source.\n"
            "- findings: Array of analytical insights. Each should be a complete sentence or short paragraph that draws an "
            "operationally relevant conclusion — ideally connecting two or more data points. Include as many as the data "
            "supports — don't pad, don't truncate. Prefer 'The concentration of X near Y suggests...' over 'There are N "
            "features of type Z.' Surface what's non-obvious.\n"
            "- tactical_commentary: A free-form analyst-voice paragraph (or two). Use this for the things that don't fit "
            "neatly into findings: hunches, second-order effects, what would worry you, what an adversary might exploit, "
            "what you'd want reconnaissance to confirm. Be direct.\n"
            "- limitations: Array of genuine data gaps and uncertainties affecting the assessment."
        )

        # Build message list — fresh inspection vs follow-up paths
        messages: list[dict[str, Any]] = []
        history = conversation_history or []
        is_follow_up = bool(history) and any(t.get("role") == "user" for t in history)

        if not is_follow_up:
            # Fresh inspection — structured JSON output
            user_content = detailed_context + fresh_inspection_ask
            if question:
                user_content += f"\n\nANALYST QUESTION: {question}"
            messages = [{"role": "user", "content": user_content}]
        else:
            # Follow-up chat — natural prose, no JSON forced
            messages.append({
                "role": "user",
                "content": (
                    brief_context
                    + "\n\nUse this as stable reference context. Answer follow-up questions conversationally and directly. "
                    "Do not force JSON, fixed sections, or a full restatement of the brief unless the user asks for it."
                ),
            })
            messages.append({
                "role": "assistant",
                "content": "Reference context loaded. I will answer follow-up questions directly and only pull in the relevant evidence."
            })
            for turn in history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})
            if question:
                messages.append({"role": "user", "content": question})

        try:
            text = await self._call_api(system_prompt, messages, max_tokens=3000)
        except Exception as exc:
            raise ValueError(f"Claude API call failed: {exc}") from exc

        if is_follow_up:
            # Free-prose conversation turn — return as summary, no findings/limitations parsing
            findings: list[str] = []
            summary = text.strip()
            limitations: list[str] = []
        else:
            parsed = _parse_llm_response(text)
            findings = parsed["findings"]
            summary = parsed["summary"] or (findings[0] if findings else text[:200])
            limitations = parsed["limitations"]
            # Tactical commentary, if present, is appended to summary as a separate paragraph
            try:
                payload = json.loads(_strip_code_fence(text))
                if isinstance(payload, dict) and payload.get("tactical_commentary"):
                    summary = f"{summary}\n\n— {str(payload['tactical_commentary']).strip()}"
            except (json.JSONDecodeError, TypeError):
                pass

        evidence_refs = _evidence_refs(package)
        structured_output = _build_structured_output(
            profile=profile_value,
            summary=summary,
            findings=findings,
            limitations=limitations,
            provider=self.provider_name,
            status="ready",
            evidence_refs=evidence_refs,
            quality=package.get("quality"),
            model=settings.anthropic_model,
        )

        return {
            "provider": self.provider_name,
            "status": "ready",
            "profile": profile_value.value,
            "question": question,
            "summary": summary,
            "findings": findings,
            "limitations": limitations,
            "model": settings.anthropic_model,
            "evidence_bundle": evidence_bundle or package.get("evidence_items", []),
            "llm_input": llm_input_payload,
            "output": structured_output.model_dump(mode="json"),
        }

    async def generate_intsum(
        self,
        *,
        data_package: dict[str, Any],
        raw_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a structured INTSUM (Intelligence Summary) for the AOI."""
        if not raw_data:
            raw_data = _build_raw_data_from_package(data_package)
        data_text = _format_data_for_prompt(raw_data, data_package)
        sel = data_package.get("selection") or {}
        area_sqkm = sel.get("area_sqkm", 0)
        label = sel.get("label") or "Unnamed AOI"
        bounds = sel.get("bounds_wgs84", [])

        system_prompt = (
            "You are a NATO intelligence officer drafting a formal INTSUM (Intelligence Summary). "
            "Write in clean military prose — concise, declarative sentences, present tense. "
            "Use OAKOC for terrain. Use PMESII-PT considerations for civil factors where relevant. "
            "Cite specific quantities and named features from the data. Never invent facts. "
            "Each section should stand on its own as professional intelligence product text — no headings, no bullets "
            "unless explicitly listed in the schema. Write paragraphs. Be specific, not generic."
        )

        user_prompt = (
            f"AREA OF INTEREST: {label}, {area_sqkm:.1f} km²"
            + (f"\nBOUNDS (WGS84 W,S,E,N): {', '.join(f'{b:.4f}' for b in bounds)}" if bounds else "")
            + "\n\nINTELLIGENCE INPUTS:\n"
            + data_text
            + "\n\nProduce an INTSUM. Return JSON with these keys (each value is a string of flowing prose, "
            "2-5 sentences per section unless noted):\n\n"
            "- situation_overview: Single-paragraph characterization of the AOI's overall operational profile.\n"
            "- terrain_observation: Observation & fields of fire — what can be seen, where lines of sight open or close.\n"
            "- terrain_approach: Avenues of approach — how forces could move into/through this area.\n"
            "- terrain_key: Key terrain — features whose seizure/retention gives marked advantage.\n"
            "- terrain_obstacles: Natural and man-made obstacles to movement.\n"
            "- terrain_cover: Cover and concealment — where forces can hide from observation and fire.\n"
            "- weather_impact: Current weather and its operational impact on UAS, mobility, optics, personnel.\n"
            "- infrastructure: Critical infrastructure assessment — power, fuel, transport, communications, logistics nodes. "
            "Name specific facilities where data provides names.\n"
            "- civil_considerations: Population, civilian density, CIVCAS risk, ROE implications.\n"
            "- ccir_answers: Answers to four standard CCIRs as a single string with each on a new line. "
            "Format each as 'CCIR-N: [question] — [answer]'. Use:\n"
            "  CCIR-1: Are enemy forces present or capable of operating in this AOI?\n"
            "  CCIR-2: What infrastructure could be commandeered, denied, or destroyed?\n"
            "  CCIR-3: What restrictions does terrain and weather impose on friendly operations?\n"
            "  CCIR-4: What civilian considerations constrain ROE?\n"
            "- assessment: Analyst's synthesized conclusion — 3-5 sentences answering 'so what?' for a commander.\n"
            "- limitations: Genuine intelligence gaps and data limitations affecting this assessment."
        )

        try:
            text = await self._call_api(system_prompt, [{"role": "user", "content": user_prompt}], max_tokens=3500)
        except Exception as exc:
            raise ValueError(f"Claude INTSUM call failed: {exc}") from exc

        sections = _parse_intsum_response(text)
        return {
            "provider": self.provider_name,
            "status": "ready",
            "model": settings.anthropic_model,
            "sections": sections,
        }

    async def _call_api(self, system: str, messages: list[dict[str, Any]], max_tokens: int = 2000) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": settings.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": settings.anthropic_model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": messages,
                },
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]


# ─── INTSUM fallback ───────────────────────────────────────────────────────────

def _rules_intsum_sections(data_package: dict[str, Any], raw_data: dict[str, Any]) -> dict[str, str]:
    """Deterministic INTSUM section generator used when Claude is unavailable."""
    sel = data_package.get("selection") or {}
    area_sqkm = float(sel.get("area_sqkm") or 0)
    nls = raw_data.get("nls", {})
    col_map = {c.get("collection", ""): c.get("count", 0) for c in nls.get("collections", [])}
    osm = raw_data.get("osm-poi", {})
    osm_col = {c.get("collection", ""): c.get("count", 0) for c in osm.get("collections", [])}
    pop = (raw_data.get("statistics-finland", {}) or {}).get("population_total", 0) or 0
    cells = (raw_data.get("opencellid", {}) or {}).get("feature_count", 0) or 0
    digi = (raw_data.get("digiroad", {}) or {}).get("feature_count", 0) or 0
    fmi_obs = (raw_data.get("fmi", {}) or {}).get("observations", {}) or {}

    def _w(name):
        v = (fmi_obs.get(name) or {}).get("latest", {}).get("value")
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    forest = col_map.get("metsamaankasvillisuus", 0)
    bog = col_map.get("suo", 0)
    water = col_map.get("jarvi", 0) + col_map.get("virtavesialue", 0)
    rocky = col_map.get("kallioalue", 0)
    roads = col_map.get("tieviiva", 0)
    builds = col_map.get("rakennus", 0) + col_map.get("taajaanrakennettualue", 0)
    open_land = col_map.get("maatalousmaa", 0)

    overview = (
        f"AOI covers {area_sqkm:.1f} km² of northern Karelian terrain. "
        f"Dominant features: {'forested' if forest else 'limited forest cover'}, "
        f"{'extensive water obstacles' if water > 5 else 'minimal hydrography'}, "
        f"{'significant bog/wetland' if bog else 'firm ground'}. "
        f"Population {int(pop)}, {digi} road infrastructure features. "
        f"Civilian comms baseline: {cells} cell towers."
    )
    obs = (
        f"Forest density ({forest} features) limits long-range observation in wooded sectors. "
        f"{'Open agricultural belts provide cleared fields of fire.' if open_land else 'Few clearings for unobstructed observation.'} "
        f"{'Rocky elevation provides natural overwatch positions.' if rocky else ''}"
    ).strip()
    approach = (
        f"{roads} road segments form primary avenues of approach. "
        f"{'Water obstacles channel cross-country movement to bridge crossings.' if water else 'Few water-imposed routing constraints.'} "
        f"{'Bog terrain blocks heavy off-road movement.' if bog else ''}"
    ).strip()
    key_t = (
        f"Road junctions, bridge crossings ({digi} Digiroad features), "
        f"{'airfield/port nodes' if osm_col.get('airfields') or osm_col.get('ports_terminals') else 'no aviation/maritime hubs'}, "
        f"and built-up areas ({builds} features) constitute key terrain."
    )
    obstacles = (
        f"Water: {water} hydrographic features. Bog: {bog} features. "
        f"Forest density and rocky terrain impose vehicle mobility constraints in {forest + rocky} feature areas."
    )
    cover = (
        f"Forest cover ({forest} features) provides concealment corridors. "
        f"{'Built-up areas offer urban cover.' if builds else ''} "
        f"{'Open terrain offers minimal concealment.' if open_land and not forest else ''}"
    ).strip()

    wind = _w("wind_speed"); temp = _w("temperature"); precip = _w("precipitation"); cloud = _w("cloud_cover")
    weather_parts = []
    if wind is not None:
        if wind > 10:
            weather_parts.append(f"Wind {wind:.0f} m/s — UAS operations not feasible.")
        elif wind > 7:
            weather_parts.append(f"Wind {wind:.0f} m/s — small UAS degraded.")
        else:
            weather_parts.append(f"Wind {wind:.0f} m/s — UAS feasible.")
    if temp is not None:
        weather_parts.append(f"Temperature {temp:.0f}°C{'  (cold-weather precautions)' if temp < 0 else ''}.")
    if precip is not None and precip > 0:
        weather_parts.append(f"Precipitation {precip:.1f} mm/h — sensor degradation, mobility reduced.")
    if cloud is not None and cloud > 70:
        weather_parts.append(f"Cloud cover {cloud:.0f}% — satellite/aerial collection degraded.")
    weather = " ".join(weather_parts) or "Weather data not available."

    infra_parts = [f"{digi} road and bridge features (Digiroad)."]
    if osm_col.get("power_infrastructure"):
        infra_parts.append(f"Power infrastructure: {osm_col['power_infrastructure']} features (substations, lines).")
    if osm_col.get("fuel_supply"):
        infra_parts.append(f"Fuel supply: {osm_col['fuel_supply']} stations.")
    if osm_col.get("airfields"):
        infra_parts.append(f"Aviation: {osm_col['airfields']} airfield features.")
    if osm_col.get("logistics"):
        infra_parts.append(f"Logistics nodes: {osm_col['logistics']} features (depots, sawmills, agrarian supply).")
    if osm_col.get("ports_terminals"):
        infra_parts.append(f"Transport hubs: {osm_col['ports_terminals']} features.")
    if cells:
        infra_parts.append(f"Civilian comms: {cells} cell towers.")
    infra = " ".join(infra_parts)

    if pop < 100:
        civil = f"Population {int(pop)} — minimal civilian presence; low CIVCAS risk."
    elif pop < 2000:
        civil = f"Population {int(pop)} — moderate civilian presence; CIVCAS precautions required."
    else:
        civil = f"Population {int(pop)} — significant civilian population; urban ROE apply; civilian protection is a primary constraint."

    ccir = (
        "CCIR-1: Are enemy forces present? — No direct indicators from OSINT inputs.\n"
        "CCIR-2: What infrastructure could be commandeered, denied, or destroyed? — " +
        (f"{osm_col.get('logistics', 0)} logistics nodes, {osm_col.get('fuel_supply', 0)} fuel, "
         f"{osm_col.get('power_infrastructure', 0)} power features identified." if osm else "Infrastructure data limited.") + "\n"
        "CCIR-3: Restrictions imposed by terrain and weather? — " +
        (f"{'High wind' if wind and wind > 10 else 'Operationally permissive weather'}; "
         f"{'water obstacles channel movement' if water else 'open routing'}.") + "\n"
        "CCIR-4: ROE constraints from civil considerations? — " +
        (f"{'High CIVCAS risk' if pop > 2000 else 'Low to moderate CIVCAS risk'}, "
         f"{int(pop)} residents in AOI.")
    )

    assessment = (
        f"This {area_sqkm:.1f} km² AOI presents a mixed terrain profile suited for "
        f"{'concealed dismounted operations' if forest else 'exposed mounted operations'}. "
        f"Mobility is {'constrained by water and bog' if water and bog else 'broadly permissive'}. "
        f"Weather is {'a limiting factor for air assets' if wind and wind > 10 else 'within operational limits'}. "
        f"Civilian considerations are {'a primary constraint on ROE' if pop > 2000 else 'manageable'}. "
        "Recommend further reconnaissance to validate enemy disposition before commitment of force."
    )

    limitations = (
        "Assessment based solely on open-source intelligence inputs. "
        "No HUMINT, SIGINT, or current enemy disposition data integrated. "
        "Terrain analysis is structural; trafficability requires ground reconnaissance to confirm. "
        "Weather covers recent observations plus FMI HARMONIE forecast up to the selected horizon."
    )

    return {
        "situation_overview": overview,
        "terrain_observation": obs,
        "terrain_approach": approach,
        "terrain_key": key_t,
        "terrain_obstacles": obstacles,
        "terrain_cover": cover,
        "weather_impact": weather,
        "infrastructure": infra,
        "civil_considerations": civil,
        "ccir_answers": ccir,
        "assessment": assessment,
        "limitations": limitations,
    }


# ─── RulesAnalyzer (military-aware deterministic fallback) ────────────────────

class RulesAnalyzer(EvidenceAnalyzer):
    provider_name = "rules"

    async def analyze(
        self,
        *,
        data_package: dict[str, Any] | None = None,
        llm_input: dict[str, Any] | None = None,
        profile: str | AnalysisProfile = AnalysisProfile.GENERAL,
        question: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        selection: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        raw_data: dict[str, Any] | None = None,
        freshness: list[dict[str, Any]] | None = None,
        evidence_bundle: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        profile_value = _coerce_profile(profile)
        package = _coerce_data_package(
            data_package=data_package,
            selection=selection,
            metrics=metrics,
            freshness=freshness,
            evidence_bundle=evidence_bundle,
        )
        llm_input_payload = _coerce_llm_input(
            llm_input=llm_input,
            data_package=package,
            profile=profile_value,
            question=question,
            conversation_history=conversation_history,
        )

        if not raw_data:
            raw_data = _build_raw_data_from_package(package)

        evidence_refs = _evidence_refs(package)
        findings: list[str] = []
        limitations: list[str] = []

        area_sqkm = _indicator_value(package, "selection_area_sqkm", (metrics or {}).get("selection_area_sqkm", 0))

        # ── NLS terrain analysis ──────────────────────────────────────────────
        nls = raw_data.get("nls", {})
        if nls.get("collections"):
            col_map = {c.get("collection", ""): c.get("count", 0) for c in nls["collections"]}

            forest_ct = col_map.get("metsamaankasvillisuus", 0)
            if forest_ct:
                density = forest_ct / area_sqkm if area_sqkm else 0
                if density > 1.5:
                    findings.append(f"Dense forest cover ({forest_ct} features, {density:.1f}/km²): high concealment value, significant off-road mobility constraint for wheeled vehicles.")
                else:
                    findings.append(f"Moderate forest cover ({forest_ct} features): provides concealment corridors; mixed trafficability.")

            bog_ct = col_map.get("suo", 0)
            if bog_ct:
                findings.append(f"Bogs/marshes present ({bog_ct} features): severe mobility constraint — off-road movement for tracked vehicles limited, wheeled vehicles likely unable to cross.")

            lake_ct = col_map.get("jarvi", 0) + col_map.get("meri", 0)
            river_ct = col_map.get("virtavesialue", 0)
            if lake_ct or river_ct:
                water_desc = []
                if lake_ct:
                    water_desc.append(f"{lake_ct} lake/sea features")
                if river_ct:
                    water_desc.append(f"{river_ct} river/stream features")
                findings.append(f"Water obstacles: {', '.join(water_desc)} — channelizes movement to bridge crossing points, creates natural defensive barriers.")

            road_ct = col_map.get("tieviiva", 0)
            if road_ct:
                findings.append(f"Road network: {road_ct} road segments — primary avenues of approach identified; route analysis required for weight/width limits.")

            rocky_ct = col_map.get("kallioalue", 0)
            if rocky_ct:
                findings.append(f"Rocky terrain ({rocky_ct} features): good natural cover and concealment; limits vehicle movement off road.")

            open_ct = col_map.get("maatalousmaa", 0)
            if open_ct:
                findings.append(f"Open agricultural terrain ({open_ct} features): wide fields of fire, minimal concealment — exposed to observation and direct fire.")

            building_ct = col_map.get("rakennus", 0) + col_map.get("taajaanrakennettualue", 0)
            if building_ct:
                findings.append(f"Urban/built terrain ({building_ct} features): potential strongpoints and cover, but elevated civilian presence and complex ROE environment.")

            power_ct = col_map.get("sahkolinja", 0)
            if power_ct:
                findings.append(f"Power line infrastructure ({power_ct} features): critical asset, helicopter routing constraint under lines.")

            airfield_ct = col_map.get("lentokenttaalue", 0) + col_map.get("satamaalue", 0)
            if airfield_ct:
                findings.append(f"Airfield/port infrastructure detected: key terrain for air assault, CASEVAC, and logistics.")

        elif not nls.get("feature_count"):
            limitations.append("NLS terrain data not loaded — load workspace data first for full terrain analysis.")

        # ── Digiroad bridge/road analysis ────────────────────────────────────
        digiroad = raw_data.get("digiroad", {})
        if digiroad.get("feature_count"):
            findings.append(f"Digiroad: {digiroad['feature_count']} transport infrastructure features in AOI — bridge weight/height limits require cross-reference before routing heavy vehicles.")
        elif not digiroad.get("feature_count"):
            limitations.append("Digiroad road/bridge data not loaded — bridge weight limits unknown.")

        # ── Weather operational impact ────────────────────────────────────────
        fmi = raw_data.get("fmi", {})
        obs = fmi.get("observations", {})
        if obs:
            def _val(name: str) -> float | None:
                v = (obs.get(name) or {}).get("latest", {}).get("value")
                try:
                    return float(v) if v is not None else None
                except (TypeError, ValueError):
                    return None

            wind = _val("wind_speed")
            gust = _val("wind_gust")
            temp = _val("temperature")
            precip = _val("precipitation")
            cloud = _val("cloud_cover")

            weather_parts: list[str] = []
            if wind is not None:
                effective_wind = gust if gust and gust > wind else wind
                if effective_wind > 10:
                    weather_parts.append(f"wind {wind:.0f} m/s gusting {gust:.0f} m/s — drone operations not feasible")
                elif effective_wind > 7:
                    weather_parts.append(f"wind {wind:.0f} m/s — small UAS ops degraded")
                else:
                    weather_parts.append(f"wind {wind:.0f} m/s — UAS operations feasible")
            if precip is not None and precip > 0:
                weather_parts.append(f"precipitation {precip:.1f} mm/h — EO sensor degradation, mobility reduced")
            if temp is not None:
                if temp < -10:
                    weather_parts.append(f"temp {temp:.0f}°C — cold weather ops: battery life reduced, equipment checks critical")
                elif temp < 0:
                    weather_parts.append(f"temp {temp:.0f}°C — below freezing, icy surfaces likely")
                else:
                    weather_parts.append(f"temp {temp:.0f}°C")
            if cloud is not None and cloud > 80:
                weather_parts.append(f"cloud cover {cloud:.0f}% — satellite/aerial imagery collection degraded")

            if weather_parts:
                findings.append(f"Current weather: {'; '.join(weather_parts)}.")
        else:
            limitations.append("FMI weather data not loaded — weather impact on operations unknown.")

        # ── Population / CIVCAS ───────────────────────────────────────────────
        pop_total = _indicator_value(package, "population_total", 0)
        if pop_total:
            if pop_total < 100:
                findings.append(f"Population {pop_total}: minimal civilian presence — low CIVCAS risk, reduced civilian interference.")
            elif pop_total < 2000:
                findings.append(f"Population {pop_total}: moderate civilian presence — CIVCAS precautions required, civilian evacuation may be needed.")
            else:
                findings.append(f"Population {pop_total}: significant civilian population — high CIVCAS risk, urban ROE apply, civilian protection is primary constraint.")

        # ── Comms infrastructure ──────────────────────────────────────────────
        cell = raw_data.get("opencellid", {})
        cell_ct = cell.get("feature_count") or 0
        if cell_ct > 5:
            findings.append(f"Cell tower coverage: {cell_ct} towers — good civilian comms baseline; SIGINT/EW opportunities present.")
        elif cell_ct > 0:
            findings.append(f"Cell tower coverage: {cell_ct} towers — sparse civilian comms; military comms must be self-sufficient.")
        else:
            findings.append("No cell towers detected — remote area with no civilian comms infrastructure; own-force comms planning critical.")

        # ── Fallback if no data at all ────────────────────────────────────────
        if not findings:
            findings.append("No source data intersects this AOI. Load workspace data using the 'Load Data' button before analysis.")
            limitations.append("All data sources empty — ensure workspace data is loaded and the AOI polygon overlaps the loaded area.")

        if question:
            findings.insert(0, f"Analyst question noted: '{question}'. Assessment below is based on available data.")

        # ── Summary ───────────────────────────────────────────────────────────
        terrain_notes = []
        if nls.get("collections"):
            col_map = {c.get("collection", ""): c.get("count", 0) for c in nls["collections"]}
            if col_map.get("metsamaankasvillisuus"):
                terrain_notes.append("forested")
            if col_map.get("suo"):
                terrain_notes.append("boggy")
            if col_map.get("jarvi") or col_map.get("virtavesialue"):
                terrain_notes.append("water obstacles")
            if col_map.get("maatalousmaa"):
                terrain_notes.append("open agricultural terrain")

        terrain_str = "/".join(terrain_notes) if terrain_notes else "terrain data not loaded"
        summary = (
            f"{area_sqkm:.1f} km² AOI. Terrain: {terrain_str}. "
            f"Population: {int(pop_total)}. "
            f"{'Weather within operational limits.' if obs and not any('not feasible' in f for f in findings) else 'Weather or data constraints apply.'} "
            f"Confidence: {package.get('quality', {}).get('overall_confidence', 'medium')}."
        )
        if question:
            summary = f"Re: '{question}'. {summary}"

        structured_output = _build_structured_output(
            profile=profile_value,
            summary=summary,
            findings=findings,
            limitations=limitations,
            provider=self.provider_name,
            status="ready",
            evidence_refs=evidence_refs,
            quality=package.get("quality"),
        )

        return {
            "provider": self.provider_name,
            "status": "ready",
            "profile": profile_value.value,
            "question": question,
            "summary": summary,
            "findings": findings,
            "limitations": limitations,
            "evidence_bundle": evidence_bundle or package.get("evidence_items", []),
            "llm_input": llm_input_payload,
            "output": structured_output.model_dump(mode="json"),
        }


# ─── Disabled / Ollama analyzers ─────────────────────────────────────────────

class DisabledAnalyzer(EvidenceAnalyzer):
    provider_name = "disabled"

    async def health(self) -> dict[str, Any]:
        return {"provider": self.provider_name, "status": "disabled"}

    async def analyze(self, *, data_package=None, llm_input=None, profile=AnalysisProfile.GENERAL,
                      question=None, conversation_history=None, selection=None, metrics=None,
                      raw_data=None, freshness=None, evidence_bundle=None) -> dict[str, Any]:
        package = _coerce_data_package(data_package=data_package, selection=selection,
                                       metrics=metrics, freshness=freshness, evidence_bundle=evidence_bundle)
        profile_value = _coerce_profile(profile)
        llm_input_payload = _coerce_llm_input(llm_input=llm_input, data_package=package,
                                               profile=profile_value, question=question,
                                               conversation_history=conversation_history)
        msg = "Analyzer disabled. Set ANALYSIS_PROVIDER=claude (or 'rules') in .env to enable."
        structured_output = _build_structured_output(
            profile=profile_value, summary=msg, findings=[msg],
            limitations=["Set ANALYSIS_PROVIDER in .env"], provider=self.provider_name,
            status="disabled", evidence_refs=_evidence_refs(package), quality=package.get("quality"),
        )
        return {
            "provider": self.provider_name, "status": "disabled", "profile": profile_value.value,
            "summary": msg, "findings": [msg], "limitations": [],
            "evidence_bundle": evidence_bundle or package.get("evidence_items", []),
            "llm_input": llm_input_payload, "output": structured_output.model_dump(mode="json"),
        }


class OllamaAnalyzer(EvidenceAnalyzer):
    provider_name = "ollama"

    async def health(self) -> dict[str, Any]:
        try:
            payload = await self._fetch_tags()
        except Exception as exc:
            return {"provider": self.provider_name, "status": "error",
                    "base_url": settings.ollama_base_url, "model": settings.ollama_model,
                    "reachable": False, "error": str(exc)}
        available_models = [item.get("name") for item in payload.get("models", []) if item.get("name")]
        model_available = settings.ollama_model in available_models
        return {
            "provider": self.provider_name,
            "status": "ready" if model_available else "degraded",
            "base_url": settings.ollama_base_url, "model": settings.ollama_model,
            "reachable": True, "model_available": model_available, "available_models": available_models,
        }

    async def analyze(self, *, data_package=None, llm_input=None, profile=AnalysisProfile.GENERAL,
                      question=None, conversation_history=None, selection=None, metrics=None,
                      raw_data=None, freshness=None, evidence_bundle=None) -> dict[str, Any]:
        profile_value = _coerce_profile(profile)
        package = _coerce_data_package(data_package=data_package, selection=selection,
                                       metrics=metrics, freshness=freshness, evidence_bundle=evidence_bundle)
        llm_input_payload = _coerce_llm_input(llm_input=llm_input, data_package=package,
                                               profile=profile_value, question=question,
                                               conversation_history=conversation_history)
        if not raw_data:
            raw_data = _build_raw_data_from_prompt(package)

        data_text = _format_data_for_prompt(raw_data, package)
        prompt = self._build_prompt(llm_input_payload, profile_value, data_text, question)
        payload = await self._generate_payload(prompt)

        text = (payload.get("response") or "").strip()
        if not text:
            raise ValueError("Ollama returned an empty response")

        parsed = _parse_llm_response(text)
        findings = parsed["findings"]
        summary = parsed["summary"] or (findings[0] if findings else text)
        limitations = parsed["limitations"]

        if not summary:
            raise ValueError("Ollama returned no usable summary")

        evidence_refs = _evidence_refs(package)
        structured_output = _build_structured_output(
            profile=profile_value, summary=summary, findings=findings[:6], limitations=limitations[:4],
            provider=self.provider_name, status="ready", evidence_refs=evidence_refs,
            quality=package.get("quality"), model=settings.ollama_model,
        )
        return {
            "provider": self.provider_name, "status": "ready", "profile": profile_value.value,
            "question": question, "summary": summary, "findings": findings[:6], "limitations": limitations[:4],
            "model": settings.ollama_model,
            "evidence_bundle": evidence_bundle or package.get("evidence_items", []),
            "llm_input": llm_input_payload, "output": structured_output.model_dump(mode="json"),
        }

    async def _fetch_tags(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout_seconds) as client:
            response = await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            return response.json()

    async def _generate_payload(self, prompt: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout_seconds) as client:
            response = await client.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/generate",
                json={"model": settings.ollama_model, "prompt": prompt, "stream": False, "format": "json"},
            )
            response.raise_for_status()
            return response.json()

    def _build_prompt(self, llm_input: dict[str, Any], profile: AnalysisProfile,
                      data_text: str, question: str | None) -> str:
        profile_focus = PROFILE_SPECS.get(profile, PROFILE_SPECS[AnalysisProfile.GENERAL])["focus"]
        q = f"\nANALYST QUESTION: {question}" if question else ""
        return (
            "You are an IPB analyst for Finnish terrain. Use only the supplied data. "
            "Apply OAKOC framework. Do not invent missing data.\n\n"
            f"Profile: {profile.value} — {profile_focus}\n\n"
            f"{data_text}{q}\n\n"
            'Return JSON: {"summary": "...", "findings": [...], "limitations": [...]}'
        )


def _build_raw_data_from_prompt(package: dict[str, Any]) -> dict[str, Any]:
    return _build_raw_data_from_package(package)


# ─── Factory ──────────────────────────────────────────────────────────────────

async def get_analyzer_health() -> dict[str, Any]:
    return await build_analyzer().health()


def build_analyzer() -> EvidenceAnalyzer:
    provider = settings.analysis_provider.strip().lower()
    if provider == "disabled":
        return DisabledAnalyzer()
    if provider == "ollama":
        return OllamaAnalyzer()
    if provider == "claude":
        if not settings.anthropic_api_key:
            # Fall through to rules if key not configured
            return RulesAnalyzer()
        return ClaudeAnalyzer()
    return RulesAnalyzer()
