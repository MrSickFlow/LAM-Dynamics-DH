from __future__ import annotations

import json
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


def _format_data_for_prompt(raw_data: dict[str, Any], data_package: dict[str, Any]) -> str:
    """Build a compact, military-relevant text description of all available data."""
    lines: list[str] = []

    # NLS terrain
    nls = raw_data.get("nls", {})
    if nls.get("feature_count") or nls.get("collections"):
        cols = nls.get("collections", [])
        if cols:
            terrain_lines = []
            for col in cols:
                label = col.get("label", col.get("collection", ""))
                count = col.get("count", 0)
                cid = col.get("collection", "")
                mil = _NLS_TERRAIN_MILITARY.get(cid)
                if mil and count:
                    terrain_lines.append(f"  - {mil[0]} ({count} features): {mil[1]}")
                elif count:
                    terrain_lines.append(f"  - {label} ({count} features)")
            if terrain_lines:
                lines.append("NLS TERRAIN DATA:")
                lines.extend(terrain_lines)

    # Digiroad roads/bridges
    digiroad = raw_data.get("digiroad", {})
    if digiroad.get("feature_count"):
        lines.append(f"ROAD INFRASTRUCTURE: {digiroad['feature_count']} Digiroad features (roads, bridges, tunnels)")
        if digiroad.get("collections"):
            for col in digiroad["collections"][:3]:
                lines.append(f"  - {col.get('label', col.get('collection', ''))}: {col.get('count', 0)}")

    # Weather
    fmi = raw_data.get("fmi", {})
    obs = fmi.get("observations", {})
    if obs:
        def _val(name: str) -> Any:
            return (obs.get(name) or {}).get("latest", {}).get("value")
        temp = _val("temperature")
        wind = _val("wind_speed")
        gust = _val("wind_gust")
        precip = _val("precipitation")
        cloud = _val("cloud_cover")
        parts = []
        if temp is not None:
            parts.append(f"temp {temp}°C")
        if wind is not None:
            parts.append(f"wind {wind} m/s" + (f" (gusts {gust} m/s)" if gust else ""))
        if precip is not None:
            parts.append(f"precip {precip} mm/h")
        if cloud is not None:
            parts.append(f"cloud cover {cloud}%")
        if parts:
            lines.append(f"WEATHER: {', '.join(parts)}")
            # Operational weather notes
            if wind is not None and float(wind) > 10:
                lines.append("  - Wind >10 m/s: drone operations severely restricted")
            elif wind is not None and float(wind) > 7:
                lines.append("  - Wind 7-10 m/s: small drone operations degraded")
            if precip is not None and float(precip) > 2:
                lines.append("  - Active precipitation: drone ops restricted, visibility reduced, mobility degraded")
            if temp is not None and float(temp) < -5:
                lines.append("  - Cold weather: equipment reliability concerns, personnel cold-weather requirements")
            if cloud is not None and float(cloud) > 80:
                lines.append("  - Heavy cloud cover: aerial surveillance and satellite observation degraded")

    # Population
    pop = raw_data.get("statistics-finland", {})
    if pop.get("population_total"):
        total = pop["population_total"]
        if total < 100:
            civil_note = "minimal civilian presence — low CIVCAS risk"
        elif total < 1000:
            civil_note = "moderate civilian presence — CIVCAS precautions required"
        else:
            civil_note = "significant civilian population — high CIVCAS risk, urban ROE apply"
        lines.append(f"POPULATION: {total} residents in AOI — {civil_note}")

    # Cell towers / comms
    cell = raw_data.get("opencellid", {})
    if cell.get("feature_count"):
        lines.append(f"COMMUNICATIONS: {cell['feature_count']} cell towers — civilian comms infrastructure present")
    elif cell.get("feature_count") == 0:
        lines.append("COMMUNICATIONS: No cell towers detected — poor civilian comms coverage, degraded SIGINT environment")

    # OSM POIs
    osm = raw_data.get("osm-poi", {})
    if osm.get("feature_count"):
        poi_cols = osm.get("collections", [])
        poi_types = ", ".join(f"{c.get('label', c.get('collection', ''))}" for c in poi_cols[:5])
        lines.append(f"POINTS OF INTEREST: {osm['feature_count']} POIs ({poi_types})")

    # Derived indicators from package
    for ind in data_package.get("derived_indicators", []):
        if "feature_density" in ind.get("indicator_id", ""):
            sid = ind["indicator_id"].replace("_feature_density", "")
            val = ind.get("value", 0)
            if val and sid not in ("nls", "digiroad", "statistics-finland", "fmi", "opencellid", "osm-poi"):
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

        system_prompt = (
            "You are an IPB (Intelligence Preparation of the Battlefield) analyst specializing in Finnish terrain. "
            "You analyze open-source data from NLS topography, Digiroad road network, FMI weather, "
            "Statistics Finland, OpenCellID, and OpenStreetMap to produce actionable military assessments. "
            "Use the OAKOC framework (Observation & fields of fire, Avenues of approach, Key terrain, "
            "Obstacles, Cover & concealment). Be specific and reference the actual data. "
            "Never invent data not present in the package. Flag data gaps explicitly."
        )

        context_message = (
            f"AOI: {area_sqkm:.1f} km²"
            + (f" | Bounds: {', '.join(f'{b:.3f}' for b in bounds)}" if bounds else "")
            + f"\nAnalysis profile: {profile_value.value} — {profile_focus}\n\n"
            + data_text
            + "\n\nProvide a military assessment. "
            "Return JSON with keys: summary (2-3 sentence tactical assessment), "
            "findings (array of 5-8 specific military-relevant findings), "
            "limitations (array of data gaps or caveats)."
        )

        # Build message list — prepend context on first turn, use history for follow-ups
        messages: list[dict[str, Any]] = []
        history = conversation_history or []

        if not history:
            # Fresh inspection
            user_content = context_message
            if question:
                user_content += f"\n\nANALYST QUESTION: {question}"
            messages = [{"role": "user", "content": user_content}]
        else:
            # Follow-up conversation — seed with context then replay history
            messages.append({"role": "user", "content": context_message})
            messages.append({
                "role": "assistant",
                "content": json.dumps({"summary": "AOI data loaded.", "findings": [], "limitations": []})
            })
            for turn in history[1:]:  # skip the first assistant turn (the initial summary)
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})
            if question:
                messages.append({"role": "user", "content": question})

        try:
            text = await self._call_api(system_prompt, messages)
        except Exception as exc:
            raise ValueError(f"Claude API call failed: {exc}") from exc

        parsed = _parse_llm_response(text)
        findings = parsed["findings"]
        summary = parsed["summary"] or (findings[0] if findings else text[:200])
        limitations = parsed["limitations"]

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

    async def _call_api(self, system: str, messages: list[dict[str, Any]]) -> str:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": settings.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": settings.anthropic_model,
                    "max_tokens": 1200,
                    "system": system,
                    "messages": messages,
                },
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]


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
