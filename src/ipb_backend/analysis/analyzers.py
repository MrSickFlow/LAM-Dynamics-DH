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
            {
                "indicator_id": "selection_area_sqkm",
                "value": (metrics or {}).get("selection_area_sqkm", 0),
            },
            {
                "indicator_id": "population_total",
                "value": (metrics or {}).get("population_total", 0),
            },
            {
                "indicator_id": "weather_station_count",
                "value": (metrics or {}).get("weather_station_count", 0),
            },
        ],
        "source_freshness": freshness or [],
        "evidence_items": [
            {
                "evidence_id": f"ev-{item.get('source_id', 'unknown')}-{index:03d}",
                **item,
            }
            for index, item in enumerate(evidence_bundle or [], start=1)
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
            id=f"imp-{index}",
            topic=profile.value,
            assessment="grounded",
            statement=finding,
            evidence_refs=evidence_refs,
            confidence=(quality or {}).get("overall_confidence", "medium"),
        )
        for index, finding in enumerate(findings[:4], start=1)
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
            id=f"lim-{index}",
            statement=item,
            evidence_refs=evidence_refs[:1],
            severity="high" if "fallback" in item.lower() else "medium",
        )
        for index, item in enumerate(limitations[:4], start=1)
    ]

    uncertainties: list[LimitationStatement] = []
    if not evidence_refs:
        uncertainties.append(
            LimitationStatement(
                id="unc-1",
                statement="No intersecting evidence items were available for interpretation.",
                evidence_refs=[],
                severity="high",
            )
        )

    metadata: dict[str, Any] = {
        "provider": provider,
        "evidence_ref_count": len(evidence_refs),
    }
    if model:
        metadata["model"] = model

    return LlmAnalysisOutput(
        profile=profile,
        status=status,
        summary=summary,
        implications=implications,
        defensive_relevance=defensive_relevance,
        limitations=limitation_items,
        uncertainties=uncertainties,
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


def _parse_ollama_response_text(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fence(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        findings = _normalize_lines(cleaned)
        return {
            "summary": findings[0] if findings else cleaned,
            "findings": findings,
            "limitations": [],
        }

    if isinstance(payload, list):
        findings = _normalize_lines(payload)
        return {
            "summary": findings[0] if findings else "",
            "findings": findings,
            "limitations": [],
        }

    if not isinstance(payload, dict):
        return {
            "summary": cleaned,
            "findings": _normalize_lines(cleaned),
            "limitations": [],
        }

    findings = _normalize_lines(payload.get("findings"))
    summary = str(payload.get("summary") or (findings[0] if findings else "")).strip()
    limitations = _normalize_lines(payload.get("limitations"))
    implications = payload.get("implications", [])
    defensive_relevance = payload.get("defensive_relevance", [])
    uncertainties = payload.get("uncertainties", [])
    return {
        "summary": summary,
        "findings": findings,
        "limitations": limitations,
        "implications": implications,
        "defensive_relevance": defensive_relevance,
        "uncertainties": uncertainties,
    }


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
        evidence_refs = _evidence_refs(package)
        findings = []

        raw_data = raw_data or {}
        nls = raw_data.get("nls", {})
        if nls.get("feature_count"):
            top = nls.get("collections", [])[:2]
            top_text = ", ".join(f"{item['label']} ({item['count']})" for item in top)
            findings.append(f"NLS shows {nls['feature_count']} intersecting terrain features. Top layers: {top_text}.")

        digiroad = raw_data.get("digiroad", {})
        if digiroad.get("feature_count"):
            findings.append(f"Digiroad contributes {digiroad['feature_count']} transport features inside the AOI.")

        population = raw_data.get("statistics-finland", {})
        if population.get("population_total"):
            findings.append(
                f"Population cells intersecting the AOI sum to {population['population_total']} residents in the current scaffold."
            )

        weather = raw_data.get("fmi", {})
        if weather.get("station_count"):
            findings.append("An FMI weather station falls inside the AOI, so current conditions are locally grounded.")

        if not findings:
            findings.append("No intersecting source data was found for the current polygon selection.")

        if question:
            findings.insert(0, f"Question focus: {question}")

        summary = (
            f"AOI covers {_indicator_value(package, 'selection_area_sqkm', (metrics or {}).get('selection_area_sqkm', 0))} km2 and intersects "
            f"{(metrics or {}).get('nls_feature_count', 0)} NLS features, {(metrics or {}).get('digiroad_feature_count', 0)} transport features, "
            f"population cells totaling {_indicator_value(package, 'population_total', (metrics or {}).get('population_total', 0))}, and "
            f"{_indicator_value(package, 'weather_station_count', (metrics or {}).get('weather_station_count', 0))} weather stations."
        )
        if question:
            summary = f"Question received: {question}. {summary}"

        structured_output = _build_structured_output(
            profile=profile_value,
            summary=summary,
            findings=findings,
            limitations=[],
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
            "limitations": [],
            "evidence_bundle": evidence_bundle or package.get("evidence_items", []),
            "llm_input": llm_input_payload,
            "output": structured_output.model_dump(mode="json"),
        }


class DisabledAnalyzer(EvidenceAnalyzer):
    provider_name = "disabled"

    async def health(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "status": "disabled",
        }

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
        package = _coerce_data_package(
            data_package=data_package,
            selection=selection,
            metrics=metrics,
            freshness=freshness,
            evidence_bundle=evidence_bundle,
        )
        profile_value = _coerce_profile(profile)
        llm_input_payload = _coerce_llm_input(
            llm_input=llm_input,
            data_package=package,
            profile=profile_value,
            question=question,
            conversation_history=conversation_history,
        )
        structured_output = _build_structured_output(
            profile=profile_value,
            summary="Analyzer disabled. Deterministic metrics and raw evidence remain available.",
            findings=["Analyzer disabled. Deterministic metrics and raw evidence remain available."],
            limitations=["Switch ANALYSIS_PROVIDER to 'rules' or 'ollama' to enable interpretation."],
            provider=self.provider_name,
            status="disabled",
            evidence_refs=_evidence_refs(package),
            quality=package.get("quality"),
        )
        return {
            "provider": self.provider_name,
            "status": "disabled",
            "profile": profile_value.value,
            "summary": "Analyzer disabled. Deterministic metrics and raw evidence remain available.",
            "findings": ["Switch ANALYSIS_PROVIDER to 'rules' or 'ollama' to enable interpretation."],
            "limitations": ["Switch ANALYSIS_PROVIDER to 'rules' or 'ollama' to enable interpretation."],
            "evidence_bundle": evidence_bundle or package.get("evidence_items", []),
            "llm_input": llm_input_payload,
            "output": structured_output.model_dump(mode="json"),
        }


class OllamaAnalyzer(EvidenceAnalyzer):
    provider_name = "ollama"

    async def health(self) -> dict[str, Any]:
        try:
            payload = await self._fetch_tags()
        except Exception as exc:
            return {
                "provider": self.provider_name,
                "status": "error",
                "base_url": settings.ollama_base_url,
                "model": settings.ollama_model,
                "reachable": False,
                "error": str(exc),
            }

        available_models = [item.get("name") for item in payload.get("models", []) if item.get("name")]
        model_available = settings.ollama_model in available_models
        return {
            "provider": self.provider_name,
            "status": "ready" if model_available else "degraded",
            "base_url": settings.ollama_base_url,
            "model": settings.ollama_model,
            "reachable": True,
            "model_available": model_available,
            "available_models": available_models,
        }

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
        prompt = self._build_prompt(llm_input_payload, profile_value)
        payload = await self._generate_payload(prompt)

        text = (payload.get("response") or "").strip()
        if not text:
            raise ValueError("Ollama returned an empty response")

        parsed = _parse_ollama_response_text(text)
        findings = parsed["findings"]
        summary = parsed["summary"] or (findings[0] if findings else text)
        limitations = parsed["limitations"]

        if not summary:
            raise ValueError("Ollama returned no usable summary")

        evidence_refs = _evidence_refs(package)
        structured_output = _build_structured_output(
            profile=profile_value,
            summary=summary,
            findings=findings[:6],
            limitations=limitations[:4],
            provider=self.provider_name,
            status="ready",
            evidence_refs=evidence_refs,
            quality=package.get("quality"),
            model=settings.ollama_model,
        )

        return {
            "provider": self.provider_name,
            "status": "ready",
            "profile": profile_value.value,
            "question": question,
            "summary": summary,
            "findings": findings[:6],
            "limitations": limitations[:4],
            "model": settings.ollama_model,
            "evidence_bundle": evidence_bundle or package.get("evidence_items", []),
            "llm_input": llm_input_payload,
            "output": structured_output.model_dump(mode="json"),
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
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
            )
            response.raise_for_status()
            return response.json()

    def _build_prompt(
        self,
        llm_input: dict[str, Any],
        profile: AnalysisProfile,
    ) -> str:
        return (
            "You are interpreting normalized geospatial evidence for a defensive planning prototype. "
            "Use only the supplied data package. Do not invent missing data. Mention source limitations explicitly.\n\n"
            f"Analysis profile: {profile.value}\n"
            f"Profile focus: {PROFILE_SPECS.get(profile, PROFILE_SPECS[AnalysisProfile.GENERAL])['focus']}\n"
            f"Optimized wrapper input: {llm_input}\n\n"
            "Return strict JSON with keys: summary (string), findings (array of 3-5 concise strings), "
            "limitations (array of concise strings)."
        )


async def get_analyzer_health() -> dict[str, Any]:
    analyzer = build_analyzer()
    return await analyzer.health()


def build_analyzer() -> EvidenceAnalyzer:
    provider = settings.analysis_provider.strip().lower()
    if provider == "disabled":
        return DisabledAnalyzer()
    if provider == "ollama":
        return OllamaAnalyzer()
    return RulesAnalyzer()