from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import httpx

from ipb_backend.config import settings


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
        selection: dict[str, Any],
        metrics: dict[str, Any],
        raw_data: dict[str, Any],
        freshness: list[dict[str, Any]],
        evidence_bundle: list[dict[str, Any]],
    ) -> dict[str, Any]:
        pass


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
    return {
        "summary": summary,
        "findings": findings,
        "limitations": limitations,
    }


class RulesAnalyzer(EvidenceAnalyzer):
    provider_name = "rules"

    async def analyze(
        self,
        *,
        selection: dict[str, Any],
        metrics: dict[str, Any],
        raw_data: dict[str, Any],
        freshness: list[dict[str, Any]],
        evidence_bundle: list[dict[str, Any]],
    ) -> dict[str, Any]:
        findings = []

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

        summary = (
            f"AOI covers {metrics['selection_area_sqkm']} km2 and intersects "
            f"{metrics['nls_feature_count']} NLS features, {metrics['digiroad_feature_count']} transport features, "
            f"population cells totaling {metrics['population_total']}, and {metrics['weather_station_count']} weather stations."
        )

        return {
            "provider": self.provider_name,
            "status": "ready",
            "summary": summary,
            "findings": findings,
            "evidence_bundle": evidence_bundle,
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
        selection: dict[str, Any],
        metrics: dict[str, Any],
        raw_data: dict[str, Any],
        freshness: list[dict[str, Any]],
        evidence_bundle: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "status": "disabled",
            "summary": "Analyzer disabled. Deterministic metrics and raw evidence remain available.",
            "findings": ["Switch ANALYSIS_PROVIDER to 'rules' or 'ollama' to enable interpretation."],
            "evidence_bundle": evidence_bundle,
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
        selection: dict[str, Any],
        metrics: dict[str, Any],
        raw_data: dict[str, Any],
        freshness: list[dict[str, Any]],
        evidence_bundle: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = self._build_prompt(selection, metrics, freshness, evidence_bundle)
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

        return {
            "provider": self.provider_name,
            "status": "ready",
            "summary": summary,
            "findings": findings[:6],
            "limitations": limitations[:4],
            "model": settings.ollama_model,
            "evidence_bundle": evidence_bundle,
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
        selection: dict[str, Any],
        metrics: dict[str, Any],
        freshness: list[dict[str, Any]],
        evidence_bundle: list[dict[str, Any]],
    ) -> str:
        return (
            "You are interpreting geospatial AOI evidence. Use only the supplied evidence. "
            "Do not invent missing data. Mention source limitations explicitly.\n\n"
            f"Selection bounds: {selection.get('bounds', [])}\n"
            f"Metrics: {metrics}\n"
            f"Freshness: {freshness}\n"
            f"Evidence bundle: {evidence_bundle}\n\n"
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