from __future__ import annotations

import json
from typing import Any

import httpx

from ipb_backend.config import settings
from ipb_backend.planning.force_model import (
    ForceComposition,
    Operation,
    PlanningResponse,
    RecommendedSite,
)


def _site_payload(site: RecommendedSite) -> dict[str, Any]:
    return {
        "rank": site.rank,
        "score": site.score,
        "feasible": site.feasible,
        "centroid": site.centroid,
        "score_breakdown": site.score_breakdown,
        "rationale": site.rationale,
        "constraint_matches": [
            {
                "name": m.name,
                "passed": m.passed,
                "observed": m.observed,
                "required": m.required,
                "detail": m.detail,
            }
            for m in site.constraint_matches
        ],
    }


def _build_prompt(
    response: PlanningResponse,
    force: ForceComposition,
    operation: Operation,
    site: RecommendedSite,
) -> str:
    force_summary = {
        "infantry": force.infantry,
        "vehicles": [v.model_dump() for v in force.vehicles],
        "drones": [d.model_dump() for d in force.drones],
        "column_movement": force.column_movement,
    }
    operation_summary = operation.model_dump()

    return (
        "You are explaining a single recommended military site to a planning officer. "
        "Use only the supplied evidence. Do not invent numbers. Keep it under 80 words. "
        "Reference the specific constraints that matched and the dominant score components.\n\n"
        f"Operation: {json.dumps(operation_summary)}\n"
        f"Force: {json.dumps(force_summary)}\n"
        f"Weights: {json.dumps(response.weights)}\n"
        f"Site: {json.dumps(_site_payload(site))}\n\n"
        "Return strict JSON: {\"narrative\": \"<one-paragraph rationale>\"}."
    )


async def _generate(prompt: str) -> str:
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
        payload = response.json()
    text = (payload.get("response") or "").strip()
    if not text:
        return ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(data, dict):
        return str(data.get("narrative") or "").strip()
    return ""


def _rule_based_narrative(
    site: RecommendedSite, operation: Operation, force: ForceComposition
) -> str:
    pieces: list[str] = []

    top_component = max(site.score_breakdown.items(), key=lambda kv: kv[1], default=(None, 0))
    if top_component[0] and top_component[1] > 0:
        pieces.append(
            f"Site #{site.rank} scores {site.score:.2f}; the leading factor is {top_component[0].replace('_', ' ')} ({top_component[1]:.2f})."
        )

    passed = [m for m in site.constraint_matches if m.passed]
    failed = [m for m in site.constraint_matches if not m.passed]
    if passed:
        pieces.append(
            "Hard constraints satisfied: " + ", ".join(m.name for m in passed) + "."
        )
    if failed:
        pieces.append(
            "Failed constraints: " + "; ".join(m.detail or m.name for m in failed) + "."
        )

    if operation.notes:
        pieces.append(f"Operation context: {operation.notes}.")

    return " ".join(pieces)


async def enrich_with_narratives(
    response: PlanningResponse,
    force: ForceComposition,
    operation: Operation,
) -> PlanningResponse:
    provider = settings.analysis_provider.strip().lower()
    use_llm = provider == "ollama"

    enriched: list[RecommendedSite] = []
    for site in response.top_sites:
        narrative: str
        if use_llm:
            try:
                prompt = _build_prompt(response, force, operation, site)
                narrative = await _generate(prompt)
            except Exception as exc:
                narrative = _rule_based_narrative(site, operation, force) + f" (LLM fallback: {exc})"
        else:
            narrative = _rule_based_narrative(site, operation, force)

        enriched.append(site.model_copy(update={"narrative": narrative or None}))

    return response.model_copy(update={"top_sites": enriched})
