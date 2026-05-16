from __future__ import annotations

from ipb_backend.llm.contracts import AnalysisProfile


PROFILE_SPECS = {
    AnalysisProfile.GENERAL: {
        "label": "General",
        "focus": "Balanced operational summary across available evidence.",
    },
    AnalysisProfile.MOBILITY: {
        "label": "Mobility",
        "focus": "Movement corridors, route constraints, and off-road access.",
    },
    AnalysisProfile.CONCEALMENT: {
        "label": "Concealment",
        "focus": "Cover, concealment, and exposure constraints.",
    },
    AnalysisProfile.SURVEILLANCE: {
        "label": "Surveillance / Observation",
        "focus": "Observation opportunities, gaps, and monitoring constraints.",
    },
    AnalysisProfile.INFRASTRUCTURE_RESILIENCE: {
        "label": "Infrastructure Resilience",
        "focus": "Dependency concentration, redundancy, and disruption sensitivity.",
    },
    AnalysisProfile.LOGISTICS: {
        "label": "Logistics",
        "focus": "Access, sustainment, and support-node implications.",
    },
    AnalysisProfile.COMMUNICATIONS: {
        "label": "Communications",
        "focus": "Cell coverage, communications nodes, and likely connectivity gaps.",
    },
}


def list_profile_specs() -> list[dict[str, str]]:
    return [
        {
            "profile": profile.value,
            "label": details["label"],
            "focus": details["focus"],
        }
        for profile, details in PROFILE_SPECS.items()
    ]