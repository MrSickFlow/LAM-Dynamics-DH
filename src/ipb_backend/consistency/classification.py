from __future__ import annotations

from dataclasses import dataclass

from ipb_backend.models import EwClassification


@dataclass(frozen=True)
class SourceEwProfile:
    ew_classification: EwClassification
    gnss_dependent: bool
    rationale: str


SOURCE_EW_PROFILES: dict[str, SourceEwProfile] = {
    "nls": SourceEwProfile(
        EwClassification.IMMUNE,
        False,
        "Pre-collected topographic and infrastructure vectors; not live GNSS telemetry.",
    ),
    "statistics-finland": SourceEwProfile(
        EwClassification.IMMUNE,
        False,
        "Census and municipal statistics; updated on publication cycles.",
    ),
    "digiroad": SourceEwProfile(
        EwClassification.MIXED,
        False,
        "Static road network with optional live-linked attributes; geometry is immune.",
    ),
    "osm-poi": SourceEwProfile(
        EwClassification.IMMUNE,
        False,
        "Crowd-sourced POI baseline; ingestion is not GPS-sensor telemetry.",
    ),
    "fmi": SourceEwProfile(
        EwClassification.VULNERABLE,
        False,
        "Live meteorological station feeds; RF or network disruption can affect reporting.",
    ),
    "opencellid": SourceEwProfile(
        EwClassification.VULNERABLE,
        True,
        "Derived from handset GNSS fixes; jamming or spoofing degrades coverage maps.",
    ),
    "satellites": SourceEwProfile(
        EwClassification.MIXED,
        True,
        "Orbital elements are immune; live tasking confirmation is GNSS-dependent.",
    ),
    "maritime-demo": SourceEwProfile(
        EwClassification.VULNERABLE,
        True,
        "AIS positions are GNSS-derived; used for cross-check demonstrations.",
    ),
}


AREA_CENTERS: dict[str, dict[str, float]] = {
    "archipelago sea": {"lat": 60.2, "lon": 22.0},
    "north karelia": {"lat": 62.8, "lon": 30.2},
    "lapland": {"lat": 68.9, "lon": 21.5},
    "lapland (kasivarren lappi)": {"lat": 68.9, "lon": 21.5},
    "kasivarren lappi": {"lat": 68.9, "lon": 21.5},
}


def normalize_area(area: str) -> str:
    return " ".join(area.lower().strip().split())


def resolve_area_center(area: str) -> dict[str, float]:
    return AREA_CENTERS.get(normalize_area(area), AREA_CENTERS["north karelia"])


def get_source_profile(source_id: str) -> SourceEwProfile:
    return SOURCE_EW_PROFILES.get(
        source_id,
        SourceEwProfile(
            EwClassification.MIXED,
            False,
            "Unclassified source; treat trust conservatively.",
        ),
    )
