from __future__ import annotations

from ipb_backend.agents.base import AnalysisAgent
from ipb_backend.ingestion.sources.digiroad import DigiroadAdapter
from ipb_backend.models import AgentRunResult

SILTA_ALIK_LABELS = {0: "bridge", 1: "underpass", -1: "tunnel"}

VEHICLE_CLASSES: list[tuple[str, int]] = [
    ("Light vehicles only", 16),
    ("Light armored / medium truck", 30),
    ("Heavy truck / IFV", 50),
    ("Main battle tank", 70),
    ("Super-heavy", 999),
]


def _classify_weight(kg: int) -> str:
    tonnes = kg / 1000
    for label, threshold in VEHICLE_CLASSES:
        if tonnes <= threshold:
            return label
    return "Unknown"


class BridgeLoadAgent(AnalysisAgent):
    agent_id = "bridge-load-agent"

    def __init__(self, adapter: DigiroadAdapter) -> None:
        self.adapter = adapter

    async def run(self, area: str, timeframe: str) -> AgentRunResult:
        record = await self.adapter.fetch(area=area, timeframe=timeframe)
        collections = record.data.get("collections", {})

        bridge_features = collections.get("dr_tielinkki_silta_alikulku_tunneli", {}).get("features", [])
        weight_rules = collections.get("dr_max_massa", {}).get("features", [])
        height_rules = collections.get("dr_max_korkeus", {}).get("features", [])
        width_rules = collections.get("dr_max_leveys", {}).get("features", [])
        axle_rules = collections.get("dr_max_akselimassa", {}).get("features", [])
        combined_rules = collections.get("dr_yhdistelman_max_massa", {}).get("features", [])

        by_link: dict[str, dict] = {}
        for rules, key in [
            (weight_rules, "max_weight_kg"),
            (height_rules, "max_height_cm"),
            (width_rules, "max_width_cm"),
            (axle_rules, "max_axle_kg"),
            (combined_rules, "max_combined_kg"),
        ]:
            for f in rules:
                p = f.get("properties", {})
                lid = p.get("link_id")
                if not lid:
                    continue
                entry = by_link.setdefault(lid, {})
                existing = entry.get(key)
                if existing is None or p.get("arvo", 0) < existing:
                    entry[key] = p.get("arvo")

        enriched: list[dict] = []
        bridge_count = 0
        underpass_count = 0
        tunnel_count = 0
        weight_dist: dict[str, int] = {}
        height_restricted: list[str] = []

        for f in bridge_features:
            p = f.get("properties", {})
            lid = p.get("link_id")
            silta_type = p.get("silta_alik", 0)
            limits = by_link.get(lid, {})

            max_weight_kg = limits.get("max_weight_kg")
            max_height_cm = limits.get("max_height_cm")
            max_width_cm = limits.get("max_width_cm")
            max_axle_kg = limits.get("max_axle_kg")

            enriched_feature = {
                "type": "Feature",
                "geometry": f.get("geometry"),
                "properties": {
                    "link_id": lid,
                    "type": SILTA_ALIK_LABELS.get(silta_type, f"unknown({silta_type})"),
                    "silta_alik": silta_type,
                    "max_weight_tonnes": round(max_weight_kg / 1000, 1) if max_weight_kg else None,
                    "max_height_m": round(max_height_cm / 100, 1) if max_height_cm else None,
                    "max_width_m": round(max_width_cm / 100, 1) if max_width_cm else None,
                    "max_axle_tonnes": round(max_axle_kg / 1000, 1) if max_axle_kg else None,
                    "vehicle_class": _classify_weight(max_weight_kg) if max_weight_kg else "unknown",
                },
            }
            enriched.append(enriched_feature)

            if silta_type == 0:
                bridge_count += 1
            elif silta_type == 1:
                underpass_count += 1
            elif silta_type == -1:
                tunnel_count += 1

            if max_weight_kg:
                cls = _classify_weight(max_weight_kg)
                weight_dist[cls] = weight_dist.get(cls, 0) + 1

            if max_height_cm and max_height_cm < 400:
                height_restricted.append(lid)

        findings = [
            f"Total bridge/tunnel structures: {len(bridge_features)} ({bridge_count} bridges, {underpass_count} underpasses, {tunnel_count} tunnels)",
        ]

        if weight_dist:
            parts = sorted(weight_dist.items(), key=lambda x: VEHICLE_CLASSES.index(next(vc for vc in VEHICLE_CLASSES if vc[0] == x[0])) if any(vc[0] == x[0] for vc in VEHICLE_CLASSES) else 99)
            findings.append("Weight capacity distribution:")
            for cls, cnt in parts:
                findings.append(f"  - {cls}: {cnt} structures")

        if height_restricted:
            findings.append(f"{len(height_restricted)} structures have height < 4.0m (may restrict military vehicles)")

        total_matched = collections.get("dr_tielinkki_silta_alikulku_tunneli", {}).get("number_matched", 0)
        findings.append(f"Total features in area: {total_matched} (sample: {len(bridge_features)} returned)")

        low_cap = sum(1 for e in enriched if e["properties"].get("max_weight_tonnes") and e["properties"]["max_weight_tonnes"] < 20)
        high_cap = sum(1 for e in enriched if e["properties"].get("max_weight_tonnes") and e["properties"]["max_weight_tonnes"] >= 60)
        findings.append(f"Route assessment: {low_cap} low-capacity (<20t), {high_cap} high-capacity (≥60t) structures")

        return AgentRunResult(
            agent_id=self.agent_id,
            area=area,
            timeframe=timeframe,
            summary=f"Bridge load capacity analysis for {area}: {bridge_count} bridges, {underpass_count} underpasses, {tunnel_count} tunnels analyzed",
            findings=findings,
            data={
                "enriched_features": enriched[:500],
                "total_features_in_area": total_matched,
            },
        )
