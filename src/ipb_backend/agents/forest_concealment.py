from __future__ import annotations

from ipb_backend.agents.base import AnalysisAgent
from ipb_backend.ingestion.sources.osm_poi import OsmPoiAdapter
from ipb_backend.models import AgentRunResult

LEAF_TYPE_LABELS = {"needleleaved": "coniferous", "broadleaved": "deciduous", "mixed": "mixed"}
LEAF_CYCLE_LABELS = {"evergreen": "year-round", "deciduous": "seasonal", "mixed": "mixed"}

CONCEALMENT_RATINGS: dict[str, str] = {
    "coniferous, year-round": "high",
    "coniferous, seasonal": "medium",
    "mixed": "medium",
    "deciduous, year-round": "medium",
    "deciduous, seasonal": "low",
}


class ForestConcealmentAgent(AnalysisAgent):
    agent_id = "forest-concealment-agent"

    def __init__(self, adapter: OsmPoiAdapter) -> None:
        self.adapter = adapter

    async def run(self, area: str, timeframe: str) -> AgentRunResult:
        record = await self.adapter.fetch(area=area, timeframe=timeframe)
        categories = record.data.get("categories", {})
        forest_features = categories.get("forest", [])

        if not forest_features:
            return AgentRunResult(
                agent_id=self.agent_id,
                area=area,
                timeframe=timeframe,
                summary=f"Forest concealment analysis for {area}: no forest data available",
                findings=["No forest features found in this area"],
            )

        by_type: dict[str, int] = {}
        by_rating: dict[str, int] = {}
        for feat in forest_features:
            tags = feat.get("tags", {})
            lt = LEAF_TYPE_LABELS.get(tags.get("leaf_type", ""), "unknown")
            lc = LEAF_CYCLE_LABELS.get(tags.get("leaf_cycle", ""), "") if tags.get("leaf_cycle") else "year-round"
            natural = tags.get("natural", tags.get("landuse", "unknown"))
            key = f"{lt}, {lc}" if lc else lt
            by_type[key] = by_type.get(key, 0) + 1
            rating = CONCEALMENT_RATINGS.get(key, "low")
            by_rating[rating] = by_rating.get(rating, 0) + 1

        total = len(forest_features)
        findings = [
            f"Total forest/woodland features: {total}",
            f"Vegetation type breakdown:",
        ]
        for key, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
            findings.append(f"  - {key}: {cnt} ({cnt/total*100:.1f}%)")

        concealment = f"Concealment rating: "
        parts = []
        for rating in ["high", "medium", "low"]:
            cnt = by_rating.get(rating, 0)
            if cnt:
                parts.append(f"{rating}: {cnt} ({cnt/total*100:.1f}%)")
        findings.append(concealment + ", ".join(parts))

        high = by_rating.get("high", 0)
        if high:
            findings.append(f"{high} features provide year-round canopy cover (high concealment)")

        return AgentRunResult(
            agent_id=self.agent_id,
            area=area,
            timeframe=timeframe,
            summary=f"Forest concealment analysis for {area}: {total} features, {by_rating.get('high', 0)} high-concealment",
            findings=findings,
        )
