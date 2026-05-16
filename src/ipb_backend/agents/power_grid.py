from __future__ import annotations

from ipb_backend.agents.base import AnalysisAgent
from ipb_backend.ingestion.sources.nls import NationalLandSurveyAdapter
from ipb_backend.models import AgentRunResult


class PowerGridAgent(AnalysisAgent):
    agent_id = "power-grid-agent"

    def __init__(self, adapter: NationalLandSurveyAdapter) -> None:
        self.adapter = adapter

    async def run(self, area: str, timeframe: str) -> AgentRunResult:
        record = await self.adapter.fetch(area=area, timeframe=timeframe)
        collections = record.data.get("collections", {})

        power_lines = collections.get("sahkolinja", {}).get("features", [])
        total_matched = collections.get("sahkolinja", {}).get("number_matched", 0)

        if not power_lines:
            return AgentRunResult(
                agent_id=self.agent_id,
                area=area,
                timeframe=timeframe,
                summary=f"Power grid analysis for {area}: no power line data available",
                findings=["No power line features found in this area"],
            )

        total_km = 0.0
        for feat in power_lines:
            coords = feat.get("geometry", {}).get("coordinates", [])
            if len(coords) > 1:
                from math import radians, cos, sin, sqrt, atan2
                seg_km = 0
                for i in range(len(coords) - 1):
                    lon1, lat1 = coords[i][0], coords[i][1]
                    lon2, lat2 = coords[i + 1][0], coords[i + 1][1]
                    dlat = radians(lat2 - lat1)
                    dlon = radians(lon2 - lon1)
                    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
                    c = 2 * atan2(sqrt(a), sqrt(1 - a))
                    seg_km += 6371.0 * c
                total_km += seg_km

        findings = [
            f"Total power line features in area: {len(power_lines)} (of {total_matched} matched)",
            f"Estimated total power line length: {total_km:.1f} km",
        ]

        if total_km > 100:
            findings.append("  - Extensive power grid present — multiple chokepoint options for disruption")
        elif total_km > 20:
            findings.append("  - Moderate power grid — limited chokepoints, localized disruption possible")
        else:
            findings.append("  - Sparse power grid — minimal infrastructure, limited disruption value")

        return AgentRunResult(
            agent_id=self.agent_id,
            area=area,
            timeframe=timeframe,
            summary=f"Power grid analysis for {area}: {len(power_lines)} line segments, {total_km:.1f} km total",
            findings=findings,
        )
