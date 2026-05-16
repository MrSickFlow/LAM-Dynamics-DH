from __future__ import annotations

from ipb_backend.agents.base import AnalysisAgent
from ipb_backend.models import AgentRunResult


class SummaryAgent(AnalysisAgent):
    agent_id = "summary-agent"

    async def run(self, area: str, timeframe: str) -> AgentRunResult:
        return AgentRunResult(
            agent_id=self.agent_id,
            area=area,
            timeframe=timeframe,
            summary="Placeholder analysis output.",
            findings=[
                "Terrain, infrastructure, and weather fusion is not implemented yet.",
                "This endpoint defines the contract for future derived analysis.",
            ],
        )
