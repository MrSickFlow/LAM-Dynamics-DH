from __future__ import annotations

from abc import ABC, abstractmethod

from ipb_backend.models import AgentRunResult


class AnalysisAgent(ABC):
    agent_id: str

    @abstractmethod
    async def run(self, area: str, timeframe: str) -> AgentRunResult:
        """Run derived analysis for the selected area and timeframe."""
