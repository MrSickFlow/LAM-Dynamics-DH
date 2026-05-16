from __future__ import annotations

from ipb_backend.agents.base import AnalysisAgent
from ipb_backend.ingestion.sources.opencellid import OpenCellIdAdapter
from ipb_backend.models import AgentRunResult


class CellTowerAgent(AnalysisAgent):
    agent_id = "celltower-agent"

    def __init__(self, adapter: OpenCellIdAdapter) -> None:
        self.adapter = adapter

    async def run(self, area: str, timeframe: str) -> AgentRunResult:
        record = await self.adapter.fetch(area=area, timeframe=timeframe)
        cells = record.data.get("cells", [])
        total = record.data.get("total_cells", 0)

        operators: dict[str, int] = {}
        technologies: dict[str, int] = {}
        for cell in cells:
            mcc = cell.get("mcc", "unknown")
            mnc = cell.get("mnc", "unknown")
            net = f"{mcc}-{mnc}"
            operators[net] = operators.get(net, 0) + 1
            radio = cell.get("radio", "unknown")
            technologies[radio] = technologies.get(radio, 0) + 1

        findings = [f"Total cell towers: {total}"]
        if operators:
            findings.append(
                f"Operators: {', '.join(f'{k}={v}' for k, v in sorted(operators.items()))}"
            )
        if technologies:
            findings.append(
                f"Technologies: {', '.join(f'{k}={v}' for k, v in sorted(technologies.items()))}"
            )

        return AgentRunResult(
            agent_id=self.agent_id,
            area=area,
            timeframe=timeframe,
            summary=f"Cell tower analysis for {area}: {total} towers found",
            findings=findings,
        )
