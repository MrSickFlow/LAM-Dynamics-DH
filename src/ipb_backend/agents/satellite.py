from __future__ import annotations

from ipb_backend.agents.base import AnalysisAgent
from ipb_backend.ingestion.sources.satellites import SatelliteTleAdapter
from ipb_backend.models import AgentRunResult


class SatelliteAgent(AnalysisAgent):
    agent_id = "satellite-agent"

    def __init__(self, adapter: SatelliteTleAdapter) -> None:
        self.adapter = adapter

    async def run(self, area: str, timeframe: str) -> AgentRunResult:
        record = await self.adapter.fetch(area=area, timeframe=timeframe)
        sats = record.data.get("satellites", {})
        total = record.data.get("total_tracked", 0)

        optical = []
        sar = []
        multispectral = []
        for name, info in sats.items():
            stype = info.get("type", "")
            passes = info.get("predicted_passes", [])
            if "optical" in stype.lower() or "commercial imaging" in stype.lower():
                optical.append({"name": name, "passes": len(passes), "type": stype})
            elif "sar" in stype.lower():
                sar.append({"name": name, "passes": len(passes), "type": stype})
            elif "multispectral" in stype.lower():
                multispectral.append({"name": name, "passes": len(passes), "type": stype})

        findings = [
            f"Total reconnaissance/imaging satellites tracked: {total}",
            f"Optical imaging: {len(optical)} satellites",
            f"SAR imaging: {len(sar)} satellites (all-weather, day/night)",
            f"Multispectral: {len(multispectral)} satellites",
        ]

        if optical:
            next_optical = optical[0]
            next_pass = sats.get(next_optical["name"], {}).get("predicted_passes", [])
            if next_pass:
                findings.append(f"Next optical pass: {next_optical['name']} at {next_pass[0]['pass_time_utc']}")
        if sar:
            next_sar = sar[0]
            findings.append(f"Next SAR pass: {next_sar['name']}")

        upcoming = []
        for name, info in sorted(sats.items()):
            for p in info.get("predicted_passes", [])[:2]:
                upcoming.append({
                    "satellite": name,
                    "type": info.get("type", ""),
                    "time_utc": p["pass_time_utc"],
                    "altitude_km": p.get("altitude_km"),
                })
        upcoming.sort(key=lambda x: x["time_utc"])

        if upcoming:
            findings.append(f"Next overpass overall: {upcoming[0]['satellite']} at {upcoming[0]['time_utc']}")

        return AgentRunResult(
            agent_id=self.agent_id,
            area=area,
            timeframe=timeframe,
            summary=f"Satellite overpass analysis for {area}: {total} imaging satellites tracked",
            findings=findings,
        )
