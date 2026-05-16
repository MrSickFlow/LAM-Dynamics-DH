from __future__ import annotations

from ipb_backend.agents.base import AnalysisAgent
from ipb_backend.ingestion.sources.statistics_finland import StatisticsFinlandAdapter
from ipb_backend.models import AgentRunResult

URBAN_RURAL_ORDER = ["Total", "Urban areas", "Inner urban area", "Outer urban area", "Peri-urban area", "Rural areas", "Local centres in rural areas", "Rural areas close to urban areas", "Rural heartland areas", "Sparsely populated rural areas", "Unknown"]


class DemographicsAgent(AnalysisAgent):
    agent_id = "demographics-agent"

    def __init__(self, adapter: StatisticsFinlandAdapter) -> None:
        self.adapter = adapter

    async def run(self, area: str, timeframe: str) -> AgentRunResult:
        record = await self.adapter.fetch(area=area, timeframe=timeframe)
        data = record.data

        total = data.get("total", 0)
        male = data.get("male", 0)
        female = data.get("female", 0)
        age_dist = data.get("age_distribution", {}).get("groups", {})
        urban_rural = data.get("urban_rural", {})

        findings: list[str] = [
            f"Total population: {total:,}",
            f"Sex distribution: {male:,} male ({male/total*100:.1f}%), {female:,} female ({female/total*100:.1f}%)",
        ]

        if age_dist:
            yng = age_dist.get("0-14", 0)
            mid = age_dist.get("15-64", 0)
            old = age_dist.get("65+", 0)
            findings.append(f"Age distribution: 0-14: {yng:,} ({yng/total*100:.1f}%), 15-64: {mid:,} ({mid/total*100:.1f}%), 65+: {old:,} ({old/total*100:.1f}%)")

        if urban_rural:
            by_class = urban_rural.get("total_by_class", {})
            total_classified = sum(v for k, v in by_class.items() if k != "Total" and k != "Unknown")
            urban_pop = by_class.get("Urban areas", 0)
            rural_pop = by_class.get("Rural areas", 0)
            if total_classified > 0:
                findings.append(f"Urban/rural split: {urban_pop:,} urban ({urban_pop/total_classified*100:.1f}%), {rural_pop:,} rural ({rural_pop/total_classified*100:.1f}%)")
            for cls in URBAN_RURAL_ORDER:
                val = by_class.get(cls)
                if val and val > 0:
                    findings.append(f"  - {cls}: {val:,} ({val/total_classified*100:.1f}%)")

            per_muni = urban_rural.get("per_municipality", {})
            for muni_code, muni_data in sorted(per_muni.items()):
                name = muni_data.get("name", muni_code)
                classes = muni_data.get("classes", {})
                muni_total = classes.get("Total", 0)
                muni_urban = classes.get("Urban areas", 0)
                muni_rural = classes.get("Rural areas", 0)
                if muni_total:
                    findings.append(f"{name}: {muni_total:,} total, {muni_urban:,} urban ({muni_urban/muni_total*100:.1f}%), {muni_rural:,} rural ({muni_rural/muni_total*100:.1f}%)")

        return AgentRunResult(
            agent_id=self.agent_id,
            area=area,
            timeframe=timeframe,
            summary=f"Demographics analysis for {area}: {total:,} inhabitants, {urban_rural.get('total_by_class', {}).get('Urban areas', 0):,} urban / {urban_rural.get('total_by_class', {}).get('Rural areas', 0):,} rural",
            findings=findings,
        )
