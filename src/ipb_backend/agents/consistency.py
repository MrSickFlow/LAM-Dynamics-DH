from __future__ import annotations

from ipb_backend.agents.base import AnalysisAgent
from ipb_backend.consistency.engine import DataConsistencyEngine
from ipb_backend.ingestion.registry import SourceRegistry
from ipb_backend.ingestion.service import IngestionService
from ipb_backend.models import AgentRunResult


class ConsistencyEngineAgent(AnalysisAgent):
    agent_id = "consistency-engine-agent"

    def __init__(
        self,
        engine: DataConsistencyEngine,
        ingestion_service: IngestionService,
        registry: SourceRegistry,
    ) -> None:
        self.engine = engine
        self.ingestion_service = ingestion_service
        self.registry = registry

    async def run(self, area: str, timeframe: str) -> AgentRunResult:
        records = _latest_records_by_source(self.ingestion_service.records)
        report = await self.engine.evaluate(
            area=area,
            timeframe=timeframe,
            records=records,
            sources=self.registry.list_sources(),
        )
        findings = [report.summary]
        if report.ew_pattern_detected:
            findings.append("EW pattern heuristic: spatial clustering or multiple high-severity anomalies detected.")
        for layer in sorted(report.layer_trust, key=lambda item: item.confidence):
            findings.append(
                f"Trust {layer.source_id}: {layer.confidence:.0%} ({layer.ew_classification.value}, "
                f"{'; '.join(layer.factors[:2])})"
            )
        for anomaly in report.anomalies[:8]:
            demo = " [demo]" if anomaly.synthetic_demo else ""
            findings.append(f"[{anomaly.severity.value}] {anomaly.title}{demo}")
        if len(report.anomalies) > 8:
            findings.append(f"... and {len(report.anomalies) - 8} more anomalies")

        return AgentRunResult(
            agent_id=self.agent_id,
            area=area,
            timeframe=timeframe,
            summary=report.summary,
            findings=findings,
            data=report.model_dump(mode="json"),
        )


def _latest_records_by_source(records):
    latest = {}
    for record in records:
        current = latest.get(record.source_id)
        if current is None or record.retrieved_at > current.retrieved_at:
            latest[record.source_id] = record
    return latest
