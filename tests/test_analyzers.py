import asyncio
import json

from ipb_backend.analysis.analyzers import OllamaAnalyzer


def test_ollama_analyzer_parses_structured_json(monkeypatch):
    async def fake_generate_payload(self, prompt):
        assert "Return strict JSON" in prompt
        return {
            "response": json.dumps(
                {
                    "summary": "AOI overlaps one transport corridor and sparse population.",
                    "findings": [
                        "One transport feature intersects the AOI.",
                        "Population cells sum to 210 residents.",
                    ],
                    "limitations": ["Weather station coverage is absent inside the AOI."],
                }
            )
        }

    monkeypatch.setattr(OllamaAnalyzer, "_generate_payload", fake_generate_payload)

    result = asyncio.run(
        OllamaAnalyzer().analyze(
            selection={"bounds": [30.0, 62.4, 30.6, 62.85]},
            metrics={
                "selection_area_sqkm": 340.4,
                "nls_feature_count": 2,
                "digiroad_feature_count": 1,
                "population_total": 210,
                "weather_station_count": 0,
            },
            raw_data={},
            freshness=[],
            evidence_bundle=[{"source_id": "digiroad", "detail": "1 feature", "support": "demo"}],
        )
    )

    assert result["provider"] == "ollama"
    assert result["status"] == "ready"
    assert result["summary"] == "AOI overlaps one transport corridor and sparse population."
    assert result["findings"][0] == "One transport feature intersects the AOI."
    assert result["limitations"] == ["Weather station coverage is absent inside the AOI."]
    assert result["output"]["profile"] == "general"
    assert result["output"]["implications"]


def test_ollama_analyzer_falls_back_to_plain_text(monkeypatch):
    async def fake_generate_payload(self, prompt):
        return {
            "response": "- First grounded finding\n- Second grounded finding"
        }

    monkeypatch.setattr(OllamaAnalyzer, "_generate_payload", fake_generate_payload)

    result = asyncio.run(
        OllamaAnalyzer().analyze(
            selection={"bounds": [30.0, 62.4, 30.6, 62.85]},
            metrics={
                "selection_area_sqkm": 340.4,
                "nls_feature_count": 2,
                "digiroad_feature_count": 1,
                "population_total": 210,
                "weather_station_count": 0,
            },
            raw_data={},
            freshness=[],
            evidence_bundle=[],
        )
    )

    assert result["summary"] == "First grounded finding"
    assert result["findings"] == ["First grounded finding", "Second grounded finding"]
    assert result["output"]["summary"] == "First grounded finding"