from __future__ import annotations

from ipb_backend.agents.base import AnalysisAgent
from ipb_backend.ingestion.sources.fmi import FmiAdapter
from ipb_backend.models import AgentRunResult


class WeatherImpactAgent(AnalysisAgent):
    agent_id = "weather-impact-agent"

    def __init__(self, adapter: FmiAdapter) -> None:
        self.adapter = adapter

    async def run(self, area: str, timeframe: str) -> AgentRunResult:
        record = await self.adapter.fetch(area=area, timeframe=timeframe)
        obs = record.data.get("observations", {})
        station = record.data.get("station", {})

        if not obs:
            return AgentRunResult(
                agent_id=self.agent_id,
                area=area,
                timeframe=timeframe,
                summary=f"Weather impact analysis for {area}: no data available",
                findings=["No weather data found for this area"],
            )

        temp = obs.get("temperature", {}).get("latest", {}).get("value")
        wind = obs.get("wind_speed", {}).get("latest", {}).get("value")
        wind_gust = obs.get("wind_gust", {}).get("latest", {}).get("value")
        humidity = obs.get("humidity", {}).get("latest", {}).get("value")
        pressure = obs.get("pressure", {}).get("latest", {}).get("value")
        precip = obs.get("precipitation", {}).get("latest", {}).get("value")
        cloud = obs.get("cloud_cover", {}).get("latest", {}).get("value")

        findings: list[str] = [
            f"Station: {station.get('name', 'unknown')} ({station.get('region', '')})",
            f"Temperature: {temp}°C" if temp is not None else "Temperature: N/A",
        ]

        if temp is not None:
            if temp < -15:
                findings.append("  - Impact: Extreme cold — risk of equipment failure, personnel frostbite")
            elif temp < 0:
                findings.append("  - Impact: Sub-zero — icy roads, reduced traction for vehicles")
            elif temp > 30:
                findings.append("  - Impact: Heat — equipment cooling concerns, personnel dehydration risk")
            else:
                findings.append("  - Impact: Moderate temperatures — no significant thermal impact")

        if wind is not None:
            findings.append(f"Wind: {wind} m/s" + (f" (gusts {wind_gust} m/s)" if wind_gust else ""))
            if wind > 15:
                findings.append("  - Impact: HIGH — drone operations severely restricted, artillery dispersion increases")
            elif wind > 10:
                findings.append("  - Impact: MODERATE — small UAV operations limited, paratrooper drop risk")
            elif wind > 5:
                findings.append("  - Impact: LOW — minor effect on small drones, crosswind on bridges")
            else:
                findings.append("  - Impact: MINIMAL — favorable for all air operations")

        if precip is not None and precip > 0:
            findings.append(f"Precipitation: {precip} mm")
            if precip > 10:
                findings.append("  - Impact: Heavy rain — visibility reduced, off-road mobility degraded, river levels rising")
            elif precip > 2:
                findings.append("  - Impact: Moderate rain — reduced visibility, wet ground slows dismounted movement")

        if cloud is not None:
            findings.append(f"Cloud cover: {cloud}%")
            if cloud > 80:
                findings.append("  - Impact: Overcast — limits satellite optical surveillance, favorable for covert movement")
            elif cloud > 50:
                findings.append("  - Impact: Partly cloudy — intermittent satellite surveillance windows")
            else:
                findings.append("  - Impact: Clear — favorable for satellite and aerial surveillance")

        if humidity is not None and humidity > 90:
            findings.append(f"Humidity: {humidity}% — fog/mist likely, reduced visibility, thermal imaging degraded")

        return AgentRunResult(
            agent_id=self.agent_id,
            area=area,
            timeframe=timeframe,
            summary=f"Weather impact analysis for {area}: {temp}°C, wind {wind} m/s" if temp is not None and wind is not None else f"Weather impact analysis for {area}",
            findings=findings,
        )
