from __future__ import annotations

import re
import unicodedata
from typing import Any

import httpx

from ipb_backend.ingestion.base import SourceAdapter
from ipb_backend.models import DatasetRecord

AREA_MUNICIPALITIES: dict[str, list[str]] = {
    "north karelia": ["KU167", "KU176", "KU260", "KU422", "KU426", "KU276"],
    "archipelago sea": ["KU445"],
    "lapland": ["KU698"],
    "lapland (kasivarren lappi)": ["KU698"],
    "kasivarren lappi": ["KU698"],
}

MUNICIPALITY_LABELS: dict[str, str] = {
    "KU167": "Joensuu",
    "KU176": "Juuka",
    "KU260": "Kitee",
    "KU422": "Lieksa",
    "KU426": "Liperi",
    "KU276": "Kontiolahti",
    "KU445": "Parainen",
    "KU698": "Rovaniemi",
}


class StatisticsFinlandAdapter(SourceAdapter):
    BASE_URL = "https://pxdata.stat.fi/PXWeb/api/v1/en/StatFin"
    TABLE_PATH = "vaerak/statfin_vaerak_pxt_11re.px"

    def _resolve_municipalities(self, area: str) -> list[str]:
        normalized = self._normalize_area(area)
        return AREA_MUNICIPALITIES.get(normalized, AREA_MUNICIPALITIES["north karelia"])

    async def fetch(self, area: str, timeframe: str) -> DatasetRecord:
        municipalities = self._resolve_municipalities(area)
        latest_year = "2024"

        async with httpx.AsyncClient(timeout=30.0) as client:
            query_total: dict[str, Any] = {
                "query": [
                    {"code": "Alue", "selection": {"filter": "item", "values": municipalities}},
                    {"code": "Ik\u00e4", "selection": {"filter": "item", "values": ["SSS"]}},
                    {"code": "Sukupuoli", "selection": {"filter": "item", "values": ["SSS", "1", "2"]}},
                    {"code": "Vuosi", "selection": {"filter": "item", "values": [latest_year]}},
                    {"code": "Tiedot", "selection": {"filter": "item", "values": ["vaesto"]}},
                ],
                "response": {"format": "json-stat2"},
            }
            resp_total = await client.post(f"{self.BASE_URL}/{self.TABLE_PATH}", json=query_total)
            resp_total.raise_for_status()
            total_data = resp_total.json()

            all_ages = ["SSS"] + [f"{i:03d}" for i in range(100)] + ["100-"]
            query_age: dict[str, Any] = {
                "query": [
                    {"code": "Alue", "selection": {"filter": "item", "values": municipalities}},
                    {"code": "Ik\u00e4", "selection": {"filter": "item", "values": all_ages}},
                    {"code": "Sukupuoli", "selection": {"filter": "item", "values": ["SSS"]}},
                    {"code": "Vuosi", "selection": {"filter": "item", "values": [latest_year]}},
                    {"code": "Tiedot", "selection": {"filter": "item", "values": ["vaesto"]}},
                ],
                "response": {"format": "json-stat2"},
            }
            resp_age = await client.post(f"{self.BASE_URL}/{self.TABLE_PATH}", json=query_age)
            resp_age.raise_for_status()
            age_data = resp_age.json()

        pop_data = self._extract_population(total_data, municipalities)
        age_dist = self._extract_age_distribution(age_data, all_ages, municipalities)
        pop_data["age_distribution"] = age_dist

        municipality_names = [MUNICIPALITY_LABELS.get(m, m) for m in municipalities]
        summary = (
            f"Statistics Finland population data for {area}: "
            f"{pop_data['total']:,} inhabitants ({', '.join(municipality_names)}, {latest_year})"
        )

        return DatasetRecord(
            source_id=self.definition.source_id,
            category=self.definition.category,
            area=area,
            timeframe=timeframe,
            summary=summary,
            data={
                "provider": "Statistics Finland (Tilastokeskus)",
                "api": "PxWeb API",
                "license": "CC BY 4.0",
                "query": {
                    "area": area,
                    "municipalities": municipalities,
                    "year": latest_year,
                },
                **pop_data,
            },
        )

    def _extract_population(self, raw: dict[str, Any], municipalities: list[str]) -> dict[str, Any]:
        values = raw.get("value", [])
        n_muni = len(municipalities)
        stride = values_per_muni(raw, n_muni)

        total = 0
        male = 0
        female = 0
        per_muni: dict[str, dict[str, int]] = {}

        for i in range(n_muni):
            base = i * stride
            t = int(values[base]) if base < len(values) else 0
            m = int(values[base + 1]) if base + 1 < len(values) else 0
            f = int(values[base + 2]) if base + 2 < len(values) else 0
            total += t
            male += m
            female += f
            code = municipalities[i]
            per_muni[code] = {
                "name": MUNICIPALITY_LABELS.get(code, code),
                "total": t,
                "male": m,
                "female": f,
            }

        return {
            "total": total,
            "male": male,
            "female": female,
            "per_municipality": per_muni,
        }

    def _extract_age_distribution(
        self, raw: dict[str, Any], all_ages: list[str], municipalities: list[str]
    ) -> dict[str, Any]:
        values = raw.get("value", [])
        n_muni = len(municipalities)
        n_ages = len(all_ages)

        age_groups: dict[str, int] = {"0-14": 0, "15-64": 0, "65+": 0}

        for m in range(n_muni):
            for a in range(1, n_ages):
                idx = m * n_ages + a
                if idx >= len(values):
                    continue
                pop_val = values[idx]
                if pop_val is None:
                    continue
                age_key = all_ages[a]
                age_val: int | None = None
                if age_key == "100-":
                    age_val = 100
                else:
                    try:
                        age_val = int(age_key)
                    except ValueError:
                        continue
                if age_val is None:
                    continue
                if age_val <= 14:
                    age_groups["0-14"] += pop_val
                elif age_val <= 64:
                    age_groups["15-64"] += pop_val
                else:
                    age_groups["65+"] += pop_val

        total_known = sum(age_groups.values())
        return {
            "groups": age_groups,
            "total_grouped": total_known,
        }

    def _normalize_area(self, area: str) -> str:
        ascii_area = unicodedata.normalize("NFKD", area).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", ascii_area).strip().lower()


def values_per_muni(raw: dict[str, Any], n_muni: int) -> int:
    sizes = raw.get("size", [])
    size_ika = sizes[1] if len(sizes) > 1 else 1
    size_sukupuoli = sizes[2] if len(sizes) > 2 else 1
    size_vuosi = sizes[3] if len(sizes) > 3 else 1
    size_tiedot = sizes[4] if len(sizes) > 4 else 1
    return size_ika * size_sukupuoli * size_vuosi * size_tiedot
