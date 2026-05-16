from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional

import httpx

from ipb_backend.ingestion.base import SourceAdapter
from ipb_backend.models import DatasetRecord
from ipb_backend.spatial import format_bbox, resolve_area_bbox

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

AREA_POPULATION_CENTERS = {
    "north karelia": [
        (0.31, 0.40, 1.0, 0.11),
        (0.43, 0.83, 0.28, 0.12),
        (0.44, 0.12, 0.24, 0.10),
    ],
    "archipelago sea": [
        (0.44, 0.48, 0.85, 0.14),
        (0.68, 0.42, 0.30, 0.10),
    ],
    "lapland": [
        (0.48, 0.34, 0.95, 0.13),
        (0.30, 0.58, 0.35, 0.12),
        (0.68, 0.62, 0.30, 0.12),
    ],
    "lapland (kasivarren lappi)": [
        (0.48, 0.34, 0.95, 0.13),
        (0.30, 0.58, 0.35, 0.12),
        (0.68, 0.62, 0.30, 0.12),
    ],
    "kasivarren lappi": [
        (0.48, 0.34, 0.95, 0.13),
        (0.30, 0.58, 0.35, 0.12),
        (0.68, 0.62, 0.30, 0.12),
    ],
}


URBAN_RURAL_LABELS = {
    "SSS": "Total",
    "KS": "Urban areas",
    "K1": "Inner urban area",
    "K2": "Outer urban area",
    "K3": "Peri-urban area",
    "MS": "Rural areas",
    "M4": "Local centres in rural areas",
    "M5": "Rural areas close to urban areas",
    "M6": "Rural heartland areas",
    "M7": "Sparsely populated rural areas",
    "X": "Unknown",
}

URBAN_RURAL_CODES = list(URBAN_RURAL_LABELS.keys())


class StatisticsFinlandAdapter(SourceAdapter):
    BASE_URL = "https://pxdata.stat.fi/PXWeb/api/v1/en/StatFin"
    TABLE_PATH = "vaerak/statfin_vaerak_pxt_11re.px"
    URBAN_RURAL_TABLE = "vaerak/statfin_vaerak_pxt_11s3.px"

    def _resolve_municipalities(self, area: str) -> list[str]:
        normalized = self._normalize_area(area)
        return AREA_MUNICIPALITIES.get(normalized, AREA_MUNICIPALITIES["north karelia"])

    async def fetch(self, area: str, timeframe: str) -> DatasetRecord:
        municipalities = self._resolve_municipalities(area)
        latest_year = "2024"
        bbox = resolve_area_bbox(area)

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

            urban_rural = await self._fetch_urban_rural(client, municipalities, latest_year)

        pop_data = self._extract_population(total_data, municipalities)
        age_dist = self._extract_age_distribution(age_data, all_ages, municipalities)
        pop_data["age_distribution"] = age_dist
        features = self._build_features(area, bbox, pop_data["total"])
        pop_data["population_total"] = pop_data["total"]
        pop_data["urban_rural"] = urban_rural

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
                    "bbox_wgs84": format_bbox(bbox),
                },
                "note": "Live municipal totals with a derived population grid for AOI clipping and map-side inspection.",
                "features": features,
                **pop_data,
            },
        )

    async def _fetch_urban_rural(self, client: httpx.AsyncClient, municipalities: list[str], year: str) -> dict:
        query = {
            "query": [
                {"code": "Alue", "selection": {"filter": "item", "values": municipalities}},
                {"code": "Kaupunki-maaseutu-luokitus", "selection": {"filter": "item", "values": URBAN_RURAL_CODES}},
                {"code": "Sukupuoli", "selection": {"filter": "item", "values": ["SSS"]}},
                {"code": "Ikä", "selection": {"filter": "item", "values": ["SSS"]}},
                {"code": "Vuosi", "selection": {"filter": "item", "values": [year]}},
                {"code": "Tiedot", "selection": {"filter": "item", "values": ["vaesto"]}},
            ],
            "response": {"format": "json-stat2"},
        }
        resp = await client.post(f"{self.BASE_URL}/{self.URBAN_RURAL_TABLE}", json=query)
        resp.raise_for_status()
        data = resp.json()

        values = data.get("value", [])
        n_classes = len(URBAN_RURAL_CODES)
        per_muni: dict[str, dict[str, int]] = {}
        totals: dict[str, int] = {}

        for m_idx, muni_code in enumerate(municipalities):
            base = m_idx * n_classes
            muni_data: dict[str, int] = {}
            for c_idx, class_code in enumerate(URBAN_RURAL_CODES):
                idx = base + c_idx
                pop_val = int(values[idx]) if idx < len(values) and values[idx] is not None else 0
                muni_data[class_code] = pop_val
                totals[class_code] = totals.get(class_code, 0) + pop_val
            per_muni[muni_code] = {
                "name": MUNICIPALITY_LABELS.get(muni_code, muni_code),
                "classes": {URBAN_RURAL_LABELS[k]: v for k, v in muni_data.items()},
            }

        return {
            "per_municipality": per_muni,
            "total_by_class": {URBAN_RURAL_LABELS[k]: v for k, v in totals.items()},
        }

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
            "per_muni": per_muni,
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
                age_val: Optional[int] = None
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

    def _build_features(
        self,
        area: str,
        bbox: tuple[float, float, float, float],
        target_population: int,
    ) -> list[dict]:
        min_x, min_y, max_x, max_y = bbox
        width = max_x - min_x
        height = max_y - min_y
        area_name = self._normalize_area(area)
        centers = AREA_POPULATION_CENTERS.get(area_name, AREA_POPULATION_CENTERS["north karelia"])
        columns = 10
        rows = 10

        def ring(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
            return [[
                min_x + width * x0,
                min_y + height * y0,
            ], [
                min_x + width * x1,
                min_y + height * y0,
            ], [
                min_x + width * x1,
                min_y + height * y1,
            ], [
                min_x + width * x0,
                min_y + height * y1,
            ], [
                min_x + width * x0,
                min_y + height * y0,
            ]]

        weighted_cells: list[dict[str, Any]] = []
        total_weight = 0.0

        for row in range(rows):
            for column in range(columns):
                x0 = column / columns
                x1 = (column + 1) / columns
                y0 = row / rows
                y1 = (row + 1) / rows
                center_x = (x0 + x1) / 2
                center_y = (y0 + y1) / 2

                weight = 0.08 + (0.12 * (1.0 - center_y))
                for hotspot_x, hotspot_y, scale, spread in centers:
                    distance_sq = (center_x - hotspot_x) ** 2 + (center_y - hotspot_y) ** 2
                    weight += scale * math.exp(-(distance_sq / (2 * spread * spread)))

                weighted_cells.append(
                    {
                        "row": row,
                        "column": column,
                        "x0": x0,
                        "x1": x1,
                        "y0": y0,
                        "y1": y1,
                        "weight": weight,
                    }
                )
                total_weight += weight

        features: list[dict] = []
        assigned_population = 0
        for index, cell in enumerate(weighted_cells):
            share = float(cell["weight"]) / total_weight if total_weight else 0.0
            urbanity = share * len(weighted_cells)
            population = max(1, round(target_population * share)) if target_population > 0 else 0
            median_age = round(46 - min(urbanity * 4.5, 8.0) + (int(cell["row"]) / rows) * 2.5)
            assigned_population += population
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [ring(float(cell["x0"]), float(cell["y0"]), float(cell["x1"]), float(cell["y1"]))],
                    },
                    "properties": {
                        "cell_id": f"pop-{index + 1}",
                        "population": population,
                        "median_age": median_age,
                    },
                }
            )

        drift = target_population - assigned_population
        if drift and features:
            peak_feature = max(features, key=lambda feature: feature["properties"]["population"])
            peak_feature["properties"]["population"] += drift

        return features


def values_per_muni(raw: dict[str, Any], n_muni: int) -> int:
    sizes = raw.get("size", [])
    size_ika = sizes[1] if len(sizes) > 1 else 1
    size_sukupuoli = sizes[2] if len(sizes) > 2 else 1
    size_vuosi = sizes[3] if len(sizes) > 3 else 1
    size_tiedot = sizes[4] if len(sizes) > 4 else 1
    return size_ika * size_sukupuoli * size_vuosi * size_tiedot
