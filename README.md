# IPB Open Source Backend

Automates intelligence preparation of the battlefield (IPB) using Finnish open data — built for the AI2PB challenge by 61N Solutions at Defence Hackathon 2026.

Given a geographic area and timeframe, the tool retrieves, processes, and surfaces operationally relevant open-source data via API and interactive map UI.

## Data sources

| Source | Data | API |
|---|---|---|
| [NLS](https://maanmittauslaitos.fi/en) | Topography, land cover, water bodies, elevation contours, power lines, admin boundaries | OGC API Features |
| [FMI](https://en.ilmatieteenlaitos.fi/open-data) | Weather observations + forecasts (temp, wind, humidity, pressure, precipitation, gusts, cloud) | WFS |
| [Digiroad](https://vayla.fi/en) | Road network, bridges, tunnels, weight/height/width limits, speed limits, pavement, traffic | OGC API Features |
| [Statistics Finland](https://stat.fi) | Population, age/sex distribution, urban-rural classification (11 classes) per municipality | PxWeb API |
| [OpenCellID](https://opencellid.org) | Cell tower locations, operator, technology (GSM/UMTS/LTE/NR), range | getInArea (BBOX) |
| [OpenStreetMap](https://overpass-api.de) | POIs (education, healthcare, water, religion, emergency, govt, transport, forest) | Overpass API |
| [Celestrak](https://celestrak.org) | TLE orbital data for 40+ reconnaissance/imaging satellites | GP |

## Analysis agents

| Agent | What it does | Endpoint |
|---|---|---|
| **CellTower** | Operator/technology breakdown per area | `POST /api/agents/celltower-agent/run` |
| **Satellite** | Overpass windows grouped by type (optical/SAR/multispectral) | `POST /api/agents/satellite-agent/run` |
| **BridgeLoad** | Joins bridge geometry with weight/height/width limits, classifies against military vehicle thresholds | `POST /api/agents/bridge-load-agent/run` |
| **Demographics** | Population totals, sex/age distribution, urban vs rural per municipality | `POST /api/agents/demographics-agent/run` |
| **ForestConcealment** | Concealment rating from OSM forest type (coniferous/deciduous/mixed) and leaf cycle | `POST /api/agents/forest-concealment-agent/run` |
| **WeatherImpact** | Maps weather to operational effects (drone restrictions, surveillance windows, mobility) | `POST /api/agents/weather-impact-agent/run` |
| **PowerGrid** | Power line density and chokepoint assessment from NLS data | `POST /api/agents/power-grid-agent/run` |
| **Summary (WIP)** | Multi-source fusion placeholder | — |
| **Data Consistency Engine** | Cross-validates EW-vulnerable vs immune sources; trust scores, anomalies, spatial clustering | `POST /api/consistency/run` |

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/sources` | List data sources with status |
| POST | `/api/ingest` | Fetch data from specified sources |
| GET | `/api/datasets` | List ingested records |
| GET | `/api/agents` | List analysis agents |
| POST | `/api/agents/{id}/run` | Run an agent |
| POST | `/api/consistency/run` | Run Data Consistency Engine (trust + anomalies) |
| GET | `/api/consistency/report` | Last consistency report |
| GET | `/api/sources?include_trust=true` | Sources with trust scores after consistency run |
| POST | `/api/aoi/inspect` | Spatial clip + fusion analysis |
| GET | `/api/weather/point` | Point weather query |
| GET | `/api/ui-demo` | Interactive map + dashboard UI |

## Quick start

```bash
git clone https://github.com/MrSickFlow/LAM-Dynamics-DH
cd LAM-Dynamics-DH
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your keys (NLS, OpenCellID, N2YO)

# Start
PYTHONPATH=src .venv/bin/uvicorn ipb_backend.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/api/ui-demo

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -v
```

22 tests covering health, ingestion, all adapters, all agents, AOI inspection, and edge cases.

## Project layout

```
src/ipb_backend/
  agents/           8 analysis agents
  analysis/         AOI engine + Ollama analyzer
  api/              FastAPI routes
  ingestion/
    sources/        7 source adapters
    base.py         SourceAdapter ABC
    registry.py     SourceRegistry
    service.py      In-memory ingestion
    scheduler.py    Auto-refresh loop
  config.py         Settings from .env
  models.py         Pydantic models
  spatial.py        Geo clipping (shapely)
  ui_placeholder.html   Leaflet map UI
tests/
  test_api.py       20 API tests
  test_analyzers.py 2 analyzer tests
```

## Target areas

Archipelago Sea, North Karelia, and Lapland (Käsivarren Lappi). The tool generalizes to any WGS84 bounding box.

## Challenge context

Built for the AI2PB challenge by [61N Solutions](https://www.61n.fi/in-english/) at Junction Defence Hackathon 2026.

Judging criteria: Uniqueness (25%), Visualization (25%), Generalization (25%), Data source breadth (25%).

IPB doctrine references:
- [MCRP 2-10B.1 (USMC)](https://www.marines.mil/Portals/1/Publications/MCRP%202-10B.1.pdf)
- [ATP 2-01.3 (Army)](https://irp.fas.org/doddir/army/atp2-01-3.pdf)
