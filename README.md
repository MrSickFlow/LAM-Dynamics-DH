# IPB Open Source Backend

Backend-first scaffold for a hackathon tool that automates open-source intelligence preparation of the battlefield (IPB).

## Phase 1 goals

- Pull data from multiple open-source providers through source-specific adapters.
- Normalize retrieved data into a common internal format.
- Track refresh policies and source availability.
- Expose the collected data and source status through a simple API.

## Included now

- FastAPI backend with health, source, ingest, UI placeholder, and agent placeholder endpoints.
- Extensible ingestion adapter interface.
- Source registry with refresh intervals and source status.
- In-memory ingestion service for quick iteration during the hackathon.
- Placeholder adapters for FMI, National Land Survey, Statistics Finland, and Digiroad.
- Placeholder UI schema and analysis agent contracts.
- Basic tests.

## Project layout

```text
src/ipb_backend/
  agents/           Placeholder analysis agents
  api/              API route registration
  ingestion/        Adapters, registry, scheduler, service
  main.py           FastAPI entrypoint
  models.py         Shared Pydantic models
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn ipb_backend.main:app --reload --app-dir src
```

Open http://127.0.0.1:8000/docs for the API.

## Recommended next steps

1. Replace adapter placeholders with real API clients and credentials where needed.
2. Persist normalized datasets to PostGIS, DuckDB, or Parquet instead of memory.
3. Add a map UI and dashboard against the existing API contract.
4. Add derived analytics for mobility, chokepoints, visibility, and weather impacts.
