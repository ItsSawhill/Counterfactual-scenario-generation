# Backend

## Current Status

The backend is a FastAPI service that exposes health, RAG, fallback simulation, and frontend-compatible scenario endpoints.

It does **not** currently serve trained DDPM inference. The current path generation code is a deterministic fallback simulator in `backend/model/scenario_generator.py`.

## Architecture

```mermaid
flowchart LR
    A[README + docs + optional data/context] --> B[backend/rag/ingest.py]
    B --> C[FAISS Vector Store]
    D[User Request] --> E[/rag/query]
    C --> E
    D --> F[/generate-with-context]
    E --> F
    F --> G[backend/model/scenario_generator.py<br/>fallback simulation]
    H[Frontend /simulate request] --> I[/simulate]
    I --> G
```

## Endpoints

- `GET /health`
- `POST /rag/ingest`
- `POST /rag/query`
- `POST /generate-with-context`
- `POST /simulate`
- `POST /refresh-live-state`

## `/simulate`

`/simulate` accepts the frontend scenario controls:

```json
{
  "inflation": 0.03,
  "interest_rate": 0.04,
  "horizon": 30,
  "path_count": 128
}
```

It returns a frontend-compatible payload:

```json
{
  "assets": ["SPY", "QQQ", "GC", "CL", "ZN"],
  "paths": [[[0.001, -0.002, 0.0005, 0.0011, -0.0003]]],
  "return_scaler_mean": [0, 0, 0, 0, 0],
  "return_scaler_scale": [1, 1, 1, 1, 1],
  "start_prices": [525, 452, 214, 78, 110],
  "horizon": 30,
  "path_count": 128,
  "model_horizon": 30,
  "live_state": {
    "market_data_as_of": "2026-05-15",
    "model_input_as_of": "2026-05-15",
    "live_fetched_at": "2026-05-15T12:00:00Z",
    "cache_status": "fresh",
    "cache_age_seconds": 0,
    "cache_ttl_seconds": 900
  },
  "fallback_generator_used": true
}
```

Current behavior:

- uses handcrafted drift, volatility, and cross-asset correlation assumptions
- is deterministic for the same input parameters
- returns paths shaped for the React UI contract
- does not load `conditional_ddpm_upgraded_best.pt`
- does not load processed windows or scalers
- does not fetch live market data for `/refresh-live-state`

## `/generate-with-context`

This route:

1. retrieves local document chunks through `backend/rag/retriever.py`
2. passes those chunks to `backend/model/scenario_generator.py`
3. returns retrieved context plus fallback-generated paths

The retrieved context currently influences only lightweight heuristic adjustment in the fallback generator. It is not DDPM conditioning.

## Planned DDPM Integration

The productionization step is to replace or wrap `backend/model/scenario_generator.py` with a real inference module that:

- imports the notebook DDPM architecture as regular Python code
- loads `counterfactual_data_build/outputs/checkpoints/conditional_ddpm_upgraded_best.pt`
- loads model config and scalers
- builds condition windows from processed/live features
- returns the existing frontend-compatible response schema

Until that exists, the backend should be described as a demo API with fallback simulation.

## Tests

Current tests validate:

- health endpoint shape
- `/simulate` response shape
- graceful missing-vector-store behavior
- mocked `/generate-with-context` response shape

They do not validate DDPM checkpoint loading, model quality, or real RAG ingestion.
