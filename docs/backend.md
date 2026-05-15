# Backend

## Architecture

The backend is a FastAPI service with two additive layers:

1. `backend/rag/`
   Handles ingestion and retrieval from local project documents.
2. `backend/model/`
   Handles scenario generation and frontend-compatible simulation responses.

```mermaid
flowchart LR
    A[README + docs + optional data/context] --> B[backend/rag/ingest.py]
    B --> C[FAISS Vector Store]
    D[User Request] --> E[/rag/query]
    C --> E
    D --> F[/generate-with-context]
    E --> F
    F --> G[backend/model/scenario_generator.py]
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

## Request And Response Examples

### `/simulate`

Request:

```json
{
  "inflation": 0.03,
  "interest_rate": 0.04,
  "horizon": 30,
  "path_count": 128
}
```

Response shape:

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
  }
}
```

### `/generate-with-context`

This route first retrieves context from the local vector store, then passes:

- `user_request`
- `scenario_parameters`
- `retrieved_context`

into the model-facing generator.

## Where RAG Connects

RAG is connected in `backend/main.py` before generation:

1. retrieve relevant context
2. pass context into `backend/model/scenario_generator.py`
3. return retrieved evidence alongside generated output

## Where Model Inference Connects

The current generator is intentionally a fallback implementation. The correct replacement point is the return-generation block in:

- [backend/model/scenario_generator.py](/Users/sahil/Counterfactual-Financial-Scenario-Generation/backend/model/scenario_generator.py)

The notebook logic that should eventually be extracted lives in:

- `live_counterfactual_lab_v1.ipynb`
- `realtime_counterfactual_generation_system.ipynb`

## TODOs

- Extract DDPM checkpoint loading and conditioning logic into importable Python modules.
- Replace fallback stochastic path generation with real notebook-derived inference.
- Add a packaged backend route for scenario-suite generation if the frontend expands beyond `/simulate`.
- Decide whether retrieved RAG context should influence prompts, scenario constraints, or direct model conditioning.
