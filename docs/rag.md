# RAG

## What RAG Does In This Project

The RAG layer retrieves relevant local project context before scenario generation or evaluation. Instead of sending a scenario request directly into a model flow without context, the backend can first search:

- `README.md`
- `docs/`
- optional local research material under `data/context/`

This gives the API a lightweight memory of methodology, architecture, assumptions, and prior evaluation notes.

## Architecture

```mermaid
flowchart LR
    A[README + docs + optional data/context] --> B[Chunking]
    B --> C[SentenceTransformer Embeddings]
    C --> D[FAISS Vector Store]
    E[User Scenario Request] --> F[FastAPI /rag/query]
    D --> F
    F --> G[Retrieved Context Chunks]
    G --> H[/generate-with-context]
    H --> I[Scenario Generation Placeholder<br/>or future DDPM inference module]
```

## How Ingestion Works

1. `POST /rag/ingest` loads config from `configs/rag_config.yaml`.
2. The ingester reads `README.md` plus text-like files under configured source directories.
3. Documents are normalized and chunked using `chunk_size` and `chunk_overlap`.
4. Chunk embeddings are created with a local sentence-transformer model.
5. Embeddings are stored in a local FAISS index, with chunk metadata saved alongside it.

By default, the vector store is written to `.rag_store/`, which is ignored by git.

## How Retrieval Works

1. `POST /rag/query` embeds the incoming query.
2. The retriever runs similarity search against the local FAISS index.
3. The API returns the top matching chunks with:
   - chunk text
   - source filename
   - similarity score

## How It Connects To Financial Scenario Generation

The current repository keeps most inference logic in notebooks, especially:

- `live_counterfactual_lab_v1.ipynb`
- `realtime_counterfactual_generation_system.ipynb`

The new `POST /generate-with-context` route already retrieves relevant context and returns it with the user request. The final production integration step is to extract scenario inference into an importable Python module and call it from the endpoint after retrieval.

The backend now routes that request through `backend/model/scenario_generator.py`. At the moment this uses a documented fallback generator that preserves the frontend/backend contract until DDPM inference is extracted from the notebooks.

This keeps the RAG layer additive:

- no notebook rewrites
- no frontend changes required
- clear path toward production inference later

## Example API Calls

Ingest:

```bash
curl -X POST http://127.0.0.1:8000/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{"force_rebuild": true}'
```

Query:

```bash
curl -X POST http://127.0.0.1:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What happens if inflation rises while unemployment increases?", "top_k": 5}'
```

Generate with context:

```bash
curl -X POST http://127.0.0.1:8000/generate-with-context \
  -H "Content-Type: application/json" \
  -d '{
    "user_request": "Evaluate a stagflation-style scenario with higher inflation and weaker labor markets.",
    "top_k": 5,
    "scenario_parameters": {
      "inflation": 0.055,
      "interest_rate": 0.0525,
      "horizon": 30,
      "path_count": 128
    }
  }'
```
