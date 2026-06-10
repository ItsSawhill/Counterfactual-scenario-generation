# RAG

## Current Status

The RAG layer retrieves local project-document context. It does not currently condition the trained DDPM model.

RAG is useful here as a document lookup layer for methodology, assumptions, architecture notes, and prior result summaries. It is not yet a retrieval-augmented diffusion system.

## What RAG Reads

By default, ingestion reads:

- `README.md`
- files under `docs/`
- optional text-like files under `data/context/` if that directory exists

Configuration lives in `configs/rag_config.yaml`.

## Architecture

```mermaid
flowchart LR
    A[README + docs + optional data/context] --> B[Chunking]
    B --> C[SentenceTransformer Embeddings]
    C --> D[FAISS Vector Store]
    E[User Query] --> F[FastAPI /rag/query]
    D --> F
    F --> G[Retrieved Document Chunks]
    H[/generate-with-context] --> I[Fallback Simulation]
    G --> H
```

## Ingestion

`POST /rag/ingest`:

1. loads `configs/rag_config.yaml`
2. reads configured source documents
3. chunks text
4. embeds chunks with `sentence-transformers/all-MiniLM-L6-v2`
5. writes a FAISS index and metadata to `.rag_store/`

`.rag_store/` is ignored by git.

## Retrieval

`POST /rag/query`:

1. embeds the query
2. searches the local FAISS index
3. returns top matching chunks with text, source, and similarity score

If `.rag_store/` is missing, querying fails with a clear error asking the user to run ingestion first.

## `/generate-with-context`

`POST /generate-with-context` currently:

1. retrieves local document chunks
2. returns those chunks alongside the generated scenario response
3. passes retrieved text into the fallback simulator

It does not:

- load a DDPM checkpoint
- alter DDPM conditioning tensors
- use retrieval to select model weights
- provide retrieval-augmented diffusion inference

## Productionization Path

To make RAG materially affect model generation, the backend first needs real DDPM inference. After that, retrieved context could be used to:

- constrain scenario parameters
- choose scenario templates
- annotate outputs with supporting methodology
- eventually influence an explicit shock-conditioning vector

That is future work.

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
