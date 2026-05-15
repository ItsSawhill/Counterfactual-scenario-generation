from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from backend.rag.ingest import ingest_project_documents, load_rag_config
from backend.rag.retriever import retrieve_context
from backend.rag.schemas import (
    GenerateWithContextRequest,
    GenerateWithContextResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    resolve_vector_store_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(
    title="Counterfactual Financial Scenario Generation API",
    version="0.1.0",
    description="Lightweight RAG and scenario-generation integration layer.",
)


def _vector_store_ready() -> tuple[bool, str]:
    config = load_rag_config(PROJECT_ROOT)
    vector_store_path = resolve_vector_store_path(PROJECT_ROOT, config)
    ready = (vector_store_path / "index.faiss").exists() and (vector_store_path / "metadata.json").exists()
    return ready, str(vector_store_path.relative_to(PROJECT_ROOT))


def _build_generation_placeholder(
    user_request: str,
    scenario_parameters: dict[str, Any],
    rag_response: QueryResponse,
) -> GenerateWithContextResponse:
    return GenerateWithContextResponse(
        user_request=user_request,
        retrieved_context=rag_response.contexts,
        generated_scenario={
            "status": "placeholder",
            "scenario_request": user_request,
            "scenario_parameters": scenario_parameters,
            "suggested_next_step": "Connect this endpoint to the live counterfactual notebook or a packaged inference module.",
        },
        integration_status="rag_connected_model_pending",
        integration_notes=[
            "Current scenario generation logic lives primarily in live_counterfactual_lab_v1.ipynb and realtime_counterfactual_generation_system.ipynb.",
            "The clean production path is to extract notebook inference into importable Python modules, then call that module here after retrieval.",
            "Retrieved context is already available in this endpoint and can be appended to scenario prompts, logged as provenance, or mapped into scenario constraints before model execution.",
        ],
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ready, vector_store_path = _vector_store_ready()
    return HealthResponse(
        status="ok",
        service="counterfactual-rag-api",
        vector_store_ready=ready,
        vector_store_path=vector_store_path,
    )


@app.post("/rag/ingest", response_model=IngestResponse)
def rag_ingest(request: IngestRequest) -> IngestResponse:
    try:
        return ingest_project_documents(PROJECT_ROOT, force_rebuild=request.force_rebuild)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/rag/query", response_model=QueryResponse)
def rag_query(request: QueryRequest) -> QueryResponse:
    try:
        return retrieve_context(PROJECT_ROOT, request.query, request.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/generate-with-context", response_model=GenerateWithContextResponse)
def generate_with_context(request: GenerateWithContextRequest) -> GenerateWithContextResponse:
    try:
        rag_response = retrieve_context(PROJECT_ROOT, request.user_request, request.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _build_generation_placeholder(
        user_request=request.user_request,
        scenario_parameters=request.scenario_parameters,
        rag_response=rag_response,
    )
