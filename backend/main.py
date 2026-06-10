from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException

from backend.model import generate_scenario, simulate_frontend_request
from backend.model.schemas import LiveStateMeta, ScenarioParameters
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
logger = logging.getLogger("counterfactual_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.warning("/simulate is running in fallback simulation mode.")
    logger.warning("DDPM artifacts are not loaded by this backend yet; checkpoint-backed inference is not implemented.")
    yield


app = FastAPI(
    title="Counterfactual Financial Scenario Generation API",
    version="0.1.0",
    description="Lightweight RAG and scenario-generation integration layer.",
    lifespan=lifespan,
)


def _vector_store_ready() -> tuple[bool, str]:
    config = load_rag_config(PROJECT_ROOT)
    vector_store_path = resolve_vector_store_path(PROJECT_ROOT, config)
    ready = (vector_store_path / "index.faiss").exists() and (vector_store_path / "metadata.json").exists()
    return ready, str(vector_store_path.relative_to(PROJECT_ROOT))


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

    generated = generate_scenario(
        user_request=request.user_request,
        scenario_parameters=request.scenario_parameters,
        retrieved_context=[context.model_dump() for context in rag_response.contexts],
    )
    return GenerateWithContextResponse(
        user_request=request.user_request,
        retrieved_context=rag_response.contexts,
        generated_scenario=generated,
        metadata={
            "generator_type": generated["metadata"]["generator_type"],
            "ddpm_enabled": generated["ddpm_enabled"],
            "retrieved_context_count": len(rag_response.contexts),
            "project_root": str(PROJECT_ROOT),
        },
        fallback_generator_used=generated["fallback_generator_used"],
        integration_notes=[
            "Current model-facing generation is connected through backend/model/scenario_generator.py.",
            "The fallback stochastic generator should be replaced with extracted DDPM inference from live_counterfactual_lab_v1.ipynb; archive/realtime_counterfactual_generation_system.ipynb is retained only as a stale legacy reference.",
            "Retrieved RAG context is already passed into the generator and can be upgraded from heuristic guidance to true model conditioning logic later.",
        ],
    )


@app.post("/simulate")
def simulate(request: ScenarioParameters) -> dict:
    return simulate_frontend_request(
        inflation=request.inflation,
        interest_rate=request.interest_rate,
        horizon=request.horizon,
        path_count=request.path_count,
    )


@app.post("/refresh-live-state", response_model=LiveStateMeta)
def refresh_live_state() -> LiveStateMeta:
    generated = simulate_frontend_request(
        inflation=0.03,
        interest_rate=0.04,
        horizon=30,
        path_count=64,
    )
    return LiveStateMeta(**generated["live_state"])
