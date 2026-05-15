from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from backend.model.schemas import ScenarioGenerationResponse


class RagConfig(BaseModel):
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store_path: str = ".rag_store"
    source_directories: list[str] = Field(default_factory=lambda: ["docs", "data/context"])


class SourceDocument(BaseModel):
    path: str
    text: str


class ChunkRecord(BaseModel):
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int


class RetrievedContext(BaseModel):
    text: str
    source: str
    score: float


class IngestRequest(BaseModel):
    force_rebuild: bool = False


class IngestResponse(BaseModel):
    status: str
    vector_store_path: str
    document_count: int
    chunk_count: int
    indexed_sources: list[str]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(5, ge=1, le=20)


class QueryResponse(BaseModel):
    query: str
    contexts: list[RetrievedContext]


class GenerateWithContextRequest(BaseModel):
    user_request: str = Field(..., min_length=3)
    top_k: int = Field(5, ge=1, le=20)
    scenario_parameters: dict[str, Any] = Field(default_factory=dict)


class GenerateWithContextResponse(BaseModel):
    user_request: str
    retrieved_context: list[RetrievedContext]
    generated_scenario: ScenarioGenerationResponse
    metadata: dict[str, Any]
    fallback_generator_used: bool
    integration_notes: list[str]


class HealthResponse(BaseModel):
    status: str
    service: str
    vector_store_ready: bool
    vector_store_path: str


def resolve_vector_store_path(project_root: Path, config: RagConfig) -> Path:
    return (project_root / config.vector_store_path).resolve()
