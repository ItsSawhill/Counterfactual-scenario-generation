from __future__ import annotations

import json
from pathlib import Path

from .ingest import INDEX_FILENAME, METADATA_FILENAME, load_rag_config
from .schemas import QueryResponse, RetrievedContext, resolve_vector_store_path


def _ensure_embedding_dependencies():
    try:
        import faiss  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "RAG dependencies are missing. Install requirements.txt before querying the vector store."
        ) from exc


def _load_store(project_root: Path):
    _ensure_embedding_dependencies()
    import faiss

    config = load_rag_config(project_root)
    vector_store_path = resolve_vector_store_path(project_root, config)
    index_path = vector_store_path / INDEX_FILENAME
    metadata_path = vector_store_path / METADATA_FILENAME

    if not index_path.exists() or not metadata_path.exists():
        raise RuntimeError(
            "Vector store not found. Run POST /rag/ingest before querying."
        )

    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return config, index, metadata


def retrieve_context(project_root: Path, query: str, top_k: int) -> QueryResponse:
    from sentence_transformers import SentenceTransformer

    config, index, metadata = _load_store(project_root)
    model = SentenceTransformer(config.embedding_model)
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    distances, indices = index.search(query_embedding, top_k)
    contexts: list[RetrievedContext] = []

    for score, match_index in zip(distances[0], indices[0], strict=False):
        if match_index < 0 or match_index >= len(metadata):
            continue
        match = metadata[match_index]
        contexts.append(
            RetrievedContext(
                text=match["text"],
                source=match["source"],
                score=float(score),
            )
        )

    return QueryResponse(query=query, contexts=contexts)
