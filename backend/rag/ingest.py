from __future__ import annotations

import json
from pathlib import Path

import yaml

from .schemas import ChunkRecord, IngestResponse, RagConfig, SourceDocument, resolve_vector_store_path


TEXT_EXTENSIONS = {".md", ".txt", ".json", ".yaml", ".yml", ".csv"}
INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def load_rag_config(project_root: Path) -> RagConfig:
    config_path = project_root / "configs" / "rag_config.yaml"
    if not config_path.exists():
        return RagConfig()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return RagConfig(**payload)


def collect_source_documents(project_root: Path, config: RagConfig) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    seen_paths: set[Path] = set()
    candidate_paths = [project_root / "README.md"]

    for source_dir in config.source_directories:
        candidate = project_root / source_dir
        if candidate.is_dir():
            candidate_paths.extend(sorted(candidate.rglob("*")))

    for path in candidate_paths:
        if not path.is_file() or path in seen_paths:
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS and path.name != "README.md":
            continue
        text = _read_text_file(path)
        if not text:
            continue
        seen_paths.add(path)
        documents.append(
            SourceDocument(
                path=str(path.relative_to(project_root)),
                text=text,
            )
        )

    return documents


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[tuple[str, int, int]] = []
    step = chunk_size - chunk_overlap
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append((chunk, start, end))
        if end >= len(normalized):
            break
        start += step

    return chunks


def build_chunk_records(documents: list[SourceDocument], config: RagConfig) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for document in documents:
        chunks = chunk_text(document.text, config.chunk_size, config.chunk_overlap)
        for chunk_index, (chunk_text_value, start, end) in enumerate(chunks):
            records.append(
                ChunkRecord(
                    chunk_id=f"{document.path}::chunk::{chunk_index}",
                    text=chunk_text_value,
                    source=document.path,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                )
            )
    return records


def _ensure_embedding_dependencies():
    try:
        import faiss  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "RAG dependencies are missing. Install requirements.txt before running ingestion."
        ) from exc


def _encode_chunks(texts: list[str], model_name: str):
    _ensure_embedding_dependencies()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def _write_vector_store(vector_store_path: Path, embeddings, chunk_records: list[ChunkRecord]) -> None:
    import faiss

    vector_store_path.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(vector_store_path / INDEX_FILENAME))

    metadata_payload = [record.model_dump() for record in chunk_records]
    (vector_store_path / METADATA_FILENAME).write_text(
        json.dumps(metadata_payload, indent=2),
        encoding="utf-8",
    )


def ingest_project_documents(project_root: Path, force_rebuild: bool = False) -> IngestResponse:
    config = load_rag_config(project_root)
    vector_store_path = resolve_vector_store_path(project_root, config)
    index_path = vector_store_path / INDEX_FILENAME
    metadata_path = vector_store_path / METADATA_FILENAME

    if not force_rebuild and index_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        indexed_sources = sorted({record["source"] for record in metadata})
        return IngestResponse(
            status="already_indexed",
            vector_store_path=str(vector_store_path.relative_to(project_root)),
            document_count=len(indexed_sources),
            chunk_count=len(metadata),
            indexed_sources=indexed_sources,
        )

    documents = collect_source_documents(project_root, config)
    if not documents:
        raise RuntimeError("No source documents were found for RAG ingestion.")

    chunk_records = build_chunk_records(documents, config)
    if not chunk_records:
        raise RuntimeError("No non-empty text chunks were generated for ingestion.")

    embeddings = _encode_chunks([record.text for record in chunk_records], config.embedding_model)
    _write_vector_store(vector_store_path, embeddings, chunk_records)

    return IngestResponse(
        status="indexed",
        vector_store_path=str(vector_store_path.relative_to(project_root)),
        document_count=len(documents),
        chunk_count=len(chunk_records),
        indexed_sources=[document.path for document in documents],
    )
