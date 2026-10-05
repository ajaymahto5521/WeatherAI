"""Load reference docs -> chunk -> embed -> store (run via scripts/ingest_documents.py).

The fitted embedder is saved next to the index so queries use the same vocabulary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.config import settings
from app.rag.chunking import chunk_directory
from app.rag.embeddings import create_embedding_model
from app.rag.vector_store import create_vector_store

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "rag_index"
EMBEDDER_FILENAME = "embedder.joblib"


@dataclass
class IngestionResult:
    n_chunks: int
    n_documents: int
    embedding_dim: int
    store_type: str


def ingest(
    reference_dir: Path = REFERENCE_DIR,
    index_dir: Path = INDEX_DIR,
    backend: str = "auto",
    embedding_backend: str | None = None,
) -> IngestionResult:
    chunks = chunk_directory(
        reference_dir,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    if not chunks:
        raise RuntimeError(f"No documents found in {reference_dir}")

    embedder = create_embedding_model(embedding_backend)
    embedder.fit([c.text for c in chunks])
    embeddings = embedder.embed_batch([c.text for c in chunks])

    index_dir.mkdir(parents=True, exist_ok=True)
    embedder.save(index_dir / EMBEDDER_FILENAME)

    store = create_vector_store(embedder.embedding_dim, index_dir, backend=backend)
    store.reset()
    store.add(chunks, embeddings)

    result = IngestionResult(
        n_chunks=store.count(),
        n_documents=len({c.document_name for c in chunks}),
        embedding_dim=embedder.embedding_dim,
        store_type=type(store).__name__,
    )
    logger.info("Ingestion complete: %s", result)
    return result
