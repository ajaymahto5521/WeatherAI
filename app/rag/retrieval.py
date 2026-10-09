"""Query-time retrieval: embed the question, search the store."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib

from app.config import settings
from app.rag.embeddings import SentenceTransformerModel, TFIDFEmbeddingModel
from app.rag.ingestion import EMBEDDER_FILENAME, INDEX_DIR
from app.rag.vector_store import SearchHit, create_vector_store

logger = logging.getLogger(__name__)


def _load_embedder(path: Path):
    backend = joblib.load(path).get("backend", "tfidf")
    cls = SentenceTransformerModel if backend == "sentence" else TFIDFEmbeddingModel
    return cls.load(path)


class Retriever:
    """Loads the embedder saved at ingestion time and searches the store."""

    def __init__(
        self,
        index_dir: Path = INDEX_DIR,
        backend: str = "auto",
        top_k: int | None = None,
    ):
        embedder_path = Path(index_dir) / EMBEDDER_FILENAME
        if not embedder_path.exists():
            raise FileNotFoundError(
                f"{embedder_path} not found — run scripts/ingest_documents.py first"
            )
        # Load the embedder fitted during ingestion so questions use the same vector space.
        self.embedder = _load_embedder(embedder_path)
        self.store = create_vector_store(self.embedder.embedding_dim, index_dir, backend=backend)
        self.top_k = top_k or settings.rag_top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        # The store ranks document chunks by similarity to the question vector.
        return self.store.search(self.embedder.embed_text(query), top_k or self.top_k)

    @staticmethod
    def to_sources(hits: list[SearchHit]) -> list[dict]:
        """Shape matching the /ask response: [{document, section}]."""
        return [{"document": h.chunk.document_name, "section": h.chunk.section} for h in hits]
