"""Chunk storage and similarity search: local numpy files or PostgreSQL/pgvector.

Embeddings are assumed L2-normalised, so cosine similarity is a dot product.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.rag.chunking import Chunk

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    """A retrieved chunk plus its cosine similarity score."""
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def reset(self) -> None: ...
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...
    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[SearchHit]: ...
    def count(self) -> int: ...


class LocalVectorStore:
    """Embeddings in a numpy array, chunk metadata in JSON. Brute-force search."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
        self._load()

    @property
    def _npy(self) -> Path:
        return self.directory / "embeddings.npy"

    @property
    def _json(self) -> Path:
        return self.directory / "chunks.json"

    def _load(self) -> None:
        if self._npy.exists() and self._json.exists():
            self._matrix = np.load(self._npy)
            self._chunks = [Chunk(**c) for c in json.loads(self._json.read_text("utf-8"))]

    def _save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        np.save(self._npy, self._matrix)
        self._json.write_text(
            json.dumps([asdict(c) for c in self._chunks], ensure_ascii=False), "utf-8"
        )

    def reset(self) -> None:
        self._chunks, self._matrix = [], None
        for p in (self._npy, self._json):
            p.unlink(missing_ok=True)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        embeddings = embeddings.astype(np.float32)
        self._matrix = embeddings if self._matrix is None else np.vstack([self._matrix, embeddings])
        self._chunks.extend(chunks)
        self._save()

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[SearchHit]:
        if self._matrix is None or not self._chunks:
            return []
        scores = self._matrix @ query_embedding.astype(np.float32)
        order = np.argsort(-scores)[:top_k]
        return [SearchHit(self._chunks[i], float(scores[i])) for i in order]

    def count(self) -> int:
        return len(self._chunks)


class PgVectorStore:
    """pgvector-backed store. Vectors go over as '[x,y,...]' text, so only psycopg2 is needed.

    HNSW indexes cap at 2000 dimensions: MiniLM (384) gets one, the larger
    TF-IDF vectors fall back to an exact scan.
    """

    TABLE = "document_chunks"

    def __init__(self, dsn: str, embedding_dim: int):
        import psycopg2  # imported lazily so the module loads without a DB driver

        self.embedding_dim = embedding_dim
        # SQLAlchemy-style URL -> plain libpq DSN
        self._conn = psycopg2.connect(dsn.replace("postgresql+psycopg2://", "postgresql://"))
        self._conn.autocommit = True
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # A table left over from a different embedder has the wrong vector size. The index
            # is rebuilt from the documents on every ingest, so drop it instead of failing.
            cur.execute(
                "SELECT a.atttypmod FROM pg_attribute a WHERE a.attrelid = to_regclass(%s) "
                "AND a.attname = 'embedding'", (self.TABLE,),
            )
            row = cur.fetchone()
            if row and row[0] != self.embedding_dim:
                logger.warning("Dropping %s: stored dim %s != %s", self.TABLE, row[0], self.embedding_dim)
                cur.execute(f"DROP TABLE {self.TABLE}")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    id            SERIAL PRIMARY KEY,
                    text          TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    source        TEXT NOT NULL,
                    section       TEXT NOT NULL,
                    chunk_index   INT  NOT NULL,
                    char_start    INT  NOT NULL,
                    char_end      INT  NOT NULL,
                    embedding     vector({self.embedding_dim}) NOT NULL
                )
                """
            )
            if self.embedding_dim <= 2000:
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {self.TABLE}_embedding_idx "
                    f"ON {self.TABLE} USING hnsw (embedding vector_cosine_ops)"
                )

    @staticmethod
    def _literal(vec: np.ndarray) -> str:
        return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

    def reset(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"TRUNCATE {self.TABLE} RESTART IDENTITY")

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        from psycopg2.extras import execute_values

        rows = [
            (c.text, c.document_name, c.source, c.section,
             c.chunk_index, c.char_start, c.char_end, self._literal(e))
            for c, e in zip(chunks, embeddings)
        ]
        with self._conn.cursor() as cur:
            execute_values(
                cur,
                f"INSERT INTO {self.TABLE} (text, document_name, source, section, "
                f"chunk_index, char_start, char_end, embedding) VALUES %s",
                rows,
                template="(%s,%s,%s,%s,%s,%s,%s,%s::vector)",
            )

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[SearchHit]:
        lit = self._literal(query_embedding)
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT text, document_name, source, section, chunk_index, char_start, "
                f"char_end, 1 - (embedding <=> %s::vector) AS score "
                f"FROM {self.TABLE} ORDER BY embedding <=> %s::vector LIMIT %s",
                (lit, lit, top_k),
            )
            return [SearchHit(Chunk(*row[:7]), float(row[7])) for row in cur.fetchall()]

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.TABLE}")
            return cur.fetchone()[0]


def create_vector_store(
    embedding_dim: int,
    local_dir: Path,
    backend: str = "auto",
    dsn: str | None = None,
) -> VectorStore:
    """backend: 'pgvector' | 'local' | 'auto' (try pgvector, fall back to local)."""
    if backend in ("pgvector", "auto"):
        try:
            if dsn is None:
                from app.config import settings
                dsn = settings.postgres_url
            store = PgVectorStore(dsn, embedding_dim)
            logger.info("Using pgvector store")
            return store
        except Exception as exc:
            if backend == "pgvector":
                raise
            logger.warning("pgvector unavailable (%s) — using local store", exc)
    logger.info("Using local vector store at %s", local_dir)
    return LocalVectorStore(local_dir)
