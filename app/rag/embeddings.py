"""Embedding backends for RAG.

Preferred backend is sentence-transformers (384-dim dense). The TF-IDF embedder
exists because corporate application control blocks the unsigned .pyd files that
sentence-transformers' tokenizer pulls in; it only needs numpy and gives
fixed-size vectors so the pgvector schema stays the same.
Pick the backend with EMBEDDING_BACKEND=tfidf|sentence (default: auto).
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Protocol, runtime_checkable

import joblib
import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingModel(Protocol):
    embedding_dim: int

    def fit(self, texts: list[str]) -> "EmbeddingModel": ...

    def embed_text(self, text: str) -> np.ndarray:
        """float32 array of shape (embedding_dim,)."""
        ...

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """float32 array of shape (N, embedding_dim)."""
        ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "EmbeddingModel": ...


class TFIDFEmbeddingModel:
    """Unigram+bigram TF-IDF vectors, L2-normalised (so cosine == dot product)."""

    STOP_WORDS = {
        "the","is","are","was","were","be","been","being","have","has","had",
        "and","but","or","nor","for","yet","so","in","on","at","to","of","a",
        "an","by","it","its","that","this","these","those","from","with","as",
        "not","can","will","would","could","should","may","might","shall","do",
        "does","did","if","when","where","how","what","which","who","whom",
        "they","their","them","we","our","you","your","he","his","she","her",
        "also","such","more","most","any","all","each","than","then","into",
        "over","under","between","through","about","after","before","during",
        "both","either","neither","there","here","very","just","only","even",
    }

    def __init__(
        self,
        embedding_dim: int = 4096,  # big enough for the whole vocab on a small corpus
        min_df: int = 1,
        max_df_ratio: float = 0.99,
    ):
        self.embedding_dim = embedding_dim
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio

        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._is_fitted = False
        self._n_docs = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        clean = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        words = [
            w for w in clean.split()
            if len(w) > 2 and w not in TFIDFEmbeddingModel.STOP_WORDS
        ]
        # bigrams keep phrases like "flash_flood" together
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        return words + bigrams

    def fit(self, texts: list[str]) -> "TFIDFEmbeddingModel":
        n = len(texts)
        self._n_docs = n

        df: Counter = Counter()
        for text in texts:
            df.update(set(self._tokenize(text)))

        max_df_abs = int(n * self.max_df_ratio)
        vocab_words = sorted(
            word for word, count in df.items()
            if self.min_df <= count <= max_df_abs
        )

        # Over capacity: keep terms that are neither singletons nor ubiquitous,
        # favouring ones that appear in roughly 15% of chunks.
        if len(vocab_words) > self.embedding_dim:
            def score(w: str) -> float:
                d = df[w]
                if d == 1:
                    return 0.0
                return math.log((n + 1) / (d + 1)) * (1 - abs(d / n - 0.15))

            vocab_words = sorted(vocab_words, key=score, reverse=True)[:self.embedding_dim]
            vocab_words.sort()

        self._vocab = {word: idx for idx, word in enumerate(vocab_words)}

        # smoothed IDF so no term gets zero weight
        idf = np.zeros(len(self._vocab), dtype=np.float32)
        for word, idx in self._vocab.items():
            idf[idx] = math.log((n + 1) / (df[word] + 1)) + 1.0

        self._idf = idf
        self._is_fitted = True

        logger.info(
            "TF-IDF fitted: %d docs, %d vocabulary words, dim=%d",
            n, len(self._vocab), self.embedding_dim,
        )
        return self

    def _tf_vector(self, text: str) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before embed_text()")

        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        total = len(tokens)
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        for token, count in Counter(tokens).items():
            if token in self._vocab:
                idx = self._vocab[token]
                vec[idx] = (count / total) * self._idf[idx]

        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec

    def embed_text(self, text: str) -> np.ndarray:
        return self._tf_vector(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        matrix = np.stack([self._tf_vector(t) for t in texts], axis=0)
        return matrix.astype(np.float32)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "vocab": self._vocab,
            "idf": self._idf,
            "embedding_dim": self.embedding_dim,
            "n_docs": self._n_docs,
            "min_df": self.min_df,
            "max_df_ratio": self.max_df_ratio,
            "backend": "tfidf",
        }
        joblib.dump(artifact, path, compress=3)
        logger.info("Embedding model saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "TFIDFEmbeddingModel":
        artifact = joblib.load(path)
        model = cls(
            embedding_dim=artifact["embedding_dim"],
            min_df=artifact["min_df"],
            max_df_ratio=artifact["max_df_ratio"],
        )
        model._vocab = artifact["vocab"]
        model._idf = artifact["idf"]
        model._n_docs = artifact["n_docs"]
        model._is_fitted = True
        logger.info(
            "Embedding model loaded: %d vocab words, dim=%d",
            len(model._vocab), model.embedding_dim,
        )
        return model


class SentenceTransformerModel:
    """all-MiniLM-L6-v2 (384-dim, ~256 token input). Weights load lazily."""

    MODEL_NAME = "all-MiniLM-L6-v2"
    embedding_dim = 384

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def fit(self, texts: list[str]) -> "SentenceTransformerModel":
        # nothing to learn, just load the pretrained weights
        self._load()
        return self

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading %s ...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Sentence transformer loaded. dim=%d", self.embedding_dim)

    def embed_text(self, text: str) -> np.ndarray:
        self._load()
        vec = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        self._load()
        vecs = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=len(texts) > 50,
        )
        return vecs.astype(np.float32)

    def save(self, path: Path) -> None:
        # only the model name is stored; weights come from the HF hub
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"backend": "sentence", "model_name": self.model_name}, path)

    @classmethod
    def load(cls, path: Path) -> "SentenceTransformerModel":
        artifact = joblib.load(path)
        model = cls(model_name=artifact["model_name"])
        model._load()
        return model


def create_embedding_model(
    backend: str | None = None,
    embedding_dim: int = 4096,
) -> TFIDFEmbeddingModel | SentenceTransformerModel:
    """backend: 'tfidf' | 'sentence' | None/'auto' (env var, else sentence if importable).

    embedding_dim only applies to the TF-IDF backend.
    """
    if backend is None:
        backend = os.environ.get("EMBEDDING_BACKEND", "auto")

    if backend == "sentence":
        logger.info("Using SentenceTransformer backend")
        return SentenceTransformerModel()

    if backend == "auto":
        try:
            import sentence_transformers  # noqa: F401
            logger.info("sentence-transformers available — using dense embeddings")
            return SentenceTransformerModel()
        except Exception:
            logger.info("sentence-transformers not available — using TF-IDF backend")

    logger.info("Using TF-IDF embedding backend (dim=%d)", embedding_dim)
    return TFIDFEmbeddingModel(embedding_dim=embedding_dim)
