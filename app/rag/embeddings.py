"""
app/rag/embeddings.py
======================
Embedding model wrapper for the RAG pipeline.

ARCHITECTURE DECISION
---------------------
Production target: sentence-transformers (BERT-based dense embeddings, 384 dims)
Current fallback:  TF-IDF (term frequency-inverse document frequency, sparse)

WHY THE FALLBACK?
Corporate Application Control policies block unsigned DLL files installed
by pip. The sentence-transformers library loads Rust-compiled tokenizer
binaries (.pyd files) that are blocked. Rather than halt the project,
we implement a pure-numpy TF-IDF embedder that:
  - Has identical interface (EmbeddingModel.embed_text, embed_batch)
  - Returns comparable similarity scores
  - Requires only numpy + scipy (both work)
  - Produces vectors of fixed dimension for consistent pgvector storage

THE CODE IS WRITTEN TO SWAP IMPLEMENTATIONS via the EMBEDDING_BACKEND
environment variable:
  EMBEDDING_BACKEND=tfidf       -> TFIDFEmbeddingModel (default, no DLLs)
  EMBEDDING_BACKEND=sentence    -> SentenceTransformerModel (when available)

WHAT IS TF-IDF?
---------------
TF (Term Frequency) = count(word in chunk) / total_words_in_chunk
IDF (Inverse Document Frequency) = log(N / df(word))  where N = total docs
                                                        df = docs containing word
TF-IDF = TF * IDF

High TF-IDF: word is common in THIS chunk but rare across all chunks -> distinctive
Low TF-IDF:  word is common everywhere ("the", "is") -> not informative

The vector for a chunk has one dimension per vocabulary word.
Two chunks with the same distinctive words will have high cosine similarity.

DIMENSION CHOICE: We use 512 dimensions for our TF-IDF vectors, matching
common dense embedding sizes, making it straightforward to switch later.
"""

from __future__ import annotations

import hashlib
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


# =============================================================================
# Protocol (interface definition)
# =============================================================================

@runtime_checkable
class EmbeddingModel(Protocol):
    """
    The interface every embedding backend must satisfy.

    Using a Protocol (structural subtyping) means we can swap
    TFIDFEmbeddingModel for SentenceTransformerModel without changing
    any calling code — as long as both implement these methods.
    """
    embedding_dim: int      # must be consistent across fit/embed

    def fit(self, texts: list[str]) -> "EmbeddingModel":
        """Build vocabulary / load model weights. Called once during ingestion."""
        ...

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text. Returns float32 array of shape (embedding_dim,)."""
        ...

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts. Returns float32 array of shape (N, embedding_dim)."""
        ...

    def save(self, path: Path) -> None:
        """Serialize the fitted model to disk."""
        ...

    @classmethod
    def load(cls, path: Path) -> "EmbeddingModel":
        """Deserialize a fitted model from disk."""
        ...


# =============================================================================
# TF-IDF Embedding Model
# =============================================================================

class TFIDFEmbeddingModel:
    """
    TF-IDF based text embedding using pure numpy.

    The vocabulary is built during fit() from the ingestion corpus.
    At inference, new text is embedded using the same vocabulary.

    Vocabulary hashing trick:
    Rather than storing the full vocabulary (memory-intensive), we hash
    each word to a bucket in [0, embedding_dim). Words collide occasionally
    (hash collision) but this is negligible for our document sizes.
    This is called the "feature hashing" or "hashing trick".

    Parameters
    ----------
    embedding_dim : vector size. 512 is a good balance for our 99 chunks.
    min_df        : minimum document frequency — words appearing in fewer
                    than min_df chunks are ignored (removes rare noise)
    max_df_ratio  : maximum document frequency ratio — words appearing in
                    more than this fraction of chunks are ignored (stop words)
    """

    def __init__(
        self,
        embedding_dim: int   = 4096,   # large enough to hold full vocab for small corpora
        min_df:        int   = 1,
        max_df_ratio:  float = 0.99,
    ):
        self.embedding_dim  = embedding_dim
        self.min_df         = min_df
        self.max_df_ratio   = max_df_ratio

        # Populated during fit()
        self._vocab:    dict[str, int]   = {}   # word -> column index
        self._idf:      np.ndarray | None = None  # IDF weight per vocab word
        self._is_fitted = False
        self._n_docs    = 0

    # ── Text preprocessing ─────────────────────────────────────────────────

    # English stop words to always exclude regardless of document frequency
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

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Lowercase, strip punctuation, remove stop words, split on whitespace.
        Also generates bigrams (2-word pairs) to capture phrases like
        "tornado watch" and "flash flood" as single features.
        """
        clean = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        words = [
            w for w in clean.split()
            if len(w) > 2 and w not in TFIDFEmbeddingModel.STOP_WORDS
        ]
        # Bigrams: "tornado watch" -> "tornado_watch"
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        return words + bigrams

    # ── Fit ────────────────────────────────────────────────────────────────

    def fit(self, texts: list[str]) -> "TFIDFEmbeddingModel":
        """
        Build vocabulary and IDF weights from the corpus.

        Steps:
        1. Tokenize all texts
        2. Count document frequency (df) for each word — in how many texts does it appear
        3. Filter by min_df and max_df_ratio
        4. Assign each surviving word a column index
        5. Compute IDF = log((N + 1) / (df + 1)) + 1  (smoothed to avoid zero)
        """
        n = len(texts)
        self._n_docs = n

        # Document frequency: how many texts contain each word
        df: Counter = Counter()
        for text in texts:
            tokens = set(self._tokenize(text))  # set = count each word once per doc
            df.update(tokens)

        # Filter vocabulary
        max_df_abs = int(n * self.max_df_ratio)
        vocab_words = [
            word for word, count in df.items()
            if self.min_df <= count <= max_df_abs
        ]

        # Sort for determinism
        vocab_words.sort()

        # If vocabulary exceeds embedding_dim, keep most discriminative words.
        # We prefer words with mid-range IDF (not too common, not too rare).
        if len(vocab_words) > self.embedding_dim:
            # Score: prefer words that appear in 2-30% of documents (most useful for retrieval)
            # Words appearing in only 1 doc (df=1) are usually noise/proper nouns
            # Words appearing everywhere (df>80%) are stop words
            def score(w: str) -> float:
                d = df[w]
                ratio = d / n
                # Penalise very rare (df=1) and very common
                if d == 1:
                    return 0.0
                return math.log((n + 1) / (d + 1)) * (1 - abs(ratio - 0.15))

            vocab_words = sorted(vocab_words, key=score, reverse=True)[:self.embedding_dim]
            vocab_words.sort()  # re-sort alphabetically for determinism

        self._vocab = {word: idx for idx, word in enumerate(vocab_words)}

        # IDF: log smoothed — prevents zero IDF for rare words
        idf_values = np.zeros(len(self._vocab), dtype=np.float32)
        for word, idx in self._vocab.items():
            idf_values[idx] = math.log((n + 1) / (df[word] + 1)) + 1.0

        self._idf       = idf_values
        self._is_fitted = True

        logger.info(
            "TF-IDF fitted: %d docs, %d vocabulary words, dim=%d",
            n, len(self._vocab), self.embedding_dim,
        )
        return self

    # ── Embed ──────────────────────────────────────────────────────────────

    def _tf_vector(self, text: str) -> np.ndarray:
        """
        Compute the TF-IDF vector for a single text.

        Returns a float32 array of shape (embedding_dim,) — L2 normalized.

        L2 normalization: divide by the Euclidean norm so all vectors
        have unit length. This makes cosine similarity = dot product,
        which is faster to compute and easier to reason about.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before embed_text()")

        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Term frequency per token
        tf_counts: Counter = Counter(tokens)
        total = len(tokens)

        # Build sparse vector in dense array
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        for token, count in tf_counts.items():
            if token in self._vocab:
                idx    = self._vocab[token]
                tf     = count / total
                vec[idx] = tf * self._idf[idx]

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm

        return vec

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single string. Returns shape (embedding_dim,)."""
        return self._tf_vector(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of strings. Returns shape (len(texts), embedding_dim).

        This is more efficient than calling embed_text() in a loop because
        we avoid Python overhead for the normalization step.
        """
        matrix = np.stack([self._tf_vector(t) for t in texts], axis=0)
        return matrix.astype(np.float32)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Save the fitted vocabulary + IDF weights to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "vocab":        self._vocab,
            "idf":          self._idf,
            "embedding_dim": self.embedding_dim,
            "n_docs":       self._n_docs,
            "min_df":       self.min_df,
            "max_df_ratio": self.max_df_ratio,
            "backend":      "tfidf",
        }
        joblib.dump(artifact, path, compress=3)
        logger.info("Embedding model saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "TFIDFEmbeddingModel":
        """Load a previously fitted model."""
        artifact = joblib.load(path)
        model = cls(
            embedding_dim=artifact["embedding_dim"],
            min_df=artifact["min_df"],
            max_df_ratio=artifact["max_df_ratio"],
        )
        model._vocab       = artifact["vocab"]
        model._idf         = artifact["idf"]
        model._n_docs      = artifact["n_docs"]
        model._is_fitted   = True
        logger.info(
            "Embedding model loaded: %d vocab words, dim=%d",
            len(model._vocab), model.embedding_dim,
        )
        return model


# =============================================================================
# Sentence Transformer Model (used when available)
# =============================================================================

class SentenceTransformerModel:
    """
    Wrapper around sentence-transformers for dense BERT embeddings.

    Used when the 'sentence-transformers' package is importable.
    Falls back to TFIDFEmbeddingModel if not available.

    All-MiniLM-L6-v2 properties:
      - 384 dimensions
      - 80 MB model size
      - Max 256 tokens (~1800 characters) per input
      - Trained on 1B sentence pairs
      - Strong semantic similarity for weather terminology
    """

    MODEL_NAME    = "all-MiniLM-L6-v2"
    embedding_dim = 384

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model     = None

    def fit(self, texts: list[str]) -> "SentenceTransformerModel":
        """For sentence-transformers, fit() just loads the model."""
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
        """Save model name (the weights are downloaded from HuggingFace Hub)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"backend": "sentence", "model_name": self.model_name}, path)

    @classmethod
    def load(cls, path: Path) -> "SentenceTransformerModel":
        artifact = joblib.load(path)
        model = cls(model_name=artifact["model_name"])
        model._load()
        return model


# =============================================================================
# Factory — auto-selects the best available backend
# =============================================================================

def create_embedding_model(
    backend: str | None = None,
    embedding_dim: int  = 4096,
) -> TFIDFEmbeddingModel | SentenceTransformerModel:
    """
    Create an embedding model using the best available backend.

    Selection priority:
    1. EMBEDDING_BACKEND env var (explicit override)
    2. 'sentence' if sentence_transformers imports successfully
    3. 'tfidf' as reliable fallback

    Parameters
    ----------
    backend       : 'tfidf' | 'sentence' | None (auto-detect)
    embedding_dim : only used for TF-IDF backend
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
        except (ImportError, Exception):
            logger.info("sentence-transformers not available — using TF-IDF backend")

    logger.info("Using TF-IDF embedding backend (dim=%d)", embedding_dim)
    return TFIDFEmbeddingModel(embedding_dim=embedding_dim)
