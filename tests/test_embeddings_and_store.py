import numpy as np
import pytest

from app.rag.chunking import Chunk
from app.rag.embeddings import TFIDFEmbeddingModel
from app.rag.vector_store import LocalVectorStore

TEXTS = ["tornado warning means a tornado is occurring", "monsoon rain arrives in june",
         "dew point measures atmospheric moisture"]


def test_tfidf_vectors_are_normalised_and_deterministic():
    m = TFIDFEmbeddingModel().fit(TEXTS)
    v = m.embed_batch(TEXTS)
    assert v.shape == (3, m.embedding_dim)
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert np.allclose(m.embed_text(TEXTS[0]), v[0])


def test_tfidf_save_load_roundtrip(tmp_path):
    m = TFIDFEmbeddingModel().fit(TEXTS)
    m.save(tmp_path / "e.joblib")
    m2 = TFIDFEmbeddingModel.load(tmp_path / "e.joblib")
    assert np.allclose(m.embed_text("monsoon"), m2.embed_text("monsoon"))


def _chunks():
    return [Chunk(t, "d.txt", "src", f"s{i}", i, 0, len(t)) for i, t in enumerate(TEXTS)]


def test_local_store_search_persist_reset(tmp_path):
    m = TFIDFEmbeddingModel().fit(TEXTS)
    store = LocalVectorStore(tmp_path)
    assert store.search(m.embed_text("x")) == []
    store.add(_chunks(), m.embed_batch(TEXTS))
    hit = store.search(m.embed_text("when does the monsoon rain arrive"), top_k=1)[0]
    assert hit.chunk.section == "s1"
    assert LocalVectorStore(tmp_path).count() == 3  # reloaded from disk
    store.reset()
    assert store.count() == 0 and LocalVectorStore(tmp_path).count() == 0


def test_store_rejects_length_mismatch(tmp_path):
    with pytest.raises(ValueError):
        LocalVectorStore(tmp_path).add(_chunks(), np.zeros((2, 4), dtype=np.float32))
