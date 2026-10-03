import pytest

from app.rag.chunking import RecursiveCharacterSplitter, chunk_directory
from tests.conftest import ROOT


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=100)


def test_chunks_respect_size_roughly():
    text = ". ".join(f"Sentence number {i} about rain and wind" for i in range(200))
    chunks = RecursiveCharacterSplitter(chunk_size=300, chunk_overlap=30).split_text(text)
    assert len(chunks) > 5
    assert all(len(c) <= 400 for c in chunks)


def test_short_text_is_one_chunk():
    assert len(RecursiveCharacterSplitter(500, 50).split_text("Short text.")) == 1


def test_reference_docs_have_metadata():
    chunks = chunk_directory(ROOT / "data" / "reference")
    assert {c.document_name for c in chunks} and len(chunks) > 20
    assert all(c.text.strip() and c.section and c.char_end > c.char_start for c in chunks)
