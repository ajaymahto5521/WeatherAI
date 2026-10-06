import pytest

from app.rag.chunking import RecursiveCharacterSplitter, SectionAwareSplitter, chunk_directory
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


def test_markdown_sections_and_tables():
    chunks = SectionAwareSplitter().split_document(
        ROOT / "data" / "reference" / "severe_wind_and_storm_sop.md"
    )
    sections = {c.section for c in chunks}
    assert any(s.endswith("3.1 Tower Crane Weather Vaning & Cut-off Limit (45 km/h Limit)") for s in sections)
    assert all("|" not in c.text and "**" not in c.text for c in chunks)
    # a table row keeps its column names, so the number still has a meaning
    cutoff = [c for c in chunks if "Strong Breeze (Cut-Off)" in c.text]
    assert cutoff and "Wind Speed (km/h): 45" in cutoff[0].text
    assert chunks[0].source.startswith("OSHA")


def test_merged_chunks_do_not_repeat_the_overlap():
    text = "\n".join(f"Row {i}: " + "x" * 200 for i in range(10))
    chunks = RecursiveCharacterSplitter(chunk_size=500, chunk_overlap=50).split_text(text)
    assert all(c.count("Row 0:") <= 1 for c in chunks)
    assert all(len(c) <= 500 for c in chunks)
