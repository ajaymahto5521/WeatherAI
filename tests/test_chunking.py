import sys
from types import SimpleNamespace

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


def test_chunk_directory_reads_nested_files_and_skips_excluded_folders(tmp_path):
    (tmp_path / "hazards").mkdir()
    (tmp_path / "_out_of_scope").mkdir()
    (tmp_path / "hazards" / "storm.txt").write_text("Severe wind safety.", encoding="utf-8")
    (tmp_path / "_out_of_scope" / "unrelated.txt").write_text("Not weather.", encoding="utf-8")

    chunks = chunk_directory(tmp_path)

    assert [chunk.document_name for chunk in chunks] == ["storm.txt"]


def test_pdf_text_is_split_into_page_sections(tmp_path, monkeypatch):
    pdf_path = tmp_path / "weather.pdf"
    pdf_path.write_bytes(b"mock PDF")
    fake_page = SimpleNamespace(extract_text=lambda: "Thunderstorms can produce lightning.")
    monkeypatch.setitem(
        sys.modules,
        "pypdf",
        SimpleNamespace(PdfReader=lambda path: SimpleNamespace(pages=[fake_page])),
    )

    chunks = SectionAwareSplitter().split_document(pdf_path)

    assert len(chunks) == 1
    assert chunks[0].section == "Page 1"
    assert "Thunderstorms can produce lightning." in chunks[0].text


def test_pdf_without_extractable_text_requires_ocr(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"mock PDF")
    fake_page = SimpleNamespace(extract_text=lambda: "")
    monkeypatch.setitem(
        sys.modules,
        "pypdf",
        SimpleNamespace(PdfReader=lambda path: SimpleNamespace(pages=[fake_page])),
    )

    with pytest.raises(ValueError, match="scanned PDFs need OCR"):
        SectionAwareSplitter().split_document(pdf_path)


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
