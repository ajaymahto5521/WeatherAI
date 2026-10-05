"""Split reference documents into overlapping chunks with section metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    text: str
    document_name: str   # e.g. "weather_alerts_definitions.txt"
    source: str          # from the document's SOURCE: header
    section: str         # nearest preceding "SECTION:" title
    chunk_index: int     # 0-based position within the document
    char_start: int      # approximate offsets into the original text
    char_end: int

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        return len(self.text)


class RecursiveCharacterSplitter:
    """Splits on the coarsest separator that works, recursing to finer ones.

    Same idea as LangChain's RecursiveCharacterTextSplitter. Small pieces are
    merged back up to chunk_size, carrying chunk_overlap chars between chunks.
    """

    SEPARATORS = ["\n\n", "\n---", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        return self._split(text, self.SEPARATORS)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not text.strip():
            return []
        if len(text) <= self.chunk_size:
            return [text.strip()]

        separator = separators[0] if separators else ""
        remaining = separators[1:]

        pieces = text.split(separator) if separator else list(text)

        good_pieces: list[str] = []
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if len(piece) <= self.chunk_size:
                good_pieces.append(piece)
            else:
                good_pieces.extend(self._split(piece, remaining))

        return self._merge(good_pieces)

    def _merge(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for piece in pieces:
            piece_len = len(piece)

            if current_len + piece_len + 1 > self.chunk_size and current:
                chunk_text = " ".join(current).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                # seed the next chunk with the tail of this one
                overlap_text = chunk_text[-self.chunk_overlap:]
                current = [overlap_text] if overlap_text.strip() else []
                current_len = len(overlap_text)

            current.append(piece)
            current_len += piece_len + 1  # +1 for the joining space

        if current:
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks


class SectionAwareSplitter:
    """Chunks a reference doc and tags each chunk with its SECTION: title.

    Docs start with a DOCUMENT:/SOURCE: header and use "SECTION: <title>" lines.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self._splitter = RecursiveCharacterSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_document(self, file_path: Path) -> list[Chunk]:
        text = file_path.read_text(encoding="utf-8")
        document_name = file_path.name
        source = self._parse_header(text).get("SOURCE", "Unknown source")

        chunks: list[Chunk] = []
        current_section = "Introduction"
        char_pos = 0

        for i, raw_text in enumerate(self._splitter.split_text(text)):
            section_match = re.search(r"SECTION:\s*([^\n]{1,80})", raw_text)
            if section_match:
                # cut at first '.' so a header with no newline doesn't swallow body text
                raw_section = re.split(r"[.\n]", section_match.group(1).strip())[0].strip()
                if raw_section:
                    current_section = raw_section

            # offsets are approximate: overlap and whitespace stripping mean the
            # chunk may not appear verbatim in the original
            char_start = text.find(raw_text[:50], char_pos)
            if char_start == -1:
                char_start = char_pos
            char_end = char_start + len(raw_text)
            char_pos = max(char_pos, char_start)

            chunks.append(
                Chunk(
                    text=raw_text,
                    document_name=document_name,
                    source=source,
                    section=current_section,
                    chunk_index=i,
                    char_start=char_start,
                    char_end=char_end,
                )
            )

        return chunks

    def _parse_header(self, text: str) -> dict[str, str]:
        meta: dict[str, str] = {}
        header_pattern = re.compile(r"^([A-Z]+):\s*(.+)$", re.MULTILINE)

        for match in header_pattern.finditer(text[:500]):
            key = match.group(1).strip()
            if key in ("DOCUMENT", "SOURCE", "VERSION"):
                meta[key] = match.group(2).strip()

        return meta


def chunk_directory(
    directory: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    extensions: tuple[str, ...] = (".txt", ".md"),
) -> list[Chunk]:
    """Chunk every matching file in a directory, in filename order."""
    splitter = SectionAwareSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    files = sorted(
        f for f in directory.iterdir()
        if f.suffix.lower() in extensions and f.name != ".gitkeep"
    )

    all_chunks: list[Chunk] = []
    for file_path in files:
        all_chunks.extend(splitter.split_document(file_path))
    return all_chunks
