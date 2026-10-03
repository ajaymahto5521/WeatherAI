"""
app/rag/chunking.py
====================
Text chunking for the RAG pipeline.

Splits large documents into smaller, overlapping pieces that:
  1. Fit within the embedding model's token limit (~512 tokens / ~2000 chars)
  2. Are semantically coherent — prefer paragraph/sentence boundaries
  3. Overlap slightly so answers aren't lost at chunk edges

We implement two strategies:

  RecursiveCharacterSplitter  — the main splitter (used for ingestion)
    Tries separators in order: section boundary > paragraph > sentence > word > char
    Falls back to the next separator if the chunk is still too large.
    This mirrors LangChain's RecursiveCharacterTextSplitter logic.

  SectionAwareSplitter        — preserves SECTION: headers as metadata
    Wraps RecursiveCharacterSplitter but also extracts section names,
    document name, and source from the document header.

WHY NOT USE LANGCHAIN'S SPLITTER DIRECTLY?
We implement our own so you understand exactly what's happening.
LangChain's version is more general but does the same thing underneath.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class Chunk:
    """
    A single piece of text ready for embedding and storage.

    Every field here becomes a column in the database.
    The text is what gets embedded. The metadata is what gets returned
    to the user as "source" in the /ask response.
    """
    text: str                    # The chunk content — this gets embedded
    document_name: str           # e.g. "weather_alerts_definitions.txt"
    source: str                  # e.g. "NWS educational summary"
    section: str                 # e.g. "Tornado Watch and Warning"
    chunk_index: int             # position within the document (0-based)
    char_start: int              # starting character offset in original document
    char_end: int                # ending character offset in original document

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        return len(self.text)


# =============================================================================
# Recursive Character Splitter
# =============================================================================

class RecursiveCharacterSplitter:
    """
    Splits text into overlapping chunks, preferring natural boundaries.

    Strategy:
      1. Try to split on the first separator (e.g. double newline = paragraph)
      2. If any resulting piece is still larger than chunk_size, recursively
         split it using the next separator down
      3. Merge small consecutive pieces together until the chunk is full
      4. When merging, carry over `overlap` characters from the previous chunk

    Separators tried in order (highest to lowest preference):
      "\\n\\n"   — paragraph break
      "\\n---"   — section divider (our document format)
      "\\n"      — line break
      ". "       — sentence end
      " "        — word boundary
      ""         — character by character (last resort)

    Parameters
    ----------
    chunk_size : int
        Target maximum characters per chunk (default 500).
        Chunks may slightly exceed this if a sentence is long.
    chunk_overlap : int
        How many characters to repeat from the end of the previous chunk
        at the start of the next chunk (default 50).
    """

    # Try these separators in order. The first one that creates pieces
    # smaller than chunk_size is used. If none work, fall to the next.
    SEPARATORS = ["\n\n", "\n---", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        """
        Split text into a list of string chunks.

        This is the core method. The rest of the class wraps it with
        document-level metadata.
        """
        return self._split(text, self.SEPARATORS)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split text using the first separator that works.

        'Works' means all resulting pieces fit within chunk_size OR
        we've run out of separators.
        """
        if not text.strip():
            return []

        # If text already fits, return it as a single chunk
        if len(text) <= self.chunk_size:
            return [text.strip()]

        # Try each separator in order
        separator = separators[0] if separators else ""
        remaining_separators = separators[1:] if len(separators) > 1 else []

        # Split the text on this separator
        if separator:
            pieces = text.split(separator)
        else:
            # Last resort: split character by character
            pieces = list(text)

        # Recursively split any piece that is still too large
        good_pieces: list[str] = []
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if len(piece) <= self.chunk_size:
                good_pieces.append(piece)
            else:
                # This piece is still too large — recurse with next separator
                sub_pieces = self._split(piece, remaining_separators)
                good_pieces.extend(sub_pieces)

        # Merge small consecutive pieces into chunks of ~chunk_size
        return self._merge(good_pieces)

    def _merge(self, pieces: list[str]) -> list[str]:
        """
        Merge small pieces into chunks of ~chunk_size with overlap.

        Algorithm:
          - Accumulate pieces into a buffer
          - When the buffer would exceed chunk_size, emit it as a chunk
          - Start the next chunk with the last `overlap` characters of
            the emitted chunk (so information isn't lost at boundaries)
        """
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for piece in pieces:
            piece_len = len(piece)

            # If adding this piece would overflow the chunk
            if current_len + piece_len + 1 > self.chunk_size and current:
                # Emit the current buffer as a chunk
                chunk_text = " ".join(current).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                # Overlap: keep the tail of the current chunk for context
                # We rebuild `current` with just enough pieces to fill `overlap`
                overlap_text = chunk_text[-self.chunk_overlap:]
                current = [overlap_text] if overlap_text.strip() else []
                current_len = len(overlap_text)

            current.append(piece)
            current_len += piece_len + 1  # +1 for the space between pieces

        # Emit whatever remains
        if current:
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks


# =============================================================================
# Section-Aware Splitter
# =============================================================================

class SectionAwareSplitter:
    """
    Wraps RecursiveCharacterSplitter and extracts structured metadata
    from our specific document format.

    Our documents have a header like:
        DOCUMENT: Weather Alert and Warning Definitions
        SOURCE: Educational summary based on NWS terminology
        ...
    And section dividers like:
        SECTION: Tornado Watch and Warning

    This splitter:
      1. Parses the header to extract document_name and source
      2. Tracks which SECTION each chunk belongs to
      3. Returns Chunk objects with full metadata

    Parameters
    ----------
    chunk_size    : target characters per chunk
    chunk_overlap : overlap characters between consecutive chunks
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self._splitter = RecursiveCharacterSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_document(self, file_path: Path) -> list[Chunk]:
        """
        Read a document file and return a list of Chunk objects.

        Parameters
        ----------
        file_path : Path
            Path to a .txt reference document.

        Returns
        -------
        list[Chunk]
            One Chunk per text piece, with metadata populated.
        """
        text = file_path.read_text(encoding="utf-8")
        document_name = file_path.name

        # Extract header metadata
        doc_meta   = self._parse_header(text)
        source     = doc_meta.get("SOURCE", "Unknown source")

        # Split the full text into raw string chunks
        raw_chunks = self._splitter.split_text(text)

        chunks: list[Chunk] = []
        current_section = "Introduction"
        char_pos = 0

        for i, raw_text in enumerate(raw_chunks):
            # Detect section changes by scanning for "SECTION:" in the chunk.
            # We take only the first line after SECTION: to get a clean title.
            section_match = re.search(r"SECTION:\s*([^\n]{1,80})", raw_text)
            if section_match:
                # Grab up to the first newline or 80 chars — avoids spilling
                # into paragraph text when the section header has no newline
                raw_section = section_match.group(1).strip()
                # Trim at the first sentence-ending punctuation or newline
                raw_section = re.split(r"[.\n]", raw_section)[0].strip()
                if raw_section:
                    current_section = raw_section

            # Estimate character position in original document
            char_start = text.find(raw_text[:50], char_pos)
            if char_start == -1:
                char_start = char_pos  # fallback
            char_end = char_start + len(raw_text)
            char_pos = max(char_pos, char_start)

            chunk = Chunk(
                text=raw_text,
                document_name=document_name,
                source=source,
                section=current_section,
                chunk_index=i,
                char_start=char_start,
                char_end=char_end,
            )
            chunks.append(chunk)

        return chunks

    def _parse_header(self, text: str) -> dict[str, str]:
        """
        Extract key:value pairs from the document header.

        Looks for lines like:
            DOCUMENT: Weather Alert and Warning Definitions
            SOURCE: Educational summary based on NWS terminology
        """
        meta: dict[str, str] = {}
        header_pattern = re.compile(r"^([A-Z]+):\s*(.+)$", re.MULTILINE)

        for match in header_pattern.finditer(text[:500]):  # only scan first 500 chars
            key   = match.group(1).strip()
            value = match.group(2).strip()
            if key in ("DOCUMENT", "SOURCE", "VERSION"):
                meta[key] = value

        return meta


# =============================================================================
# Convenience function
# =============================================================================

def chunk_directory(
    directory: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    extensions: tuple[str, ...] = (".txt", ".md"),
) -> list[Chunk]:
    """
    Chunk all documents in a directory.

    Parameters
    ----------
    directory   : path to folder containing reference documents
    chunk_size  : target characters per chunk
    chunk_overlap: overlap between chunks
    extensions  : file extensions to process

    Returns
    -------
    list[Chunk]
        All chunks from all documents, in document order.
    """
    splitter = SectionAwareSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: list[Chunk] = []
    files = sorted(
        f for f in directory.iterdir()
        if f.suffix.lower() in extensions and f.name != ".gitkeep"
    )

    for file_path in files:
        doc_chunks = splitter.split_document(file_path)
        all_chunks.extend(doc_chunks)

    return all_chunks
