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
        return self._merge(self._pieces(text, self.SEPARATORS))

    def _pieces(self, text: str, separators: list[str]) -> list[str]:
        """Cut at the coarsest separator, going finer only for pieces still over chunk_size."""
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text] if text else []

        pieces: list[str] = []
        for part in text.split(separators[0]) if separators[0] else list(text):
            pieces.extend(self._pieces(part, separators[1:]))
        return pieces

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
                if " " in overlap_text:  # don't start the next chunk mid-word
                    overlap_text = overlap_text.split(" ", 1)[1]
                current = [overlap_text] if overlap_text.strip() else []
                current_len = len(overlap_text)

            current.append(piece)
            current_len += piece_len + 1  # +1 for the joining space

        if current:
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks


# "## Heading" (markdown, any depth) or the older "SECTION: title" line
HEADING = re.compile(r"^(?:(#{1,6})[ 	]+(.+?)|SECTION:[ 	]*(.+?))[ 	]*$", re.MULTILINE)
SOURCE = re.compile(r"^\**(?:SOURCE|Regulatory Standard):\**[ 	]*(.+?)[ 	]*$", re.MULTILINE)


def clean_markdown(text: str) -> str:
    """Plain text for embedding: no bold markers or rules, tables turned into sentences.

    Each table row becomes "Column: value; Column: value" so a chunk still
    says what its numbers mean once it is cut away from the header row.
    """
    out: list[str] = []
    header: list[str] | None = None
    for line in text.replace("**", "").splitlines():
        if line.lstrip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if header is None:
                header = cells
            elif not all(re.fullmatch(r":?-+:?", c) for c in cells):  # skip the |---| row
                out.append("; ".join(f"{h}: {c}" for h, c in zip(header, cells)))
        else:
            header = None
            if line.strip() != "---":
                out.append(line)
    return "\n".join(out)


class SectionAwareSplitter:
    """Splits a document at its headings, then chunks each section on its own.

    Every chunk starts with its section path, e.g. "Section 3: Crane Protocols >
    3.1 Tower Crane Cut-off", so it still makes sense when read in isolation.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self._splitter = RecursiveCharacterSplitter(chunk_size, chunk_overlap)

    def split_document(self, file_path: Path) -> list[Chunk]:
        if file_path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError(
                    "PDF ingestion requires pypdf; install project requirements."
                ) from exc

            pages = PdfReader(file_path).pages
            page_texts = [
                (page_number, page.extract_text() or "")
                for page_number, page in enumerate(pages, 1)
            ]
            if not any(page_text.strip() for _, page_text in page_texts):
                raise ValueError(
                    f"No extractable text found in {file_path}; scanned PDFs need OCR."
                )
            text = "\n\n".join(
                f"## Page {page_number}\n{page_text}"
                for page_number, page_text in page_texts
                if page_text.strip()
            )
        else:
            text = file_path.read_text(encoding="utf-8")
        source_match = SOURCE.search(text[:1000])
        source = source_match.group(1) if source_match else "Unknown source"

        # (section path, start of its body, end of its body)
        headings = list(HEADING.finditer(text))
        bounds = [0] + [m.start() for m in headings] + [len(text)]
        titles = ["Introduction"]
        stack: list[tuple[int, str]] = []  # (heading level, title) of the open headings
        for m in headings:
            level = len(m.group(1)) if m.group(1) else 2
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2) or m.group(3)))
            # the document's own "# Title" is left out of the path unless it is the only heading
            path = [t for lv, t in stack if lv > 1] or [stack[-1][1]]
            titles.append(" > ".join(path))

        chunks: list[Chunk] = []
        for i, (title, start, end) in enumerate(zip(titles, bounds, bounds[1:])):
            body = text[start:end]
            if i:  # every part after the first begins with its own heading line
                body = HEADING.sub("", body, count=1)
            for piece in self._splitter.split_text(clean_markdown(body)):
                # offsets are approximate: cleaning and overlap change the text
                char_start = max(text.find(piece[:30], start, end), start)
                chunks.append(Chunk(
                    text=f"{title}\n{piece}",
                    document_name=file_path.name,
                    source=source,
                    section=title,
                    chunk_index=len(chunks),
                    char_start=char_start,
                    char_end=char_start + len(piece),
                ))
        return chunks


def chunk_directory(
    directory: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    extensions: tuple[str, ...] = (".txt", ".md", ".pdf"),
) -> list[Chunk]:
    """Chunk supported documents recursively, excluding explicitly out-of-scope folders."""
    splitter = SectionAwareSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    excluded_directories = {"_optional", "_out_of_scope", "_unprocessed"}

    files = sorted(
        (
            f for f in directory.rglob("*")
            if f.is_file()
            and f.suffix.lower() in extensions
            and f.name != ".gitkeep"
            and not excluded_directories.intersection(f.relative_to(directory).parts[:-1])
        ),
        key=lambda f: f.relative_to(directory).as_posix().casefold(),
    )

    all_chunks: list[Chunk] = []
    for file_path in files:
        all_chunks.extend(splitter.split_document(file_path))
    return all_chunks
