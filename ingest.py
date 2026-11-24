"""Document ingestion: text extraction + chunking with overlap.

Supported inputs: ``.pdf`` (via PyPDF) and ``.txt`` / ``.md`` (plain text).
Each piece of text is split into overlapping character chunks so that an
answer spanning a chunk boundary is never cut in half. Every chunk keeps
its source filename and page number, which is what makes citations possible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    """One retrievable piece of a document."""

    text: str
    source: str  # filename, e.g. "sample_guide.txt"
    page: int = 1  # 1-based page number (1 for plain-text files)
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)

    def citation(self) -> str:
        """Short human-readable citation, e.g. ``sample_guide.txt, p. 1``."""
        return f"{self.source}, p. {self.page}"


def extract_text_from_pdf(path: str) -> list[tuple[int, str]]:
    """Return ``[(page_number, page_text), ...]`` for a PDF file.

    Raises ImportError with a clear message if PyPDF is not installed.
    """
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - exercised only without pypdf
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Reading PDFs needs PyPDF: pip install pypdf"
            ) from exc
    reader = PdfReader(path)
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        pages.append((i, page.extract_text() or ""))
    return pages


def extract_text_from_txt(path: str) -> list[tuple[int, str]]:
    """Return ``[(1, full_text)]`` for a plain-text / Markdown file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [(1, f.read())]


def extract_text(path: str) -> list[tuple[int, str]]:
    """Extract ``(page, text)`` pairs from a supported file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext in {".txt", ".md"}:
        return extract_text_from_txt(path)
    raise ValueError(f"Unsupported file type: {ext!r} (use .pdf, .txt or .md)")


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    source: str = "",
    page: int = 1,
) -> list[DocumentChunk]:
    """Split ``text`` into overlapping chunks.

    Sliding window: each chunk is up to ``chunk_size`` characters long and
    the next chunk starts ``chunk_size - chunk_overlap`` characters later,
    so the last ``chunk_overlap`` characters of one chunk repeat at the
    start of the next. Short texts produce a single chunk; empty text
    produces none.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [DocumentChunk(text=text, source=source, page=page, chunk_index=0)]

    chunks: list[DocumentChunk] = []
    step = chunk_size - chunk_overlap
    start = 0
    index = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                DocumentChunk(text=piece, source=source, page=page, chunk_index=index)
            )
            index += 1
        if end >= len(text):
            break
        start += step
    return chunks


def load_document(
    path: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[DocumentChunk]:
    """Extract a file and return all of its chunks (source + page kept)."""
    source = os.path.basename(path)
    chunks: list[DocumentChunk] = []
    for page, page_text in extract_text(path):
        chunks.extend(
            chunk_text(
                page_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source=source,
                page=page,
            )
        )
    # Re-number chunk indices across the whole document.
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i
    return chunks


def load_text(
    text: str, source: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[DocumentChunk]:
    """Chunk already-extracted text (handy for tests and the Streamlit app)."""
    return chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                      source=source, page=1)
