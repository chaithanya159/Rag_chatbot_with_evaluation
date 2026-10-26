"""
loaders/document_loader.py
---------------------------
Loads a file from disk, extracts text, and splits it into chunked
Document objects (page_content + metadata), same role as the reference
repo's document_loader but with our own chunking logic (no LangChain).
"""

import os
import re
from dataclasses import dataclass, field
from typing import List

from pypdf import PdfReader
import docx

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

# Defaults used when the caller does not pass chunk settings.
# 200 / 50 came out best in the retrieval experiments (see experiment.py).
CHUNK_SIZE = 200   # words per chunk
CHUNK_OVERLAP = 50 # words repeated between chunks


@dataclass
class Document:
    """Minimal stand-in for a langchain-style Document."""
    page_content: str
    metadata: dict = field(default_factory=dict)


# ---------- raw text extraction ----------

def _load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_pdf(file_path: str) -> List[tuple]:
    """Returns list of (page_number, text) so chunks can carry page metadata."""
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def _load_docx(file_path: str) -> str:
    document = docx.Document(file_path)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


# ---------- chunking ----------

def _split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_plain_text(text: str, source: str, page: int = None,
                      chunk_size: int = None, chunk_overlap: int = None) -> List[Document]:
    # Resolve defaults at call time so changing the module constants still works.
    size = chunk_size or CHUNK_SIZE
    overlap = CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    overlap = max(0, min(overlap, size // 2))  # an overlap above half a chunk stops chunks making progress

    paragraphs = _split_paragraphs(text)
    pieces = []
    for para in paragraphs:
        if len(para.split()) <= size:
            pieces.append(para)
        else:
            pieces.extend(_split_sentences(para))

    docs = []
    current_words: List[str] = []
    chunk_id = 0

    def flush():
        nonlocal chunk_id
        if current_words:
            meta = {"source": source, "chunk_id": chunk_id}
            if page is not None:
                meta["page"] = page
            docs.append(Document(page_content=" ".join(current_words), metadata=meta))
            chunk_id += 1

    for piece in pieces:
        piece_words = piece.split()
        if current_words and len(current_words) + len(piece_words) > size:
            flush()
            overlap_words = current_words[-overlap:] if overlap else []
            current_words = overlap_words + piece_words
        else:
            current_words.extend(piece_words)
    flush()

    return docs


# ---------- public entry point ----------

def load_document(file_path: str, chunk_size: int = None, chunk_overlap: int = None) -> List[Document]:
    """
    Loads a file and returns a list of chunked Document objects.
    chunk_size / chunk_overlap are in words; when omitted the module defaults apply.
    Raises ValueError for unsupported/unreadable files.
    """
    ext = os.path.splitext(file_path)[1].lower()
    source = os.path.basename(file_path)

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    if ext == ".pdf":
        pages = _load_pdf(file_path)
        if not pages:
            raise ValueError("No extractable text found in PDF.")
        docs: List[Document] = []
        for page_num, page_text in pages:
            docs.extend(_chunk_plain_text(page_text, source, page=page_num,
                                          chunk_size=chunk_size, chunk_overlap=chunk_overlap))
        return docs

    elif ext == ".docx":
        text = _load_docx(file_path)
    else:  # .txt / .md
        text = _load_txt(file_path)

    if not text.strip():
        raise ValueError(f"No extractable text found in {source}.")

    return _chunk_plain_text(text, source, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
