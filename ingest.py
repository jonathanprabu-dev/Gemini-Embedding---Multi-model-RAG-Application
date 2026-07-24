"""Ingestion pipeline: detect modality, extract/chunk, upload original, embed, insert.

Public entry points:
  * ingest_file(filename, data)  -> dict summary (used by the GUI)
  * ingest_path(path)            -> dict summary (used for bulk local files in data/)
"""
from __future__ import annotations

import io
import mimetypes
import os
import uuid
from typing import Dict, List, Tuple

from config import CHUNK_OVERLAP, CHUNK_SIZE, STORAGE_BUCKET, supabase_client
from embeddings import embed_media, embed_texts

# ---- Modality detection by extension ----
TEXT_EXT = {".txt", ".md", ".markdown", ".csv", ".json", ".html", ".htm", ".rst", ".log"}
PDF_EXT = {".pdf"}
DOCX_EXT = {".docx"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}

MIME_OVERRIDES = {
    ".md": "text/markdown",
    ".mov": "video/quicktime",
    ".m4a": "audio/mp4",
    ".webp": "image/webp",
}


def _ext(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()


def detect_modality(filename: str) -> str:
    e = _ext(filename)
    if e in IMAGE_EXT:
        return "image"
    if e in VIDEO_EXT:
        return "video"
    if e in AUDIO_EXT:
        return "audio"
    if e in PDF_EXT:
        return "pdf"
    if e in TEXT_EXT or e in DOCX_EXT:
        return "text"
    return "other"


def guess_mime(filename: str) -> str:
    e = _ext(filename)
    if e in MIME_OVERRIDES:
        return MIME_OVERRIDES[e]
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


# ---- Text extraction ----
def extract_text(filename: str, data: bytes) -> str:
    e = _ext(filename)
    if e in PDF_EXT:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if e in DOCX_EXT:
        try:
            from docx import Document  # python-docx (optional)
        except ImportError as exc:
            raise RuntimeError("Install python-docx to ingest .docx files.") from exc
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    # plain text-ish formats
    return data.decode("utf-8", errors="replace")


def chunk_text(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - CHUNK_OVERLAP  # sliding window with overlap
    return chunks


# ---- Storage ----
def upload_original(filename: str, data: bytes) -> str:
    """Upload the original file to the private Storage bucket; return its path."""
    sb = supabase_client()
    path = f"{uuid.uuid4().hex}/{filename}"
    sb.storage.from_(STORAGE_BUCKET).upload(
        path=path,
        file=data,
        file_options={"content-type": guess_mime(filename), "upsert": "false"},
    )
    return path


# ---- Row insertion ----
def _insert_rows(rows: List[Dict]) -> int:
    if not rows:
        return 0
    supabase_client().table("documents").insert(rows).execute()
    return len(rows)


def ingest_file(filename: str, data: bytes) -> Dict:
    """Ingest one file (bytes already in memory). Returns a summary dict."""
    modality = detect_modality(filename)
    storage_path = upload_original(filename, data)
    rows: List[Dict] = []

    if modality in ("text", "pdf"):
        text = extract_text(filename, data)
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError(f"No extractable text found in {filename!r}.")
        vectors = embed_texts(chunks)
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            rows.append(
                {
                    "source_name": filename,
                    "modality": "pdf" if modality == "pdf" else "text",
                    "storage_path": storage_path,
                    "chunk_index": i,
                    "content": chunk,
                    "metadata": {"chunks": len(chunks)},
                    "embedding": vec,
                }
            )
    elif modality in ("image", "video", "audio"):
        vec = embed_media(data, guess_mime(filename))
        rows.append(
            {
                "source_name": filename,
                "modality": modality,
                "storage_path": storage_path,
                "chunk_index": 0,
                "content": None,
                "metadata": {"mime_type": guess_mime(filename)},
                "embedding": vec,
            }
        )
    else:
        raise ValueError(f"Unsupported file type for {filename!r} (modality=other).")

    inserted = _insert_rows(rows)
    return {
        "filename": filename,
        "modality": modality,
        "chunks": inserted,
        "storage_path": storage_path,
    }


def ingest_path(path: str) -> Dict:
    with open(path, "rb") as f:
        data = f.read()
    return ingest_file(os.path.basename(path), data)


def ingest_directory(directory: str) -> List[Dict]:
    """Ingest every supported file in a directory (non-recursive)."""
    results: List[Dict] = []
    for name in sorted(os.listdir(directory)):
        full = os.path.join(directory, name)
        if not os.path.isfile(full) or name.startswith("."):
            continue
        results.append(ingest_path(full))
    return results
