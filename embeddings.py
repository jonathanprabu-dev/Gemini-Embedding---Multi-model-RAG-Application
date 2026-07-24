"""Gemini Embedding 2 wrappers.

Invariants (do not break):
  * Every embedding is requested at output_dimensionality=EMBED_DIM (MRL truncation),
    then defensively truncated to EMBED_DIM and **manually L2-normalized** — Gemini
    returns non-unit vectors whenever output_dimensionality < 3072.
  * Stored content uses task_type RETRIEVAL_DOCUMENT; search queries use RETRIEVAL_QUERY.
"""
from __future__ import annotations

from typing import List

import numpy as np

from config import EMBED_DIM, EMBED_MODEL, gemini_client


def _normalize(vec: List[float]) -> List[float]:
    arr = np.asarray(vec, dtype=np.float32)[:EMBED_DIM]  # MRL truncation safeguard
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        raise ValueError("Received a zero-norm embedding from Gemini.")
    return (arr / norm).tolist()


def _embed(contents, task_type: str) -> List[List[float]]:
    """Call Gemini embed_content and return L2-normalized EMBED_DIM vectors."""
    from google.genai import types

    resp = gemini_client().models.embed_content(
        model=EMBED_MODEL,
        contents=contents,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBED_DIM,
        ),
    )
    return [_normalize(e.values) for e in resp.embeddings]


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of document text chunks (RETRIEVAL_DOCUMENT)."""
    if not texts:
        return []
    return _embed(texts, task_type="RETRIEVAL_DOCUMENT")


def embed_media(data: bytes, mime_type: str) -> List[float]:
    """Embed a single media item (image/video/audio/pdf) as a document.

    Uses the multimodal capability of gemini-embedding-2 — the bytes are passed
    as an inline Part and land in the same vector space as text.
    """
    from google.genai import types

    part = types.Part.from_bytes(data=data, mime_type=mime_type)
    return _embed(part, task_type="RETRIEVAL_DOCUMENT")[0]


def embed_query(text: str) -> List[float]:
    """Embed a search query (RETRIEVAL_QUERY)."""
    return _embed(text, task_type="RETRIEVAL_QUERY")[0]
