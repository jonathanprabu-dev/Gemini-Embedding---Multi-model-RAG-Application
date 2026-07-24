"""Retrieval: embed the query, call match_documents, attach signed media URLs."""
from __future__ import annotations

from typing import Dict, List, Optional

from config import MATCH_COUNT, STORAGE_BUCKET, supabase_client
from embeddings import embed_query

SIGNED_URL_TTL = 3600  # seconds


def _signed_url(storage_path: Optional[str]) -> Optional[str]:
    if not storage_path:
        return None
    try:
        res = supabase_client().storage.from_(STORAGE_BUCKET).create_signed_url(
            storage_path, SIGNED_URL_TTL
        )
        return res.get("signedURL") or res.get("signedUrl")
    except Exception:
        return None


def retrieve(query: str, match_count: int = MATCH_COUNT, filter: Optional[Dict] = None) -> List[Dict]:
    """Return the top-k matching rows, each augmented with a `signed_url`."""
    q_emb = embed_query(query)
    resp = supabase_client().rpc(
        "match_documents",
        {
            "query_embedding": q_emb,
            "match_count": match_count,
            "filter": filter or {},
        },
    ).execute()

    hits: List[Dict] = resp.data or []
    for hit in hits:
        hit["signed_url"] = _signed_url(hit.get("storage_path"))
    return hits
