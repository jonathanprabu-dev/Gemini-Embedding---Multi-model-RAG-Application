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


def retrieve(
    query: str,
    match_count: int = MATCH_COUNT,
    filter: Optional[Dict] = None,
    match_threshold: float = 0.0,
    modality: Optional[str] = None,
) -> List[Dict]:
    """Return the top-k matching rows, each augmented with a `signed_url`.

    `match_threshold` drops hits below the given cosine similarity (0..1) and
    `modality` restricts results to a single context type (text, pdf, image,
    video, audio). Both are applied in SQL by the `match_documents` RPC.
    """
    q_emb = embed_query(query)
    resp = supabase_client().rpc(
        "match_documents",
        {
            "query_embedding": q_emb,
            "match_count": match_count,
            "filter": filter or {},
            "match_threshold": match_threshold,
            "modality_filter": modality,
        },
    ).execute()

    hits: List[Dict] = resp.data or []
    for hit in hits:
        hit["signed_url"] = _signed_url(hit.get("storage_path"))
    return hits


def list_documents() -> List[Dict]:
    """List ingested documents grouped by source_name, for the Browse tab.

    Each source may span several chunk rows; they're collapsed into one entry
    with a `chunks` count and a `signed_url` for previewing the original file.
    """
    resp = (
        supabase_client()
        .table("documents")
        .select("source_name, modality, storage_path")
        .execute()
    )
    rows = resp.data or []
    docs: Dict[str, Dict] = {}
    for row in rows:
        name = row["source_name"]
        doc = docs.setdefault(
            name,
            {
                "source_name": name,
                "modality": row["modality"],
                "storage_path": row.get("storage_path"),
                "chunks": 0,
            },
        )
        doc["chunks"] += 1
    ordered = sorted(docs.values(), key=lambda d: d["source_name"].lower())
    for doc in ordered:
        doc["signed_url"] = _signed_url(doc.get("storage_path"))
    return ordered


def database_stats() -> Dict:
    """Return a summary of ingested documents for the sidebar."""
    resp = supabase_client().table("documents").select("modality, source_name").execute()
    rows = resp.data or []
    by_modality: Dict[str, int] = {}
    sources = set()
    for row in rows:
        by_modality[row["modality"]] = by_modality.get(row["modality"], 0) + 1
        sources.add(row["source_name"])
    return {
        "total_vectors": len(rows),
        "total_sources": len(sources),
        "by_modality": dict(sorted(by_modality.items())),
    }
