"""Answer generation with OpenAI gpt-5.1 over the retrieved multimodal context.

Text/PDF hits are injected as text context. Image hits are attached to the message
so the vision-capable model can actually look at them. Video/audio hits are cited by
name (their transcript/frames are not fed inline), so the model knows they were retrieved.
"""
from __future__ import annotations

import base64
from typing import Dict, List, Tuple

from config import OPENAI_MODEL, STORAGE_BUCKET, openai_client, supabase_client

SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Answer the user's question using ONLY the "
    "retrieved context provided (text snippets and any attached images). Cite sources by "
    "their [n] index. If the context is insufficient, say so plainly rather than guessing."
)


def _download(storage_path: str) -> bytes:
    return supabase_client().storage.from_(STORAGE_BUCKET).download(storage_path)


def _build_user_content(query: str, hits: List[Dict]) -> Tuple[list, List[Dict]]:
    """Build the OpenAI multimodal user-message content and a citation list."""
    parts: list = [{"type": "text", "text": f"Question: {query}\n\nRetrieved context:"}]
    citations: List[Dict] = []

    for i, hit in enumerate(hits, start=1):
        modality = hit.get("modality")
        source = hit.get("source_name")
        sim = hit.get("similarity")
        citations.append({"index": i, "source_name": source, "modality": modality, "similarity": sim})

        header = f"\n[{i}] source={source} modality={modality} similarity={sim:.3f}"
        if hit.get("content"):
            parts.append({"type": "text", "text": f"{header}\n{hit['content']}"})
        elif modality == "image" and hit.get("storage_path"):
            parts.append({"type": "text", "text": f"{header} (image attached below)"})
            try:
                raw = _download(hit["storage_path"])
                mime = (hit.get("metadata") or {}).get("mime_type", "image/png")
                b64 = base64.b64encode(raw).decode("ascii")
                parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                )
            except Exception as exc:
                parts.append({"type": "text", "text": f"(could not load image: {exc})"})
        else:
            parts.append(
                {"type": "text", "text": f"{header} (media of type {modality}; not shown inline)"}
            )

    return parts, citations


def answer(query: str, hits: List[Dict]) -> Dict:
    """Generate an answer from retrieved hits. Returns {answer, citations}."""
    if not hits:
        return {"answer": "No relevant documents were found in the vector store.", "citations": []}

    user_content, citations = _build_user_content(query, hits)
    resp = openai_client().chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return {"answer": resp.choices[0].message.content, "citations": citations}
