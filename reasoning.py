"""Answer generation over the retrieved multimodal context.

Reasoning runs on a local Ollama model (default qwen3.5:4b) — no external API
or billing. Text/PDF hits are injected as text context. Image hits are attached to the
message so the vision model can actually look at them (Ollama takes images as a list of
base64 strings on the message, not interleaved content parts). Video/audio hits are cited
by name (their transcript/frames are not fed inline), so the model knows they were retrieved.
"""
from __future__ import annotations

import base64
from typing import Dict, List, Tuple

from config import OLLAMA_MODEL, STORAGE_BUCKET, ollama_client, supabase_client

SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Answer the user's question using ONLY the "
    "retrieved context provided (text snippets and any attached images). Cite sources by "
    "their [n] index. If the context is insufficient, say so plainly rather than guessing."
)


def _download(storage_path: str) -> bytes:
    return supabase_client().storage.from_(STORAGE_BUCKET).download(storage_path)


def _build_prompt(query: str, hits: List[Dict]) -> Tuple[str, List[str], List[Dict]]:
    """Build the text prompt, a list of base64 images, and a citation list."""
    lines: List[str] = [f"Question: {query}", "", "Retrieved context:"]
    images: List[str] = []
    citations: List[Dict] = []

    for i, hit in enumerate(hits, start=1):
        modality = hit.get("modality")
        source = hit.get("source_name")
        sim = hit.get("similarity")
        citations.append({"index": i, "source_name": source, "modality": modality, "similarity": sim})

        header = f"\n[{i}] source={source} modality={modality} similarity={sim:.3f}"
        if hit.get("content"):
            lines.append(f"{header}\n{hit['content']}")
        elif modality == "image" and hit.get("storage_path"):
            lines.append(f"{header} (image attached)")
            try:
                raw = _download(hit["storage_path"])
                images.append(base64.b64encode(raw).decode("ascii"))
            except Exception as exc:
                lines.append(f"(could not load image: {exc})")
        else:
            lines.append(f"{header} (media of type {modality}; not shown inline)")

    return "\n".join(lines), images, citations


def answer(query: str, hits: List[Dict]) -> Dict:
    """Generate an answer from retrieved hits. Returns {answer, citations}."""
    if not hits:
        return {"answer": "No relevant documents were found in the vector store.", "citations": []}

    prompt, images, citations = _build_prompt(query, hits)
    user_msg: Dict = {"role": "user", "content": prompt}
    if images:
        user_msg["images"] = images

    resp = ollama_client().chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            user_msg,
        ],
    )
    return {"answer": resp["message"]["content"], "citations": citations}
