"""Central configuration and shared clients.

All environment-driven constants live here so the embedding dimension and model
IDs are defined exactly once. EMBED_DIM must match the pgvector column
`embedding vector(1536)` and the `match_documents` RPC signature in Supabase.
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

# override=True so keys in .env take precedence over any stale system env vars.
load_dotenv(override=True)


def _require(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise RuntimeError(
            f"Missing required environment variable {name!r}. "
            f"Add it to your .env file (see .env.example)."
        )
    return val


# ---- Models ----
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-2-preview")

# Reasoning runs on a local Ollama model (no external API / billing).
# gemma3 is multimodal and can look at retrieved images; override in .env to swap models.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ---- Embedding invariants ----
# MRL-truncated + L2-normalized dimension. MUST equal the DB vector() size.
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))

# ---- Supabase ----
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://tvbvjzvvkzznnhoepfyb.supabase.co")
SUPABASE_PROJECT_ID = os.getenv("SUPABASE_PROJECT_ID", "tvbvjzvvkzznnhoepfyb")
STORAGE_BUCKET = os.getenv("STORAGE_BUCKET", "documents")

# ---- Retrieval ----
MATCH_COUNT = int(os.getenv("MATCH_COUNT", "5"))

# Chunking (characters)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


@lru_cache(maxsize=1)
def gemini_client():
    from google import genai

    return genai.Client(api_key=_require("GEMINI_API_KEY"))


@lru_cache(maxsize=1)
def ollama_client():
    from ollama import Client

    return Client(host=OLLAMA_HOST)


@lru_cache(maxsize=1)
def supabase_client():
    from supabase import create_client

    return create_client(SUPABASE_URL, _require("SUPABASE_SERVICE_ROLE_KEY"))
