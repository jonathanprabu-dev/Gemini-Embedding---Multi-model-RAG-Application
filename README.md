# Multimodal RAG — Gemini Embedding 2 + Supabase + Ollama

Embed text, PDFs, images, video, and audio into one vector space with **Gemini Embedding 2**,
store them in **Supabase pgvector**, and answer questions with a **local Ollama** vision model
over the retrieved context. A **Streamlit** GUI drives the whole flow.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env     # then fill in your keys
ollama pull gemma3:4b    # the local reasoning model
```

Set these in `.env`:

- `GEMINI_API_KEY` — Google AI Studio key with Gemini Embedding 2 access
- `SUPABASE_SERVICE_ROLE_KEY` — Supabase → Project Settings → API Keys → `service_role` (server-side only)

Reasoning is local, so there is no second API key or billing. [Ollama](https://ollama.com) must be
running; set `OLLAMA_MODEL` / `OLLAMA_HOST` in `.env` to point elsewhere.

## Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

### Start on boot (Windows)

`scripts\start-rag.cmd` starts Ollama (if it isn't already running), starts the Streamlit
server, waits for it to respond, then opens it in Chrome. A shortcut to it lives in
`shell:startup`, so it runs at every logon.

To disable, delete **Multimodal RAG.lnk** from `shell:startup` (Win+R → `shell:startup`).
To re-enable, put a shortcut to the script back there. Running the script by hand is safe —
it reuses an already-running server rather than starting a second one. It appends to
`scripts\start-rag.log`, which is where to look if a tab doesn't appear.

## Use

- **📤 Upload & Embed** — drag-and-drop files, or drop them in `data/` and click *Embed everything in data/*.
- **💬 Query** — ask a question; the app retrieves the top matches and answers with citations.
  Top-k, similarity threshold, and context-type filter live in the sidebar; the reasoning model
  can be switched off to see retrieved sources only.
- **📚 Browse** — list every ingested document by title and preview the stored original.

Supported: text (`.txt .md .csv .json .html`), PDF, `.docx`, images, video, and audio.

## How it works

1. Files are embedded at **1536 dims** (Matryoshka-truncated + L2-normalized) and stored in `public.documents`.
2. A query is embedded, matched by cosine similarity via the `match_documents` RPC, and the top-k rows
   are passed to the local Ollama model (images attached for the vision model).

| File | Role |
|------|------|
| `config.py` | env + clients, `EMBED_DIM=1536` |
| `embeddings.py` | Gemini embed (truncate + normalize) |
| `ingest.py` | modality detection, extraction, chunking, upload |
| `retrieve.py` | query embedding → `match_documents` |
| `reasoning.py` | build prompt → local Ollama answer |
| `app.py` | Streamlit GUI |

## Notes

- `EMBED_DIM` is fixed at 1536 and must match the DB `vector(1536)` column and the `match_documents` signature.
- `.env` is gitignored. The `service_role` key bypasses RLS — keep it server-side only.
