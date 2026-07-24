# Multimodal RAG — Gemini Embedding 2 + Supabase + OpenAI

Embed text, PDFs, images, video, and audio into one vector space with **Gemini Embedding 2**,
store them in **Supabase pgvector**, and answer questions with **OpenAI gpt-5.1** over the
retrieved context. A **Streamlit** GUI drives the whole flow.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env     # then fill in your keys
```

Set these in `.env`:

- `GEMINI_API_KEY` — Google AI Studio key with Gemini Embedding 2 access
- `OPENAI_API_KEY` — OpenAI key
- `SUPABASE_SERVICE_ROLE_KEY` — Supabase → Project Settings → API Keys → `service_role` (server-side only)

## Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Use

- **📤 Upload & Embed** — drag-and-drop files, or drop them in `data/` and click *Embed everything in data/*.
- **💬 Query** — ask a question; the app retrieves the top matches and answers with citations.

Supported: text (`.txt .md .csv .json .html`), PDF, `.docx`, images, video, and audio.

## How it works

1. Files are embedded at **1536 dims** (Matryoshka-truncated + L2-normalized) and stored in `public.documents`.
2. A query is embedded, matched by cosine similarity via the `match_documents` RPC, and the top-k rows
   are passed to gpt-5.1 (images attached for the vision model).

| File | Role |
|------|------|
| `config.py` | env + clients, `EMBED_DIM=1536` |
| `embeddings.py` | Gemini embed (truncate + normalize) |
| `ingest.py` | modality detection, extraction, chunking, upload |
| `retrieve.py` | query embedding → `match_documents` |
| `reasoning.py` | build prompt → gpt-5.1 answer |
| `app.py` | Streamlit GUI |

## Notes

- `EMBED_DIM` is fixed at 1536 and must match the DB `vector(1536)` column and the `match_documents` signature.
- `.env` is gitignored. The `service_role` key bypasses RLS — keep it server-side only.
