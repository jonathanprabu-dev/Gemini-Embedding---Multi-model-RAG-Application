# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A multimodal RAG app: files of any modality (text/PDF/docx/image/video/audio) are embedded into **one shared vector space** with Gemini Embedding 2, stored in Supabase pgvector + Storage, retrieved by cosine similarity, and answered by a **local Ollama** vision model. Streamlit is the only UI.

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in GEMINI_API_KEY + SUPABASE_SERVICE_ROLE_KEY
streamlit run app.py          # http://localhost:8501
```

Headless run (for driving with Playwright/screenshots):
```bash
python -m streamlit run app.py --server.headless true --server.port 8501
```

Reasoning requires a running Ollama daemon with the configured model pulled (`ollama pull gemma3:4b`, or whatever `OLLAMA_MODEL` is set to). If Ollama is down, retrieval still works — the app catches the model failure and shows the retrieved sources anyway.

`scripts\start-rag.cmd` is the logon launcher (shortcut in `shell:startup`): starts Ollama if needed, starts Streamlit on **port 8501** (the canonical port — the launcher, README, and any Chrome startup page all use it; ad-hoc `streamlit run` may pick another), waits for the port, opens Chrome. It is intentionally batch — an earlier PowerShell version that polled the port with `Net.Sockets.TcpClient` in a retry loop was **blocked and deleted by Defender's AMSI** as a port scanner. Don't port it back to PowerShell.

Chrome's "open on startup" pages (`session.startup_urls` / `restore_on_startup`) are **protected preferences** — Chrome HMAC-signs them and resets any value written directly into its `Preferences` JSON by an outside process. You cannot script this; it must be set through Chrome's Settings UI (`chrome://settings/onStartup`). Point it at the launcher's port (8501), and remember the tab only loads if the Streamlit server is already up.

There is no test suite and no linter configured. Verification is done by running the Streamlit app and driving it.

## Architecture

Data flows in one direction through five modules; each has a single responsibility and imports only downward.

```
config.py      env vars + lru_cached clients (gemini / ollama / supabase). Every model ID,
               dimension, and tunable is defined here exactly once — never hardcode them elsewhere.
embeddings.py  the ONLY place that calls Gemini embed_content
ingest.py      modality detection → extract/chunk → upload original to Storage → embed → insert rows
retrieve.py    retrieve(): embed query → match_documents RPC → signed URLs.
               list_documents(): group rows by source_name for the Browse tab.
reasoning.py   hits → prompt (+ base64 images) → Ollama chat → {answer, citations}
app.py         Streamlit: Upload & Embed / Query / Browse tabs
```

### Embedding invariants (breaking these silently corrupts retrieval)

- `EMBED_DIM` (1536) must equal **all three** of: the `documents.embedding vector(1536)` column, the `match_documents` RPC signature, and `output_dimensionality` in the embed call.
- Gemini returns **non-unit** vectors whenever `output_dimensionality < 3072`, so `embeddings._normalize` does manual L2 normalization after MRL truncation. Cosine similarity in Postgres assumes unit vectors — do not remove this.
- Task types are asymmetric on purpose: stored content uses `RETRIEVAL_DOCUMENT`, queries use `RETRIEVAL_QUERY`. Mixing them degrades ranking.

### Supabase contract

- Table `public.documents`: `source_name, modality, storage_path, chunk_index, content, metadata (jsonb), embedding vector(1536)`. Text/PDF rows carry `content`; image/video/audio rows have `content = NULL` and are located only by their vector.
- **`source_name` is the human display title, NOT the filename.** `ingest_file(..., title=)` defaults it to the filename but the Upload tab lets the user override it (with several files staged, each title is suffixed with `(filename)` to avoid collisions). The real filename always lives in `metadata.filename` and is what drives the storage path + MIME sniffing. Browse groups rows by `source_name`, so all chunks of one doc collapse to one entry.
- RPC `match_documents(query_embedding, match_count, filter, match_threshold, modality_filter)` — the last two do threshold and modality filtering **in SQL**. Changing the Python call site means changing the SQL function signature too (via a migration), and vice versa.
- Storage bucket `documents` is private; originals are uploaded under `{uuid}/{filename}` and surfaced through short-lived signed URLs (`retrieve._signed_url`, 1h TTL).
- `SUPABASE_SERVICE_ROLE_KEY` bypasses RLS — server-side only, never into anything browser-reachable.

### Reasoning quirk

Ollama takes images as a **list of base64 strings on the message** (`msg["images"]`), not as interleaved content parts like the OpenAI/Anthropic APIs. `reasoning._build_prompt` downloads image hits from Storage and base64-encodes them for this. Video/audio hits are cited by name only — nothing is fed inline for them.

**Model compatibility trap:** the default `OLLAMA_MODEL` is `gemma3:4b` (a multimodal model on Ollama's engine), NOT `llama3.2-vision`. Llama 3.2 Vision's `mllama` architecture fails to load on the installed Ollama 0.32.3 (`unknown model architecture: 'mllama'`) — the model pulls fine but errors at inference. If you swap `OLLAMA_MODEL`, pick one this Ollama build actually runs. First inference after a fresh pull also does a slow cold load (~5 min); later calls reuse the loaded model.

## Streamlit gotchas already solved here

Several non-obvious workarounds in `app.py` exist for real bugs; don't "clean them up":

- **`uploader_key` remount.** Streamlit's dropzone latches into a drag-active state on a `dragenter` with no matching `dragleave`, and an invisible overlay then swallows clicks. Bumping the widget key to remount it is the only Python-side fix. Bumping it after an embed also prevents double-ingesting the same file.
- **Search button is not `disabled` on empty input.** `st.text_input` only commits on Enter/blur, so a disabled button would eat the first click.
- **`st.rerun()` after ingest** is needed because the sidebar stats are computed at the top of the script, before the button handler runs.
- Retrieval and reasoning are wrapped in **separate** try/except blocks so a model failure never discards the retrieved sources.

## Notes

- The repo root holds many `rag-*.png` screenshots from manual UI verification; they are untracked scratch, not fixtures.
