# Multimodal RAG — Gemini Embedding 2 + Supabase + OpenAI

A Retrieval-Augmented Generation app that embeds **text, PDFs, images, video, and audio**
into a single vector space with **Gemini Embedding 2** (`gemini-embedding-2-preview`),
stores them in **Supabase pgvector**, and answers questions with **OpenAI gpt-5.1** reasoning
over the retrieved context. A **Streamlit** GUI drives the whole flow.

```
Upload ──▶ Gemini Embedding 2 (1536-dim, L2-normalized) ──▶ Supabase pgvector
                                                                   │
Question ──▶ Gemini query embedding ──▶ match_documents (cosine) ──┘
                                                │
                                                ▼
                                   OpenAI gpt-5.1  ──▶  Answer + cited sources
```

## Architecture

| Piece | What it does | File |
|-------|--------------|------|
| Config | Loads `.env`, holds `EMBED_DIM=1536`, builds clients | `config.py` |
| Embeddings | Gemini embed (MRL-truncate + **L2-normalize**, task-type prefixes) | `embeddings.py` |
| Ingestion | Modality detection, text/PDF/docx extraction, chunking, Storage upload, row insert | `ingest.py` |
| Retrieval | Query embedding → `match_documents` RPC → signed media URLs | `retrieve.py` |
| Reasoning | Build multimodal prompt → gpt-5.1 → answer + citations | `reasoning.py` |
| GUI | Upload / embed / query / view | `app.py` |

Supabase project: **`tvbvjzvvkzznnhoepfyb`** ("image"). Table `public.documents`
(`embedding vector(1536)`, HNSW cosine index), RPC `match_documents`, private Storage
bucket `documents`. The schema is already applied.

## 1. Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in the keys
```

Edit `.env` and provide:

- `GEMINI_API_KEY` — Google AI Studio key (with Gemini Embedding 2 access).
- `OPENAI_API_KEY` — OpenAI key.
- `SUPABASE_SERVICE_ROLE_KEY` — Supabase → **Project Settings → API → `service_role`**.
  Server-side only; it bypasses RLS, so never ship it to a browser.

`SUPABASE_URL`, `SUPABASE_PROJECT_ID`, and the model/dim settings are pre-filled.

## 2. Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## 3. Where to upload documents

Two ways, both in the **📤 Upload & Embed** tab:

1. **Drag-and-drop** any supported file into the uploader (multi-select is fine).
2. **Bulk folder** — drop files into the local `data/` folder and click
   **“Embed everything in data/”**.

Supported types:

| Modality | Extensions | How it's embedded |
|----------|-----------|-------------------|
| Text | `.txt .md .csv .json .html .rst .log` | text extracted → chunked → embedded |
| PDF | `.pdf` | text extracted (`pypdf`) → chunked → embedded (handles >6 pages) |
| Word | `.docx` | text extracted (`python-docx`) → chunked → embedded |
| Image | `.png .jpg .jpeg .webp .gif .bmp` | image bytes embedded directly (multimodal) |
| Video | `.mp4 .mov .webm .avi .mkv` | media embedded directly (model limit ≈120s) |
| Audio | `.mp3 .wav .m4a .aac .ogg .flac` | media embedded directly |

## 4. How documents are processed and embedded

1. **Modality detection** by extension (`ingest.detect_modality`).
2. The **original file** is uploaded to the private Supabase Storage bucket `documents`;
   its path is saved on each row.
3. **Text/PDF/docx**: text is extracted and split into ~1500-char chunks with ~200-char
   overlap, one row per chunk.
4. **Media (image/video/audio)**: the raw bytes go straight to Gemini's multimodal
   embedder, one row per file.
5. Every embedding is requested at **`output_dimensionality=1536`** (Matryoshka truncation),
   then **manually L2-normalized** — required because Gemini returns non-unit vectors below
   3072 dims. Stored content uses task type `RETRIEVAL_DOCUMENT`.
6. Rows (`source_name, modality, storage_path, chunk_index, content, metadata, embedding`)
   are inserted into `public.documents`.

## 5. How to run queries

In the **💬 Query** tab: type a question, choose how many results to retrieve, click
**Search**. Under the hood:

1. The query is embedded with task type `RETRIEVAL_QUERY` (truncated + normalized).
2. `match_documents(query_embedding, match_count, filter)` runs a cosine-similarity search
   over the HNSW index and returns the top-k rows with a `similarity` score.
3. Retrieved text is passed to **gpt-5.1**; retrieved **images are attached** to the prompt
   so the vision model can read them. Video/audio hits are cited by name.
4. The GUI shows the **answer** plus each **source** with its score and an inline preview
   (image/video/audio player or a link to the original).

### Query from Python (optional)

```python
from retrieve import retrieve
from reasoning import answer

hits = retrieve("What are the key findings?", match_count=5)
print(answer("What are the key findings?", hits)["answer"])
```

## Notes

- The embedding dimension is fixed at **1536** in `config.py` and must match the DB column
  and the `match_documents` signature. To change it, update all three and re-create the table.
- Video/audio/PDF have model-side limits (≈120s video, PNG/JPEG images, ≤6 native PDF pages);
  PDFs beyond 6 pages are handled here via text extraction. Oversized media surfaces a clear error.
- `.env` is gitignored. The service-role key is used only server-side.
