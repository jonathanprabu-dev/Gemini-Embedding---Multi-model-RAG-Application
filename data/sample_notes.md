# Project Meeting Notes — Vector Search Pilot

**Date:** 2026-07-24
**Attendees:** Data platform team

## Decisions

1. Store embeddings at **1536 dimensions** to keep a clean pgvector HNSW index while
   preserving retrieval quality.
2. Use **cosine similarity** as the distance metric.
3. Keep original files in a **private Supabase Storage bucket**; the database rows only
   hold extracted text plus the embedding vector.

## Action items

- Ingest the Q3 research PDFs and product screenshots.
- Benchmark retrieval latency with the HNSW index at 10k, 100k, and 1M rows.
- Draft guidance on chunk size; current default is 1500 characters with 200 overlap.

## Open question

Should video clips be embedded whole or split into scene segments? For clips under two
minutes, whole-clip embedding is the current recommendation.
