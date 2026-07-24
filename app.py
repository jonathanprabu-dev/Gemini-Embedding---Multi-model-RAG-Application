"""Streamlit GUI for the multimodal RAG system.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import traceback

import streamlit as st

import config
from ingest import ingest_directory, ingest_file
from reasoning import answer
from retrieve import retrieve

st.set_page_config(page_title="Gemini Embedding 2 RAG", page_icon="🔎", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _keys_ready() -> bool:
    return all(os.getenv(k, "").strip() for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "SUPABASE_SERVICE_ROLE_KEY"))


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Configuration")
    st.write(f"**Embed model:** `{config.EMBED_MODEL}`")
    st.write(f"**Reasoning model:** `{config.OPENAI_MODEL}`")
    st.write(f"**Embedding dim:** `{config.EMBED_DIM}`")
    st.write(f"**Supabase project:** `{config.SUPABASE_PROJECT_ID}`")
    st.write(f"**Storage bucket:** `{config.STORAGE_BUCKET}`")
    st.divider()

    def _mark(name: str) -> str:
        return "✅" if os.getenv(name, "").strip() else "❌"

    st.subheader("API keys")
    st.write(f"{_mark('GEMINI_API_KEY')} GEMINI_API_KEY")
    st.write(f"{_mark('OPENAI_API_KEY')} OPENAI_API_KEY")
    st.write(f"{_mark('SUPABASE_SERVICE_ROLE_KEY')} SUPABASE_SERVICE_ROLE_KEY")
    if not _keys_ready():
        st.warning("Fill in the missing keys in your `.env`, then restart the app.")

st.title("🔎 Multimodal RAG — Gemini Embedding 2")
st.caption("Upload text, PDFs, images, video, or audio → embed → ask questions → view sources.")

upload_tab, query_tab = st.tabs(["📤 Upload & Embed", "💬 Query"])

# ---------------- Upload & Embed ----------------
with upload_tab:
    st.subheader("Upload documents")
    files = st.file_uploader(
        "Drop files of any supported type (text, md, pdf, docx, png, jpg, mp4, mov, mp3, wav, ...)",
        accept_multiple_files=True,
    )

    if st.button("Embed uploaded files", type="primary", disabled=not files):
        if not _keys_ready():
            st.error("Missing API keys — see the sidebar.")
        else:
            progress = st.container()
            for f in files:
                with progress:
                    with st.spinner(f"Embedding {f.name} ..."):
                        try:
                            res = ingest_file(f.name, f.getvalue())
                            st.success(
                                f"**{res['filename']}** — {res['modality']}, "
                                f"{res['chunks']} vector(s) stored."
                            )
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Failed on {f.name}: {exc}")
                            st.code(traceback.format_exc())

    st.divider()
    st.subheader("Or bulk-embed the local `data/` folder")
    st.caption(f"Drop files into `{DATA_DIR}` and click below.")
    if st.button("Embed everything in data/"):
        if not _keys_ready():
            st.error("Missing API keys — see the sidebar.")
        else:
            with st.spinner("Ingesting data/ ..."):
                try:
                    results = ingest_directory(DATA_DIR)
                    if not results:
                        st.info("No files found in data/.")
                    for res in results:
                        st.success(f"**{res['filename']}** — {res['modality']}, {res['chunks']} vector(s).")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
                    st.code(traceback.format_exc())

# ---------------- Query ----------------
with query_tab:
    st.subheader("Ask a question")
    question = st.text_input("Your question", placeholder="What does the document say about ...?")
    k = st.slider("Number of results to retrieve", 1, 15, config.MATCH_COUNT)

    if st.button("Search", type="primary", disabled=not question):
        if not _keys_ready():
            st.error("Missing API keys — see the sidebar.")
        else:
            try:
                with st.spinner("Retrieving ..."):
                    hits = retrieve(question, match_count=k)
                with st.spinner("Reasoning with the model ..."):
                    result = answer(question, hits)

                st.markdown("### Answer")
                st.markdown(result["answer"] or "_(empty response)_")

                st.markdown("### Retrieved sources")
                if not hits:
                    st.info("No matching documents.")
                for i, hit in enumerate(hits, start=1):
                    sim = hit.get("similarity")
                    with st.expander(
                        f"[{i}] {hit.get('source_name')} — {hit.get('modality')} "
                        f"(similarity {sim:.3f})",
                        expanded=(i == 1),
                    ):
                        if hit.get("content"):
                            st.write(hit["content"])
                        url = hit.get("signed_url")
                        modality = hit.get("modality")
                        if url and modality == "image":
                            st.image(url)
                        elif url and modality == "video":
                            st.video(url)
                        elif url and modality == "audio":
                            st.audio(url)
                        elif url:
                            st.markdown(f"[Open original file]({url})")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                st.code(traceback.format_exc())
