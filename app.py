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
from retrieve import database_stats, list_documents, retrieve

st.set_page_config(page_title="Gemini Embedding 2 RAG", page_icon="🔎", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _keys_ready() -> bool:
    return all(os.getenv(k, "").strip() for k in ("GEMINI_API_KEY", "SUPABASE_SERVICE_ROLE_KEY"))


# ---------------- Sidebar ----------------
CONTEXT_TYPES = ["All", "text", "pdf", "image", "video", "audio"]

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-k results", 1, 50, config.MATCH_COUNT)
    similarity_threshold = st.slider(
        "Similarity threshold", 0.0, 1.0, 0.0, 0.05,
        help="Drop retrieved sources below this cosine similarity.",
    )
    context_type = st.selectbox("Context type filter", CONTEXT_TYPES, index=0)
    modality_filter = None if context_type == "All" else context_type
    use_reasoning = st.checkbox(
        f"Reasoning model (`{config.OLLAMA_MODEL}`)",
        value=True,
        help="When on, the local Ollama model generates an answer. When off, only the "
        "retrieved sources are shown — no answer is generated.",
    )

    st.divider()
    st.subheader("Database stats")
    if st.button("🔄 Refresh stats", type="primary"):
        st.rerun()
    try:
        stats = database_stats()
        c1, c2 = st.columns(2)
        c1.metric("Vectors", stats["total_vectors"])
        c2.metric("Sources", stats["total_sources"])
        if stats["by_modality"]:
            st.caption("By context type")
            for modality, count in stats["by_modality"].items():
                st.write(f"- **{modality}**: {count}")
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Stats unavailable: {exc}")

st.title("🔎 Multimodal RAG — Gemini Embedding 2")
st.caption("Upload text, PDFs, images, video, or audio → embed → ask questions → view sources.")

upload_tab, query_tab, browse_tab = st.tabs(["📤 Upload & Embed", "💬 Query", "📚 Browse"])

# ---------------- Upload & Embed ----------------
with upload_tab:
    st.subheader("Upload documents")

    # The uploader's key is bumped to force a remount. Streamlit's dropzone
    # (react-dropzone) latches into its "drag active" state on a dragenter that
    # never gets a matching dragleave — e.g. you drag a file over the app and
    # then drop it outside the box or press Esc. In that state an overlay covers
    # the widget and swallows clicks, so the Upload button stops responding and
    # drag-and-drop stops accepting files. Remounting is the only way to clear
    # it from the Python side.
    st.session_state.setdefault("uploader_key", 0)

    files = st.file_uploader(
        "Drop files of any supported type (text, md, pdf, docx, png, jpg, mp4, mov, mp3, wav, ...)",
        accept_multiple_files=True,
        key=f"uploader_{st.session_state['uploader_key']}",
    )

    doc_title = st.text_input(
        "Document title",
        placeholder="Optional — defaults to the filename",
        key=f"doc_title_{st.session_state['uploader_key']}",
        help="The title shown in Browse and in the retrieved-sources list. With "
        "several files staged, each title is suffixed with its filename to keep "
        "them distinguishable.",
    )

    if st.button(
        "↺ Reset uploader",
        help="Click if the box is stuck showing 'Drag and drop files here' and the "
        "Upload button no longer responds.",
    ):
        st.session_state["uploader_key"] += 1
        st.session_state.pop("embed_results", None)
        st.rerun()

    # Staging a file is not the same as ingesting it — say so, otherwise a
    # successful drag-and-drop looks like nothing happened.
    if files:
        st.caption(
            f"**{len(files)} file(s) staged.** Nothing is embedded until you click "
            "**Embed uploaded files** below."
        )

    if st.button("Embed uploaded files", type="primary", disabled=not files):
        if not _keys_ready():
            st.error("Missing API keys — add them to your `.env` and restart.")
        else:
            results: list[tuple[str, str]] = []
            title = doc_title.strip()
            for f in files:
                # One title across several files would collide, so keep the
                # filename as a disambiguating suffix in that case.
                if not title:
                    file_title = None
                elif len(files) == 1:
                    file_title = title
                else:
                    file_title = f"{title} ({f.name})"

                with st.spinner(f"Embedding {f.name} ..."):
                    try:
                        res = ingest_file(f.name, f.getvalue(), title=file_title)
                        results.append(
                            ("ok", f"**{res['title']}** — {res['modality']}, "
                                   f"{res['chunks']} vector(s) stored.")
                        )
                    except Exception as exc:  # noqa: BLE001
                        results.append(("err", f"Failed on {f.name}: {exc}\n\n```\n"
                                               f"{traceback.format_exc()}```"))
            # Rerun so the sidebar "Database stats" (computed at the top of the
            # script, before this handler) reflect the rows just inserted.
            # Bumping the key also clears the staged files, so clicking the
            # button twice can't ingest the same file again.
            st.session_state["embed_results"] = results
            st.session_state["uploader_key"] += 1
            st.rerun()

    for kind, msg in st.session_state.get("embed_results", []):
        (st.success if kind == "ok" else st.error)(msg)

    st.divider()
    st.subheader("Or bulk-embed the local `data/` folder")
    st.caption(f"Drop files into `{DATA_DIR}` and click below.")
    if st.button("Embed everything in data/"):
        if not _keys_ready():
            st.error("Missing API keys — add them to your `.env` and restart.")
        else:
            with st.spinner("Ingesting data/ ..."):
                try:
                    results, skipped = ingest_directory(DATA_DIR)
                    if not results and not skipped:
                        st.info("No files found in data/.")
                    for res in results:
                        st.success(f"**{res['filename']}** — {res['modality']}, {res['chunks']} vector(s).")
                    for note in skipped:
                        st.warning(f"Skipped {note}")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
                    st.code(traceback.format_exc())

# ---------------- Query ----------------
with query_tab:
    st.subheader("Ask a question")
    question = st.text_input("Your question", placeholder="What does the document say about ...?")
    st.caption(
        f"Retrieving top **{top_k}** · similarity ≥ **{similarity_threshold:.2f}** · "
        f"context type **{context_type}** — change these in the sidebar."
    )

    # Deliberately NOT disabled on an empty question: st.text_input only commits
    # its value on Enter/blur, so a disabled button would swallow the very first
    # click (the click blurs the box, the value commits, the button enables —
    # but that click never fired the search, so you'd have to click twice).
    if st.button("Search", type="primary"):
        if not question.strip():
            st.warning("Type a question first.")
        elif not _keys_ready():
            st.error("Missing API keys — add them to your `.env` and restart.")
        else:
            # Retrieval and reasoning are handled separately so a failure in the
            # model call (e.g. an OpenAI quota error) still shows the retrieved
            # sources instead of discarding them.
            hits = None
            try:
                with st.spinner("Retrieving ..."):
                    hits = retrieve(
                        question,
                        match_count=top_k,
                        match_threshold=similarity_threshold,
                        modality=modality_filter,
                    )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Retrieval failed: {exc}")
                st.code(traceback.format_exc())

            if hits is not None:
                if use_reasoning:
                    st.markdown("### Answer")
                    try:
                        with st.spinner(f"Reasoning with `{config.OLLAMA_MODEL}` (local) ..."):
                            result = answer(question, hits)
                        st.markdown(result["answer"] or "_(empty response)_")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Answer generation failed: {exc}")
                        st.caption("Retrieved sources are still shown below.")
                        st.code(traceback.format_exc())
                else:
                    st.caption("Reasoning model is off — showing retrieved sources only.")

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

# ---------------- Browse ----------------
with browse_tab:
    st.subheader("Browse documents")
    search_title = st.text_input(
        "Search by title", placeholder="Filter documents by title ..."
    )
    if not _keys_ready():
        st.error("Missing API keys — add them to your `.env` and restart.")
    else:
        try:
            docs = list_documents()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load documents: {exc}")
            docs = []

        if search_title:
            needle = search_title.lower()
            docs = [d for d in docs if needle in d["source_name"].lower()]

        st.caption(f"{len(docs)} document(s)")
        if not docs:
            st.info("No documents found.")
        for d in docs:
            with st.expander(
                f"📄 {d['source_name']} — {d['modality']} ({d['chunks']} vector(s))"
            ):
                url = d.get("signed_url")
                modality = d.get("modality")
                if url and modality == "image":
                    st.image(url)
                elif url and modality == "video":
                    st.video(url)
                elif url and modality == "audio":
                    st.audio(url)
                elif url:
                    st.markdown(f"[Open original file]({url})")
                else:
                    st.caption("No stored file to preview.")
