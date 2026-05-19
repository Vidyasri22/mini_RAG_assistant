"""
Streamlit UI for the Mini-RAG Assistant.

Run with:  streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

from rag_pipeline import (
    VectorStore,
    answer_question,
    build_corpus,
)


st.set_page_config(page_title="Mini-RAG Assistant", page_icon="📚", layout="wide")

st.title("📚 Mini-RAG Assistant")
st.caption(
    "Upload documents, ask questions, get answers grounded in your knowledge base "
    " - with confidence scores and source citations."
)

# ---------- Educational intro ----------

with st.expander("📖 New here? Start with this", expanded=False):
    st.markdown(
        """
        **What is this?**
        A *Retrieval-Augmented Generation* (RAG) assistant. You give it documents,
        ask questions, and it answers using **only** what's in those documents - 
        with citations so you can verify every claim without any manual searching.

        **What is a "knowledge base"?**
        It's just the collection of documents you upload.

        **Three steps to use:**
        1. Paste your Groq API key in the sidebar (free at console.groq.com)
        2. Upload your documents and click **Build Index**
        3. Ask questions in the main panel
        """
    )

with st.expander("✨ Key features"):
    st.markdown(
        """
        - ✅ **Grounded answers** - only uses your documents, no hallucinations
        - ✅ **Inline citations** like `[1]`, `[2]` - every claim traces to a source passage
        - ✅ **Confidence score** (High / Medium / Low) - know when to trust an answer
        - ✅ **Refusal handling** - says *"I don't have enough information"* instead of guessing
        - ✅ **Multi-document search** - all uploaded files are searched together
        - ✅ **Free to run** - local embeddings (MiniLM) + free Llama via Groq

        **Each answer you receive includes:**

        🟢 A confidence badge &nbsp;•&nbsp; 📝 The answer with inline `[n]` citations &nbsp;•&nbsp; 🔍 The exact source passages the model used (expandable)
        """
    )
# ---------- Sidebar: config + upload ----------

with st.sidebar:
    st.header("⚙️ Setup")

    st.markdown("**🔑 Groq API Key**")
    st.caption(
        "Needed for the Llama model that writes the answer. Your documents stay "
        "local. The key lives only in this browser session - it isn't saved anywhere."
    )
    api_key = st.text_input(
        "API Key",
        type="password",
        label_visibility="collapsed",
        help="Get a free key at console.groq.com (no credit card required)",
    )
    if not api_key:
        st.info("👆 Paste your key above to enable answering.")

    st.divider()

    st.markdown("**🎯 Chunks to retrieve (k)**")
    st.caption(
        "How many pieces that your documents are split into. So that the assistant look at when answering a "
        "question. Compare the modes below, then pick what fits your use case.Default each chunk size is <=500 words"
    )

    st.markdown(
        """
        | k | Mode | Best for | Trade-off |
        |---|------|----------|-----------|
        | **1–2** | 🎯 Focused | Specific factual lookups (*"what's the refund window?"*) | May miss context if the answer spans sections |
        | **3–5** | ⚖️ Balanced *(recommended)* | Most everyday questions | Sweet spot of quality vs. cost |
        | **6–8** | 📚 Broad | Summaries, multi-doc comparisons | Slower, more API tokens, can dilute the answer |
        """
    )

    k = st.slider("k", 1, 8, 4, label_visibility="collapsed")

    st.divider()

    st.header("📄 Knowledge Base")
    st.caption(
        "These are the documents the assistant is allowed to quote from. "
        "Supported: **PDF** (text-based, not scanned), **TXT**, **MD**. Upload "
        "as many as you like - they're all searched together."
    )
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.caption(f"📎 {len(uploaded_files)} file(s) ready to index:")
        for f in uploaded_files:
            size_kb = len(f.getvalue()) / 1024
            st.caption(f"  • `{f.name}` ({size_kb:.1f} KB)")

    build_clicked = st.button(
        "Build Index",
        type="primary",
        disabled=not uploaded_files,
        help="Reads your files, chunks them, and creates the searchable vector index.",
    )

# ---------- Session state ----------

if "store" not in st.session_state:
    st.session_state.store = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, RAGResponse)

# ---------- Build index ----------

if build_clicked and uploaded_files:
    with st.spinner("Reading files and building vector index..."):
        # Streamlit's UploadedFile is in-memory; save to temp paths so we can reuse
        # the same file-reading code as the CLI.
        tmp_dir = Path(tempfile.mkdtemp())
        paths = []
        for f in uploaded_files:
            p = tmp_dir / f.name
            p.write_bytes(f.read())
            paths.append(p)

        corpus = build_corpus(paths)
        store = VectorStore()
        store.build(corpus)
        st.session_state.store = store
        st.session_state.corpus_stats = {
            "files": len(paths),
            "chunks": len(corpus),
        }
    st.sidebar.success(
        f"✅ Indexed {st.session_state.corpus_stats['chunks']} chunks "
        f"from {st.session_state.corpus_stats['files']} file(s)."
    )

# ---------- Main panel: ask questions ----------

if st.session_state.store is None:
    st.info("👈 Upload documents and click **Build Index** in the sidebar to get started.")
    st.markdown(
        """
        #### Don't have documents handy? Try these:
        - A **product manual PDF** (router, appliance, software)
        - **Class or meeting notes** (ask: *what did we decide about X?*)
        - **A policy document** (terms of service, employee handbook, lease)
        - **A Wikipedia article** saved as PDF (File → Print → Save as PDF)
        """
    )
else:
    stats = st.session_state.corpus_stats
    # col1, col2, col3 = st.columns(3)
    # col1.metric("📄 Files indexed", stats["files"])
    # col2.metric("🧩 Total chunks", stats["chunks"])
    # col3.metric("🎯 Retrieving per question", k)

    # ---- Sample questions: click to auto-fill the input ----
    st.caption("💡 Try a sample question, or type your own below:")
    sample_questions = [
        "What is the refund policy?",
        "How long does shipping take?",
        "Summarize the key features of the product.",
    ]

    def _fill_question(sample: str):
        st.session_state.question_box = sample

    sample_cols = st.columns(len(sample_questions))
    for col, sample in zip(sample_cols, sample_questions):
        col.button(
            sample,
            on_click=_fill_question,
            args=(sample,),
            use_container_width=True,
            help="Click to copy into the question box below",
        )

    question = st.text_input(
        "Ask a question about your documents:",
        key="question_box",
        placeholder="e.g. What is the refund policy?",
        help="Phrase it naturally - semantic search doesn't need exact keywords from the doc.",
    )

    ask_clicked = st.button("Ask", type="primary",
                            disabled=not (question and api_key))

    if not api_key:
        st.warning("Enter your Groq API key in the sidebar first.")

    if ask_clicked and question and api_key:
        with st.spinner("Retrieving context and generating answer..."):
            try:
                result = answer_question(
                    question,
                    st.session_state.store,
                    api_key=api_key,
                    k=k,
                )
                st.session_state.history.append((question, result))
            except Exception as e:
                st.error(f"Error: {e}")

    # Show conversation history, most recent first
    for q, result in reversed(st.session_state.history):
        st.divider()
        st.markdown(f"### ❓ {q}")

        # Confidence badge - color by label
        color = {"High": "green", "Medium": "orange", "Low": "red"}[
            result.confidence_label
        ]
        confidence_tip = {
            "High": "Cited chunks closely match your question - answer is likely accurate.",
            "Medium": "Partial match - verify against the retrieved sources below.",
            "Low": "Weak match - the docs may not contain a good answer. Treat with skepticism.",
        }[result.confidence_label]
        st.markdown(
            f"**Confidence:** :{color}[{result.confidence_label} "
            f"({result.confidence:.2f})]  &nbsp; *{confidence_tip}*"
        )

        st.markdown("**Answer:**")
        st.write(result.answer)

        with st.expander(
            f"🔍 Retrieved sources ({len(result.retrieved)}) - click to inspect what the model saw"
        ):
            st.caption(
                "These are the document passages pulled by similarity search. "
                "The model answered using only these - anything not here was not "
                "available to it. Higher similarity = more relevant to your question."
            )
            for i, r in enumerate(result.retrieved, 1):
                st.markdown(
                    f"**[{i}] {r.chunk.source}** - similarity: `{r.similarity:.3f}`"
                )
                preview = r.chunk.text[:400] + ("..." if len(r.chunk.text) > 400 else "")
                st.text(preview)

    if st.session_state.history:
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
