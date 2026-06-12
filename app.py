"""
app.py — AetherRAG Streamlit UI

A premium RAG chatbot interface powered by ChromaDB, Sentence-Transformers,
and multiple LLM backends (Gemini, OpenAI, Ollama, Mock).
"""

# !! Must be set before ANY chromadb import to prevent slow telemetry network calls
import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import logging
import sys

import streamlit as st

# Configure logging early so all modules see it
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("app")

from config import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_K,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
)
from rag_engine import (
    retrieve,
    get_rag_prompt,
    generate_llm_response_stream,
    add_document_to_rag,
    get_collection_stats,
    clear_collection,
    _l2_to_similarity,
    condense_query,
)

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="chat bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Premium CSS ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1e293b;
}

/* -- Background -- */
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 60%, #f5f3ff 100%);
    background-attachment: fixed;
}

# /* -- Sidebar -- */
# section[data-testid="stSidebar"] {
#     background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
#     border-right: 1px solid rgba(99, 102, 241, 0.12) !important;
#     backdrop-filter: blur(16px);
# }
# section[data-testid="stSidebar"] hr {
#     border-top: 1px solid rgba(0, 0, 0, 0.08) !important;
# }

/* -- Typography -- */
h1 {
    font-weight: 800 !important;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #4338ca 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.1rem !important;
}
h2, h3 { font-weight: 600 !important; color: #0f172a !important; }

/* -- Glass Card -- */
.glass-card {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(99, 102, 241, 0.08);
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(148, 163, 184, 0.08);
    transition: box-shadow 0.2s ease;
}
.glass-card:hover { box-shadow: 0 12px 40px rgba(99, 102, 241, 0.12); }

/* -- Empty state banner -- */
.empty-state {
    text-align: center;
    padding: 40px 20px;
    border: 1px dashed rgba(99, 102, 241, 0.4);
    border-radius: 16px;
    background: rgba(99, 102, 241, 0.03);
    margin: 24px 0;
}
.empty-state-icon { font-size: 3rem; margin-bottom: 12px; }
.empty-state-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #4f46e5;
    margin-bottom: 6px;
}
.empty-state-subtitle { color: #475569; font-size: 0.92rem; }

/* -- Source cards -- */
.source-container { display: flex; flex-direction: column; gap: 10px; margin: 10px 0; }

.source-card {
    background: rgba(255, 255, 255, 0.65);
    border-left: 3px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 10px 14px;
    font-size: 0.87rem;
    border-top: 1px solid rgba(99, 102, 241, 0.06);
    border-right: 1px solid rgba(99, 102, 241, 0.06);
    border-bottom: 1px solid rgba(99, 102, 241, 0.06);
    transition: all 0.18s ease;
}
.source-card:hover {
    background: rgba(99, 102, 241, 0.06);
    border-left-color: #4f46e5;
    transform: translateX(3px);
}
.source-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.source-title { font-weight: 600; color: #4f46e5; font-size: 0.88rem; }
.source-score {
    font-size: 0.73rem;
    color: #4f46e5;
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.2);
    padding: 2px 7px;
    border-radius: 99px;
    font-weight: 500;
}
.source-score.high { background: rgba(16,185,129,0.1); border-color: rgba(16,185,129,0.3); color: #059669; }
.source-score.mid  { background: rgba(245,158,11,0.1); border-color: rgba(245,158,11,0.3); color: #d97706; }
.source-score.low  { background: rgba(239,68,68,0.08);  border-color: rgba(239,68,68,0.25);  color: #dc2626; }
.source-body {
    color: #334155;
    line-height: 1.55;
    font-style: italic;
    background: rgba(255, 255, 255, 0.95);
    padding: 7px 11px;
    border-radius: 6px;
    margin-top: 4px;
    font-size: 0.84rem;
    border: 1px solid rgba(99, 102, 241, 0.05);
}

/* -- Status pills -- */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #047857;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-bottom: 12px;
}
.status-pill.warning {
    background: rgba(245, 158, 11, 0.15);
    border-color: rgba(245, 158, 11, 0.35);
    color: #b45309;
}
.status-pill.error {
    background: rgba(239, 68, 68, 0.12);
    border-color: rgba(239, 68, 68, 0.3);
    color: #b91c1c;
}
.status-pill.offline {
    background: rgba(100, 116, 139, 0.12);
    border-color: rgba(100, 116, 139, 0.3);
    color: #475569;
}

/* -- Chat input -- */
div[data-testid="stChatInput"] {
    background-color: rgba(255, 255, 255, 0.85) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(14px);
    box-shadow: 0 4px 24px rgba(148, 163, 184, 0.08);
}

/* -- Stat number -- */
.stat-big {
    font-size: 2rem;
    font-weight: 700;
    color: #4f46e5;
    line-height: 1;
}
.stat-label {
    font-size: 0.78rem;
    color: #64748b;
    margin-bottom: 2px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.stat-sub {
    font-size: 1.1rem;
    font-weight: 600;
    color: #4f46e5;
}

/* -- Hide Deploy button, header, MainMenu, and footer completely -- */
.stAppDeployButton, .stDeployButton, [data-testid="stAppDeployButton"] {
    display: none !important;
    visibility: hidden !important;
}
#MainMenu, header {
    visibility: hidden !important;
    display: none !important;
}
footer {
    visibility: hidden !important;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session state bootstrap
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db_stats" not in st.session_state:
    st.session_state.db_stats = None  # lazy-load


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_db_stats(force: bool = False) -> dict:
    """Cache DB stats in session state to avoid repeated ChromaDB calls."""
    if st.session_state.db_stats is None or force:
        st.session_state.db_stats = get_collection_stats()
    return st.session_state.db_stats


def _similarity_class(sim: float) -> str:
    if sim >= 0.65:
        return "high"
    if sim >= 0.4:
        return "mid"
    return "low"


def _render_sources(sources: dict) -> None:
    """Render the retrieved source passages in an expander."""
    docs = (sources.get("documents") or [[]])[0]
    metas = (sources.get("metadatas") or [[]])[0]
    distances = (sources.get("distances") or [[]])[0]

    if not docs:
        return

    with st.expander(f"📄 Retrieved Sources ({len(docs)} passage{'s' if len(docs) != 1 else ''})"):
        st.markdown("<div class='source-container'>", unsafe_allow_html=True)
        for doc, meta, dist in zip(docs, metas, distances):
            source_name = (meta or {}).get("source", "Unknown")
            chunk_idx = (meta or {}).get("chunk_index", "?")
            sim = _l2_to_similarity(dist if dist is not None else 0.0)
            sim_pct = sim * 100
            sim_cls = _similarity_class(sim)

            truncated = doc.strip()
            if len(truncated) > 350:
                truncated = truncated[:350] + "..."

            st.markdown(f"""
            <div class='source-card'>
                <div class='source-header'>
                    <span class='source-title'>📄 {source_name}
                        <span style='color:#64748b;font-weight:400;font-size:0.78rem;'> · chunk #{chunk_idx}</span>
                    </span>
                    <span class='source-score {sim_cls}'>Similarity {sim_pct:.1f}%</span>
                </div>
                <div class='source-body'>"{truncated}"</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def _render_prompt_debugger(prompt: str) -> None:
    if prompt:
        with st.expander("🛠️ RAG Prompt Inspector"):
            st.code(prompt, language="text")


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown("<h1>🌌 chat bot</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#475569;font-size:1.05rem;margin-bottom:1.5rem;'>"
    "Knowledge-base RAG chatbot · ChromaDB · Sentence-Transformers · Multi-LLM"
    "</p>",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️setting")

    # -- LLM Provider ---------------------------------------------------------
    _PROVIDER_LABELS = {
        "Ollama (Local Server)": "ollama",
        "Offline / Mock Generator": "mock",
    }
    provider_label = st.selectbox(
        "LLM Provider",
        options=list(_PROVIDER_LABELS.keys()),
        index=0,
        help="Select the language model backend to use for generating answers.",
    )
    provider = _PROVIDER_LABELS[provider_label]

    # -- Provider-specific settings --------------------------------------------
    model_name: str | None = None
    ollama_url: str | None = None

    if provider == "ollama":
        ollama_url = st.text_input(
            "Ollama Host URL",
            value=DEFAULT_OLLAMA_URL,
            help="Base URL of your running Ollama server.",
        )
        model_name = st.text_input(
            "Model Name",
            value=DEFAULT_OLLAMA_MODEL,
            help="Name of the model pulled in Ollama (e.g. llama3, mistral).",
        )
        st.caption("ℹ️ Ensure Ollama is running and the model is pulled (`ollama pull llama3`).")

    else:  # mock
        st.markdown(
            "<div class='status-pill offline'>⚙️ Mock Generator (Offline)</div>",
            unsafe_allow_html=True,
        )
        st.caption("No API key needed. Extracts relevant sentences locally from retrieved context.")

    st.markdown("---")

    # -- Retrieval settings ----------------------------------------------------
    st.markdown("### 🔍 Retrieval")
    k_val = st.slider(
        "Passages to Retrieve (K)",
        min_value=1, max_value=10, value=DEFAULT_K,
        help="How many document chunks to retrieve per query.",
    )
    
    use_reranking = st.checkbox(
        "Enable Re-ranking 🚀",
        value=False,
        help="Use a Cross-Encoder neural model to re-sort retrieved chunks for higher quality context."
    )
    
    stats_data = load_db_stats()
    db_sources = stats_data.get("sources", [])
    if db_sources:
        selected_sources = st.multiselect(
            "Filter by Sources 📁",
            options=db_sources,
            default=None,
            help="Select specific documents to restrict search. Leave empty to query all files."
        )
    else:
        selected_sources = []

    st.markdown("---")

    # -- Document ingestion ----------------------------------------------------
    st.markdown("### 📥 Ingest Documents")
    uploaded_file = st.file_uploader(
        "Upload .txt or .md files to the knowledge base",
        type=["txt", "md"],
        help="Documents are chunked and embedded into the vector database.",
    )

    if uploaded_file is not None:
        chunk_mode = st.selectbox(
            "Chunking Mode",
            options=["Paragraph-aware (Recommended)", "Word Tokens", "Characters"],
            index=0,
            help="Choose how the document text is split into chunks."
        )
        _MODE_MAP = {
            "Paragraph-aware (Recommended)": "para",
            "Word Tokens": "token",
            "Characters": "char"
        }
        selected_mode = _MODE_MAP[chunk_mode]

        col1, col2 = st.columns(2)
        with col1:
            chunk_sz = st.number_input(
                "Chunk Size", min_value=100, max_value=4000, value=DEFAULT_CHUNK_SIZE, step=100,
                help="Size of each chunk (words for paragraph/token mode, characters for char mode)."
            )
        with col2:
            overlap_sz = st.number_input(
                "Overlap", min_value=0, max_value=1000, value=DEFAULT_OVERLAP, step=25,
                help="Overlap between consecutive chunks (same unit as Chunk Size)."
            )

        if st.button("🚀 Process & Index File", use_container_width=True, type="primary"):
            file_name = uploaded_file.name
            try:
                file_content = uploaded_file.read().decode("utf-8", errors="replace")
                if not file_content.strip():
                    st.error("⚠️ The uploaded file appears to be empty.")
                else:
                    with st.spinner(f"Chunking & indexing '{file_name}'..."):
                        num_chunks = add_document_to_rag(file_name, file_content, chunk_sz, overlap_sz, selected_mode)
                    if num_chunks > 0:
                        st.success(f"✅ Indexed **{num_chunks}** chunks from '{file_name}'!")
                        load_db_stats(force=True)
                        st.rerun()
                    else:
                        st.error("❌ No chunks were produced. Check the file content.")
            except ValueError as ve:
                st.error(f"⚠️ Validation error: {ve}")
            except RuntimeError as re_:
                st.error(f"❌ Indexing failed: {re_}")
            except Exception as exc:
                st.error(f"❌ Unexpected error: {exc}")
                logger.exception("File ingestion failed for '%s'", uploaded_file.name)

    st.markdown("---")

    # -- Database statistics ---------------------------------------------------
    st.markdown("### 📊 Knowledge Base")
    stats = load_db_stats()

    if "error" in stats:
        st.warning(f"⚠️ Could not load DB stats: {stats['error']}")
    else:
        db_count = stats["count"]
        db_sources = stats["sources"]
        st.markdown(f"""
        <div class='glass-card'>
            <div class='stat-label'>Indexed Chunks</div>
            <div class='stat-big'>{db_count}</div>
            <div style='margin-top:10px;'>
                <div class='stat-label'>Unique Sources</div>
                <div class='stat-sub'>{len(db_sources)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if db_sources:
            with st.expander(f"📁 Indexed Files ({len(db_sources)})"):
                for src in db_sources:
                    st.markdown(f"📄 `{src}`")

    st.markdown("---")

    # -- Actions ---------------------------------------------------------------
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear DB", use_container_width=True, type="secondary", help="Wipe all indexed documents from the vector database."):
            with st.spinner("Clearing vector database..."):
                try:
                    clear_collection()
                    load_db_stats(force=True)
                    st.toast("🗑️ Database cleared!", icon="✅")
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ Failed to clear DB: {exc}")
    with col_b:
        if st.button("💬 Clear Chat", use_container_width=True, type="secondary", help="Clear the chat history."):
            st.session_state.messages = []
            st.rerun()


# -----------------------------------------------------------------------------
# MAIN PANEL — Status + Chat
# -----------------------------------------------------------------------------

# -- Connection status pill ----------------------------------------------------
def _render_status_pill() -> None:
    if provider == "mock":
        st.markdown(
            "<div class='status-pill offline'>⚙️ Offline — Mock Generator</div>",
            unsafe_allow_html=True,
        )
    elif provider == "ollama":
        st.markdown(
            f"<div class='status-pill'>🦙 Ollama · {model_name or DEFAULT_OLLAMA_MODEL} · {ollama_url or DEFAULT_OLLAMA_URL}</div>",
            unsafe_allow_html=True,
        )


_render_status_pill()

# -- Empty state (no documents indexed) ---------------------------------------
fresh_stats = load_db_stats()
if fresh_stats.get("count", 0) == 0 and not st.session_state.messages:
    st.markdown("""
    <div class='empty-state'>
        <div class='empty-state-icon'>📚</div>
        <div class='empty-state-title'>Your knowledge base is empty</div>
        <div class='empty-state-subtitle'>
            Upload a <code>.txt</code> or <code>.md</code> file in the sidebar, or run
            <code>python ingest.py</code> to populate the database.
        </div>
    </div>
    """, unsafe_allow_html=True)

# -- Chat history --------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            if message.get("sources"):
                _render_sources(message["sources"])
            if message.get("prompt"):
                _render_prompt_debugger(message["prompt"])

# -- Chat input ----------------------------------------------------------------
query = st.chat_input("Ask a question about your documents...")

if query:
    query = query.strip()
    if not query:
        st.stop()

    # 1. Render user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # 2. Retrieve context
    results: dict = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    prompt = ""
    context_text = ""

    with st.spinner("🔍 Retrieving relevant context..."):
        try:
            # Rewrite query using history to handle conversational follow-ups (e.g. pronouns like "it", "they")
            condensed_query = condense_query(
                query=query,
                chat_history=st.session_state.messages[:-1],
                provider=provider,
                api_key=GEMINI_API_KEY if provider == "gemini" else (OPENAI_API_KEY if provider == "openai" else None),
                model_name=model_name,
                ollama_url=ollama_url,
            )
            
            results = retrieve(
                query=condensed_query,
                k=k_val,
                source_filters=selected_sources,
                rerank=use_reranking,
            )
            has_docs = bool((results.get("documents") or [[]])[0])
            context_text = "\n".join((results.get("documents") or [[]])[0]) if has_docs else ""
            prompt = get_rag_prompt(condensed_query, results)
        except ValueError as ve:
            st.error(f"⚠️ Invalid query: {ve}")
            st.stop()
        except RuntimeError as rte:
            st.error(f"❌ Retrieval error: {rte}")
            logger.error("Retrieval failed: %s", rte)
            st.stop()
        except Exception as exc:
            st.error(f"❌ Unexpected retrieval error: {exc}")
            logger.exception("Unexpected retrieval error.")
            st.stop()

    # 3. Stream LLM response
    full_response = ""
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        try:
            stream = generate_llm_response_stream(
                prompt=prompt,
                provider=provider,
                model_name=model_name,
                ollama_url=ollama_url,
                query=condensed_query,
                context=context_text,
            )
            full_response = response_placeholder.write_stream(stream)

        except ValueError as ve:
            full_response = f"⚠️ **Configuration error:** {ve}"
            response_placeholder.warning(full_response)
        except RuntimeError as rte:
            full_response = f"❌ **{provider.capitalize()} error:** {rte}"
            response_placeholder.error(full_response)
            logger.error("LLM streaming error (%s): %s", provider, rte)
        except Exception as exc:
            full_response = f"❌ **Unexpected error:** {exc}"
            response_placeholder.error(full_response)
            logger.exception("Unexpected LLM error.")

    # 4. Save to chat history and refresh
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": results,
        "prompt": prompt,
    })
    st.rerun()
