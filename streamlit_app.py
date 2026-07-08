"""Streamlit frontend for the Legal GraphRAG Assistant.

This module is a pure presentation layer. It does not implement any
retrieval, ranking, or graph logic itself -- it only wires together the
already-existing, already-working project components:

    BM25Retriever      (src/retriever.py)
    SemanticRetriever  (src/semantic_retriever.py)
    HybridRetriever    (src/hybrid_retriever.py)
    CrossEncoderReranker (src/cross_encoder.py)
    GraphRetriever     (src/graph/graph_retriever.py)

No existing project file is imported for its private/internal helpers,
and no existing file is modified. All component constructors, method
names, arguments, and return fields used below were verified against the
uploaded source of these modules.

Run with:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import faulthandler
faulthandler.enable()

import traceback
import sys

def excepthook(exc_type, exc_value, exc_tb):
    traceback.print_exception(exc_type, exc_value, exc_tb)

sys.excepthook = excepthook
# ---------------------------------------------------------------------------
# Optional .env support (best-effort only, never required)
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Import existing project components (never modified).
#
# The project layout uses a ``src`` package (confirmed by the import
# statements inside hybrid_retriever.py and graph_retriever.py), but we
# fall back to top-level imports so this frontend still runs if the
# project is executed from inside ``src`` or with a different sys.path
# layout. Either way, no existing file is edited.
# ---------------------------------------------------------------------------

try:
    from src.retriever import BM25Retriever
    from src.hybrid_retriever import HybridRetriever
except ImportError:
    from retriever import BM25Retriever
    from hybrid_retriever import HybridRetriever
try:
    from src.graph.graph_retriever import GraphRetriever
except ImportError:
    try:
        from graph.graph_retriever import GraphRetriever  # type: ignore
    except ImportError:
        from graph_retriever import GraphRetriever  # type: ignore

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama  # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_CHUNKS_PATH = Path(
    os.getenv(
        "PROCESSED_CHUNKS_PATH",
        BASE_DIR / "data" / "processed_chunks.json",
    )
)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")

HYBRID_TOP_K = 20
RERANK_TOP_K = 5
GRAPH_TOP_K = 5
GRAPH_MAX_HOPS = 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("legal_graphrag_streamlit")


# ---------------------------------------------------------------------------
# Component initialization (runs exactly once per server process)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_processed_chunks() -> Optional[List[Dict[str, Any]]]:
    """Load the already-processed corpus. Never parses or chunks documents."""
    path = Path(PROCESSED_CHUNKS_PATH)
    if not path.exists():
        logger.error("processed_chunks.json not found at %s", path.resolve())
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception:
        logger.exception("Failed to read processed_chunks.json")
        return None

    if not isinstance(chunks, list) or not chunks:
        logger.error("processed_chunks.json did not contain a non-empty list")
        return None

    logger.info("Loaded %d processed chunks", len(chunks))
    return chunks


@st.cache_resource(show_spinner=False)
def init_components() -> Dict[str, Any]:
    """Construct every retrieval component exactly once.

    Returns a dict with keys: bm25, semantic, hybrid, cross_encoder,
    graph_retriever (may be None), llm (may be None), and any startup
    error/warning strings for display in the UI.
    """
    warnings: List[str] = []

    chunks = load_processed_chunks()
    if not chunks:
        return {
            "ready": False,
            "warnings": [
                f"processed_chunks.json could not be loaded from "
                f"'{PROCESSED_CHUNKS_PATH}'. Run the existing ingestion "
                f"pipeline first."
            ],
        }

    # BM25
    try:
        bm25 = BM25Retriever(chunks)
    except Exception as exc:
        logger.exception("Failed to construct BM25Retriever")
        return {"ready": False, "warnings": [f"BM25Retriever failed to initialize: {exc}"]}

    # Semantic (loads existing ChromaDB; only rebuilds if missing/mismatched)
    try:
        semantic = SemanticRetriever()
        semantic.initialize(chunks)
    except Exception as exc:
        logger.exception("Failed to construct/initialize SemanticRetriever")
        return {
            "ready": False,
            "warnings": [f"SemanticRetriever failed to initialize: {exc}"],
        }

    # Hybrid
    try:
        hybrid = HybridRetriever(bm25=bm25, semantic=semantic)
    except Exception as exc:
        logger.exception("Failed to construct HybridRetriever")
        return {"ready": False, "warnings": [f"HybridRetriever failed to initialize: {exc}"]}

    # Cross encoder
    try:
        cross_encoder = CrossEncoderReranker()
        cross_encoder.initialize()
    except Exception as exc:
        logger.exception("Failed to construct/initialize CrossEncoderReranker")
        return {
            "ready": False,
            "warnings": [f"CrossEncoderReranker failed to initialize: {exc}"],
        }

    # Graph retriever (optional -- app must still work if Neo4j is unavailable)
    graph_retriever = None
    try:
        graph_retriever = GraphRetriever()
    except Exception as exc:
        logger.warning("GraphRetriever unavailable: %s", exc)
        warnings.append(
            "Knowledge graph is unavailable (Neo4j not reachable or not "
            "configured). Continuing with hybrid retrieval + cross-encoder "
            "only."
        )

    # LLM
    llm = None
    try:
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    except Exception as exc:
        logger.warning("ChatOllama unavailable: %s", exc)
        warnings.append(
            f"Ollama model '{OLLAMA_MODEL}' could not be initialized. "
            f"Answers cannot be generated until Ollama is running."
        )

    return {
        "ready": True,
        "bm25": bm25,
        "semantic": semantic,
        "hybrid": hybrid,
        "cross_encoder": cross_encoder,
        "graph_retriever": graph_retriever,
        "llm": llm,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Query pipeline
# ---------------------------------------------------------------------------


def _format_hybrid_context(reranked: List[Dict[str, Any]]) -> str:
    """Render reranked hybrid/cross-encoder candidates as prompt evidence."""
    if not reranked:
        return "No relevant contract clauses were retrieved."

    blocks = []
    for doc in reranked:
        title = doc.get("title") or "Untitled"
        section = doc.get("section") or ""
        chunk_id = doc.get("chunk_id", "")
        text = doc.get("text", "")
        header = f"[Clause | {title}"
        if section:
            header += f" | Section: {section}"
        header += f" | chunk_id={chunk_id}]"
        blocks.append(f"{header}\n{text}")

    return "\n\n".join(blocks)


def _format_graph_context(graph_context: List[Dict[str, Any]]) -> str:
    """Render GraphRetriever.retrieve() output as prompt evidence."""
    if not graph_context:
        return "No related knowledge graph entities were found."

    blocks = []
    for node in graph_context:
        name = node.get("name", "Unknown")
        label = node.get("label", "Unknown")
        connections = node.get("connections", [])

        conn_lines = []
        for conn in connections:
            relationship = conn.get("relationship", "")
            target = conn.get("target", "")
            target_label = conn.get("target_label", "")
            conn_lines.append(f"  -[{relationship}]-> {target} ({target_label})")

        block = f"[Entity | {name} ({label})]"
        if conn_lines:
            block += "\n" + "\n".join(conn_lines)
        blocks.append(block)

    return "\n\n".join(blocks)


def _build_prompt(query: str, hybrid_context: str, graph_context: str) -> str:
    """Build a grounded legal-assistant prompt from retrieved evidence only."""
    return (
        "You are a precise legal assistant answering questions about "
        "contracts using ONLY the evidence provided below. Never rely on "
        "outside knowledge and never invent facts, clauses, or citations.\n\n"
        "If the evidence below is insufficient to answer confidently, say "
        "so explicitly instead of guessing.\n\n"
        "=== RETRIEVED CONTRACT CLAUSES ===\n"
        f"{hybrid_context}\n\n"
        "=== RELATED KNOWLEDGE GRAPH CONTEXT ===\n"
        f"{graph_context}\n\n"
        "=== QUESTION ===\n"
        f"{query}\n\n"
        "Provide a clear, well-organized answer. Reference clause titles "
        "or sections where relevant."
    )


def run_query_pipeline(components: Dict[str, Any], query: str) -> str:
    """Execute the full retrieval -> rerank -> graph -> LLM pipeline."""
    overall_start = time.perf_counter()
    timings: Dict[str, float] = {}

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(query)

    hybrid: HybridRetriever = components["hybrid"]
    cross_encoder: CrossEncoderReranker = components["cross_encoder"]
    graph_retriever = components.get("graph_retriever")
    llm = components.get("llm")

    # ---------------- Hybrid retrieval ----------------
    stage_start = time.perf_counter()
    try:
        hybrid_results = hybrid.search(query, top_k=HYBRID_TOP_K, debug=True)
    except Exception:
        logger.exception("Hybrid retrieval failed")
        hybrid_results = []
    timings["hybrid_retrieval"] = time.perf_counter() - stage_start

    print("\n" + "=" * 80)
    print("HYBRID RETRIEVAL")
    print("=" * 80)
    print(f"Candidates returned: {len(hybrid_results)}")

    # ---------------- Cross-encoder rerank ----------------
    stage_start = time.perf_counter()
    reranked: List[Dict[str, Any]] = []
    if hybrid_results:
        try:
            reranked = cross_encoder.rerank(
                query, hybrid_results, top_k=RERANK_TOP_K, debug=True
            )
        except Exception:
            logger.exception("Cross-encoder rerank failed")
            reranked = hybrid_results[:RERANK_TOP_K]
    timings["cross_encoder_rerank"] = time.perf_counter() - stage_start

    # ---------------- Graph retrieval ----------------
    stage_start = time.perf_counter()
    graph_context: List[Dict[str, Any]] = []
    if graph_retriever is not None:
        try:
            graph_context = graph_retriever.retrieve(
                query, top_k=GRAPH_TOP_K, max_hops=GRAPH_MAX_HOPS
            )
        except Exception:
            logger.exception("Graph retrieval failed")
            graph_context = []
    timings["graph_retrieval"] = time.perf_counter() - stage_start

    print("\n" + "=" * 80)
    print("GRAPH RETRIEVAL")
    print("=" * 80)
    print(f"Entities returned: {len(graph_context)}")

    # ---------------- Merge contexts ----------------
    hybrid_context_str = _format_hybrid_context(reranked)
    graph_context_str = _format_graph_context(graph_context)

    print("\n" + "=" * 80)
    print("MERGED CONTEXT")
    print("=" * 80)
    print(hybrid_context_str)
    print("-" * 40)
    print(graph_context_str)

    # ---------------- Prompt ----------------
    prompt = _build_prompt(query, hybrid_context_str, graph_context_str)

    print("\n" + "=" * 80)
    print("PROMPT")
    print("=" * 80)
    print(prompt)

    # ---------------- LLM ----------------
    stage_start = time.perf_counter()
    if llm is None:
        answer = (
            "The language model (Ollama) is not currently available, so I "
            "cannot generate an answer. Please ensure Ollama is running "
            f"with the '{OLLAMA_MODEL}' model and try again."
        )
    elif not reranked and not graph_context:
        answer = (
            "I could not find sufficient evidence in the contract corpus "
            "or knowledge graph to answer this question."
        )
    else:
        try:
            response = llm.invoke(prompt)
            answer = getattr(response, "content", str(response))
        except Exception:
            logger.exception("LLM invocation failed")
            answer = (
                "An error occurred while generating the answer from the "
                "language model. Please check that Ollama is running and "
                "try again."
            )
    timings["llm_generation"] = time.perf_counter() - stage_start
    timings["total"] = time.perf_counter() - overall_start

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)

    print("\n" + "=" * 80)
    print("TIMINGS")
    print("=" * 80)
    for stage, duration in timings.items():
        print(f"{stage:<20}: {duration:.3f}s")
    print("=" * 80 + "\n")

    return answer


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0e1117;
            color: #e6e6e6;
        }
        .block-container {
            max-width: 800px;
            padding-top: 2rem;
            margin: 0 auto;
        }
        .app-title {
            text-align: center;
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .app-subtitle {
            text-align: center;
            color: #9aa0a6;
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        '<div class="app-title">⚖️ Legal GraphRAG Assistant</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="app-subtitle">Hybrid RAG + Knowledge Graph + '
        "Cross Encoder + Ollama</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.write("Entered main")
    st.set_page_config(
        page_title="Legal GraphRAG Assistant",
        page_icon="⚖️",
        layout="centered",
    )

    _inject_theme()
    _render_header()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.spinner("Loading retrieval components..."):
        components = init_components()

    for warning_msg in components.get("warnings", []):
        st.warning(warning_msg)

    if not components.get("ready"):
        st.error(
            "The application could not start because required components "
            "are unavailable. Please check the terminal logs."
        )
        return

    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask a question about the contracts...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = run_query_pipeline(components, user_query)
                except Exception:
                    logger.exception("Unhandled error in query pipeline")
                    answer = (
                        "An unexpected error occurred while processing your "
                        "question. Please try again."
                    )
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})



    main()