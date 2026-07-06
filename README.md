# ⚖️ Legal RAG: Hybrid Ensemble Retrieval & Knowledge Graph

An advanced Retrieval-Augmented Generation (RAG) pipeline designed to process, index, and query complex commercial legal contracts. 

This project solves the core limitations of standard AI search in the legal and financial sectors by combining **Exact Lexical Matching (BM25)**, **Semantic Vector Search (ChromaDB)**, and **Entity Relationship Mapping (PostgreSQL Knowledge Graph)**.

---

## 🎯 The Problem & The Solution

Lawyers and compliance officers waste hundreds of hours manually searching massive repositories of past contracts. Standard AI retrieval systems fail in three ways:
1. **Semantic Failure:** Keyword searches for "extreme weather exemption" miss clauses titled "Force Majeure" or "Act of God".
2. **Exact-Match Failure:** Vector embeddings blur distinct alphanumeric IDs, failing to retrieve exact statutes like "Section 409A" or "10-Q".
3. **Relational Failure:** Neither method can answer multi-hop questions like, *"Has Company A been sued by Company B over an IP clause?"*

**The Solution:** This architecture implements an **Ensemble Retrieval System** orchestrated by LangChain. It routes queries simultaneously to a BM25 index (for exact keywords), a Vector Database (for semantic meaning), and a Knowledge Graph (for relationships), applying Reciprocal Rank Fusion (RRF) to merge the results into a legally precise context window.

---

## 🏗️ Architecture & Roadmap

### ✅ Phase 1: Data Preparation & Lexical Retrieval (Current State)
- **Parser:** Ingests the SQuAD-style CUAD dataset and prepends structured metadata headers (Contract Type, Parties, Date) to the contract body.
- **Chunker:** Utilizes `RecursiveCharacterTextSplitter` with custom text cleaning that aggressively removes standard stop-words while strictly preserving **legal modifiers** (e.g., *not, unless, shall, provided that*).
- **Lexical Engine:** Implements a robust `BM25Okapi` index with custom tokenization (preserving hyphens and periods) for exact-match statutory lookup.

### ⏳ Phase 2: Dense Semantic Search (Next Up)
- Integrate **ChromaDB** and Hugging Face embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
- Establish the dual-retrieval LangChain router.

### ⏳ Phase 3: Knowledge Graph Construction
- Extract entity relationships (Subject -> Predicate -> Object) using LLMs.
- Store and query relationship triples using **PostgreSQL**.

### ⏳ Phase 4: Generation & QLoRA Fine-Tuning
- Fine-tune a **LLaMA-3** model on legal QA pairs using **Unsloth** (4-bit quantization).
- Deploy as the final generation head to output citation-backed legal analysis.

---

## 🗂️ Repository Structure

```text
.
├── data/
│   ├── CUAD_v1.json            # Raw dataset (Not included in repo)
│   └── processed_chunks.json   # Generated output for Phase 2/3
├── src/
│   ├── parser.py               # Extracts full text & metadata from JSON
│   ├── chunker.py              # LangChain splitting & legal token cleaning
│   └── retriever.py            # BM25 Inverted Index & search execution
├── main.py                     # Phase 1 Pipeline Execution
├── requirements.txt            # Project dependencies
└── README.md


drive link: https://drive.google.com/drive/folders/1ul41vBCSh5xJhafas_O9PN9f65oeJgEJ?usp=drive_link
Contract Understanding Atticus Dataset (CUAD)

