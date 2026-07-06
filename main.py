import os
import time
import json
from pathlib import Path

from src.parser import parse_cuad
from src.chunker import chunk_contracts
from src.retriever import BM25Retriever

from evaluation.test_queries import TEST_QUERIES
from evaluation.retrieval_eval import (
    hit_rate,
    recall_at_k,
    reciprocal_rank,
)


def main():
    """
    Executes Phase 1:
    Data Preparation + Lexical Retrieval Pipeline
    """

    # ============================================================
    # Configuration
    # ============================================================

    DATA_PATH = Path("data") / "CUAD_v1.json"
    OUTPUT_CHUNKS_PATH = Path("data") / "processed_chunks.json"

    MAX_CONTRACTS = 200

    print("Initializing Phase 1 Pipeline...")
    start_time = time.time()

    # ============================================================
    # Parsing
    # ============================================================

    print(f"➔ Parsing the top {MAX_CONTRACTS} contracts from {DATA_PATH}...")

    try:
        parsed_documents = parse_cuad(
            input_path=DATA_PATH,
            n_contracts=MAX_CONTRACTS,
        )

    except FileNotFoundError:

        print(
            f"Error: Could not find dataset at {DATA_PATH}."
        )

        return

    # ============================================================
    # Bridge parser -> chunker
    # ============================================================

    for doc in parsed_documents:

        doc["title"] = doc.get(
            "raw_title",
            "Untitled Document",
        )

    # ============================================================
    # Chunking
    # ============================================================

    print(
        "➔ Applying text cleaning, custom stop-word removal, and paragraph chunking..."
    )

    chunked_corpus = chunk_contracts(
        parsed_documents
    )

    print(
        f"   Generated {len(chunked_corpus)} total document chunks."
    )

    # ============================================================
    # Save processed chunks
    # ============================================================

    print(
        f"➔ Saving chunks to {OUTPUT_CHUNKS_PATH}..."
    )

    OUTPUT_CHUNKS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_CHUNKS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            chunked_corpus,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ============================================================
    # Build BM25 Index
    # ============================================================

    print(
        "➔ Building the exact-match retrieval engine index..."
    )

    retriever = BM25Retriever(
        chunked_corpus
    )

    # ============================================================
    # Demo Query
    # ============================================================

    test_query = "force majeure or act of god"

    print(
        f"\n➔ Executing Test Query: '{test_query}'"
    )

    results = retriever.search(
        query=test_query,
        top_k=5,
    )

    # ============================================================
    # Display Results
    # ============================================================

    print(
        "\n--- Top 5 Lexical Retrieval Results ---"
    )

    for i, result in enumerate(results, 1):

        print("\n" + "=" * 80)

        print(f"[Result {i}]")

        print(f"Score          : {result['score']:.4f}")

        print(f"Chunk ID       : {result['chunk_id']}")

        print(f"Contract Type  : {result['contract_type']}")

        print(f"Title          : {result['title']}")

        print("\nExcerpt:\n")

        print(result["text"][:350] + "...")

    # ============================================================
    # Retrieval Evaluation
    # ============================================================

    print("\n")
    print("=" * 80)
    print("Running Retrieval Evaluation")
    print("=" * 80)

    avg_hit = 0
    avg_recall = 0
    avg_mrr = 0

    for test in TEST_QUERIES:

        results = retriever.search(
            query=test["query"],
            top_k=5,
        )

        retrieved_docs = [
            r["text"]
            for r in results
        ]

        hit = hit_rate(
            retrieved_docs,
            test["expected"],
        )

        recall = recall_at_k(
            retrieved_docs,
            test["expected"],
        )

        rr = reciprocal_rank(
            retrieved_docs,
            test["expected"],
        )

        avg_hit += hit
        avg_recall += recall
        avg_mrr += rr

        print("\n" + "-" * 60)

        print(f"Query      : {test['query']}")

        print(f"Hit@5      : {hit}")

        print(f"Recall@5   : {recall:.2f}")

        print(f"MRR        : {rr:.2f}")

    total_queries = len(TEST_QUERIES)

    print("\n")
    print("=" * 80)

    print(
        f"Average Hit@5      : {avg_hit / total_queries:.3f}"
    )

    print(
        f"Average Recall@5   : {avg_recall / total_queries:.3f}"
    )

    print(
        f"Average MRR        : {avg_mrr / total_queries:.3f}"
    )

    # ============================================================
    # Summary
    # ============================================================

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)

    print(
        f"Pipeline execution completed in {elapsed:.2f} seconds."
    )

    print(
        f"Indexed Documents : {len(parsed_documents)}"
    )

    print(
        f"Indexed Chunks    : {len(chunked_corpus)}"
    )


if __name__ == "__main__":
    main()