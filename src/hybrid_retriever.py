from typing import List, Dict, Any, Set

from src.retriever import BM25Retriever
from src.semantic_retriever import SemanticRetriever


class HybridRetriever:
    """
    Hybrid Retriever using Reciprocal Rank Fusion (RRF).

    Combines:
        • BM25 Retriever
        • Semantic Retriever

    Returns the same output interface as both retrievers.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        semantic: SemanticRetriever,
        rrf_k: int = 60,
    ):
        self.bm25 = bm25
        self.semantic = semantic
        self.rrf_k = rrf_k

    # ==========================================================
    # Reciprocal Rank Fusion
    # ==========================================================

    def _rrf_score(self, rank: int) -> float:
        return 1.0 / (self.rrf_k + rank)

    # ==========================================================
    # Internal helper
    # ==========================================================

    def _add_result(
        self,
        fused: Dict[str, Dict[str, Any]],
        doc: Dict[str, Any],
        rank: int,
        source: str,
    ) -> None:
        """
        Insert or update a retrieval result inside the fused dictionary.
        """

        chunk = doc.get("chunk_id")

        if chunk is None:
            return

        if chunk not in fused:

            fused[chunk] = {
                **doc,
                "bm25_rank": None,
                "semantic_rank": None,
                "bm25_score": None,
                "semantic_similarity": None,
                "semantic_distance": None,
                "hybrid_score": 0.0,
                "retrieval_sources": {source},  # type: Set[str]
            }

        entry = fused[chunk]

        entry["hybrid_score"] += self._rrf_score(rank)

        retrieval_sources: Set[str] = entry["retrieval_sources"]
        retrieval_sources.add(source)

        if source == "bm25":
            entry["bm25_rank"] = rank
            entry["bm25_score"] = doc.get("score")

        elif source == "semantic":
            entry["semantic_rank"] = doc.get(
                "semantic_rank",
                rank,
            )
            entry["semantic_similarity"] = doc.get("similarity")
            entry["semantic_distance"] = doc.get("distance")

    # ==========================================================
    # Search
    # ==========================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:

        if not query.strip():
            return []

        if top_k <= 0:
            return []

        retrieval_depth = max(top_k * 3, 20)

        bm25_results = self.bm25.search(
            query,
            top_k=retrieval_depth,
        )

        semantic_results = self.semantic.search(
            query,
            top_k=retrieval_depth,
        )

        fused: Dict[str, Dict[str, Any]] = {}

        # -----------------------------
        # BM25 Results
        # -----------------------------

        for rank, doc in enumerate(
            bm25_results,
            start=1,
        ):
            self._add_result(
                fused=fused,
                doc=doc,
                rank=rank,
                source="bm25",
            )

        # -----------------------------
        # Semantic Results
        # -----------------------------

        for rank, doc in enumerate(
            semantic_results,
            start=1,
        ):
            self._add_result(
                fused=fused,
                doc=doc,
                rank=rank,
                source="semantic",
            )

        bm25_only = 0
        semantic_only = 0
        both = 0

        for result in fused.values():

            sources: Set[str] = result["retrieval_sources"]

            if len(sources) == 1:
                if "bm25" in sources:
                    bm25_only += 1
                else:
                    semantic_only += 1
            else:
                both += 1

        if debug:
            print("\n========== Hybrid Retrieval Diagnostics ==========")
            print(f"Total BM25 candidates      : {len(bm25_results)}")
            print(f"Total Semantic candidates  : {len(semantic_results)}")
            print(f"Overlap                    : {both}")
            print(f"BM25 only                  : {bm25_only}")
            print(f"Semantic only              : {semantic_only}")
            print(f"Total fused candidates     : {len(fused)}")

        ranked = sorted(
            fused.values(),
            key=lambda x: (
                x["hybrid_score"],
                x.get("semantic_similarity", 0) or 0,
                x.get("chunk_id", ""),
            ),
            reverse=True,
        )

        for result in ranked:
            result["retrieval_sources"] = sorted(
                result["retrieval_sources"]
            )

        if debug:
            print("\n========== Final Ranking ==========")

            for result in ranked:

                print(
                    f"Hybrid Score      : {result['hybrid_score']:.6f}"
                )
                print(
                    f"BM25 Rank         : {result.get('bm25_rank')}"
                )
                print(
                    f"Semantic Rank     : {result.get('semantic_rank')}"
                )
                print(
                    f"Chunk ID          : {result.get('chunk_id')}"
                )
                print(
                    f"Title             : {result.get('title')}"
                )
                print(
                    f"Retrieval Sources : {result.get('retrieval_sources')}"
                )
                print(
                    f"Similarity        : {result.get('semantic_similarity')}"
                )
                print(
                    f"Distance          : {result.get('semantic_distance')}"
                )
                print("-" * 60)

        return ranked[:top_k]