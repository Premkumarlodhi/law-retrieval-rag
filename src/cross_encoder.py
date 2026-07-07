from typing import List, Dict, Any

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Cross-Encoder reranker.

    Rescores Hybrid Retriever candidates using full query-document
    interaction.

    Input:
        Hybrid Retriever output

    Output:
        Same candidates with

            cross_score
            rerank_position

    sorted by cross_score.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None

    # ==========================================================
    # Initialization
    # ==========================================================

    def initialize(self) -> None:

        if self.model is None:
            self.model = CrossEncoder(self.model_name)

    # ==========================================================
    # Score query-document pairs
    # ==========================================================

    def _score_pairs(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[float]:

        pairs = [
            (
                query,
                doc.get("text", ""),
            )
            for doc in candidates
        ]

        return self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        ).tolist()

    # ==========================================================
    # Rerank
    # ==========================================================

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:

        if self.model is None:
            raise RuntimeError(
                "CrossEncoder not initialized."
            )

        if not query.strip():
            return []

        if not candidates:
            return []

        if top_k <= 0:
            return []

        scores = self._score_pairs(
            query,
            candidates,
        )

        reranked = []

        for doc, score in zip(candidates, scores):

            entry = dict(doc)

            entry["cross_score"] = float(score)

            reranked.append(entry)

        reranked.sort(
            key=lambda x: (
                x["cross_score"],
                x.get("hybrid_score", 0),
                x.get("semantic_similarity", 0) or 0,
                x.get("chunk_id", ""),
            ),
            reverse=True,
        )

        for rank, doc in enumerate(
            reranked,
            start=1,
        ):
            doc["rerank_position"] = rank

        if debug:

            print("\n========== Cross Encoder Ranking ==========\n")

            for doc in reranked:

                print(
                    f"Rank              : {doc['rerank_position']}"
                )
                print(
                    f"Cross Score       : {doc['cross_score']:.4f}"
                )
                print(
                    f"Hybrid Score      : {doc.get('hybrid_score'):.6f}"
                )
                print(
                    f"BM25 Rank         : {doc.get('bm25_rank')}"
                )
                print(
                    f"Semantic Rank     : {doc.get('semantic_rank')}"
                )
                print(
                    f"Chunk ID          : {doc.get('chunk_id')}"
                )
                print(
                    f"Title             : {doc.get('title')}"
                )
                print(
                    f"Sources           : {doc.get('retrieval_sources')}"
                )
                print("-" * 60)

        return reranked[:top_k]