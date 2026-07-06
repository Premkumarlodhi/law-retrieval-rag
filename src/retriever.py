import re
from typing import List, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Production-ready BM25 Retriever.

    Stores the complete document objects while indexing only the text.

    Returns ranked document objects with metadata and BM25 scores,
    making it directly compatible with:
        • Hybrid Retrieval (RRF)
        • ChromaDB
        • LangChain
        • Evaluation
        • Knowledge Graph
    """

    def __init__(self, documents: List[Dict[str, Any]]):

        if not documents:
            raise ValueError("Document list cannot be empty.")

        self.documents = documents

        self.corpus = [
            doc["text"]
            for doc in documents
        ]

        tokenized_corpus = [
            self._tokenize(text)
            for text in self.corpus
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        """
        Preserve legal identifiers like:

        Section-10
        409.A
        10-Q
        Force-Majeure
        """

        text = text.lower()

        return re.findall(
            r"\b[\w\.\-]+\b",
            text,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:

        if not query.strip():
            return []

        tokenized_query = self._tokenize(query)

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_idx = np.argsort(scores)[::-1]

        results = []

        for idx in ranked_idx[:top_k]:

            doc = self.documents[idx]

            results.append(
                {
                    "doc_id":
                        doc["doc_id"],

                    "chunk_id":
                        doc["chunk_id"],

                    "title":
                        doc["title"],

                    "raw_title":
                        doc["raw_title"],

                    "contract_type":
                        doc["contract_type"],

                    "section":
                        doc.get(
                            "section",
                            "",
                        ),

                    "score":
                        float(scores[idx]),

                    "text":
                        doc["text"],

                    "metadata":
                        doc["metadata"],
                }
            )

        return results