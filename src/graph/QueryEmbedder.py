"""
graph_query_embedder.py

Query embedding module for GraphRAG.

Responsibilities
----------------
1. Load the SentenceTransformer model
2. Embed user queries
3. Embed batches of queries

This module does NOT

- connect to Neo4j
- create vector indexes
- perform retrieval
- build graphs
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class QueryEmbedder:
    """
    Generates semantic embeddings for GraphRAG queries.

    The embedding model is loaded only once and reused
    throughout the lifetime of the application.
    """

    _model: SentenceTransformer | None = None

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:

        if QueryEmbedder._model is None:

            logger.info(
                "Loading embedding model: %s",
                model_name,
            )

            QueryEmbedder._model = SentenceTransformer(
                model_name
            )

        self.model = QueryEmbedder._model

    # ============================================================
    # Single Query
    # ============================================================

    def encode(
        self,
        query: str,
    ) -> List[float]:
        """
        Generate an embedding for a single query.

        Parameters
        ----------
        query : str

        Returns
        -------
        List[float]
        """

        if not query.strip():
            return []

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding.astype(np.float32).tolist()

    # ============================================================
    # Batch Queries
    # ============================================================

    def encode_batch(
        self,
        queries: List[str],
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple queries.

        Parameters
        ----------
        queries : List[str]

        Returns
        -------
        List[List[float]]
        """

        if not queries:
            return []

        embeddings = self.model.encode(
            queries,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.astype(np.float32).tolist()

    # ============================================================
    # Embedding Dimension
    # ============================================================

    @property
    def dimension(self) -> int:
        """
        Returns the embedding dimension.
        """

        return self.model.get_sentence_embedding_dimension()