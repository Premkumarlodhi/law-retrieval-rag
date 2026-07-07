"""
graph_embeddings.py

Generates vector embeddings for Neo4j graph nodes.

Responsibilities
----------------
1. Connect to Neo4j Aura
2. Load SentenceTransformer
3. Read graph nodes
4. Generate embeddings
5. Store embeddings back into Neo4j

This module does NOT

- build graphs
- retrieve graphs
- execute GraphRAG queries
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from sentence_transformers import SentenceTransformer

load_dotenv()

logger = logging.getLogger(__name__)


class GraphEmbeddingGenerator:
    """
    Generates embeddings for every node in Neo4j.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:

        logger.info("Connecting to Neo4j...")

        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
        )

        logger.info("Loading embedding model...")

        self.model = SentenceTransformer(model_name)

    # ============================================================
    # Read Nodes
    # ============================================================

    def get_nodes(self) -> List[Dict[str, Any]]:
        """
        Fetch every node that has a 'name' property.
        """

        query = """
        MATCH (n)
        WHERE exists(n.name)
        RETURN
            elementId(n) AS id,
            labels(n) AS labels,
            properties(n) AS properties
        """

        return self.graph.query(query)

    # ============================================================
    # Build Embedding Text
    # ============================================================

    @staticmethod
    def build_text(node: Dict[str, Any]) -> str:
        """
        Converts a node into semantic text.
        """

        props = node["properties"]

        parts = []

        labels = node.get("labels", [])

        if labels:
            parts.append(f"Type: {labels[0]}")

        if props.get("name"):
            parts.append(f"Name: {props['name']}")

        if props.get("description"):
            parts.append(f"Description: {props['description']}")

        if props.get("contract_type"):
            parts.append(f"Contract Type: {props['contract_type']}")

        if props.get("section"):
            parts.append(f"Section: {props['section']}")

        return "\n".join(parts)

    # ============================================================
    # Generate Embedding
    # ============================================================

    def embed(self, text: str) -> List[float]:

        return self.model.encode(
            text,
            normalize_embeddings=True,
        ).tolist()

    # ============================================================
    # Update Node
    # ============================================================

    def update_embedding(
        self,
        node_id: str,
        embedding: List[float],
    ) -> None:

        query = """
        MATCH (n)
        WHERE elementId(n) = $id
        SET n.embedding = $embedding
        """

        self.graph.query(
            query,
            params={
                "id": node_id,
                "embedding": embedding,
            },
        )

    # ============================================================
    # Generate All Embeddings
    # ============================================================

    def build(self) -> None:

        nodes = self.get_nodes()

        logger.info(
            "Generating embeddings for %d nodes...",
            len(nodes),
        )

        for idx, node in enumerate(nodes, start=1):

            try:

                text = self.build_text(node)

                embedding = self.embed(text)

                self.update_embedding(
                    node["id"],
                    embedding,
                )

                if idx % 50 == 0:

                    logger.info(
                        "Embedded %d/%d nodes",
                        idx,
                        len(nodes),
                    )

            except Exception as e:

                logger.exception(
                    "Failed embedding node %s : %s",
                    node["id"],
                    str(e),
                )

        logger.info(
            "Finished generating graph embeddings."
        )