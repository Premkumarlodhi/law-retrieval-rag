"""
graph_embeddings.py

Generates vector embeddings for Neo4j graph nodes.
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

        query = """
        MATCH (n)
        RETURN
            elementId(n) AS id,
            labels(n) AS labels,
            properties(n) AS properties
        """

        nodes = self.graph.query(query)

        logger.info(
            "Found %d graph nodes.",
            len(nodes),
        )

        return nodes

    # ============================================================
    # Build Embedding Text
    # ============================================================

    @staticmethod
    def build_text(
        node: Dict[str, Any],
    ) -> str:

        props = node["properties"]
        labels = node["labels"]

        parts = []

        if labels:
            parts.append(f"Label: {labels[0]}")

        if props.get("name"):
            parts.append(f"Name: {props['name']}")

        if props.get("title"):
            parts.append(f"Title: {props['title']}")

        if props.get("contract_type"):
            parts.append(
                f"Contract Type: {props['contract_type']}"
            )

        if props.get("type"):
            parts.append(
                f"Type: {props['type']}"
            )

        if props.get("date"):
            parts.append(
                f"Date: {props['date']}"
            )

        if props.get("section"):
            parts.append(
                f"Section: {props['section']}"
            )

        if props.get("description"):
            parts.append(
                props["description"]
            )

        if props.get("text"):
            parts.append(
                props["text"][:3000]
            )

        if not parts:
            parts.append(str(props))

        return "\n".join(parts)

    # ============================================================
    # Generate Embedding
    # ============================================================

    def embed(
        self,
        text: str,
    ) -> List[float]:

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
            id=node_id,
            embedding=embedding,
        )

    # ============================================================
    # Generate All Embeddings
    # ============================================================

    def build(self) -> None:

        nodes = self.get_nodes()

        if not nodes:

            logger.warning(
                "No graph nodes found."
            )

            return

        logger.info(
            "Generating embeddings for %d nodes...",
            len(nodes),
        )

        embedded = 0

        for idx, node in enumerate(
            nodes,
            start=1,
        ):

            try:

                text = self.build_text(node)

                if not text.strip():
                    continue

                embedding = self.embed(text)

                self.update_embedding(
                    node["id"],
                    embedding,
                )

                embedded += 1

                if idx % 25 == 0:

                    logger.info(
                        "Embedded %d/%d nodes",
                        idx,
                        len(nodes),
                    )

            except Exception:

                logger.exception(
                    "Failed embedding node %s",
                    node["id"],
                )

        logger.info(
            "Successfully embedded %d/%d nodes.",
            embedded,
            len(nodes),
        )