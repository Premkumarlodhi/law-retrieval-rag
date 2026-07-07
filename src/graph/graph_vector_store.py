"""
graph_vector_store.py

Neo4j Vector Store

Responsibilities
----------------
1. Connect to Neo4j Aura
2. Create vector index
3. Delete vector index
4. Check index existence
5. Perform vector similarity search

This module does NOT

- generate embeddings
- build graphs
- retrieve graph neighborhoods
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv()

logger = logging.getLogger(__name__)


class GraphVectorStore:

    def __init__(
        self,
        index_name: str = "legal_graph_index",
        embedding_dimension: int = 384,
    ) -> None:

        self.index_name = index_name
        self.dimension = embedding_dimension

        logger.info("Connecting to Neo4j Aura...")

        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
        )

    # ==========================================================
    # Execute Cypher
    # ==========================================================

    def run_query(
        self,
        query: str,
        params: Dict[str, Any] | None = None,
    ) -> List[Dict]:

        if params is None:
            params = {}

        return self.graph.query(
            query,
            params=params,
        )

    # ==========================================================
    # Check if vector index exists
    # ==========================================================

    def index_exists(self) -> bool:

        query = """
        SHOW VECTOR INDEXES
        YIELD name
        WHERE name = $name
        RETURN name
        """

        result = self.run_query(
            query,
            {
                "name": self.index_name,
            },
        )

        return len(result) > 0

    # ==========================================================
    # Create Vector Index
    # ==========================================================

    def create_index(self) -> None:

        if self.index_exists():

            logger.info(
                "Vector index already exists."
            )

            return

        logger.info(
            "Creating vector index..."
        )

        query = f"""
        CREATE VECTOR INDEX {self.index_name}
        IF NOT EXISTS
        FOR (n)
        ON (n.embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.dimension},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """

        self.run_query(query)

        logger.info(
            "Vector index created."
        )

    # ==========================================================
    # Drop Vector Index
    # ==========================================================

    def drop_index(self) -> None:

        if not self.index_exists():

            logger.info(
                "Vector index does not exist."
            )

            return

        logger.info(
            "Dropping vector index..."
        )

        query = f"""
        DROP INDEX {self.index_name}
        """

        self.run_query(query)

        logger.info(
            "Vector index removed."
        )

    # ==========================================================
    # Vector Search
    # ==========================================================

    def vector_search(
        self,
        embedding: List[float],
        top_k: int = 10,
    ) -> List[Dict]:

        query = """
        CALL db.index.vector.queryNodes(
            $index,
            $top_k,
            $embedding
        )

        YIELD node, score

        RETURN
            elementId(node) AS id,
            labels(node) AS labels,
            properties(node) AS properties,
            score

        ORDER BY score DESC
        """

        return self.run_query(
            query,
            {
                "index": self.index_name,
                "top_k": top_k,
                "embedding": embedding,
            },
        )