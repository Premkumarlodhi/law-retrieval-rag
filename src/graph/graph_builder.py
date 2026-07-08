"""
graph_builder.py

Builds a Legal Knowledge Graph using:

    Contracts
        ↓
LangChain Documents
        ↓
LLMGraphTransformer
        ↓
GraphDocuments
        ↓
Neo4j Aura
"""
from __future__ import annotations
from langchain_text_splitters import RecursiveCharacterTextSplitter


import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama
from langchain_neo4j import Neo4jGraph

from .graph_schema import get_transformer_kwargs

load_dotenv()

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
    ):

        logger.info("Connecting to Neo4j...")

        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
        )

        logger.info("Loading Groq model...")

        self.llm = ChatOllama(
        model="qwen3:8b",
        temperature=0,
        )

        logger.info("Initializing Graph Transformer...")

        self.transformer = LLMGraphTransformer(
            llm=self.llm,
            **get_transformer_kwargs(),
        )
        self.graph_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=200,
        separators=[
        "\n\n",
        "\n",
        ". ",
        "; ",
        ", ",
        " ",
        "",
        ],
        )
    # ================================================================
    # Convert contracts to LangChain Documents
    # ================================================================

    def _convert_documents(
        self,
        contracts: List[Dict[str, Any]],
    ) -> List[Document]:

        docs = []

        for contract in contracts:

            text = contract.get("text", "").strip()

            if not text:
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "doc_id": contract.get("doc_id"),
                        "title": contract.get("title"),
                        "contract_type": contract.get("contract_type"),
                    },
                )
            )

        logger.info(
            "Prepared %d LangChain documents.",
            len(docs),
        )

        return docs

    # ================================================================
    # Build Graph Documents
    # ================================================================

    def build_graph_documents(
        self,
        contracts: List[Dict[str, Any]],
    ):

        documents = self._convert_documents(
            contracts
        )

        logger.info(
            "Extracting graph with LLM..."
        )

        graph_docs = self.transformer.convert_to_graph_documents(
            documents
        )
        logger.info(
        "Average chars: %.2f",
        sum(len(c["text"]) for c in contracts) / len(contracts)
        )
        logger.info(
            "Generated %d GraphDocuments.",
            len(graph_docs),
        )

        return graph_docs

    # ================================================================
    # Upload to Neo4j
    # ================================================================

    def upload(
        self,
        graph_documents,
    ):

        logger.info(
            "Uploading graph to Neo4j..."
        )

        self.graph.add_graph_documents(
            graph_documents,
            include_source=True,
        )

        logger.info(
            "Graph upload complete."
        )

    # ================================================================
    # One-shot Pipeline
    # ================================================================

    def build(
        self,
        contracts: List[Dict[str, Any]],
        batch_size: int = 1,
    ):

        total = len(contracts)

        logger.info(
            "Building Knowledge Graph for %d contracts...",
            total,
        )

        total_uploaded = 0

        for start in range(0, total, batch_size):

            end = min(start + batch_size, total)

            logger.info(
                "Processing batch %d-%d of %d",
                start + 1,
                end,
                total,
            )

            batch = contracts[start:end]

            try:

                graph_documents = self.build_graph_documents(
                    batch
                )

                if not graph_documents:

                    logger.warning(
                        "No graph documents generated for batch %d-%d",
                        start + 1,
                        end,
                    )

                    continue

                self.upload(
                    graph_documents
                )

                total_uploaded += len(graph_documents)

                logger.info(
                    "Uploaded %d GraphDocuments.",
                    len(graph_documents),
                )

            except Exception as e:

                logger.exception(
                    "Failed processing batch %d-%d : %s",
                    start + 1,
                    end,
                    str(e),
                )

                # Continue processing remaining batches
                continue

        logger.info("=" * 70)

        logger.info(
            "Knowledge Graph construction complete."
        )

        logger.info(
            "Contracts Processed : %d",
            total,
        )

        logger.info(
            "GraphDocuments Uploaded : %d",
            total_uploaded,
        )

        logger.info("=" * 70)