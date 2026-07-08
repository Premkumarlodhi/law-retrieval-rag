import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from src import cross_encoder
from src.cross_encoder import CrossEncoderReranker
from evaluation.retrieval_eval import hit_rate, reciprocal_rank, recall_at_k
from evaluation.test_queries import TEST_QUERIES
from src.chunker import chunk_contracts
from src.graph.graph_builder import KnowledgeGraphBuilder
from src.graph.graph_embeddings import GraphEmbeddingGenerator
from src.graph.graph_vector_store import GraphVectorStore
from src.graph.graph_retriever import GraphRetriever
from src.hybrid_retriever import HybridRetriever
from src.parser import parse_cuad
from src.retriever import BM25Retriever
from src.semantic_retriever import SemanticRetriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    repository_root: Path
    data_path: Path
    output_chunks_path: Path
    max_contracts: int = 200
    demo_query: str = "force majeure or act of god"
    top_k: int = 5
    eval_top_k: int = 5
    rebuild_graph: bool = False

class RetrievalPipeline:
    """Orchestrates the document ingestion, chunking, indexing, retrieval, and evaluation stages."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.parsed_documents: List[Dict[str, Any]] = []
        self.chunked_corpus: List[Dict[str, Any]] = []
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.semantic_retriever: Optional[SemanticRetriever] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.cross_encoder: Optional[CrossEncoderReranker] = None
        self.graph_builder: Optional[KnowledgeGraphBuilder] = None
        self.graph_embeddings: Optional[GraphEmbeddingGenerator] = None
        self.graph_vector_store: Optional[GraphVectorStore] = None
        self.graph_retriever: Optional[GraphRetriever] = None

    def run(self) -> Dict[str, Any]:
        start_time = time.time()
        logger.info("Initializing retrieval pipeline...")

        parsed_documents = self._parse_stage()
        prepared_documents = self._prepare_stage(parsed_documents)
        chunked_corpus = self._chunk_stage(prepared_documents)
        self._persist_chunks_stage(chunked_corpus)

        (
            self.bm25_retriever,
            self.semantic_retriever,
            self.hybrid_retriever,
            self.cross_encoder,
        ) = self._build_retrieval_stage(chunked_corpus)
        
        self._initialize_graph()

        if self.config.rebuild_graph:
            self._build_graph_stage()
        
        self._demo_query_stage(self.config.demo_query, self.config.top_k)
        evaluation_metrics = self._evaluate_stage(self.config.eval_top_k)

        elapsed = time.time() - start_time
        logger.info("Pipeline execution completed in %.2f seconds.", elapsed)
        logger.info("Indexed Documents : %d", len(self.parsed_documents))
        logger.info("Indexed Chunks    : %d", len(self.chunked_corpus))

        return {
            "documents": self.parsed_documents,
            "chunks": self.chunked_corpus,
            "evaluation": evaluation_metrics,
            "elapsed_seconds": elapsed,
        }

    def _parse_stage(self) -> List[Dict[str, Any]]:
        if not self.config.data_path.exists():
            raise FileNotFoundError(
                f"Could not find dataset at {self.config.data_path}."
            )

        logger.info(
            "Parsing the top %d contracts from %s...",
            self.config.max_contracts,
            self.config.data_path,
        )
        parsed_documents = parse_cuad(
            input_path=self.config.data_path,
            n_contracts=self.config.max_contracts,
        )
        self.parsed_documents = parsed_documents
        return parsed_documents

    def _prepare_stage(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info("Normalizing document metadata and titles...")
        prepared_documents: List[Dict[str, Any]] = []

        for document in documents:
            prepared_document = dict(document)
            prepared_document["title"] = prepared_document.get(
                "raw_title",
                "Untitled Document",
            )
            prepared_document["metadata"] = self._sanitize_metadata(
                prepared_document.get("metadata", {})
            )
            prepared_documents.append(prepared_document)

        self.parsed_documents = prepared_documents
        return prepared_documents

    def _chunk_stage(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(
            "Applying text cleaning, stop-word removal, and paragraph chunking..."
        )
        chunked_corpus = chunk_contracts(documents)
        self.chunked_corpus = chunked_corpus
        logger.info("Generated %d document chunks.", len(chunked_corpus))
        return chunked_corpus

    def _persist_chunks_stage(self, chunked_corpus: List[Dict[str, Any]]) -> None:
        logger.info("Saving processed chunks to %s...", self.config.output_chunks_path)
        self.config.output_chunks_path.parent.mkdir(parents=True, exist_ok=True)

        with self.config.output_chunks_path.open("w", encoding="utf-8") as handle:
            json.dump(chunked_corpus, handle, indent=2, ensure_ascii=False)

    def _build_retrieval_stage(
        self,
        chunked_corpus: List[Dict[str, Any]],
    ) -> Tuple[BM25Retriever, SemanticRetriever, HybridRetriever, CrossEncoderReranker]:
        logger.info("Building lexical retrieval engine...")
        bm25_retriever = BM25Retriever(chunked_corpus)

        logger.info("Building semantic retrieval engine...")
        semantic_retriever = SemanticRetriever()
        semantic_retriever.initialize(chunked_corpus)

        logger.info("Assembling hybrid retrieval engine...")
        hybrid_retriever = HybridRetriever(
            bm25=bm25_retriever,
            semantic=semantic_retriever,
        )
        logger.info("Loading Cross Encoder...")

        cross_encoder = CrossEncoderReranker()
        cross_encoder.initialize()

        return (
        bm25_retriever,
        semantic_retriever,
        hybrid_retriever,
        cross_encoder,
        )

    def _demo_query_stage(self, query: str, top_k: int) -> None:
        logger.info("Executing demo query: %s", query)
        if self.hybrid_retriever is None:
            raise RuntimeError("Hybrid retriever has not been initialized.")

        hybrid_results = self.hybrid_retriever.search(
        query=query,
        top_k=max(top_k * 4, 20),
        )

        results = self.cross_encoder.rerank(
        query=query,
        candidates=hybrid_results,
        top_k=top_k,
        debug=True,
        )
        graph_context = self._graph_search(
            query=query,
            top_k=5,
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info("GRAPH RETRIEVAL")
        logger.info("=" * 80)

        for node in graph_context:

            logger.info(
                "%s (%s)",
                node["name"],
                node["label"],
            )
        self._print_results(results, title="Hybrid Retrieval Results")

    def _evaluate_stage(self, top_k: int) -> Dict[str, float]:
        logger.info("Running retrieval evaluation...")
        if self.hybrid_retriever is None:
            raise RuntimeError("Hybrid retriever has not been initialized.")

        avg_hit = 0.0
        avg_recall = 0.0
        avg_mrr = 0.0

        for test_case in TEST_QUERIES:
            hybrid_results = self.hybrid_retriever.search(
            query=test_case["query"],
            top_k=max(top_k * 4, 20),
            )

            results = self.cross_encoder.rerank(
            query=test_case["query"],
            candidates=hybrid_results,
            top_k=top_k,
            )
            retrieved_docs = [result["text"] for result in results]

            hit = hit_rate(retrieved_docs, test_case["expected"])
            recall = recall_at_k(retrieved_docs, test_case["expected"])
            rr = reciprocal_rank(retrieved_docs, test_case["expected"])

            avg_hit += hit
            avg_recall += recall
            avg_mrr += rr

            logger.info("-" * 60)
            logger.info("Query      : %s", test_case["query"])
            logger.info("Hit@%d      : %s", top_k, hit)
            logger.info("Recall@%d   : %.2f", top_k, recall)
            logger.info("MRR        : %.2f", rr)

        total_queries = len(TEST_QUERIES)
        metrics = {
            "avg_hit_at_k": avg_hit / total_queries,
            "avg_recall_at_k": avg_recall / total_queries,
            "avg_mrr": avg_mrr / total_queries,
        }
        logger.info("Average Hit@%d      : %.3f", top_k, metrics["avg_hit_at_k"])
        logger.info("Average Recall@%d   : %.3f", top_k, metrics["avg_recall_at_k"])
        logger.info("Average MRR        : %.3f", metrics["avg_mrr"])
        return metrics
    def _initialize_graph(self) -> None:

        logger.info("Initializing Knowledge Graph...")

        self.graph_builder = KnowledgeGraphBuilder()

        self.graph_embeddings = GraphEmbeddingGenerator()

        self.graph_vector_store = GraphVectorStore()

        self.graph_retriever = GraphRetriever()
    def _build_graph_stage(self) -> None:

        logger.info("Building Knowledge Graph...")

        if self.graph_builder is None:
            raise RuntimeError("Graph Builder not initialized.")

        if self.graph_embeddings is None:
            raise RuntimeError("Graph Embeddings not initialized.")

        if self.graph_vector_store is None:
            raise RuntimeError("Graph Vector Store not initialized.")

        self.graph_builder.build(
            self.parsed_documents
        )

        self.graph_embeddings.build()

        self.graph_vector_store.create_index()
    def _graph_search(
        self,
        query: str,
        top_k: int = 5,
    ):

        if self.graph_retriever is None:
            raise RuntimeError(
                "Graph Retriever not initialized."
            )

        return self.graph_retriever.retrieve(
            query=query,
            top_k=top_k,
            max_hops=2,
        )
    @staticmethod
    def _sanitize_metadata(metadata: Any) -> Any:
        if isinstance(metadata, dict):
            sanitized: Dict[str, Any] = {}
            for key, value in metadata.items():
                if isinstance(value, dict):
                    sanitized[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
                elif isinstance(value, (list, tuple)):
                    sanitized[key] = [
                        RetrievalPipeline._sanitize_metadata(item)
                        if isinstance(item, dict)
                        else item
                        for item in value
                    ]
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    sanitized[key] = value
                else:
                    sanitized[key] = str(value)
            return sanitized

        if isinstance(metadata, (str, int, float, bool)) or metadata is None:
            return metadata

        return str(metadata)

    @staticmethod
    def _print_results(results: List[Dict[str, Any]], title: str) -> None:
        logger.info("")
        logger.info("%s", "=" * 80)
        logger.info(title)
        logger.info("%s", "=" * 80)

        for index, result in enumerate(results, start=1):
            logger.info("")
            logger.info("[Result %d]", index)
            logger.info("Hybrid Score   : %.6f", result.get("hybrid_score", 0.0))
            logger.info("BM25 Rank      : %s", result.get("bm25_rank"))
            logger.info("Semantic Rank  : %s", result.get("semantic_rank"))
            logger.info("Chunk ID       : %s", result.get("chunk_id"))
            logger.info("Contract Type  : %s", result.get("contract_type"))
            logger.info("Title          : %s", result.get("title"))
            logger.info("")
            logger.info("Excerpt:")
            logger.info("%s", result.get("text", "")[:350] + "...")


def build_default_config() -> PipelineConfig:
    repository_root = Path(__file__).resolve().parent
    return PipelineConfig(
        repository_root=repository_root,
        data_path=repository_root / "data" / "CUAD_v1.json",
        output_chunks_path=repository_root / "data" / "processed_chunks.json",
        rebuild_graph=True, 
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        pipeline = RetrievalPipeline(build_default_config())
        pipeline.run()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
    except Exception as exc:  # pragma: no cover - defensive boundary for CLI execution
        logger.exception("Pipeline execution failed: %s", exc)


if __name__ == "__main__":
    main()