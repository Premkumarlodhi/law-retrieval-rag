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
from langchain_ollama import ChatOllama
import time
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MILLISECONDS_PER_SECOND = 1000.0

OLLAMA_MODEL_NAME = "qwen3:8b"
OLLAMA_TEMPERATURE = 0

PROMPT_SECTION_BAR = "=" * 52

SYSTEM_PROMPT_INSTRUCTIONS = (
    "You are an expert legal assistant.\n\n"
    "Answer ONLY using the supplied retrieved context.\n\n"
    "Never fabricate information.\n\n"
    "If the answer is not present in the context, say:\n"
    "\"The provided documents do not contain enough information.\"\n\n"
    "Quote relevant clauses whenever possible."
)


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
        self.llm: Optional[ChatOllama] = None
    def _time_stage(
        self,
        stage_name: str,
        func,
        *args,
        **kwargs,
    ):
        start = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start

        logger.info(
            "%-30s : %.3f s",
            stage_name,
            elapsed,
        )

        return result
    
    def run(self) -> Dict[str, Any]:
        pipeline_start = time.perf_counter()
        logger.info("")
        logger.info("=" * 70)
        logger.info("PIPELINE TIMINGS")
        logger.info("=" * 70)

        parsed_documents = self._time_stage(
            "Dataset Parsing",
            self._parse_stage,
        )

        prepared_documents = self._time_stage(
            "Document Preparation",
            self._prepare_stage,
            parsed_documents,
        )

        chunked_corpus = self._time_stage(
            "Chunk Generation",
            self._chunk_stage,
            prepared_documents,
        )

        self._time_stage(
            "Chunk Persistence",
            self._persist_chunks_stage,
            chunked_corpus,
        )

        (
            self.bm25_retriever,
            self.semantic_retriever,
            self.hybrid_retriever,
            self.cross_encoder,
        ) = self._time_stage(
            "Retriever Initialization",
            self._build_retrieval_stage,
            chunked_corpus,
        )

        self._time_stage(
            "Graph Initialization",
            self._initialize_graph,
        )

        self._time_stage(
            "LLM Initialization",
            self._initialize_llm,
        )

        if self.config.rebuild_graph:

            self._time_stage(
                "Knowledge Graph Build",
                self._build_graph_stage,
            )

        query_timings = self._time_stage(
            "Demo Query",
            self._demo_query_stage,
            self.config.demo_query,
            self.config.top_k,
        )

        evaluation_metrics = self._time_stage(
            "Evaluation",
            self._evaluate_stage,
            self.config.eval_top_k,
        )

        if query_timings:
            evaluation_metrics["query_latency_ms"] = query_timings["end_to_end_query_ms"]

        total_pipeline_seconds = time.perf_counter() - pipeline_start
        self._print_query_latency_summary(query_timings, total_pipeline_seconds)

        return {
            "documents": self.parsed_documents,
            "chunks": self.chunked_corpus,
            "evaluation": evaluation_metrics,
            "elapsed_seconds": total_pipeline_seconds,
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

    def _demo_query_stage(self, query: str, top_k: int) -> Dict[str, float]:
        logger.info("Executing demo query: %s", query)
        if self.hybrid_retriever is None:
            raise RuntimeError("Hybrid retriever has not been initialized.")

        query_start = time.perf_counter()

        start = time.perf_counter()

        hybrid_results = self.hybrid_retriever.search(
            query=query,
            top_k=max(top_k * 4, 20),
        )

        hybrid_retrieval_ms = (time.perf_counter() - start) * MILLISECONDS_PER_SECOND

        start = time.perf_counter()

        results = self.cross_encoder.rerank(
            query=query,
            candidates=hybrid_results,
            top_k=top_k,
            debug=True,
        )

        cross_encoder_ms = (time.perf_counter() - start) * MILLISECONDS_PER_SECOND

        start = time.perf_counter()

        graph_context = self._graph_search(
            query=query,
            top_k=5,
        )

        graph_retrieval_ms = (time.perf_counter() - start) * MILLISECONDS_PER_SECOND

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

        start = time.perf_counter()
        prompt = self._build_prompt(query, results, graph_context)
        prompt_construction_ms = (time.perf_counter() - start) * MILLISECONDS_PER_SECOND

        start = time.perf_counter()
        self._invoke_llm(prompt)
        llm_generation_ms = (time.perf_counter() - start) * MILLISECONDS_PER_SECOND

        end_to_end_query_ms = (time.perf_counter() - query_start) * MILLISECONDS_PER_SECOND

        return {
            "hybrid_retrieval_ms": hybrid_retrieval_ms,
            "cross_encoder_ms": cross_encoder_ms,
            "graph_retrieval_ms": graph_retrieval_ms,
            "prompt_construction_ms": prompt_construction_ms,
            "llm_generation_ms": llm_generation_ms,
            "end_to_end_query_ms": end_to_end_query_ms,
        }

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

    def _initialize_llm(self) -> None:
        """Instantiate the local Ollama chat model exactly once for reuse across queries."""
        logger.info("Initializing local LLM (Ollama)...")

        self.llm = ChatOllama(
            model=OLLAMA_MODEL_NAME,
            temperature=OLLAMA_TEMPERATURE,
        )

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
    def _build_prompt(
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        graph_context: List[Dict[str, Any]],
    ) -> str:
        """Construct the final LLM prompt from retrieval and graph evidence.

        Parameters
        ----------
        query : str
            The user's natural language question.
        retrieved_documents : list of dict
            Reranked hybrid retrieval results, each with ``title``,
            ``contract_type``, and ``text`` keys.
        graph_context : list of dict
            Graph retrieval nodes, each with ``label`` and ``name`` keys.

        Returns
        -------
        str
            The complete prompt to send to the LLM.
        """
        document_blocks = []
        for index, document in enumerate(retrieved_documents, start=1):
            document_blocks.append(
                f"[Document {index}]\n"
                f"Title          : {document.get('title')}\n"
                f"Contract Type  : {document.get('contract_type')}\n"
                f"Text           : {document.get('text', '')}"
            )
        documents_section = (
            "\n\n".join(document_blocks)
            if document_blocks
            else "No documents retrieved."
        )

        graph_blocks = [
            f"{node.get('label')}: {node.get('name')}" for node in graph_context
        ]
        graph_section = "\n".join(graph_blocks) if graph_blocks else "No graph context retrieved."

        return (
            f"{SYSTEM_PROMPT_INSTRUCTIONS}\n\n"
            f"=== RETRIEVED DOCUMENTS ===\n{documents_section}\n\n"
            f"=== GRAPH CONTEXT ===\n{graph_section}\n\n"
            f"=== QUESTION ===\n{query}\n\n"
            f"=== FINAL ANSWER ===\n"
        )

    def _invoke_llm(self, prompt: str) -> str:
        """Send a constructed prompt to the local Ollama model and return its answer.

        Prints the full prompt and the model's answer to the console for
        debugging, as required.

        Parameters
        ----------
        prompt : str
            The complete prompt produced by ``_build_prompt``.

        Returns
        -------
        str
            The LLM's answer text.
        """
        if self.llm is None:
            raise RuntimeError("LLM has not been initialized.")

        logger.info("")
        logger.info(PROMPT_SECTION_BAR)
        logger.info("LLM PROMPT")
        logger.info(PROMPT_SECTION_BAR)
        logger.info("")
        logger.info("%s", prompt)

        response = self.llm.invoke(prompt)
        answer = response.content

        logger.info("")
        logger.info(PROMPT_SECTION_BAR)
        logger.info("LLM ANSWER")
        logger.info(PROMPT_SECTION_BAR)
        logger.info("")
        logger.info("%s", answer)

        return answer

    def _llm_stage(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        graph_context: List[Dict[str, Any]],
    ) -> str:
        """Build the prompt, invoke the local LLM, and return its answer.

        This is the single orchestration entry point for LLM answer
        generation: it composes ``_build_prompt`` and ``_invoke_llm`` and
        contains no retrieval logic of its own.

        Parameters
        ----------
        query : str
            The user's natural language question.
        retrieved_docs : list of dict
            Reranked hybrid retrieval results.
        graph_context : list of dict
            Graph retrieval nodes.

        Returns
        -------
        str
            The LLM's answer text.
        """
        prompt = self._build_prompt(query, retrieved_docs, graph_context)
        return self._invoke_llm(prompt)

    def _print_query_latency_summary(
        self,
        query_timings: Optional[Dict[str, float]],
        total_pipeline_seconds: float,
    ) -> None:
        """Print the final query-latency breakdown and total pipeline runtime.

        Parameters
        ----------
        query_timings : dict or None
            Per-stage millisecond timings produced by ``_demo_query_stage``.
        total_pipeline_seconds : float
            Total wall-clock runtime of the full pipeline, in seconds.
        """
        logger.info("")
        logger.info(PROMPT_SECTION_BAR)
        logger.info("QUERY LATENCY SUMMARY")
        logger.info(PROMPT_SECTION_BAR)

        if not query_timings:
            logger.info("No demo query timings were recorded.")
        else:
            logger.info("Hybrid Retrieval      : %.2f ms", query_timings["hybrid_retrieval_ms"])
            logger.info("Cross Encoder         : %.2f ms", query_timings["cross_encoder_ms"])
            logger.info("Graph Retrieval       : %.2f ms", query_timings["graph_retrieval_ms"])
            logger.info("Prompt Construction   : %.2f ms", query_timings["prompt_construction_ms"])
            logger.info("LLM Generation        : %.2f ms", query_timings["llm_generation_ms"])
            logger.info("End-to-End Query      : %.2f ms", query_timings["end_to_end_query_ms"])

        logger.info("Total Pipeline        : %.2f s", total_pipeline_seconds)

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
        rebuild_graph=False, 
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
