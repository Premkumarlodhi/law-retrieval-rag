import json
from pathlib import Path
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class SemanticRetriever:
    """
    Semantic Retrieval Engine using:

    • HuggingFace MiniLM Embeddings
    • ChromaDB

    Returns the same interface as BM25Retriever for
    easy Hybrid Retrieval integration.
    """

    def __init__(
        self,
        persist_directory: str = "vector_store/chroma_db",
        collection_name: str = "legal_contracts",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = None

    # ==========================================================
    # Build Vector Index
    # ==========================================================

    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        # Validate required document fields before constructing Chroma documents.
        docs: List[Document] = []

        for doc in documents:
            required = ("doc_id", "chunk_id", "title", "text")
            missing = [
                field for field in required if field not in doc or doc.get(field) is None
            ]
            if missing:
                raise ValueError(f"Missing required fields {missing} in document: {doc}")

            metadata = {
                "doc_id": doc["doc_id"],
                "chunk_id": doc["chunk_id"],
                "title": doc["title"],
                "raw_title": doc.get("raw_title", ""),
                "contract_type": doc.get("contract_type", ""),
                "section": doc.get("section", ""),
                "metadata": self._sanitize_metadata(doc.get("metadata", {})),
            }

            docs.append(Document(page_content=doc["text"], metadata=metadata))

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )

        print(f"Indexed {len(docs)} chunks into '{self.collection_name}'.")

    @staticmethod
    def _sanitize_metadata(value: Any) -> Any:
        # Chroma requires metadata values to be primitive or list-like scalars.
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, (list, tuple)):
            return [
                SemanticRetriever._sanitize_metadata(item) if isinstance(item, dict) else item
                for item in value
            ]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    # ==========================================================
    # Load Existing Index
    # ==========================================================

    def load_existing(self) -> None:
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding,
            collection_name=self.collection_name,
        )

    # ==========================================================
    # Initialize Vector Store
    # ==========================================================

    def initialize(self, documents: List[Dict[str, Any]], force_rebuild: bool = False) -> None:
        chroma_file = Path(self.persist_directory) / "chroma.sqlite3"

        if force_rebuild:
            print("Rebuilding ChromaDB index...")
            self.clear_index()
            self.build_index(documents)
            return

        if chroma_file.exists():
            print("Loading existing ChromaDB...")
            self.load_existing()

            collection = getattr(self.vectorstore, "_collection", None)
            if collection is None:
                raise RuntimeError("Chroma collection could not be loaded.")

            stored_docs = collection.count()
            print(f"Loaded {stored_docs} embedded chunks.")

            if stored_docs != len(documents):
                print("Existing index does not match current corpus.")
                print("Rebuilding ChromaDB...")
                self.clear_index()
                self.build_index(documents)
            return

        print("Creating new ChromaDB...")
        self.build_index(documents)

    # ==========================================================
    # Semantic Search
    # ==========================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        if self.vectorstore is None:
            raise RuntimeError(
                "SemanticRetriever has not been initialized. "
                "Call initialize() before performing a search."
            )

        collection = getattr(self.vectorstore, "_collection", None)
        if collection is None:
            raise RuntimeError("Chroma collection is unavailable.")

        if collection.count() == 0:
            raise RuntimeError("Chroma index is empty. Build or load the index before searching.")

        if not query.strip():
            return []

        if top_k <= 0:
            return []

        results = self.vectorstore.similarity_search_with_score(query=query, k=top_k)

        output: List[Dict[str, Any]] = []
        for rank, (doc, distance) in enumerate(results, start=1):
            distance = float(distance)
            similarity = 1.0 / (1.0 + distance)

            output.append(
                {
                    "doc_id": doc.metadata.get("doc_id"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "title": doc.metadata.get("title"),
                    "raw_title": doc.metadata.get("raw_title", ""),
                    "contract_type": doc.metadata.get("contract_type", ""),
                    "section": doc.metadata.get("section", ""),
                    "distance": distance,
                    "similarity": similarity,
                    "semantic_rank": rank,
                    "text": doc.page_content,
                    "metadata": doc.metadata.get("metadata", {}),
                }
            )

        if debug:
            print("\n" + "=" * 80)
            print("Semantic Retrieval Results")
            print("=" * 80)
            for result in output:
                print(f"Semantic Rank : {result['semantic_rank']}")
                print(f"Similarity    : {result['similarity']:.4f}")
                print(f"Distance      : {result['distance']:.4f}")
                print(f"Doc ID        : {result['doc_id']}")
                print(f"Chunk ID      : {result['chunk_id']}")
                print(f"Title         : {result['title']}")
                print("-" * 80)

        return output

    # ==========================================================
    # Index Information
    # ==========================================================

    def index_size(self) -> int:
        """
        Returns the number of indexed chunks stored in ChromaDB.
        """
        if self.vectorstore is None:
            return 0

        collection = getattr(self.vectorstore, "_collection", None)
        if collection is None:
            return 0

        return collection.count()

    # ==========================================================
    # Clear Existing Index
    # ==========================================================

    def clear_index(self) -> None:
        # Clear the persisted Chroma directory before rebuilding the index.
        import shutil

        path = Path(self.persist_directory)
        if path.exists():
            shutil.rmtree(path)
            print("Deleted existing ChromaDB index.")

    def delete_index(self) -> None:
        # Backward-compatible alias for callers that still use the old API name.
        self.clear_index()