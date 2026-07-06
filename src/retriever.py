import re
from typing import List
from rank_bm25 import BM25Okapi

class BM25Retriever:
    """
    A Lexical Retrieval Engine utilizing the BM25 algorithm.
    
    This class ingests a corpus of preprocessed text chunks, tokenizes them, 
    and builds an inverted index using BM25Okapi. It evaluates search queries 
    against the index to return the most lexically relevant chunks.
    """

    def __init__(self, corpus: List[str]):
        """
        Initializes the retriever, tokenizes the corpus, and builds the BM25 index.
        """
        if not corpus:
            raise ValueError("Corpus cannot be empty.")
            
        self.corpus = corpus
        
        # Tokenize the entire corpus to build the index
        tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        
        # Initialize the BM25 model
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text, preserving hyphens and periods inside words.
        Ensures exact-match viability for IDs like "Section 10-Q" or "409.A".
        """
        text = text.lower()
        # Updated Regex: preserves alphanumeric, hyphens, and internal periods 
        tokens = re.findall(r'\b[\w\.\-]+\b', text)
        return tokens

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """
        Evaluates a user string against the BM25 index and returns the highest-scoring chunks.
        """
        if not query.strip():
            return []

        # The query must be tokenized using the exact same logic as the corpus
        tokenized_query = self._tokenize(query)
        
        # Fetch the top N documents directly using rank_bm25's built-in method
        top_chunks = self.bm25.get_top_n(tokenized_query, self.corpus, n=top_k)
        
        return top_chunks

# ==========================================
# Example Usage / Testing Block
# ==========================================
if __name__ == "__main__":
    sample_corpus = [
        "Vector search engines capture semantic meaning using embeddings.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document.",
        "TF-IDF was the predecessor to BM25, but BM25 handles term frequency saturation better.",
        "Lexical search relies on exact keyword matching between the query and the document index.",
        "Hybrid search combines lexical retrieval like BM25 with semantic vector search to improve overall recall.",
        "Machine learning models require clean, preprocessed data to function optimally."
    ]

    retriever = BM25Retriever(sample_corpus)
    user_query = "How does BM25 compare to TF-IDF and handle term frequency?"
    print(f"Query: '{user_query}'\n")
    
    results = retriever.search(user_query, top_k=3)
    print("Top Results:")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res}")