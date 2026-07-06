import os
import time
import json
from pathlib import Path

# Fix 1 & 2: Import the actual function names defined in your modules
from src.parser import parse_cuad
from src.chunker import chunk_contracts
from src.retriever import BM25Retriever

def main():
    """
    Executes Phase 1: Data Preparation and Lexical Retrieval pipeline.
    """
    # 1. Configuration
    # Convert DATA_PATH to a Path object, which parser.py requires
    DATA_PATH = Path(os.path.join("data", "CUAD_v1.json"))
    OUTPUT_CHUNKS_PATH = Path(os.path.join("data", "processed_chunks.json"))
    MAX_CONTRACTS = 200 
    
    print("Initializing Phase 1 Pipeline...")
    start_time = time.time()

    # 2. Ingestion & Parsing
    print(f"➔ Parsing the top {MAX_CONTRACTS} contracts from {DATA_PATH}...")
    try:
        # Fix 1: Use parse_cuad and map to its correct argument names
        parsed_documents = parse_cuad(input_path=DATA_PATH, n_contracts=MAX_CONTRACTS)
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {DATA_PATH}. Please ensure it is downloaded.")
        return

    # 3. Preprocessing & Chunking
    print("➔ Applying text cleaning, custom stop-word removal, and paragraph chunking...")
    
    # Fix 2: parser.py outputs 'raw_title', but chunker.py expects 'title'. 
    # We bridge that gap here before passing the data forward.
    for doc in parsed_documents:
        doc["title"] = doc.get("raw_title", "Untitled Document")
        
    chunked_corpus = chunk_contracts(parsed_documents)
    print(f"   Generated {len(chunked_corpus)} total document chunks.")

    # NEW: Save the chunks for Phase 2 and Phase 3
    print(f"➔ Saving chunks to {OUTPUT_CHUNKS_PATH}...")
    OUTPUT_CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunked_corpus, f, indent=2)

    # 4. Lexical Retrieval Engine Initialization
    print("➔ Building the exact-match retrieval engine index...")
    
    # Fix 3: Extract just the 'text' fields, as BM25Retriever expects a List[str]
    corpus_texts = [chunk["text"] for chunk in chunked_corpus]
    retriever = BM25Retriever(corpus_texts)

    # 5. Pipeline Testing
    test_query = "force majeure or act of god"
    print(f"\n➔ Executing Test Query: '{test_query}'")
    
    # Retriever returns a List[str] of the best chunks
    results = retriever.search(query=test_query, top_k=5)

    print("\n--- Top 5 Lexical Retrieval Results ---")
    for idx, result_text in enumerate(results, 1):
        print(f"\n[Result {idx}]")
        # Since chunker.py automatically prepends "Title: [Name]" to the text,
        # we can just print the string excerpt directly without needing a separate metadata field.
        print(f"Excerpt: {result_text[:250]}...")

    elapsed_time = time.time() - start_time
    print(f"\nPipeline execution completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()