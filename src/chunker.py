# 2_chunker.py
"""
Contract Chunking Pipeline

Input:
    List[Dict], where each dictionary is expected to look similar to:

    {
        "title": "Master Service Agreement",
        "text": "Full contract text..."
    }

Output:
    List[Dict]

    [
        {
            "title": "...",
            "raw_title": "...",
            "contract_type": "...",
            "chunk_id": "...",
            "text": "Title: Master Service Agreement\n\n<chunk>"
        },
        ...
    ]

Features
--------
1. Custom text cleaning
2. Custom stop-word removal (NO NLTK stopwords)
3. Preserves legally important modifiers:
       not, except, unless, shall, if, etc.
4. RecursiveCharacterTextSplitter
       chunk_size = 800
       chunk_overlap = 100
5. Prepends document title to every chunk
"""

import re
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter


# =============================================================================
# Custom Stop Words
# =============================================================================

CUSTOM_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "is", "are", "was",
    "were", "be", "been", "being", "am", "to", "of", "and", "or", "as", "at",
    "by", "for", "from", "in", "into", "on", "onto", "with", "about", "over",
    "under", "up", "down", "through", "during", "after", "before", "between",
    "among", "it", "its", "they", "them", "their", "he", "she", "his", "her",
    "you", "your", "yours", "we", "our", "ours", "i", "me", "my", "mine",
    "can", "could", "would", "should", "very", "also", "just", "than", "then",
    "there", "here", "such", "some", "any", "each", "every",
}


# =============================================================================
# Legal Keywords To Preserve
# =============================================================================

LEGAL_PRESERVE = {
    "not", "no", "nor", "never", "except", "unless", "shall", "must", "may",
    "will", "if", "provided", "provided that", "subject", "subject to",
    "only", "without", "including", "excluding", "whether", "where",
    "whereas", "upon",
}


# =============================================================================
# Text Cleaning
# =============================================================================

def clean_text(text: str) -> str:
    """
    Cleans contract text while preserving legal meaning.
    """
    if not text:
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    # Remove non-printable characters
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Tokenization
    tokens = re.findall(r"\b[\w'-]+\b", text)

    cleaned = []
    for token in tokens:
        lower = token.lower()
        if lower in LEGAL_PRESERVE:
            cleaned.append(token)
            continue
        if lower in CUSTOM_STOPWORDS:
            continue
        cleaned.append(token)

    return " ".join(cleaned)


# =============================================================================
# LangChain Text Splitter
# =============================================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
)


# =============================================================================
# Chunk Contracts
# =============================================================================

from src.utils.document_schema import create_document


def chunk_contracts(contract_list: List[Dict]) -> List[Dict]:
    """
    Parameters
    ----------
    contract_list : List[Dict]

    Returns
    -------
    List[Dict]
    """

    all_chunks = []

    for document in contract_list:

        title = document.get(
            "title",
            "Untitled Document"
        )

        text = document.get(
            "text",
            ""
        )

        contract_type = document.get(
            "contract_type",
            "Unknown"
        )

        raw_title = document.get(
            "raw_title",
            title
        )

        cleaned_text = clean_text(text)

        chunks = splitter.split_text(
            cleaned_text
        )

        metadata = {
            "title": title,
            "raw_title": raw_title,
            "contract_type": contract_type,

            # placeholders for future KG extraction
            "party_1":
                document.get(
                    "party_1",
                    ""
                ),

            "party_2":
                document.get(
                    "party_2",
                    ""
                ),

            "date":
                document.get(
                    "date",
                    ""
                ),
        }

        for idx, chunk in enumerate(chunks):

            final_chunk = (
                f"Title: {title}\n\n"
                f"{chunk}"
            )

            chunk_document = create_document(
                text=final_chunk,
                metadata=metadata,
                chunk_index=idx,
                section="contract_body",
            )

            all_chunks.append(
                chunk_document
            )

    return all_chunks


# =============================================================================
# Example
# =============================================================================

if __name__ == "__main__":
    contracts = [
        {
            "title": "Employment Agreement",
            "text": """
            This Agreement is made between the Company and the Employee.
            The Employee shall not disclose confidential information
            unless required by law.
            The Company may terminate employment if the Employee
            breaches any confidentiality obligation.
            Except where otherwise required, all notices shall be
            delivered in writing.
            """,
        }
    ]

    chunks = chunk_contracts(contracts)
    for chunk in chunks:
        print("=" * 80)
        print(chunk["text"])