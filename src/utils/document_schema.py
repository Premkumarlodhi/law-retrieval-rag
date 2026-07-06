# src/utils/document_schema.py

import uuid


def create_document(
    text,
    metadata,
    chunk_index,
    section="contract_body",
):

    doc_id = metadata.get(
        "raw_title",
        str(uuid.uuid4())
    )

    return {
        "doc_id":
            doc_id,

        "chunk_id":
            f"{doc_id}_{chunk_index}",

        "title":
            metadata.get("title", ""),

        "raw_title":
            metadata.get("raw_title", ""),

        "contract_type":
            metadata.get("contract_type", ""),

        "company_a":
            metadata.get("party_1", ""),

        "company_b":
            metadata.get("party_2", ""),

        "contract_date":
            metadata.get("date", ""),

        "section":
            section,

        "text":
            text,

        "metadata":
            metadata,
    }