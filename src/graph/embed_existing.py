"""
embed_existing_graph.py

Embeds every node already present in Neo4j.

Usage:
    python embed_existing_graph.py
"""

import os
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD"),
    ),
)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def build_text(props, labels):

    parts = []

    if labels:
        parts.append(f"Label: {labels[0]}")

    priority = [
        "name",
        "title",
        "contract_type",
        "type",
        "date",
        "section",
        "description",
        "text",
    ]

    for key in priority:

        if key not in props:
            continue

        value = props[key]

        if value is None:
            continue

        if key == "text":
            value = str(value)[:3000]

        parts.append(f"{key}: {value}")

    # include remaining properties
    for key, value in props.items():

        if key in priority:
            continue

        if key == "embedding":
            continue

        parts.append(f"{key}: {value}")

    return "\n".join(parts)


with driver.session() as session:

    result = session.run("""
        MATCH (n)
        RETURN
            elementId(n) AS id,
            labels(n) AS labels,
            properties(n) AS props
    """)

    nodes = list(result)

    print(f"Found {len(nodes)} nodes")

    for i, record in enumerate(nodes, start=1):

        node_id = record["id"]
        labels = record["labels"]
        props = record["props"]

        text = build_text(
            props,
            labels,
        )

        embedding = model.encode(
            text,
            normalize_embeddings=True,
        ).tolist()

        session.run(
            """
            MATCH (n)
            WHERE elementId(n)=$id
            SET n.embedding=$embedding
            """,
            id=node_id,
            embedding=embedding,
        )

        if i % 25 == 0:

            print(
                f"Embedded {i}/{len(nodes)}"
            )

print("Finished embedding graph.")
driver.close()