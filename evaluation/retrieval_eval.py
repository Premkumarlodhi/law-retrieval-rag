from typing import List


def hit_rate(
    retrieved_docs: List[str],
    expected_terms: List[str],
):

    for doc in retrieved_docs:

        lower = doc.lower()

        for term in expected_terms:

            if term.lower() in lower:
                return 1

    return 0


def recall_at_k(
    retrieved_docs: List[str],
    expected_terms: List[str],
):

    found = 0

    for term in expected_terms:

        for doc in retrieved_docs:

            if term.lower() in doc.lower():

                found += 1
                break

    return found / len(expected_terms)


def reciprocal_rank(
    retrieved_docs: List[str],
    expected_terms: List[str],
):

    for idx, doc in enumerate(retrieved_docs):

        lower = doc.lower()

        for term in expected_terms:

            if term.lower() in lower:
                return 1 / (idx + 1)

    return 0