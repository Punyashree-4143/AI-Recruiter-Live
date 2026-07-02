from functools import lru_cache

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.skill_matcher import normalize_text
from pathlib import Path


MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _embedding_model():
    return SentenceTransformer(MODEL_NAME)


def _tokenize(value):
    return normalize_text(value).split()


def semantic_search(query, top_k=20):
    model = _embedding_model()
    query_embedding = model.encode(query)

    VECTOR_DB_PATH = (
        Path(__file__).resolve().parent.parent.parent
        / "vector_dbdemo"
    )

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_PATH)
    )
    collection = client.get_collection(
        name="candidates"
    )
    result_count = min(
        max(int(top_k), 1),
        collection.count(),
    )

    if result_count <= 0:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    return collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=result_count,
    )


def bm25_search(query, documents, top_k=20):
    if not documents:
        return []

    tokenized_docs = [
        _tokenize(document)
        for document in documents
    ]
    tokenized_query = _tokenize(query)

    if not tokenized_query:
        return []

    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)

    return sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:top_k]
