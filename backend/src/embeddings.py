from pathlib import Path

import chromadb


VECTOR_DB_PATH = (
    Path(__file__).resolve().parent.parent
    / "vector_dbdemo"
)


def create_vector_store(
    documents,
    metadata
):

    print("\nConnecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_PATH)
    )

    # Delete old collection
    try:

        client.delete_collection(
            name="candidates"
        )

        print(
            "Existing collection deleted."
        )

    except Exception:
        pass

    collection = (
        client.get_or_create_collection(
            name="candidates"
        )
    )

    total_docs = len(documents)

    print(
        f"\nStarting indexing of "
        f"{total_docs} candidates..."
    )

    BATCH_SIZE = 100

    for start_idx in range(
        0,
        total_docs,
        BATCH_SIZE
    ):

        end_idx = min(
            start_idx + BATCH_SIZE,
            total_docs
        )

        batch_documents = (
            documents[start_idx:end_idx]
        )

        batch_metadata = (
            metadata[start_idx:end_idx]
        )

        print(
            f"\nProcessing batch "
            f"{start_idx // BATCH_SIZE + 1}"
            f" | Records "
            f"{start_idx} - {end_idx}"
        )

        ids = [
            item["candidate_id"]
            for item in batch_metadata
        ]

        collection.add(
            ids=ids,
            documents=batch_documents,
            metadatas=batch_metadata
        )

        print(
            f"Stored "
            f"{len(batch_documents)} "
            f"candidates"
        )

    print(
        f"\nSuccessfully indexed "
        f"{total_docs} candidates"
    )

    return collection


def load_collection():

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_PATH)
    )

    collection = client.get_collection(
        name="candidates"
    )

    return collection