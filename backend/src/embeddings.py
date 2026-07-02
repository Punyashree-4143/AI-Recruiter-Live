import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000


def create_vector_store(
    documents,
    metadata
):
    """
    Create ChromaDB vector store using
    batch processing.
    """

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Connecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path="../vector_dbdemo"
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

        embeddings = model.encode(
            batch_documents,
            show_progress_bar=False
        )

        ids = [
            item["candidate_id"]
            for item in batch_metadata
        ]

        collection.add(
            ids=ids,
            documents=batch_documents,
            metadatas=batch_metadata,
            embeddings=embeddings.tolist()
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
    """
    Load existing ChromaDB collection.
    """

    client = chromadb.PersistentClient(
        path="../vector_db_demo"
    )


    collection = client.get_collection(
        name="candidates"
    )

    return collection