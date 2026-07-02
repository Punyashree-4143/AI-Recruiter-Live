from src.ingestion import load_and_prepare_data
from src.embeddings import create_vector_store


def main():

    print("Loading demo candidates...")

    documents, metadata = load_and_prepare_data(
        "../data/candidates_demo.jsonl"
    )

    print(
        f"Loaded {len(documents)} demo candidates"
    )

    create_vector_store(
        documents,
        metadata
    )

    print(
        "\nDemo vector database created successfully."
    )


if __name__ == "__main__":
    main()