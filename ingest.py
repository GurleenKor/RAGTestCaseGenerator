from rag_pipeline import build_vector_store, load_requirements_text, split_requirements_text


def main() -> None:
    text = load_requirements_text("data/login_requirements.pdf")
    chunks = split_requirements_text(text)
    build_vector_store(chunks, persist_directory="chroma_db")
    print("Embeddings stored successfully!")


if __name__ == "__main__":
    main()
