import os

from rag_pipeline import generate_test_cases


def main() -> None:
    query = """
Generate test cases for B2C Sign In functionality.

Include:
- positive scenarios
- negative scenarios
- validation rules
- security scenarios
"""

    response_text = generate_test_cases(
        query,
        persist_directory="chroma_db",
        model_name=os.getenv("OLLAMA_MODEL", "llama3"),
    )
    print("\n========== GENERATED TEST CASES ==========\n")
    print(response_text)


if __name__ == "__main__":
    main()
