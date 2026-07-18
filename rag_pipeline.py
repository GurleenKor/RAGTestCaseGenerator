from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from prompt import load_prompt_template

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF_PATH = BASE_DIR / "data" / "login_requirements.pdf"
DEFAULT_DB_PATH = BASE_DIR / "chroma_db"


def resolve_path(path_value: str | os.PathLike[str] | None, base_dir: Path | None = None) -> Path:
    base = (base_dir or BASE_DIR).resolve()
    if path_value is None:
        return DEFAULT_PDF_PATH.resolve()

    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (base / candidate).resolve()


def load_requirements_text(pdf_path: str | os.PathLike[str] | None = None) -> str:
    path = resolve_path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Requirements file not found: {path}")

    reader = PdfReader(str(path))
    text_parts = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(part for part in text_parts if part).strip()

    if not text:
        raise ValueError(f"No readable text found in PDF: {path}")

    return text


def split_requirements_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    if not text.strip():
        raise ValueError("No requirements text available to split")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def build_vector_store(
    text_chunks: list[str],
    persist_directory: str | os.PathLike[str] | None = None,
) -> Chroma:
    if not text_chunks:
        raise ValueError("No text chunks available to build the vector store")

    persist_dir = resolve_path(persist_directory, BASE_DIR) if persist_directory is not None else DEFAULT_DB_PATH
    persist_dir.mkdir(parents=True, exist_ok=True)
    

    embeddings = get_embeddings()
    return Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )

def get_embeddings():
    load_dotenv()

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_vector_store(persist_directory=None):
    persist_dir = resolve_path(persist_directory, BASE_DIR) if persist_directory is not None else DEFAULT_DB_PATH

    embeddings = get_embeddings()

    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings
    )

def validate_ollama_model(model_name: Optional[str]) -> str:
    if not model_name or not str(model_name).strip():
        raise ValueError("OLLAMA_MODEL is required to generate test cases")
    return str(model_name).strip()


def build_test_case_prompt(context: str, focus_area: str = "B2C Sign In functionality") -> str:
    if not context.strip():
        raise ValueError("No retrieval context available for prompt generation")

    template = load_prompt_template("test_case_generation")
    return template.format(context=context, focus_area=focus_area)


def build_api_error_message(error: Exception) -> str:
    return (
        "Unable to generate test cases because the Ollama request failed.\n"
        f"Details: {error}\n"
        "Please verify that Ollama is running and the selected model is available."
    )


def generate_test_cases(
    query_text: str,
    persist_directory: str | os.PathLike[str] | None = None,
    model_name: Optional[str] = None,
    client: object | None = None,
) -> str:
    load_dotenv()

    resolved_model = model_name or os.getenv("OLLAMA_MODEL")
    if not resolved_model:
        try:
            resolved_model = validate_ollama_model(resolved_model)
        except ValueError as exc:
            return build_api_error_message(exc)

    if client is None:
        from ollama import Client

        client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

    vector_db = load_vector_store(persist_directory)
    results = vector_db.similarity_search(query_text, k=5)

    context = "\n\n".join(
        result.page_content for result in results if getattr(result, "page_content", "").strip()
    )

    if not context:
        raise ValueError("No relevant requirements were found in the vector database")

    prompt = build_test_case_prompt(context, focus_area=query_text)
    try:
        response = client.generate(model=resolved_model, prompt=prompt)
        if isinstance(response, dict):
            return response.get("response", "")
        return str(response)
    except Exception as exc:  # pragma: no cover - exercised via runtime integration
        return build_api_error_message(exc)
