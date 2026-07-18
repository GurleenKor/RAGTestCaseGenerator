import rag_pipeline
from rag_pipeline import (
    build_test_case_prompt,
    generate_test_cases,
    load_prompt_template,
    resolve_path,
    validate_ollama_model,
)


def test_resolve_path_supports_relative_paths():
    resolved = resolve_path("data/login_requirements.pdf")

    assert resolved.is_absolute()
    assert resolved.name == "login_requirements.pdf"


def test_build_test_case_prompt_includes_context_and_focus():
    context = "The sign-in page validates email and password."
    prompt = build_test_case_prompt(context, "Generate sign-in test cases")

    assert "The sign-in page validates email and password." in prompt
    assert "Generate sign-in test cases" in prompt
    assert "Senior SDET" in prompt


def test_build_test_case_prompt_requires_grounded_non_generic_output():
    context = "The sign-in page validates email and password."
    prompt = build_test_case_prompt(context, "Generate sign-in test cases")

    assert "Use ONLY the information provided" in prompt
    assert "Do not provide generic" in prompt
    assert "Grounding Evidence" in prompt


def test_validate_ollama_model_requires_value(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    try:
        validate_ollama_model(None)
    except ValueError as exc:
        assert "OLLAMA_MODEL" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing model name")


def test_generate_test_cases_returns_fallback_when_ollama_fails(monkeypatch):
    class DummyVectorStore:
        def similarity_search(self, query_text, k=5):
            return [type("Result", (), {"page_content": "Sign-in requirements"})()]

    class DummyClient:
        def generate(self, **kwargs):
            raise RuntimeError("Ollama connection failed")

    monkeypatch.setattr(rag_pipeline, "load_vector_store", lambda *args, **kwargs: DummyVectorStore())
    monkeypatch.setattr(rag_pipeline, "load_dotenv", lambda: None)

    response_text = generate_test_cases(
        "Generate sign-in test cases",
        model_name="llama3",
        client=DummyClient(),
    )

    assert "Unable to generate test cases" in response_text
    assert "Ollama connection failed" in response_text
