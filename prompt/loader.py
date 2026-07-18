from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def load_prompt_template(name: str) -> str:
    prompt_path = BASE_DIR / f"{name}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")
