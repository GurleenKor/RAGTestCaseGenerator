import os

from dotenv import load_dotenv
from ollama import Client

load_dotenv()

client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
model_name = os.getenv("OLLAMA_MODEL", "llama3")

try:
    response = client.generate(
        model=model_name,
        prompt="Generate 5 login test cases for an ecommerce application.",
    )
    print(response.get("response", ""))
except Exception as exc:
    print(f"Ollama is unavailable: {exc}")