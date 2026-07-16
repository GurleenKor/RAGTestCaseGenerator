import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import google.generativeai as genai


# Load API Key
load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)


# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load vector database
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)


# User question
query = """
Generate test cases for B2C Sign In functionality.

Include:
- positive scenarios
- negative scenarios
- validation rules
- security scenarios
"""


# Retrieve relevant requirements
results = vector_db.similarity_search(
    query,
    k=5
)


# Combine retrieved chunks
context = ""

for result in results:
    context += result.page_content
    context += "\n\n"


# Create QA prompt

prompt = f"""
You are a Senior SDET.

Analyze the following software requirements:

-------------------
{context}
-------------------

Generate detailed test cases.

For each test case include:

1. Test Case ID
2. Scenario
3. Preconditions
4. Test Steps
5. Expected Result
6. Test Type (Positive/Negative/Security/Validation)

Focus on B2C Sign In functionality.
"""


# Call Gemini

model = genai.GenerativeModel(
    "gemini-2.0-flash"
)


response = model.generate_content(prompt)


print("\n========== GENERATED TEST CASES ==========\n")

print(response.text)