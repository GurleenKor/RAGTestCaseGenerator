from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

pdf_path = "data/login_requirements.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    text += page.extract_text()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
text_chunks = text_splitter.split_text(text)

# for i, chunk in enumerate(text_chunks):
#     print(f"Chunk {i+1}")
#     print(chunk)
#     print("=" * 50)

# create embeddings once for all chunks
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = Chroma.from_texts(
    texts=text_chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Embeddings stored successfully!")