from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.routes import testcase


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="RAG Test Case Generator API"
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


app.include_router(
    testcase.router,
    prefix="/api",
    tags=["Test Generation"]
)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "UP"
    }