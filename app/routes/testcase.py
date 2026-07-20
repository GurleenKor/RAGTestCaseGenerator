import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from rag_pipeline import (
    UPLOAD_DB_DIR,
    build_vector_store,
    clear_previous_upload_dbs,
    extract_requirement_text,
    generate_test_cases,
    split_requirements_text,
)


router = APIRouter()


@router.post("/generate-testcases")
def generate(
    requirement: str | None = Form(default=None),
    model_name: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
):
    try:
        if file is not None:
            requirement_text = extract_requirement_text(file.file.read(), file.filename)
            chunks = split_requirements_text(requirement_text)

            UPLOAD_DB_DIR.mkdir(parents=True, exist_ok=True)
            clear_previous_upload_dbs(UPLOAD_DB_DIR)
            upload_dir = UPLOAD_DB_DIR / Path(file.filename or "upload").stem
            if upload_dir.exists():
                shutil.rmtree(upload_dir)
            upload_dir.mkdir(parents=True, exist_ok=True)

            build_vector_store(chunks, persist_directory=upload_dir)
            result = generate_test_cases(
                query_text=requirement_text,
                model_name=model_name,
                persist_directory=upload_dir,
                document_text=requirement_text,
            )
        elif requirement:
            requirement_text = requirement
            result = generate_test_cases(
                query_text=requirement_text,
                model_name=model_name,
                use_uploaded_context=True,
                document_text=requirement_text,
            )
        else:
            raise ValueError("Please upload a requirement document or provide requirement text")

        return {
            "status": "success",
            "test_cases": result
        }

    except Exception as e:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )