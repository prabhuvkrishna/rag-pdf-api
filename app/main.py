from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pypdf import PdfReader

from app import rag_pipeline
from app.rag_pipeline import chunk_text, store_chunks

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "RAG PDF API is running"}


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    print("Endpoint Hit")

    contents = await file.read()

    with open("data/temp.pdf", "wb") as f:
        f.write(contents)

    reader = PdfReader("data/temp.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    chunks = chunk_text(text)

    store_chunks(chunks, filename=file.filename)

    return {
        "filename": file.filename,
        "total_chunks": len(chunks),
        "message": "Document stored successfully."
    }


class QueryRequest(BaseModel):
    query: str


@app.post("/ask/")
def ask_question(request: QueryRequest):
    response = rag_pipeline.generate_answer(request.query)

    return {
        "query": request.query,
        "answer": response["answer"],
        "sources": response["sources"],
    }