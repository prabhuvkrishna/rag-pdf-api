from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader

from app import rag_pipeline
from app.rag_pipeline import chunk_text, create_vector_store

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

    # Split into raw chunks
    raw_chunks = chunk_text(text)

    # Convert to structured chunks
    structured_chunks = [
        {"text": chunk, "source": file.filename}
        for chunk in raw_chunks
    ]

    # Create vector store using text only
    index, embeddings = create_vector_store(
        [chunk["text"] for chunk in structured_chunks]
    )

    # Store inside rag_pipeline module
    rag_pipeline.vector_index = index
    rag_pipeline.stored_chunks = structured_chunks

    return {
        "filename": file.filename,
        "total_chunks": len(structured_chunks),
        "embedding_dimension": embeddings.shape[1]
    }


class QueryRequest(BaseModel):
    query: str


@app.post("/ask/")
def ask_question(request: QueryRequest):
    print("Vector index in ask: ", rag_pipeline.vector_index)
    if rag_pipeline.vector_index is None:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded. Please upload a PDF first."
        )

    user_query = request.query

    response = rag_pipeline.generate_answer(user_query)

    return {
        "query": user_query,
        "answer": response["answer"],
        "sources": response["sources"]
    }