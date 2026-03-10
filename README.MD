PDF-Based RAG API

Overview

This project implements a Retrieval-Augmented Generation (RAG) system using FastAPI.  It allows users to upload a PDF and ask questions about its content. The system retrieves relevant document chunks using vector similarity search and generates answers grounded in the retrieved context.



Architecture

Upload Flow
PDF → Text Extraction → Chunking → Embedding → FAISS Index Storage

Query Flow
Question → Embedding → FAISS Retrieval → Context + Question → T5 Generation → Answer



Tech Stack

FastAPI (Backend API)
SentenceTransformers (Embeddings)
FAISS (Vector Search)
HuggingFace Transformers (T5-small)
PyTorch



How to Run

1. Create virtual environment:
python -m venv .venv

2. Activate:
.venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

4. Start server:
uvicorn app.main:app --reload


End points

-POST `/upload → Upload a PDF
-POST `/ask → Ask a question about uploaded PDF

Limitations

(i)Single document at a time only
(ii)No persistent vector storage
(iii)Uses lightweight T5-small model
