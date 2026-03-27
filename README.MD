
RAG PDF API

Multi-Document Retrieval-Augmented Generation System

What It Does
A production-ready REST API that lets you upload multiple PDF documents and ask natural language questions across all of them. Answers are strictly grounded in the uploaded content — no hallucination.

Upload one or multiple PDF documents via REST API
Ask natural language questions across all uploaded documents
Get accurate answers with source references and similarity scores
Zero hallucination — model is instructed to use only uploaded content
Persistent vector storage — documents survive server restarts

Architecture
The system follows a standard RAG pipeline with a persistent vector store:

User Query
    ↓
FastAPI Endpoint  (/ask/)
    ↓
SentenceTransformer  (all-MiniLM-L6-v2)  →  Query Embedding
    ↓
ChromaDB Vector Store  →  Top-K Semantic Search
    ↓
Retrieved Chunks + Source Metadata
    ↓
Cohere LLM  (command-a-03-2025)  →  Grounded Answer
    ↓
JSON Response  { answer, sources, scores }

Tech Stack

Component
Technology
Purpose
API Framework
FastAPI
REST endpoints, async file handling
PDF Parsing
PyPDF + cryptography
Text extraction from encrypted/standard PDFs
Embedding Model
all-MiniLM-L6-v2
Converts text chunks to semantic vectors
Vector Store
ChromaDB (persistent)
Multi-document storage, semantic search
LLM
Cohere command-a-03-2025
Grounded answer generation
Environment
python-dotenv
Secure API key management


API Endpoints
GET /
Health check. Confirms the API is running.

POST /upload-pdf/
Upload a PDF document. Chunks the text, generates embeddings, and stores them in ChromaDB.

Request: multipart/form-data with a PDF file
curl -X POST http://localhost:8000/upload-pdf/ -F "file=@document.pdf"

Response:
{ "filename": "document.pdf", "total_chunks": 42, "message": "Document stored successfully." }

POST /ask/
Ask a question across all uploaded documents.

Request:
{ "query": "What is the applicant's date of birth?" }

Response:
{ "query": "...", "answer": "The applicant's date of birth is 15 March 2002.", "sources": [...] }

Key Design Decisions
ChromaDB over FAISS
FAISS stores the index in RAM — every new upload overwrites the previous one and data is lost on server restart. ChromaDB persists to disk, enabling multi-document support and stateful storage across sessions.

Sentence-Boundary Chunking
Raw character-based chunking cuts sentences mid-thought, degrading retrieval quality. The system uses regex-based sentence boundary detection to preserve semantic coherence within each chunk, with configurable overlap between chunks.

Grounded LLM Prompting
The prompt explicitly instructs the model to use only the retrieved context and respond with a standard message if the answer isn't present. This eliminates hallucination entirely — the model cannot fabricate information not in the uploaded documents.

Project Structure
rag-pdf-api/
├── app/
│   ├── main.py           # FastAPI app and endpoint definitions
│   └── rag_pipeline.py   # Chunking, embedding, search, generation
├── data/                 # Temporary PDF storage during processing
├── chroma_db/            # Persistent vector store (auto-created)
├── .env                  # API keys (never committed to git)
├── .gitignore
├── requirements.txt
└── README.md

How to Run Locally

1. Clone the repository
git clone https://github.com/prabhuvkrishna/rag-pdf-api.git
cd rag-pdf-api

2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

3. Install dependencies
pip install -r requirements.txt

4. Set up environment variables
Create a .env file in the root directory:
COHERE_API_KEY=your_cohere_api_key_here

5. Start the server
uvicorn app.main:app --reload

6. Open interactive API docs
http://localhost:8000/docs

Future Improvements
Docker support for containerized deployment
Cloud deployment on Railway or Render
Support for DOCX and TXT file formats
Authentication and rate limiting
Frontend UI for non-technical users
Hybrid search combining semantic and keyword matching

