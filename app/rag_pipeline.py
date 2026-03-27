import os
import re
import numpy as np

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")



# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_length + sentence_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_chunk = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) <= overlap:
                    overlap_chunk.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current_chunk = overlap_chunk
            current_length = overlap_len

        current_chunk.append(sentence)
        current_length += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ---------------------------------------------------------------------------
# Store chunks into ChromaDB
# ---------------------------------------------------------------------------
def store_chunks(chunks: list[str], filename: str):
    embeddings = embedding_model.encode(chunks, show_progress_bar=False).tolist()

    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename} for _ in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def search(query: str, k: int = 5) -> list[dict]:
    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for i in range(len(results["documents"][0])):
        output.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "score": round(results["distances"][0][i], 4),
        })

    return output


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------
def generate_answer(query: str) -> dict:
    if collection.count() == 0:
        return {
            "answer": "No documents uploaded yet. Please upload a PDF first.",
            "sources": [],
        }

    results = search(query)

    if not results:
        return {
            "answer": "No relevant information found in the uploaded documents.",
            "sources": [],
        }

    context = "\n\n".join([
        f"[Source: {r['source']}]\n{r['text']}" for r in results
    ])

    prompt = (
        "You are a precise document assistant. "
        "Answer the user's question using ONLY the context provided below. "
        "Do not use any outside knowledge. "
        "If the answer is not present in the context, say: "
        "'I could not find an answer to that in the uploaded documents.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

    response = co.chat(
        model="command-a-03-2025",
        message=prompt,
    )

    return {
        "answer": response.text,
        "sources": results,
    }
