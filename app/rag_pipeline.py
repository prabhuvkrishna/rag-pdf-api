from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration


gen_model_name = "t5-small"

gen_tokenizer = T5Tokenizer.from_pretrained(gen_model_name)
gen_model = T5ForConditionalGeneration.from_pretrained(gen_model_name)
import torch
import faiss
import numpy as np
vector_index = None
stored_chunks = []
# Load model once (important)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Generation Model, local LLM
gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def create_vector_store(chunks):
    embeddings = embedding_model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index, embeddings

def search(index, query, chunks, k=5, threshold=1.5):
    query_embedding = embedding_model.encode([query])
    D, I = index.search(query_embedding, k)

    filtered_results = []

    for idx, i in enumerate(I[0]):
        score = float(D[0][idx])

        if score < threshold:
            filtered_results.append({
                "text": chunks[i]["text"],
                "source": chunks[i]["source"],
                "score": score
            })

    return filtered_results

def generate_answer(query):
    global vector_index, stored_chunks

    if vector_index is None:
        return {
            "answer": "No document has been uploaded yet.",
            "sources": []
        }

    results = search(vector_index, query, stored_chunks)

    if not results:
        return {
            "answer": "No relevant information found.",
            "sources": []
        }

    context = "\n".join([r["text"] for r in results])

    input_text = f"question: {query}  context: {context}"
    input_ids = gen_tokenizer.encode(input_text, return_tensors="pt")

    outputs = gen_model.generate(input_ids, max_length=200)
    response = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "answer": response,
        "sources": results
    }