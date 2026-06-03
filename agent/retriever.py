import json
import re
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "arxiv_chunks"
CHUNKS_PATH = "data/chunks.jsonl"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

print("Loading retriever components...")
embedder = SentenceTransformer(EMBED_MODEL)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)

# Load all chunks for BM25
print("Loading chunks for BM25...")
all_chunks = []
with open(CHUNKS_PATH) as f:
    for line in f:
        all_chunks.append(json.loads(line))

# Build BM25 index
print("Building BM25 index...")
tokenized_corpus = [c["text"].lower().split() for c in all_chunks]
bm25 = BM25Okapi(tokenized_corpus)
print("Retriever ready!")

def clean_paper_id(pid):
    return re.sub(r'v\d+$', '', pid)

def retrieve_dense(query, top_k=TOP_K):
    embedding = embedder.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text": doc,
            "paper_id": clean_paper_id(meta["paper_id"])
        })
    return chunks

def retrieve_bm25(query, top_k=TOP_K):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    chunks = []
    for idx in top_indices:
        if scores[idx] > 0:
            c = all_chunks[idx]
            chunks.append({
                "text": c["text"],
                "paper_id": clean_paper_id(c["paper_id"])
            })
    return chunks

def retrieve_hybrid(query, top_k=TOP_K):
    dense_chunks = retrieve_dense(query, top_k=top_k)
    bm25_chunks = retrieve_bm25(query, top_k=top_k)

    # Combine and deduplicate, dense results first
    seen = set()
    combined = []
    for c in dense_chunks + bm25_chunks:
        if c["text"] not in seen:
            seen.add(c["text"])
            combined.append(c)

    return combined[:top_k * 2]

def retrieve(query, top_k=TOP_K, use_hybrid=False):
    if use_hybrid:
        return retrieve_hybrid(query, top_k=top_k)
    else:
        return retrieve_dense(query, top_k=top_k)
