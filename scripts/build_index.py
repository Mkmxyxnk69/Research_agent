import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNKS_PATH = Path("data/chunks.jsonl")
CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "arxiv_chunks"
BATCH_SIZE = 100
EMBED_MODEL = "all-MiniLM-L6-v2"

def load_chunks():
    chunks = []
    with open(CHUNKS_PATH) as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def main():
    print(f"Loading chunks from {CHUNKS_PATH}...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    print(f"\nLoading embedding model: {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"\nSetting up ChromaDB at {CHROMA_DIR}...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection if rebuilding
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection")
    except:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"\nEmbedding and indexing {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[i:i+BATCH_SIZE]
        texts = [c["text"] for c in batch]
        ids = [f"{c['paper_id']}__chunk{c['chunk_id']}" for c in batch]
        metadatas = [{"paper_id": c["paper_id"], "chunk_id": c["chunk_id"]} for c in batch]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    print(f"\nDone! Indexed {collection.count()} chunks into ChromaDB")
    print(f"Index saved at {CHROMA_DIR}")

if __name__ == "__main__":
    main()
