import os
import re
import json
from pathlib import Path

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

QUESTIONS_PATH = "eval/questions.jsonl"
OUTPUT_PATH = "predictions/baseline.jsonl"

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "arxiv_chunks"
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 5

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment. Put it in your .env file.")

groq_client = Groq(api_key=groq_api_key)


def load_questions(path):
    questions = []
    with open(path, "r") as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def clean_paper_id(pid):
    return re.sub(r'v\d+$', '', pid)


def clean_cited_papers(cited_papers):
    return sorted(list(set(clean_paper_id(pid) for pid in cited_papers)))


def retrieve(query, top_k=TOP_K):
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


def llm_call(prompt, max_tokens=1024):
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()


def run_baseline(question, q_type="factoid"):
    chunks = retrieve(question, top_k=TOP_K)
    tool_calls = 1

    if q_type == "factoid":
        length_guide = "1-3 sentences"
        max_tokens = 256
    elif q_type == "comparative":
        length_guide = "100-300 words"
        max_tokens = 700
    else:
        length_guide = "250-600 words"
        max_tokens = 1200

    context_parts = []
    for c in chunks:
        context_parts.append(f"[{c['paper_id']}]: {c['text'][:600]}")
    context = "\n\n".join(context_parts)

    prompt = f"""You are a research assistant. Answer the question using ONLY the provided context.
After each important claim, cite the paper using [arxiv_id] format.
Keep the answer concise and grounded in the evidence.
Length: {length_guide}

Question: {question}

Context:
{context}

Answer:"""

    answer = llm_call(prompt, max_tokens=max_tokens)
    tool_calls += 1

    cited_ids = list(set(re.findall(r'\[(\d{4}\.\d+)\]', answer)))
    if not cited_ids:
        cited_ids = [clean_paper_id(c["paper_id"]) for c in chunks[:3]]

    return {
        "answer": answer,
        "cited_papers": clean_cited_papers(cited_ids),
        "tool_calls": tool_calls,
        "chunks_used": len(chunks)
    }


def answer_question(question_obj):
    qid = question_obj["id"]
    question = question_obj["question"]
    q_type = question_obj.get("type", "factoid")

    result = run_baseline(question, q_type=q_type)

    return {
        "id": qid,
        "answer": result["answer"],
        "cited_papers": result["cited_papers"]
    }


def main():
    Path("predictions").mkdir(exist_ok=True)

    questions = load_questions(QUESTIONS_PATH)
    print(f"Loaded {len(questions)} questions")

    with open(OUTPUT_PATH, "w") as f:
        for i, q in enumerate(questions, start=1):
            print(f"[{i}/{len(questions)}] Running {q['id']} ({q['type']})")
            result = answer_question(q)
            f.write(json.dumps(result) + "\n")
            print(f"Saved {result['id']}")

    print(f"\nDone. Saved predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()