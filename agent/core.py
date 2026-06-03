import os
import json
import re
import time
from groq import Groq
from dotenv import load_dotenv
from agent.retriever import retrieve as retriever_retrieve

load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"
MAX_ROUNDS = 3
TOP_K = 5

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def llm_call(prompt, max_tokens=1024, retries=5):
    for attempt in range(retries):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(f"LLM call failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt == retries - 1:
                raise
            time.sleep(wait)


def clean_paper_id(pid):
    return re.sub(r"v\d+$", "", pid)


def retrieve_with_config(query, top_k=TOP_K, use_hybrid=False):
    return retriever_retrieve(query, top_k=top_k, use_hybrid=use_hybrid)


def planner(question):
    prompt = f"""Break this research question into 2-3 clear sub-questions that together would fully answer it.
Return only a JSON list of strings, nothing else.

Question: {question}

Example output: ["sub-question 1", "sub-question 2", "sub-question 3"]"""
    result = llm_call(prompt, max_tokens=256)
    try:
        start = result.find("[")
        end = result.rfind("]") + 1
        sub_questions = json.loads(result[start:end])
        if isinstance(sub_questions, list) and len(sub_questions) > 0:
            return sub_questions
        return [question]
    except Exception:
        return [question]


def reflector(question, chunks):
    context = "\n".join([c["text"][:300] for c in chunks[:5]])
    prompt = f"""You are a research assistant. Given this question and retrieved context, decide if there is enough information to write a good answer.

Question: {question}

Retrieved context (first 300 chars each):
{context}

Reply with only YES or NO."""
    result = llm_call(prompt, max_tokens=10)
    return "YES" in result.upper()


def synthesizer(question, chunks, q_type="factoid"):
    context_parts = []
    for c in chunks:
        pid = clean_paper_id(c["paper_id"])
        context_parts.append(f"[{pid}]: {c['text'][:600]}")
    context = "\n\n".join(context_parts)

    if q_type == "factoid":
        length_guide = "1-3 sentences"
    elif q_type == "comparative":
        length_guide = "100-300 words"
    else:
        length_guide = "250-600 words"

    prompt = f"""You are a research assistant. Answer the question using ONLY the provided context.
After each claim, cite the paper using [arxiv_id] format.
Length: {length_guide}

Question: {question}

Context:
{context}

Answer:"""
    return llm_call(prompt, max_tokens=1024)


def citation_verifier(answer, chunks):
    cited_ids = re.findall(r"\[(\d{4}\.\d+)\]", answer)
    verified = []
    for cid in cited_ids:
        matching = [c for c in chunks if clean_paper_id(c["paper_id"]) == cid]
        if not matching:
            continue
        chunk_text = matching[0]["text"][:400]
        prompt = f"""Does this text support the use of [{cid}] as a citation in the answer?
Text: {chunk_text}
Answer excerpt: {answer[:300]}
Reply YES or NO only."""
        result = llm_call(prompt, max_tokens=10)
        if "YES" in result.upper():
            verified.append(cid)
    return list(set(verified))


def run_agent(question, q_type="factoid", config=None):
    if config is None:
        config = {
            "use_planner": True,
            "use_reflector": True,
            "use_citation_verifier": True,
            "use_hybrid_retrieval": False
        }

    all_chunks = []
    tool_calls = 0

    if config.get("use_planner", True):
        sub_questions = planner(question)
        tool_calls += 1
    else:
        sub_questions = [question]

    for sq in sub_questions:
        chunks = retrieve_with_config(
            sq,
            top_k=TOP_K,
            use_hybrid=config.get("use_hybrid_retrieval", False)
        )
        tool_calls += 1
        all_chunks.extend(chunks)

    seen = set()
    unique_chunks = []
    for c in all_chunks:
        if c["text"] not in seen:
            seen.add(c["text"])
            unique_chunks.append(c)
    all_chunks = unique_chunks

    if config.get("use_reflector", True):
        for _ in range(MAX_ROUNDS - 1):
            has_enough = reflector(question, all_chunks)
            tool_calls += 1
            if has_enough:
                break

            more_chunks = retrieve_with_config(
                question,
                top_k=3,
                use_hybrid=config.get("use_hybrid_retrieval", False)
            )
            tool_calls += 1

            for c in more_chunks:
                if c["text"] not in seen:
                    seen.add(c["text"])
                    all_chunks.append(c)

    answer = synthesizer(question, all_chunks[:10], q_type)
    tool_calls += 1

    if config.get("use_citation_verifier", True):
        verified_ids = citation_verifier(answer, all_chunks)
        tool_calls += 1
    else:
        verified_ids = list(set(re.findall(r"\[(\d{4}\.\d+)\]", answer)))

    all_paper_ids = list(set([clean_paper_id(c["paper_id"]) for c in all_chunks]))

    return {
        "answer": answer,
        "cited_papers": verified_ids if verified_ids else all_paper_ids[:5],
        "tool_calls": tool_calls,
        "chunks_used": len(all_chunks)
    }


if __name__ == "__main__":
    result = run_agent(
        "What does ACI stand for in the SWE-agent paper?",
        q_type="factoid",
        config={
            "use_planner": True,
            "use_reflector": True,
            "use_citation_verifier": True,
            "use_hybrid_retrieval": True
        }
    )
    print("\nAnswer:", result["answer"])
    print("Cited papers:", result["cited_papers"])
    print("Tool calls:", result["tool_calls"])