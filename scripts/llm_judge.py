import json
import os
import time
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Only judge these 5 questions (one of each type)
JUDGE_IDS = ["q01", "q04", "q07", "q11", "q21"]

PREDICTIONS_DIR = Path("predictions")
QUESTIONS_PATH = Path("eval/questions.jsonl")

def load_questions():
    questions = {}
    with open(QUESTIONS_PATH) as f:
        for line in f:
            q = json.loads(line)
            questions[q["id"]] = q
    return questions

def load_predictions(path):
    preds = {}
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            preds[p["id"]] = p
    return preds

def llm_judge(question, answer):
    prompt = f"""You are a strict research evaluator. Score this answer.

Question: {question}
Answer: {answer}

Score from 0-10:
1. Accuracy: Is the answer factually correct and complete?
2. Faithfulness: Does it stay grounded in evidence without hallucinating?

Reply in this exact JSON only:
{{"accuracy": <0-10>, "faithfulness": <0-10>, "reason": "<one line why>"}}"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.0
        )
        result = response.choices[0].message.content.strip()
        start = result.find("{")
        end = result.rfind("}") + 1
        scores = json.loads(result[start:end])
        return scores
    except Exception as e:
        print(f"  Judge failed: {e}")
        return {"accuracy": 0, "faithfulness": 0, "reason": "error"}

def main():
    questions = load_questions()
    configs = [
        "baseline",
        "full_agent",
        "no_planner",
        "no_reflector",
        "no_citation_verifier",
        "no_hybrid"
    ]

    print("LLM JUDGE RESULTS")
    print("="*80)
    print(f"{'Config':<25} {'QID':<6} {'Accuracy':>10} {'Faithful':>10} Reason")
    print("-"*80)

    summary = {}
    for config in configs:
        pred_path = PREDICTIONS_DIR / f"{config}.jsonl"
        if not pred_path.exists():
            continue
        preds = load_predictions(pred_path)
        acc_scores = []
        faith_scores = []

        for qid in JUDGE_IDS:
            if qid not in preds or qid not in questions:
                continue
            q = questions[qid]
            pred = preds[qid]
            scores = llm_judge(q["question"], pred["answer"])
            acc_scores.append(scores["accuracy"])
            faith_scores.append(scores["faithfulness"])
            print(f"{config:<25} {qid:<6} {scores['accuracy']:>10} {scores['faithfulness']:>10} {scores.get('reason','')[:30]}")
            time.sleep(3)

        avg_acc = round(sum(acc_scores)/len(acc_scores), 2) if acc_scores else 0
        avg_faith = round(sum(faith_scores)/len(faith_scores), 2) if faith_scores else 0
        summary[config] = {"accuracy": avg_acc, "faithfulness": avg_faith}

    print("\n" + "="*80)
    print("SUMMARY (LLM Judge)")
    print("="*80)
    print(f"{'Config':<25} {'Accuracy':>10} {'Faithfulness':>10}")
    print("-"*80)
    for config, scores in summary.items():
        print(f"{config:<25} {scores['accuracy']:>10} {scores['faithfulness']:>10}")
    print("="*80)

    # Save results
    with open("data/llm_judge_scores.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved to data/llm_judge_scores.json")

if __name__ == "__main__":
    main()
