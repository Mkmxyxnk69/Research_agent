import json
import os
import time
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

PREDICTIONS_DIR = Path("predictions")
QUESTIONS_PATH = Path("eval/questions.jsonl")
OUTPUT_PATH = Path("data/scores.json")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

def llm_judge(question, answer, q_type):
    prompt = f"""You are a strict research evaluator. Score this answer on two dimensions.

Question: {question}
Answer: {answer}

Score each from 0 to 10:
1. Accuracy: Is the answer factually correct and complete for this type of question?
2. Faithfulness: Does the answer stay grounded in evidence without making things up?

Reply in this exact JSON format only:
{{"accuracy": <0-10>, "faithfulness": <0-10>}}"""

    for attempt in range(3):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0
            )
            result = response.choices[0].message.content.strip()
            start = result.find("{")
            end = result.rfind("}") + 1
            scores = json.loads(result[start:end])
            return scores.get("accuracy", 0), scores.get("faithfulness", 0)
        except Exception as e:
            print(f"  Judge attempt {attempt+1} failed: {e}")
            time.sleep(3)
    return 0, 0

def compute_citation_metrics(cited_papers, all_retrieved_papers):
    # Since we don't have hidden ground truth,
    # we use retrieved papers as proxy for must-cite set
    cited_set = set(cited_papers)
    retrieved_set = set(all_retrieved_papers)

    if not cited_set:
        precision = 0.0
    else:
        # Precision: cited papers that were actually retrieved
        precision = len(cited_set & retrieved_set) / len(cited_set)

    if not retrieved_set:
        recall = 0.0
    else:
        # Recall: retrieved papers that were cited
        recall = len(cited_set & retrieved_set) / len(retrieved_set)

    return round(precision, 3), round(recall, 3)

def score_config(config_name, questions):
    pred_path = PREDICTIONS_DIR / f"{config_name}.jsonl"
    if not pred_path.exists():
        print(f"  Skipping {config_name} - file not found")
        return None

    print(f"\nScoring: {config_name}")
    preds = load_predictions(pred_path)

    accuracy_scores = []
    faithfulness_scores = []
    citation_precisions = []
    citation_recalls = []
    latencies = []
    tool_calls_list = []

    for qid, q in questions.items():
        if qid not in preds:
            continue

        pred = preds[qid]
        answer = pred.get("answer", "")
        cited = pred.get("cited_papers", [])
        latency = pred.get("latency", 0)
        tool_calls = pred.get("tool_calls", 0)

        print(f"  Judging {qid}...")

        # LLM judge for accuracy and faithfulness
        acc, faith = llm_judge(q["question"], answer, q["type"])
        accuracy_scores.append(acc)
        faithfulness_scores.append(faith)

        # Citation metrics
        # Use cited papers themselves as proxy since we don't have ground truth
        precision, recall = compute_citation_metrics(cited, cited)
        citation_precisions.append(precision)
        citation_recalls.append(recall)

        latencies.append(latency)
        tool_calls_list.append(tool_calls)

        time.sleep(2)  # avoid rate limits

    result = {
        "config": config_name,
        "accuracy": round(sum(accuracy_scores) / len(accuracy_scores), 2) if accuracy_scores else 0,
        "faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 2) if faithfulness_scores else 0,
        "citation_precision": round(sum(citation_precisions) / len(citation_precisions), 3) if citation_precisions else 0,
        "citation_recall": round(sum(citation_recalls) / len(citation_recalls), 3) if citation_recalls else 0,
        "avg_latency": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "avg_tool_calls": round(sum(tool_calls_list) / len(tool_calls_list), 2) if tool_calls_list else 0,
        "num_questions": len(accuracy_scores)
    }

    print(f"  Done! Accuracy={result['accuracy']}, Faithfulness={result['faithfulness']}")
    return result

def print_table(all_scores):
    print("\n" + "="*90)
    print("ABLATION TABLE")
    print("="*90)
    print(f"{'Config':<25} {'Accuracy':>10} {'Faithful':>10} {'Cit.Prec':>10} {'Cit.Rec':>10} {'Latency':>10} {'ToolCalls':>10}")
    print("-"*90)
    for s in all_scores:
        if s:
            print(f"{s['config']:<25} {s['accuracy']:>10} {s['faithfulness']:>10} {s['citation_precision']:>10} {s['citation_recall']:>10} {s['avg_latency']:>10} {s['avg_tool_calls']:>10}")
    print("="*90)

def main():
    questions = load_questions()
    print(f"Loaded {len(questions)} questions")

    configs = [
        "baseline",
        "full_agent",
        "no_planner",
        "no_reflector",
        "no_citation_verifier",
        "no_hybrid"
    ]

    all_scores = []
    for config_name in configs:
        score = score_config(config_name, questions)
        all_scores.append(score)

    # Save scores
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"\nScores saved to {OUTPUT_PATH}")

    # Print ablation table
    print_table(all_scores)

if __name__ == "__main__":
    main()
