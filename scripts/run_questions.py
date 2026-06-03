import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import json
import time
import os
from pathlib import Path

Path("predictions").mkdir(exist_ok=True)

QUESTIONS_PATH = "eval/questions.jsonl"

def load_questions():
    questions = []
    with open(QUESTIONS_PATH) as f:
        for line in f:
            questions.append(json.loads(line))
    return questions

def save_predictions(results, config_name):
    out_path = f"predictions/{config_name}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} predictions to {out_path}")

def run_config(config_name, config, questions):
    print(f"\n{'='*50}")
    print(f"Running config: {config_name}")
    print(f"{'='*50}")

    if config_name == "baseline":
        from agent.baseline import run_baseline
    else:
        from agent.core import run_agent

    results = []
    for i, q in enumerate(questions):
        print(f"\n[{i+1}/30] {q['id']} ({q['type']}): {q['question'][:60]}...")
        start = time.time()
        try:
            if config_name == "baseline":
                result = run_baseline(q["question"], q_type=q["type"])
            else:
                result = run_agent(q["question"], q_type=q["type"], config=config)

            latency = round(time.time() - start, 2)
            results.append({
                "id": q["id"],
                "answer": result["answer"],
                "cited_papers": result["cited_papers"],
                "latency": latency,
                "tool_calls": result["tool_calls"]
            })
            print(f"  Done in {latency}s | tool_calls={result['tool_calls']}")
            print(f"  Answer preview: {result['answer'][:100]}...")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "id": q["id"],
                "answer": "Error generating answer.",
                "cited_papers": [],
                "latency": 0,
                "tool_calls": 0
            })

        # Sleep to avoid Groq rate limits
        time.sleep(2)

    save_predictions(results, config_name)
    return results

CONFIGS = {
    "baseline": {
        "use_planner": False,
        "use_reflector": False,
        "use_citation_verifier": False,
        "use_hybrid_retrieval": False
    },

    "full_agent": {
        "use_planner": True,
        "use_reflector": True,
        "use_citation_verifier": True,
        "use_hybrid_retrieval": True
    },

    "no_planner": {
        "use_planner": False,
        "use_reflector": True,
        "use_citation_verifier": True,
        "use_hybrid_retrieval": True
    },

    "no_reflector": {
        "use_planner": True,
        "use_reflector": False,
        "use_citation_verifier": True,
        "use_hybrid_retrieval": True
    },

    "no_citation_verifier": {
        "use_planner": True,
        "use_reflector": True,
        "use_citation_verifier": False,
        "use_hybrid_retrieval": True
    },

    "no_hybrid": {
        "use_planner": True,
        "use_reflector": True,
        "use_citation_verifier": True,
        "use_hybrid_retrieval": False
    }
}
if __name__ == "__main__":
    import sys
    questions = load_questions()
    print(f"Loaded {len(questions)} questions")

    # Get config from command line or run all
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        if config_name == "baseline":
            run_config("baseline", {}, questions)
        elif config_name in CONFIGS:
            run_config(config_name, CONFIGS[config_name], questions)
        else:
            print(f"Unknown config: {config_name}")
            print(f"Available: baseline, {', '.join(CONFIGS.keys())}")
    else:
        # Run all configs
        print("Running ALL configs...")
        run_config("baseline", {}, questions)
        for config_name, config in CONFIGS.items():
            run_config(config_name, config, questions)
        print("\nAll configs done!")
