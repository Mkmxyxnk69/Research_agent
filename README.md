# Agentic Deep Research over arXiv (AIMS DTU Research Intern 2026)

This project builds an agentic deep-research system over a corpus of recent LLM-agent arXiv papers (2024–2026), then evaluates how much each component contributes to answer quality.

## Features

- arXiv paper corpus collection and indexing
- Dense + hybrid retrieval
- Agentic loop with:
  - planner
  - retriever
  - reflector
  - synthesizer
  - citation verifier
- Non-agentic baseline
- Component ablations:
  - no_planner
  - no_reflector
  - no_citation_verifier
  - no_hybrid

## Repository structure

```bash
agentic-ai/
├── agent/
│   ├── __init__.py
│   ├── core.py
│   ├── retriever.py
│   └── baseline.py
├── scripts/
│   ├── run_questions.py
│   └── ...
├── data/
│   ├── chunks.jsonl
│   ├── chroma_db/
│   └── ...
├── eval/
│   ├── questions.jsonl
│   └── SUBMISSION_FORMAT.pdf
├── predictions/
│   ├── baseline.jsonl
│   ├── full_agent.jsonl
│   ├── no_planner.jsonl
│   ├── no_reflector.jsonl
│   ├── no_citation_verifier.jsonl
│   └── no_hybrid.jsonl
├── report.pdf
├── run_all.sh
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Run individual configs

```bash
python scripts/run_questions.py baseline
python scripts/run_questions.py full_agent
python scripts/run_questions.py no_planner
python scripts/run_questions.py no_reflector
python scripts/run_questions.py no_citation_verifier
python scripts/run_questions.py no_hybrid
```

## Reproduce all results

```bash
bash run_all.sh
```

## Output format

Each predictions file is stored under:

```bash
predictions/<config>.jsonl
```

Each line follows:

```json
{"id":"q01","answer":"...","cited_papers":["2504.19413"]}
```

## Notes

- `full_agent` uses planner + reflector + citation verifier + hybrid retrieval
- `baseline` is a non-agentic retrieval + answer system
- arXiv IDs are stored without version suffixes