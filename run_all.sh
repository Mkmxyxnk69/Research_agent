#!/bin/bash
set -e

echo "Running baseline..."
python scripts/run_questions.py baseline

echo "Running full_agent..."
python scripts/run_questions.py full_agent

echo "Running no_planner..."
python scripts/run_questions.py no_planner

echo "Running no_reflector..."
python scripts/run_questions.py no_reflector

echo "Running no_citation_verifier..."
python scripts/run_questions.py no_citation_verifier

echo "Running no_hybrid..."
python scripts/run_questions.py no_hybrid

echo "Done. Predictions saved in predictions/"