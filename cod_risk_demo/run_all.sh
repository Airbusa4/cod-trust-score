#!/usr/bin/env bash
# Runs the whole COD Risk Score demo end to end (synthetic data, for illustration only).
# Usage: ./run_all.sh     (uses .venv if it exists, otherwise `python`)
set -euo pipefail
cd "$(dirname "$0")"
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=python; fi
export PYTHONIOENCODING=utf-8

# Data (deterministic, seed 42): the same files every time.
$PY -m src.generate_data
$PY -m src.build_features
$PY -m src.validate_data > /dev/null
# Model, evaluation, decisions, explanations, demo page, summary.
$PY -m src.train
$PY -m src.evaluate
$PY -m src.decide
$PY -m src.explain
$PY -m src.build_demo
$PY -m src.summary
