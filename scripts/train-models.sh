#!/bin/sh
set -eu
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
PYTHON="${PYTHON:-.venv/bin/python}"
"$PYTHON" ml/prepare.py
"$PYTHON" ml/train_linear.py
"$PYTHON" ml/train_neural.py --size tiny --epochs 8
"$PYTHON" ml/train_neural.py --size mini --epochs 8
"$PYTHON" ml/export_neural.py --size tiny
"$PYTHON" ml/export_neural.py --size mini
"$PYTHON" ml/check_exports.py
"$PYTHON" ml/evaluate.py
"$PYTHON" ml/browser_fixtures.py
npm run build
node tests/tokenizer-conformance.mjs
"$PYTHON" -m unittest discover -s tests -p 'test_*.py'
npm test
