#!/usr/bin/env bash
# Run all three problems from scratch (CPU only, ~1 hour total).
# Outputs: results/*.json, figures/*.png, rks_test_predictions.csv, mlp_test_predictions.csv
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -q -r requirements.txt
mkdir -p results figures
python q1_taxi.py    | tee results/q1.log
python q2_retail.py  | tee results/q2.log
python q3_rks_mlp.py | tee results/q3.log
