#!/usr/bin/env bash
# End-to-end pipeline driver — runs every stage and writes analytics artefacts.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python data_preparation/unzip_kseq.py        || true
python run_pipeline.py "$@"
echo "Pipeline complete. Outputs in data_storage/outputs/"
