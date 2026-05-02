"""Re-run BST training-data preparation (legacy ``prepare_train_on_*`` scripts).

This is a thin dispatcher: pick a target dataset and shell out to the
appropriate legacy script (kept verbatim under
``stage_4_stroke_classification/feature_extraction/preparing_data/``).
The legacy scripts walk the BST source repo (ShuttleSet / BadmintonDB /
TenniSet directory tree) and write ``.npy`` tensors that BST consumes.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREP_DIR = os.path.join(
    ROOT, "stage_4_stroke_classification", "feature_extraction", "preparing_data")

DISPATCH = {
    "shuttleset":         "prepare_train_on_shuttleset.py",
    "shuttleset_merged":  "prepare_train_on_shuttleset_merged.py",
    "badmintondb":        "prepare_train_on_badDB.py",
    "tenniset":           "prepare_train_on_tenniSet.py",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=sorted(DISPATCH))
    p.add_argument("--", dest="passthrough", nargs=argparse.REMAINDER, default=[])
    args = p.parse_args()

    script = os.path.join(PREP_DIR, DISPATCH[args.dataset])
    if not os.path.exists(script):
        sys.exit(f"Missing legacy script: {script}")

    cmd = [sys.executable, script, *args.passthrough]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PREP_DIR)


if __name__ == "__main__":
    main()
