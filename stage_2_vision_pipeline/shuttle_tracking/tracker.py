"""Thin wrapper around the legacy TrackNetV2 prediction CLI.

The original repo exposes ``predict10.py`` as a CLI; this module wraps it
in a Python-callable function that produces a CSV of (frame, visibility, x, y).

If the model weights are missing, ``track_to_csv`` raises and the caller
(VisionModule) falls back to an empty CSV.
"""
from __future__ import annotations

import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICT_SCRIPT = os.path.join(THIS_DIR, "tracknetv2", "gray10", "predict10.py")
DENOISE_SCRIPT = os.path.join(THIS_DIR, "tracknetv2", "gray10", "denoise10_custom.py")


def track_to_csv(clip_path: str, weights: str, out_csv: str) -> str:
    if not os.path.exists(weights):
        raise FileNotFoundError(f"TrackNetV2 weights missing: {weights}")
    if not os.path.exists(PREDICT_SCRIPT):
        raise FileNotFoundError(f"predict10.py missing at {PREDICT_SCRIPT}")
    out_dir = os.path.dirname(out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        [sys.executable, PREDICT_SCRIPT,
         "--video_name", clip_path,
         "--load_weights", weights,
         "--save_dir", out_dir],
        check=True,
    )
    # predict10 writes <basename>_predict.csv next to the video — point our
    # CSV at it for the caller.
    base = os.path.splitext(os.path.basename(clip_path))[0]
    produced = os.path.join(out_dir, base + "_predict.csv")
    if os.path.exists(produced) and produced != out_csv:
        os.replace(produced, out_csv)
    return out_csv
