"""Build the OptimusPrime training set from KSeq score_*.json annotations.

The KSeq dataset stores each rally as ``score_<n>.json`` containing a list
of frames with 17-joint coordinates and direction tokens. The original
training pipeline lived inside the Automated-Hit-frame-Detection repo's
notebooks; this script consolidates the bits we need (joint stacking +
StandardScaler) into one CLI.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def collect_joints(kseq_root: str):
    """Walk ``kseq_root`` and concatenate all ``frames[].joint`` arrays."""
    joints = []
    for score_path in glob.glob(os.path.join(kseq_root, "*", "score_*.json")):
        with open(score_path) as f:
            data = json.load(f)
        for frame in data.get("frames", []):
            j = np.asarray(frame.get("joint", []), dtype=np.float32)
            if j.size > 0:
                joints.append(j.reshape(-1))
    if not joints:
        return np.zeros((0,), dtype=np.float32)
    max_len = max(len(j) for j in joints)
    out = np.zeros((len(joints), max_len), dtype=np.float32)
    for i, j in enumerate(joints):
        out[i, : len(j)] = j
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kseq_dir", required=True,
                   help="Path to KSeq_train_data folder.")
    p.add_argument("--scaler_out", required=True,
                   help="Pickle output for the fitted StandardScaler.")
    args = p.parse_args()

    print(f"Reading {args.kseq_dir}…")
    X = collect_joints(args.kseq_dir)
    print(f"Collected {X.shape[0]} joint frames of dim {X.shape[1] if X.ndim==2 else 0}")
    scaler = StandardScaler().fit(X) if X.size else StandardScaler()
    os.makedirs(os.path.dirname(args.scaler_out) or ".", exist_ok=True)
    with open(args.scaler_out, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler → {args.scaler_out}")


if __name__ == "__main__":
    main()
