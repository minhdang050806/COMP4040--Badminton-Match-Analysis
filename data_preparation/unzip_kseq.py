"""Unpack KSeq train/test archives into ``data_storage/raw_videos``.

Mirrors the AICUP 2023 dataset layout described in the original
``Dataset/`` README:

    Dataset/KSeq_train_data.zip   — labelled rally clips + score_*.json
    Dataset/KSeq_test_dataset.zip — unlabelled clips for submission

Run after placing the raw zips into ``data_storage/raw_videos/_zips/``.
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.config import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "pipeline.yaml"))
    p.add_argument("--zip_dir", default=None,
                   help="Directory containing KSeq_*.zip; "
                        "defaults to <video.source_dir>/_zips")
    args = p.parse_args()

    cfg = load_config(args.config)
    zip_dir = args.zip_dir or os.path.join(cfg.video.source_dir, "_zips")
    if not os.path.isdir(zip_dir):
        print(f"No zips found in {zip_dir}; nothing to do.")
        return

    out_dir = cfg.video.source_dir
    os.makedirs(out_dir, exist_ok=True)

    for fn in sorted(os.listdir(zip_dir)):
        if not fn.endswith(".zip"):
            continue
        target = os.path.join(out_dir, fn.replace(".zip", ""))
        if os.path.isdir(target):
            print(f"  skip {fn} (already extracted)")
            continue
        print(f"  extract {fn} -> {target}")
        with zipfile.ZipFile(os.path.join(zip_dir, fn)) as z:
            z.extractall(target)


if __name__ == "__main__":
    main()
