"""Stage-2 entry point.

Usage:
    python -m stage_2_vision_pipeline.run
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.config import load_config                    # noqa: E402
from common.contracts import RallySegment                 # noqa: E402
from common.io import load_json, save_json                # noqa: E402
from stage_2_vision_pipeline.vision_module import VisionModule  # noqa: E402


def _load_rallies(rallies_dir):
    out = []
    for video_name in sorted(os.listdir(rallies_dir)):
        meta = os.path.join(rallies_dir, video_name, "rallies.json")
        if not os.path.exists(meta):
            continue
        for rec in load_json(meta):
            out.append(RallySegment(**rec))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "pipeline.yaml"))
    args = p.parse_args()
    cfg = load_config(args.config)

    rallies = _load_rallies(cfg.storage.rallies_dir)
    print(f"[Stage 2] Loaded {len(rallies)} rally segments from Stage 1.")
    vm = VisionModule(cfg)
    enriched = vm.process_all(rallies)

    # Write enriched RallySegments back to disk for downstream stages.
    by_video = {}
    for r in enriched:
        v = os.path.splitext(os.path.basename(r.video_path))[0]
        by_video.setdefault(v, []).append(r.to_dict())
    for v, recs in by_video.items():
        save_json(os.path.join(cfg.storage.rallies_dir, v, "rallies.json"), recs)
    print(f"[Stage 2] Enriched {len(enriched)} rallies with joints/court/shuttle.")


if __name__ == "__main__":
    main()
