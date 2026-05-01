"""Stage-3 entry point."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.config import load_config                        # noqa: E402
from common.contracts import RallySegment                    # noqa: E402
from common.io import load_json                              # noqa: E402
from stage_3_hit_frame_detection.hit_detector import HitDetector  # noqa: E402


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
    print(f"[Stage 3] Loaded {len(rallies)} rallies.")
    det = HitDetector(cfg)
    events = det.detect_all(rallies)
    print(f"[Stage 3] Detected {len(events)} hit events.")


if __name__ == "__main__":
    main()
