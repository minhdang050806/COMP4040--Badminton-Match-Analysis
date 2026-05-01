"""Stage-4 entry point."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.config import load_config                            # noqa: E402
from common.contracts import RallySegment, HitEvent              # noqa: E402
from common.io import load_json, ensure_dir                      # noqa: E402
from stage_4_stroke_classification import StrokePipeline         # noqa: E402


def _load_rallies(rallies_dir):
    out = []
    for video_name in sorted(os.listdir(rallies_dir)):
        meta = os.path.join(rallies_dir, video_name, "rallies.json")
        if not os.path.exists(meta):
            continue
        for rec in load_json(meta):
            out.append(RallySegment(**rec))
    return out


def _load_hit_events(hit_dir):
    p = os.path.join(hit_dir, "hit_events.json")
    if not os.path.exists(p):
        return []
    return [HitEvent(**rec) for rec in load_json(p)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "pipeline.yaml"))
    args = p.parse_args()
    cfg = load_config(args.config)

    rallies = _load_rallies(cfg.storage.rallies_dir)
    events = _load_hit_events(cfg.storage.hit_events_dir)
    print(f"[Stage 4] Inputs: {len(rallies)} rallies, {len(events)} events.")

    pipe = StrokePipeline(cfg)
    records = pipe.run(rallies, events)

    out_csv = os.path.join(ensure_dir(cfg.storage.strokes_csv_dir),
                           "strokes.csv")
    pipe.export_csv(records, out_csv)
    print(f"[Stage 4] Wrote {len(records)} stroke records to {out_csv}")


if __name__ == "__main__":
    main()
