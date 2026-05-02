"""End-to-end Badminton Match Analysis pipeline.

Runs Stages 1 → 5 in order, passing in-memory contracts between stages
and persisting intermediate artefacts to ``data_storage/``.

Usage:
    python run_pipeline.py --config configs/pipeline.yaml
    python run_pipeline.py --video /path/to/match.mp4
    python run_pipeline.py --skip 1            # resume from disk for stage 1
    python run_pipeline.py --only 5            # only run analytics on existing CSV
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from common.config import load_config                                # noqa: E402
from common.contracts import RallySegment, HitEvent                  # noqa: E402
from common.io import ensure_dir, load_json, save_json               # noqa: E402


def _banner(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _load_rallies(rallies_dir):
    out = []
    if not os.path.isdir(rallies_dir):
        return out
    for video_name in sorted(os.listdir(rallies_dir)):
        meta = os.path.join(rallies_dir, video_name, "rallies.json")
        if os.path.exists(meta):
            for rec in load_json(meta):
                out.append(RallySegment(**rec))
    return out


def _persist_rallies(rallies, rallies_dir):
    by_video = {}
    for r in rallies:
        v = os.path.splitext(os.path.basename(r.video_path))[0]
        by_video.setdefault(v, []).append(r.to_dict())
    for v, recs in by_video.items():
        save_json(os.path.join(rallies_dir, v, "rallies.json"), recs)


def main():
    ap = argparse.ArgumentParser(description="Run the full pipeline.")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "pipeline.yaml"))
    ap.add_argument("--video",  default=None,
                    help="Process a single video. Default: video.source_dir.")
    ap.add_argument("--skip",   nargs="*", type=int, default=[],
                    help="Stage numbers to skip (load output from disk).")
    ap.add_argument("--only",   nargs="*", type=int, default=None,
                    help="If set, only run these stages (others must be on disk).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for d in (cfg.storage.rallies_dir, cfg.storage.joints_dir,
              cfg.storage.shuttle_dir, cfg.storage.hit_events_dir,
              cfg.storage.features_dir, cfg.storage.hit_images_dir,
              cfg.storage.strokes_csv_dir, cfg.storage.analytics_dir):
        ensure_dir(d)

    def should_run(stage):
        if args.only is not None:
            return stage in args.only
        return stage not in args.skip

    rallies, hit_events, records = [], [], []

    # ---------- Stage 1 ----------
    if should_run(1):
        _banner("Stage 1 — Ingestion & Rally Segmentation")
        from stage_1_ingestion_rally_segmentation import SegmentationModule
        seg = SegmentationModule(cfg)
        t0 = time.time()
        if args.video:
            rallies = seg.process(args.video)
        else:
            rallies = seg.process_dir(cfg.video.source_dir)
        print(f"[Stage 1] {len(rallies)} rallies in {time.time()-t0:.1f}s")
    else:
        rallies = _load_rallies(cfg.storage.rallies_dir)
        print(f"[Stage 1] loaded {len(rallies)} rallies from disk")

    # ---------- Stage 2 ----------
    if should_run(2) and rallies:
        _banner("Stage 2 — Vision Pipeline (court + pose + shuttle)")
        from stage_2_vision_pipeline import VisionModule
        vm = VisionModule(cfg)
        rallies = vm.process_all(rallies)
        _persist_rallies(rallies, cfg.storage.rallies_dir)
        print(f"[Stage 2] enriched {len(rallies)} rallies")
    else:
        rallies = _load_rallies(cfg.storage.rallies_dir)

    # ---------- Stage 3 ----------
    if should_run(3) and rallies:
        _banner("Stage 3 — Hit-Frame Detection")
        from stage_3_hit_frame_detection import HitDetector
        det = HitDetector(cfg)
        hit_events = det.detect_all(rallies)
        print(f"[Stage 3] {len(hit_events)} hit events")
    else:
        p = os.path.join(cfg.storage.hit_events_dir, "hit_events.json")
        hit_events = [HitEvent(**r) for r in load_json(p)] if os.path.exists(p) else []
        print(f"[Stage 3] loaded {len(hit_events)} events from disk")

    # ---------- Stage 4 ----------
    out_csv = os.path.join(cfg.storage.strokes_csv_dir, "strokes.csv")
    if should_run(4) and rallies and hit_events:
        _banner("Stage 4 — Stroke Classification + ViT Attributes")
        from stage_4_stroke_classification import StrokePipeline
        sp = StrokePipeline(cfg)
        records = sp.run(rallies, hit_events)
        sp.export_csv(records, out_csv)
        print(f"[Stage 4] wrote {len(records)} stroke rows → {out_csv}")
    else:
        print(f"[Stage 4] using existing CSV → {out_csv}")

    # ---------- Stage 5 ----------
    if should_run(5):
        _banner("Stage 5 — Analytics & Insights")
        import subprocess
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "stage_5_analytics", "run.py"),
             "--config", args.config, "--strokes_csv", out_csv],
            check=True,
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
