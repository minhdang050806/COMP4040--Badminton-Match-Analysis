"""Stage-1 entry point.

Usage:
    python -m stage_1_ingestion_rally_segmentation.run
    # or
    python stage_1_ingestion_rally_segmentation/run.py --config configs/pipeline.yaml
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.config import load_config                       # noqa: E402
from stage_1_ingestion_rally_segmentation.segmentation_module import (  # noqa: E402
    SegmentationModule,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "pipeline.yaml"))
    p.add_argument("--video", default=None,
                   help="Process a single video. If omitted, runs over video.source_dir.")
    args = p.parse_args()

    cfg = load_config(args.config)
    mod = SegmentationModule(cfg)
    if args.video:
        rallies = mod.process(args.video)
    else:
        rallies = mod.process_dir(cfg.video.source_dir)
    print(f"[Stage 1] Generated {len(rallies)} rally segments")


if __name__ == "__main__":
    main()
