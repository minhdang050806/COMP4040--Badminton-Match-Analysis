# data_storage

Centralised, gitignored data root for the pipeline. **Do not commit large
binaries here.** Subdirectories:

```
raw_videos/                    # input MP4s + extracted KSeq archives
intermediate/
  rallies/<video>/rallies.json # Stage 1 → Stage 2 hand-off (RallySegment[])
  rallies/<video>/clips/*.mp4  # exported per-rally MP4 clips
  joints/rally_<id>.json       # Stage 2 court+human keypoint output
  shuttle/rally_<id>.csv       # Stage 2 TrackNetV2 shuttle trajectory
  hit_events/hit_events.json   # Stage 3 → Stage 4 (HitEvent[])
  features_npy/                # Stage 4 BST input tensors (training only)
  hit_frame_images/            # Stage 4 ViT-input crops
outputs/
  strokes_csv/strokes.csv      # Stage 4 → Stage 5 structured records
  analytics/                   # Stage 5 player clusters, patterns, heatmaps
weights/                       # all model checkpoints referenced in pipeline.yaml
```

## Required model weights

Place these in `data_storage/weights/`:

| File | Source |
|---|---|
| `sacnn.pt` | from `Automated-Hit-frame-Detection-for-Badminton-Match-Analysis/src/models/weights/` |
| `scaler.pickle` | same |
| `kpRCNN.pth` | same (download from project release) |
| `court_kpRCNN.pth` | same |
| `opt.pt` | same |
| `tracknet.pt` | from `A-New-Perspective-for-Shuttlecock-Hitting-Event-Detection` |
| `bst_cg_ap.pt` | trained via `data_preparation/prepare_bst_dataset.py` + `bst/main_on_shuttleset/bst_main.py` |
| `vit_<attr>/` | per-attribute ViT-B_16 ensemble checkpoint dirs (Hitter, Backhand, BallType, BallHeight, Winner) |
| `yolov8x-pose-p6.pt` | Ultralytics |
