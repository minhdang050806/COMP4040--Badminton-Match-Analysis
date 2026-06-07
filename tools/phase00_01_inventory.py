from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "project" / "outputs" / "inventory"


def run_command(args: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
        return {
            "command": args,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except FileNotFoundError as exc:
        return {"command": args, "returncode": None, "stdout": "", "stderr": str(exc)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path, role: str) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path.relative_to(ROOT)),
        "role": role,
        "exists": exists,
        "is_file": path.is_file() if exists else False,
        "is_dir": path.is_dir() if exists else False,
        "size_bytes": path.stat().st_size if exists and path.is_file() else None,
        "sha256": sha256(path) if exists and path.is_file() and path.stat().st_size < 1024 * 1024 * 1024 else None,
    }


def package_status() -> list[dict[str, Any]]:
    packages = [
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "cv2",
        "mmpose",
        "mmcv",
        "mmengine",
        "scipy",
        "matplotlib",
        "PIL",
        "yaml",
        "sklearn",
    ]
    rows: list[dict[str, Any]] = []
    for name in packages:
        present = importlib.util.find_spec(name) is not None
        version = None
        if present:
            try:
                module = __import__(name)
                version = getattr(module, "__version__", None)
            except Exception as exc:  # noqa: BLE001 - inventory must record import failures.
                version = f"import_error: {exc}"
        rows.append({"package": name, "present": present, "version": version})
    return rows


def torch_status() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"present": False}
    import torch

    return {
        "present": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def repo_inventory() -> list[dict[str, Any]]:
    repos = [
        ("external_repos/BST-Badminton-Stroke-type-Transformer", "BST code and data-prep scripts"),
        ("external_repos/TrackNetV3", "shuttle tracking and hit-frame scripts"),
        ("external_repos/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis", "SA-CNN rally filtering source"),
        ("external_repos/monotrack", "court detection source"),
        ("external_repos/mmpose", "pose estimation source"),
    ]
    rows: list[dict[str, Any]] = []
    for rel, role in repos:
        path = ROOT / rel
        rec = file_record(path, role)
        rec["git_head"] = None
        if path.is_dir() and (path / ".git").exists():
            result = run_command(["git", "-C", rel, "rev-parse", "--short", "HEAD"])
            rec["git_head"] = result["stdout"] if result["returncode"] == 0 else None
        rows.append(rec)
    return rows


def checkpoint_inventory() -> list[dict[str, Any]]:
    checkpoints = [
        (
            "external_repos/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis/src/models/weights/sacnn.pt",
            "Phase 05 rally / shot-angle filtering",
        ),
        (
            "external_repos/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis/src/models/weights/scaler.pickle",
            "legacy Automated-Hit-frame pipeline scaler; not needed for standalone SA-CNN",
        ),
        ("external_repos/TrackNetV3/exp/model_best.pt", "Phase 06 shuttle tracking"),
        (
            "project/weights/on_ShuttleSet/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt",
            "Phase 04 / Phase 10 BST_CG_AP stroke classification",
        ),
    ]
    rows = [file_record(ROOT / rel, role) for rel, role in checkpoints]
    bst_dir = ROOT / "project" / "weights" / "on_ShuttleSet"
    rows.append(
        {
            "path": str(bst_dir.relative_to(ROOT)),
            "role": "all local ShuttleSet BST/baseline weights",
            "exists": bst_dir.exists(),
            "is_file": False,
            "is_dir": bst_dir.is_dir(),
            "size_bytes": None,
            "sha256": None,
            "pt_file_count": len(list(bst_dir.glob("*.pt"))) if bst_dir.is_dir() else 0,
        }
    )
    return rows


def raw_video_manifest() -> list[dict[str, Any]]:
    raw_dir = ROOT / "project" / "dataset" / "ShuttleSet_raw_videos"
    rows: list[dict[str, Any]] = []
    for mp4 in sorted(raw_dir.glob("*.mp4")):
        info = mp4.with_suffix(".info.json")
        rows.append(
            {
                "video_file": str(mp4.relative_to(ROOT)),
                "size_bytes": mp4.stat().st_size,
                "info_json": str(info.relative_to(ROOT)) if info.exists() else None,
                "info_json_exists": info.exists(),
                "id_prefix": mp4.name.split(" - ", 1)[0],
            }
        )
    return rows


def annotation_inventory() -> dict[str, Any]:
    set_root = ROOT / "project" / "dataset" / "ShuttleSet" / "set"
    match_csv = set_root / "match.csv"
    homography_csv = set_root / "homography.csv"
    set_csvs = sorted(set_root.glob("* /set*.csv"))
    if not set_csvs:
        set_csvs = sorted(set_root.glob("*/set*.csv"))

    set_rows = []
    stroke_rows = 0
    for path in set_csvs:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = sum(1 for _ in reader)
            fieldnames = reader.fieldnames or []
        stroke_rows += count
        set_rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "match_name": path.parent.name,
                "set_file": path.name,
                "row_count": count,
                "column_count": len(fieldnames),
            }
        )

    match_df = pd.read_csv(match_csv)
    homography_df = pd.read_csv(homography_csv)
    return {
        "summary": {
            "match_rows": int(len(match_df)),
            "homography_rows": int(len(homography_df)),
            "match_directories": len([p for p in set_root.iterdir() if p.is_dir()]),
            "set_csv_files": len(set_csvs),
            "stroke_rows": stroke_rows,
            "match_csv_sha256": sha256(match_csv),
            "homography_csv_sha256": sha256(homography_csv),
            "match_columns": list(match_df.columns),
            "homography_columns": list(homography_df.columns),
        },
        "set_csvs": set_rows,
    }


def feature_inventory() -> dict[str, Any]:
    root = ROOT / "project" / "dataset" / "ShuttleSet"
    variants = sorted([p for p in root.iterdir() if p.is_dir() and p.name != "set"])
    split_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []

    for variant in variants:
        for split in ("train", "val", "test"):
            split_dir = variant / split
            if not split_dir.exists():
                continue
            classes = sorted([p for p in split_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
            joints = sorted(split_dir.glob("*/*_joints.npy"))
            pos = sorted(split_dir.glob("*/*_pos.npy"))
            shuttle = sorted(split_dir.glob("*/*_shuttle.npy"))
            branches = {str(p).removesuffix("_joints.npy") for p in joints}
            branches.update(str(p).removesuffix("_pos.npy") for p in pos)
            branches.update(str(p).removesuffix("_shuttle.npy") for p in shuttle)
            missing = 0
            for branch in branches:
                if not Path(branch + "_joints.npy").exists():
                    missing += 1
                if not Path(branch + "_pos.npy").exists():
                    missing += 1
                if not Path(branch + "_shuttle.npy").exists():
                    missing += 1

            sample_branch = None
            sample_shapes: dict[str, Any] = {"joints": None, "pos": None, "shuttle": None}
            if joints:
                sample_branch = str(joints[0]).removesuffix("_joints.npy")
                sample_shapes["joints"] = list(np.load(sample_branch + "_joints.npy", mmap_mode="r").shape)
                sample_shapes["pos"] = list(np.load(sample_branch + "_pos.npy", mmap_mode="r").shape)
                sample_shapes["shuttle"] = list(np.load(sample_branch + "_shuttle.npy", mmap_mode="r").shape)

            split_rows.append(
                {
                    "variant": variant.name,
                    "split": split,
                    "classes": len(classes),
                    "joints_files": len(joints),
                    "pos_files": len(pos),
                    "shuttle_files": len(shuttle),
                    "clip_branches": len(branches),
                    "missing_triple_files": missing,
                    "sample_branch": str(Path(sample_branch).relative_to(ROOT)) if sample_branch else None,
                    "sample_joints_shape": sample_shapes["joints"],
                    "sample_pos_shape": sample_shapes["pos"],
                    "sample_shuttle_shape": sample_shapes["shuttle"],
                }
            )

            for cls in classes:
                class_rows.append(
                    {
                        "variant": variant.name,
                        "split": split,
                        "class_name": cls.name,
                        "clip_count": len(list(cls.glob("*_joints.npy"))),
                    }
                )

    return {"splits": split_rows, "classes": class_rows}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    env = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "conda_default_env": run_command(["printenv", "CONDA_DEFAULT_ENV"])["stdout"] or None,
        "cuda_visible_devices": run_command(["printenv", "CUDA_VISIBLE_DEVICES"])["stdout"] or None,
        "commands": {
            "uname": run_command(["uname", "-a"]),
            "gcc": run_command(["gcc", "--version"]),
            "cmake": run_command(["cmake", "--version"]),
            "ffmpeg": run_command(["ffmpeg", "-version"]),
            "nvidia_smi": run_command(["nvidia-smi"]),
        },
        "packages": package_status(),
        "torch": torch_status(),
    }
    repos = repo_inventory()
    checkpoints = checkpoint_inventory()
    raw_videos = raw_video_manifest()
    annotations = annotation_inventory()
    features = feature_inventory()

    summary = {
        "generated_at_utc": env["generated_at_utc"],
        "raw_video_count": len(raw_videos),
        "raw_info_json_count": sum(1 for row in raw_videos if row["info_json_exists"]),
        **annotations["summary"],
        "feature_variants": sorted({row["variant"] for row in features["splits"]}),
        "feature_split_rows": features["splits"],
        "checkpoint_rows": checkpoints,
        "repo_rows": repos,
        "torch_cuda_available": env["torch"].get("cuda_available"),
        "torch_cuda_device_count": env["torch"].get("cuda_device_count"),
    }

    (OUT / "phase00_environment_baseline.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "phase01_dataset_inventory_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(OUT / "phase00_external_repos.csv", repos)
    write_csv(OUT / "phase00_checkpoint_inventory.csv", checkpoints)
    write_csv(OUT / "phase01_raw_video_manifest.csv", raw_videos)
    write_csv(OUT / "phase01_set_csv_inventory.csv", annotations["set_csvs"])
    write_csv(OUT / "phase01_feature_split_inventory.csv", features["splits"])
    write_csv(OUT / "phase01_feature_class_inventory.csv", features["classes"])

    print(json.dumps({
        "output_dir": str(OUT.relative_to(ROOT)),
        "raw_videos": len(raw_videos),
        "set_csv_files": annotations["summary"]["set_csv_files"],
        "stroke_rows": annotations["summary"]["stroke_rows"],
        "feature_split_rows": len(features["splits"]),
        "feature_class_rows": len(features["classes"]),
        "torch_cuda_available": env["torch"].get("cuda_available"),
    }, indent=2))


if __name__ == "__main__":
    main()
