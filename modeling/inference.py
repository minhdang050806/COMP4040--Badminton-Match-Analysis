from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
BST_ROOT = ROOT / "external_repos" / "BST-Badminton-Stroke-type-Transformer" / "stroke_classification"
DEFAULT_COLLATED = ROOT / "project" / "outputs" / "bst_collated" / "merged_seq100_between_2_hits_with_max_limits"
DEFAULT_CHECKPOINT = ROOT / "project" / "weights" / "on_ShuttleSet" / "bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt"
DEFAULT_PHASE02_TABLE = ROOT / "project" / "outputs" / "tables" / "shuttleset_ground_truth_strokes.csv"
DEFAULT_OUTPUT_PREDICTIONS = ROOT / "project" / "outputs" / "predictions"
DEFAULT_OUTPUT_TABLES = ROOT / "project" / "outputs" / "tables"


def install_positional_encoding_shim() -> None:
    """Provide the tiny API that upstream BST imports when the package is absent.

    The real checkpoint overwrites the embedding tensors immediately after model
    construction, so the shim only needs to make initialization possible.
    """

    if "positional_encodings.torch_encodings" in sys.modules:
        return

    package = types.ModuleType("positional_encodings")
    submodule = types.ModuleType("positional_encodings.torch_encodings")

    class PositionalEncoding1D:
        def __init__(self, channels: int) -> None:
            self.channels = channels

        def __call__(self, tensor: Tensor) -> Tensor:
            # tensor shape is (1, length, channels) in this project.
            _, length, channels = tensor.shape
            device = tensor.device
            dtype = tensor.dtype
            position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, channels, 2, device=device, dtype=dtype)
                * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / max(channels, 1))
            )
            pe = torch.zeros((1, length, channels), device=device, dtype=dtype)
            pe[0, :, 0::2] = torch.sin(position * div_term[: pe[0, :, 0::2].shape[-1]])
            if channels > 1:
                pe[0, :, 1::2] = torch.cos(position * div_term[: pe[0, :, 1::2].shape[-1]])
            return pe

    submodule.PositionalEncoding1D = PositionalEncoding1D
    package.torch_encodings = submodule
    sys.modules["positional_encodings"] = package
    sys.modules["positional_encodings.torch_encodings"] = submodule


def get_merged_stroke_types() -> list[str]:
    class_ls = [
        "放小球",
        "擋小球",
        "殺球",
        "挑球",
        "長球",
        "平球",
        "切球",
        "推球",
        "撲球",
        "勾球",
        "發短球",
        "發長球",
    ]
    return ["未知球種"] + ["Top_" + s for s in class_ls] + ["Bottom_" + s for s in class_ls]


class CollatedBstDataset(Dataset):
    def __init__(self, root: Path, split: str, pose_style: str) -> None:
        split_dir = root / split
        self.human_pose = np.load(split_dir / f"{pose_style}.npy", mmap_mode="r")
        self.pos = np.load(split_dir / "pos.npy", mmap_mode="r")
        self.shuttle = np.load(split_dir / "shuttle.npy", mmap_mode="r")
        self.videos_len = np.load(split_dir / "videos_len.npy", mmap_mode="r")
        self.labels = np.load(split_dir / "labels.npy", mmap_mode="r")

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, index: int) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.int64, np.int64, int]:
        return (
            self.human_pose[index],
            self.pos[index],
            self.shuttle[index],
        ), self.videos_len[index], self.labels[index], index


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_metadata(root: Path, split: str) -> list[dict[str, str]]:
    rows = read_csv(root / f"{split}_metadata.csv")
    rows.sort(key=lambda row: int(row["row_index"]))
    return rows


def build_model(seq_len: int, n_classes: int, device: torch.device) -> torch.nn.Module:
    install_positional_encoding_shim()
    sys.path.insert(0, str(BST_ROOT))
    from model.bst import BST_CG_AP
    from preparing_data.shuttleset_dataset import get_bone_pairs

    in_dim = (17 + len(get_bone_pairs())) * 2
    model = BST_CG_AP(
        in_dim=in_dim,
        n_class=n_classes,
        seq_len=seq_len,
        depth_tem=2,
        depth_inter=1,
    )
    return model.to(device)


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    indices_out: list[np.ndarray] = []
    video_lens_out: list[np.ndarray] = []
    with torch.no_grad():
        for (human_pose, pos, shuttle), video_len, labels, indices in loader:
            human_pose = human_pose.to(device=device, dtype=torch.float32)
            pos = pos.to(device=device, dtype=torch.float32)
            shuttle = shuttle.to(device=device, dtype=torch.float32)
            video_len = video_len.to(device=device, dtype=torch.long)
            labels = labels.to(device=device, dtype=torch.long)

            human_pose = human_pose.view(*human_pose.shape[:-2], -1)
            logits = model(human_pose, shuttle, pos, video_len)
            if logits.shape[-1] != n_classes:
                raise RuntimeError(f"Expected {n_classes} logits, got {logits.shape[-1]}")

            logits_out.append(logits.detach().cpu().numpy())
            labels_out.append(labels.detach().cpu().numpy())
            indices_out.append(indices.detach().cpu().numpy())
            video_lens_out.append(video_len.detach().cpu().numpy())

    return (
        np.concatenate(indices_out),
        np.concatenate(logits_out),
        np.concatenate(labels_out),
        np.concatenate(video_lens_out),
    )


def make_prediction_rows(
    metadata: list[dict[str, str]],
    class_names: list[str],
    indices: np.ndarray,
    logits: np.ndarray,
    true_labels: np.ndarray,
    video_lens: np.ndarray,
) -> list[dict[str, Any]]:
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred_labels = np.argmax(logits, axis=1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    rows: list[dict[str, Any]] = []
    for offset, row_index in enumerate(indices):
        meta = metadata[int(row_index)]
        clip_id = meta["clip_id"]
        parts = clip_id.split("_")
        stable_key = ":".join(parts[:4]) if len(parts) >= 4 else ""
        pred = int(pred_labels[offset])
        true = int(true_labels[offset])
        rows.append(
            {
                "row_index": int(row_index),
                "split": meta["split"],
                "clip_id": clip_id,
                "stable_key": stable_key,
                "source_branch": meta["source_branch"],
                "source_class_name": meta["class_name"],
                "true_label": true,
                "true_label_name": class_names[true],
                "predicted_label": pred,
                "predicted_label_name": class_names[pred],
                "confidence": float(probs[offset, pred]),
                "correct": pred == true,
                "video_len": int(video_lens[offset]),
                "original_len": int(meta["original_len"]),
                "top1_label": int(top3[offset, 0]),
                "top1_label_name": class_names[int(top3[offset, 0])],
                "top1_score": float(probs[offset, top3[offset, 0]]),
                "top2_label": int(top3[offset, 1]),
                "top2_label_name": class_names[int(top3[offset, 1])],
                "top2_score": float(probs[offset, top3[offset, 1]]),
                "top3_label": int(top3[offset, 2]),
                "top3_label_name": class_names[int(top3[offset, 2])],
                "top3_score": float(probs[offset, top3[offset, 2]]),
            }
        )
    rows.sort(key=lambda row: int(row["row_index"]))
    return rows


def confusion_rows(prediction_rows: list[dict[str, Any]], class_names: list[str]) -> list[dict[str, Any]]:
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for row in prediction_rows:
        matrix[int(row["true_label"]), int(row["predicted_label"])] += 1

    rows = []
    for true_idx, true_name in enumerate(class_names):
        out: dict[str, Any] = {"true_label": true_idx, "true_label_name": true_name}
        for pred_idx, pred_name in enumerate(class_names):
            out[f"pred_{pred_idx}_{pred_name}"] = int(matrix[true_idx, pred_idx])
        rows.append(out)
    return rows


def metric_rows(prediction_rows: list[dict[str, Any]], class_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    total = len(prediction_rows)
    correct_total = sum(1 for row in prediction_rows if row["correct"])
    rows.append(
        {
            "label": "ALL",
            "label_name": "ALL",
            "support": total,
            "correct": correct_total,
            "accuracy": correct_total / total if total else None,
        }
    )
    for label, label_name in enumerate(class_names):
        class_rows = [row for row in prediction_rows if int(row["true_label"]) == label]
        correct = sum(1 for row in class_rows if row["correct"])
        rows.append(
            {
                "label": label,
                "label_name": label_name,
                "support": len(class_rows),
                "correct": correct,
                "accuracy": correct / len(class_rows) if class_rows else None,
            }
        )
    return rows


def joined_phase02_rows(
    phase02_path: Path,
    prediction_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phase02_rows = read_csv(phase02_path)
    predictions_by_clip = {row["clip_id"]: row for row in prediction_rows}
    joined = []
    matched = 0
    for row in phase02_rows:
        pred = predictions_by_clip.get(row["clip_id"])
        out = dict(row)
        if pred is None:
            out.update(
                {
                    "bst_split": "",
                    "bst_true_label": "",
                    "bst_true_label_name": "",
                    "bst_predicted_label": "",
                    "bst_predicted_label_name": "",
                    "bst_confidence": "",
                    "bst_correct": "",
                    "bst_top2_label_name": "",
                    "bst_top2_score": "",
                    "bst_top3_label_name": "",
                    "bst_top3_score": "",
                }
            )
        else:
            matched += 1
            out.update(
                {
                    "bst_split": pred["split"],
                    "bst_true_label": pred["true_label"],
                    "bst_true_label_name": pred["true_label_name"],
                    "bst_predicted_label": pred["predicted_label"],
                    "bst_predicted_label_name": pred["predicted_label_name"],
                    "bst_confidence": pred["confidence"],
                    "bst_correct": pred["correct"],
                    "bst_top2_label_name": pred["top2_label_name"],
                    "bst_top2_score": pred["top2_score"],
                    "bst_top3_label_name": pred["top3_label_name"],
                    "bst_top3_score": pred["top3_score"],
                }
            )
        joined.append(out)
    return joined, {
        "phase02_rows": len(phase02_rows),
        "prediction_rows": len(prediction_rows),
        "matched_prediction_rows": matched,
        "unmatched_predictions": len(prediction_rows) - matched,
        "phase02_rows_without_prediction": len(phase02_rows) - matched,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 04 BST inference on collated ShuttleSet features.")
    parser.add_argument("--collated-root", type=Path, default=DEFAULT_COLLATED)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--phase02-table", type=Path, default=DEFAULT_PHASE02_TABLE)
    parser.add_argument("--output-predictions", type=Path, default=DEFAULT_OUTPUT_PREDICTIONS)
    parser.add_argument("--output-tables", type=Path, default=DEFAULT_OUTPUT_TABLES)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is False.")
        device = torch.device("cuda")
    elif args.device == "auto" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    class_names = get_merged_stroke_types()
    dataset = CollatedBstDataset(args.collated_root, args.split, "JnB_bone")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    metadata = load_metadata(args.collated_root, args.split)
    if len(metadata) != len(dataset):
        raise RuntimeError(f"Metadata rows {len(metadata)} != dataset rows {len(dataset)}")

    model = build_model(seq_len=100, n_classes=len(class_names), device=device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={missing}, unexpected={unexpected}")

    start = time.time()
    indices, logits, true_labels, video_lens = run_inference(model, loader, device, len(class_names))
    elapsed = time.time() - start
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    prediction_rows = make_prediction_rows(metadata, class_names, indices, logits, true_labels, video_lens)

    args.output_predictions.mkdir(parents=True, exist_ok=True)
    args.output_tables.mkdir(parents=True, exist_ok=True)

    prediction_csv = args.output_predictions / f"bst_{args.split}_predictions.csv"
    logits_npy = args.output_predictions / f"bst_{args.split}_logits.npy"
    probabilities_npy = args.output_predictions / f"bst_{args.split}_probabilities.npy"
    confusion_csv = args.output_predictions / f"bst_{args.split}_confusion_matrix.csv"
    metrics_csv = args.output_predictions / f"bst_{args.split}_metrics_by_class.csv"
    summary_json = args.output_predictions / f"phase04_bst_{args.split}_summary.json"
    joined_csv = args.output_tables / "shuttleset_strokes_with_bst_predictions.csv"

    prediction_columns = [
        "row_index",
        "split",
        "clip_id",
        "stable_key",
        "source_branch",
        "source_class_name",
        "true_label",
        "true_label_name",
        "predicted_label",
        "predicted_label_name",
        "confidence",
        "correct",
        "video_len",
        "original_len",
        "top1_label",
        "top1_label_name",
        "top1_score",
        "top2_label",
        "top2_label_name",
        "top2_score",
        "top3_label",
        "top3_label_name",
        "top3_score",
    ]
    write_csv(prediction_csv, prediction_rows, prediction_columns)
    np.save(logits_npy, logits)
    np.save(probabilities_npy, probabilities)
    write_csv(confusion_csv, confusion_rows(prediction_rows, class_names))
    metrics = metric_rows(prediction_rows, class_names)
    write_csv(metrics_csv, metrics, ["label", "label_name", "support", "correct", "accuracy"])

    joined, join_summary = joined_phase02_rows(args.phase02_table, prediction_rows)
    write_csv(joined_csv, joined)

    total = len(prediction_rows)
    correct = sum(1 for row in prediction_rows if row["correct"])
    confidence_values = [float(row["confidence"]) for row in prediction_rows]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "collated_root": str(args.collated_root.relative_to(ROOT)),
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "phase02_table": str(args.phase02_table.relative_to(ROOT)),
        "split": args.split,
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_threads": args.num_threads,
        "elapsed_seconds": elapsed,
        "prediction_rows": total,
        "correct_predictions": correct,
        "accuracy": correct / total if total else None,
        "class_count": len(class_names),
        "label_min": int(np.min(true_labels)),
        "label_max": int(np.max(true_labels)),
        "pred_min": int(np.min([row["predicted_label"] for row in prediction_rows])),
        "pred_max": int(np.max([row["predicted_label"] for row in prediction_rows])),
        "confidence_min": min(confidence_values),
        "confidence_max": max(confidence_values),
        "confidence_mean": sum(confidence_values) / len(confidence_values),
        "outputs": {
            "predictions_csv": str(prediction_csv.relative_to(ROOT)),
            "logits_npy": str(logits_npy.relative_to(ROOT)),
            "probabilities_npy": str(probabilities_npy.relative_to(ROOT)),
            "confusion_matrix_csv": str(confusion_csv.relative_to(ROOT)),
            "metrics_by_class_csv": str(metrics_csv.relative_to(ROOT)),
            "joined_phase02_csv": str(joined_csv.relative_to(ROOT)),
        },
        "join_summary": join_summary,
        "validation_status": "passed"
        if total == len(dataset)
        and join_summary["matched_prediction_rows"] == total
        and join_summary["unmatched_predictions"] == 0
        and 0 <= int(np.min([row["predicted_label"] for row in prediction_rows]))
        and int(np.max([row["predicted_label"] for row in prediction_rows])) < len(class_names)
        else "review",
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_predictions": str(args.output_predictions.relative_to(ROOT)),
                "prediction_rows": total,
                "accuracy": summary["accuracy"],
                "device": str(device),
                "elapsed_seconds": elapsed,
                "validation_status": summary["validation_status"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
