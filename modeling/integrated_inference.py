from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from modeling.inference import build_model
    from modeling.label_names import get_merged_stroke_types_english
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from modeling.inference import build_model
    from modeling.label_names import get_merged_stroke_types_english


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = ROOT / "project" / "outputs" / "integration" / "bst_tracked_phase09_41_44" / "inputs"
DEFAULT_CHECKPOINT = ROOT / "project" / "weights" / "bst_shuttleset_finetuned_protocol.pt"
DEFAULT_OUTPUT_ROOT = ROOT / "project" / "outputs" / "integration" / "bst_tracked_phase09_41_44"
DEFAULT_BASELINE_SUMMARY = ROOT / "project" / "outputs" / "integration" / "baseline_old_phase09" / "phase10_summary.json"

PREDICTION_COLUMNS = [
    "row_index",
    "video_id",
    "rally_id",
    "event_rank",
    "clip_id",
    "feature_alias_source_clip_id",
    "window_start_original",
    "window_end_original",
    "event_frame_original",
    "event_frame_source",
    "feature_source",
    "window_strategy",
    "player_order",
    "event_offset_frames",
    "quality_group",
    "zero_pose_frames",
    "zero_pose_rate",
    "pose_missing_rate",
    "position_missing_rate",
    "p1_pose_valid_rate",
    "p2_pose_valid_rate",
    "p1_position_valid_rate",
    "p2_position_valid_rate",
    "both_pose_missing_rate",
    "both_position_missing_rate",
    "pose_tensor_zero_rate",
    "position_tensor_zero_rate",
    "shuttle_visible_rate",
    "shuttle_visible_frames",
    "player_pos_out_of_unit_range",
    "player_side_inconsistent_rate",
    "has_phase02_label",
    "stable_key",
    "match_id",
    "set_id",
    "gt_rally_id",
    "ball_round_id",
    "stroke_type_ground_truth",
    "player",
    "server",
    "true_label",
    "true_label_name",
    "reference_split",
    "predicted_label",
    "predicted_label_name",
    "predicted_player_side",
    "predicted_stroke_type",
    "confidence",
    "correct",
    "side_correct",
    "stroke_type_correct",
    "top3_correct",
    "top2_label_name",
    "top2_score",
    "top3_label_name",
    "top3_score",
    "original_len",
    "video_len_after_collation",
    "joints_npy",
    "pos_npy",
    "shuttle_npy",
    "source_video",
    "validity_npz",
]


class Phase10Dataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.pose = np.load(root / "JnB_bone.npy", mmap_mode="r")
        self.pos = np.load(root / "pos.npy", mmap_mode="r")
        self.shuttle = np.load(root / "shuttle.npy", mmap_mode="r")
        self.videos_len = np.load(root / "videos_len.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.videos_len)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.int64, int]:
        return self.pose[index], self.pos[index], self.shuttle[index], self.videos_len[index], index


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    rows.sort(key=lambda row: int(row["row_index"]))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def select_device(name: str, gpu_id: int) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if name == "cuda":
            raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is False.")
        return torch.device("cpu")
    if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
        raise RuntimeError(f"Requested GPU {gpu_id}, but only {torch.cuda.device_count()} device(s) are visible.")
    torch.cuda.set_device(gpu_id)
    return torch.device(f"cuda:{gpu_id}")


def split_class_name(name: str) -> tuple[str, str]:
    if "_" not in name:
        return "", name
    side, stroke = name.split("_", 1)
    return side, stroke


@torch.no_grad()
def infer(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    indices: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    video_lens: list[np.ndarray] = []
    for human_pose, pos, shuttle, video_len, index in loader:
        human_pose = human_pose.to(device=device, dtype=torch.float32, non_blocking=True)
        pos = pos.to(device=device, dtype=torch.float32, non_blocking=True)
        shuttle = shuttle.to(device=device, dtype=torch.float32, non_blocking=True)
        video_len = video_len.to(device=device, dtype=torch.long, non_blocking=True)
        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        batch_logits = model(human_pose, shuttle, pos, video_len)
        indices.append(index.numpy())
        logits.append(batch_logits.cpu().numpy())
        video_lens.append(video_len.cpu().numpy())
    return np.concatenate(indices), np.concatenate(logits), np.concatenate(video_lens)


def make_prediction_rows(
    metadata: list[dict[str, str]],
    class_names: list[str],
    indices: np.ndarray,
    logits: np.ndarray,
) -> list[dict[str, Any]]:
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    top3 = np.argsort(-probabilities, axis=1)[:, :3]
    rows: list[dict[str, Any]] = []
    for offset, row_index in enumerate(indices):
        meta = metadata[int(row_index)]
        predicted = int(top3[offset, 0])
        true_label = int(meta["true_label"])
        side, stroke = split_class_name(class_names[predicted])
        true_side, true_stroke = split_class_name(meta["true_label_name"])
        top3_labels = [int(label) for label in top3[offset]]
        rows.append(
            {
                **meta,
                "predicted_label": predicted,
                "predicted_label_name": class_names[predicted],
                "predicted_player_side": side,
                "predicted_stroke_type": stroke,
                "confidence": float(probabilities[offset, predicted]),
                "correct": predicted == true_label if true_label >= 0 else "",
                "side_correct": side == true_side if true_label >= 0 else "",
                "stroke_type_correct": stroke == true_stroke if true_label >= 0 else "",
                "top3_correct": true_label in top3_labels if true_label >= 0 else "",
                "top2_label_name": class_names[int(top3[offset, 1])],
                "top2_score": float(probabilities[offset, top3[offset, 1]]),
                "top3_label_name": class_names[int(top3[offset, 2])],
                "top3_score": float(probabilities[offset, top3[offset, 2]]),
            }
        )
    rows.sort(key=lambda row: int(row["row_index"]))
    return rows


def evaluation_metrics(rows: list[dict[str, Any]], class_count: int) -> dict[str, Any]:
    labeled = [row for row in rows if int(row["true_label"]) >= 0]
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for row in labeled:
        confusion[int(row["true_label"]), int(row["predicted_label"])] += 1
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = true_positive / np.maximum(predicted, 1)
    recall = true_positive / np.maximum(support, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    supported = support > 0
    return {
        "rows": len(rows),
        "labeled_rows": len(labeled),
        "accuracy": float(true_positive.sum() / max(confusion.sum(), 1)),
        "top3_accuracy": (
            float(sum(str(row.get("top3_correct", "")).lower() == "true" for row in labeled) / len(labeled))
            if labeled
            else None
        ),
        "player_side_accuracy": (
            float(sum(str(row.get("side_correct", "")).lower() == "true" for row in labeled) / len(labeled))
            if labeled
            else None
        ),
        "stroke_type_accuracy": (
            float(sum(str(row.get("stroke_type_correct", "")).lower() == "true" for row in labeled) / len(labeled))
            if labeled
            else None
        ),
        "macro_precision_supported": float(precision[supported].mean()) if np.any(supported) else None,
        "macro_recall_supported": float(recall[supported].mean()) if np.any(supported) else None,
        "macro_f1_supported": float(f1[supported].mean()) if np.any(supported) else None,
        "supported_class_count": int(supported.sum()),
    }


def evaluation_rows(predictions: list[dict[str, Any]], class_count: int) -> list[dict[str, Any]]:
    def value(row: dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default))
        except (TypeError, ValueError):
            return default

    def clean(row: dict[str, Any]) -> bool:
        return row.get("feature_alias_source_clip_id", "") == ""

    groups: list[tuple[str, list[dict[str, Any]]]] = [
        ("all_candidates", predictions),
        ("exact_label_all", [row for row in predictions if int(row["true_label"]) >= 0]),
        ("exact_label_clean", [row for row in predictions if int(row["true_label"]) >= 0 and clean(row)]),
        ("simultaneous_event_alias_diagnostic", [row for row in predictions if not clean(row)]),
        ("reference_test_primary", [row for row in predictions if row["reference_split"] == "test" and clean(row)]),
        ("reference_test_all_including_aliases", [row for row in predictions if row["reference_split"] == "test"]),
        ("reference_val_diagnostic", [row for row in predictions if row["reference_split"] == "val" and clean(row)]),
        ("reference_val_all_including_aliases", [row for row in predictions if row["reference_split"] == "val"]),
        (
            "exact_label_pose_dropout_lt_50",
            [row for row in predictions if int(row["true_label"]) >= 0 and value(row, "pose_missing_rate") < 0.5],
        ),
        (
            "reference_test_pose_dropout_lt_50",
            [
                row
                for row in predictions
                if row["reference_split"] == "test" and clean(row) and value(row, "pose_missing_rate") < 0.5
            ],
        ),
        (
            "exact_label_pose_dropout_lt_25",
            [row for row in predictions if int(row["true_label"]) >= 0 and value(row, "pose_missing_rate") < 0.25],
        ),
        (
            "reference_test_pose_dropout_lt_25",
            [
                row
                for row in predictions
                if row["reference_split"] == "test" and clean(row) and value(row, "pose_missing_rate") < 0.25
            ],
        ),
        (
            "exact_label_all_pose_missing",
            [row for row in predictions if int(row["true_label"]) >= 0 and value(row, "pose_missing_rate") == 1.0],
        ),
        (
            "reference_test_top_pose_valid_ge_75",
            [
                row
                for row in predictions
                if row["reference_split"] == "test" and clean(row) and value(row, "p1_pose_valid_rate") >= 0.75
            ],
        ),
        (
            "reference_test_top_pose_valid_lt_50",
            [
                row
                for row in predictions
                if row["reference_split"] == "test" and clean(row) and value(row, "p1_pose_valid_rate") < 0.50
            ],
        ),
        (
            "reference_test_bottom_pose_valid_ge_75",
            [
                row
                for row in predictions
                if row["reference_split"] == "test" and clean(row) and value(row, "p2_pose_valid_rate") >= 0.75
            ],
        ),
        (
            "reference_test_both_pose_missing_lt_10",
            [
                row
                for row in predictions
                if row["reference_split"] == "test" and clean(row) and value(row, "both_pose_missing_rate") < 0.10
            ],
        ),
    ]
    for video_id in sorted({row["video_id"] for row in predictions}, key=int):
        groups.append(
            (
                f"exact_label_video_{video_id}",
                [
                    row
                    for row in predictions
                    if row["video_id"] == video_id and int(row["true_label"]) >= 0 and clean(row)
                ],
            )
        )
    return [{"evaluation_group": name, **evaluation_metrics(rows, class_count)} for name, rows in groups]


def class_evaluation_rows(predictions: list[dict[str, Any]], class_names: list[str]) -> list[dict[str, Any]]:
    labeled = [
        row
        for row in predictions
        if int(row["true_label"]) >= 0 and row.get("feature_alias_source_clip_id", "") == ""
    ]
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for row in labeled:
        confusion[int(row["true_label"]), int(row["predicted_label"])] += 1
    rows: list[dict[str, Any]] = []
    for label, name in enumerate(class_names):
        support = int(confusion[label].sum())
        predicted = int(confusion[:, label].sum())
        true_positive = int(confusion[label, label])
        precision = true_positive / max(predicted, 1)
        recall = true_positive / max(support, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                "label": label,
                "class_name": name,
                "support": support,
                "predicted_count": predicted,
                "true_positive": true_positive,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def primary_from_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("primary_evaluation"):
        return summary["primary_evaluation"]
    return next(
        (row for row in summary.get("evaluation", []) if row.get("evaluation_group") == "reference_test_primary"),
        None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 10 BST inference over prepared Phase 09 features.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs/model/checkpoint without running inference.")
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    device = select_device(args.device, args.gpu_id)
    metadata = read_csv(args.input_root / "phase10_input_metadata.csv")
    dataset = Phase10Dataset(args.input_root)
    if len(metadata) != len(dataset):
        raise RuntimeError(f"Metadata rows {len(metadata)} != prepared rows {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    class_names = get_merged_stroke_types_english()
    model = build_model(seq_len=100, n_classes=len(class_names), device=device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={missing}, unexpected={unexpected}")
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validation_passed",
                    "input_root": relative(args.input_root),
                    "checkpoint": relative(args.checkpoint),
                    "rows": len(dataset),
                    "metadata_rows": len(metadata),
                    "device": str(device),
                    "class_count": len(class_names),
                },
                indent=2,
            )
        )
        return

    started = time.time()
    indices, logits, video_lens = infer(model, loader, device)
    elapsed = time.time() - started
    expected_video_lens = np.asarray([int(metadata[int(index)]["video_len_after_collation"]) for index in indices])
    if not np.array_equal(video_lens, expected_video_lens):
        raise RuntimeError("Inferred video-length order does not match Phase 10 metadata.")
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predictions = make_prediction_rows(metadata, class_names, indices, logits)
    evaluation = evaluation_rows(predictions, len(class_names))
    class_evaluation = class_evaluation_rows(predictions, class_names)
    primary_evaluation = next(
        (row for row in evaluation if row["evaluation_group"] == "reference_test_primary"),
        None,
    )
    baseline_primary = primary_from_summary(args.baseline_summary)
    baseline_comparison = None
    if primary_evaluation is not None and baseline_primary is not None:
        baseline_comparison = {
            "baseline_summary": relative(args.baseline_summary),
            "baseline_primary": baseline_primary,
            "accuracy_delta": primary_evaluation["accuracy"] - baseline_primary["accuracy"],
            "macro_f1_supported_delta": (
                primary_evaluation["macro_f1_supported"] - baseline_primary["macro_f1_supported"]
            ),
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_root / "phase10_predictions.csv"
    structured_path = args.output_root / "phase10_structured_strokes.csv"
    evaluation_path = args.output_root / "phase10_evaluation_by_group.csv"
    class_evaluation_path = args.output_root / "phase10_evaluation_by_class.csv"
    logits_path = args.output_root / "phase10_logits.npy"
    probabilities_path = args.output_root / "phase10_probabilities.npy"
    summary_path = args.output_root / "phase10_summary.json"
    write_csv(prediction_path, predictions, PREDICTION_COLUMNS)
    structured = sorted(
        predictions,
        key=lambda row: (
            int(row["video_id"]),
            int(row["window_start_original"]),
            int(row["window_end_original"]),
            int(row["row_index"]),
        ),
    )
    write_csv(structured_path, structured, PREDICTION_COLUMNS)
    write_csv(
        evaluation_path,
        evaluation,
        [
            "evaluation_group",
            "rows",
            "labeled_rows",
            "accuracy",
            "top3_accuracy",
            "player_side_accuracy",
            "stroke_type_accuracy",
            "macro_precision_supported",
            "macro_recall_supported",
            "macro_f1_supported",
            "supported_class_count",
        ],
    )
    write_csv(
        class_evaluation_path,
        class_evaluation,
        ["label", "class_name", "support", "predicted_count", "true_positive", "precision", "recall", "f1"],
    )
    np.save(logits_path, logits)
    np.save(probabilities_path, probabilities)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "modeling/integrated_inference.py",
        "status": "passed" if len(predictions) == len(metadata) else "review",
        "input_root": relative(args.input_root),
        "checkpoint": relative(args.checkpoint),
        "output_root": relative(args.output_root),
        "device": str(device),
        "torch_version": torch.__version__,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "elapsed_seconds": elapsed,
        "prediction_rows": len(predictions),
        "labeled_evaluation_rows": sum(int(row["true_label"]) >= 0 for row in predictions),
        "candidate_only_rows": sum(int(row["true_label"]) < 0 for row in predictions),
        "class_count": len(class_names),
        "evaluation": evaluation,
        "primary_evaluation": primary_evaluation,
        "baseline_comparison": baseline_comparison,
        "outputs": {
            "predictions": relative(prediction_path),
            "structured_strokes": relative(structured_path),
            "evaluation_by_group": relative(evaluation_path),
            "evaluation_by_class": relative(class_evaluation_path),
            "logits": relative(logits_path),
            "probabilities": relative(probabilities_path),
            "summary": relative(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": relative(args.output_root), "rows": len(predictions), "status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()
