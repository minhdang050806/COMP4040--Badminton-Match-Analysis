from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader

from phase04_run_bst_inference import (
    ROOT,
    DEFAULT_CHECKPOINT,
    DEFAULT_COLLATED,
    CollatedBstDataset,
    build_model,
    get_merged_stroke_types,
    write_csv,
)


DEFAULT_OUTPUT_ROOT = ROOT / "project" / "outputs" / "bst_training_all44"
DEFAULT_WEIGHT_OUT = ROOT / "project" / "weights" / "bst_shuttleset_finetuned_protocol.pt"
DEFAULT_POSE_STYLE = "JnB_bone"
DEFAULT_SEQ_LEN = 100
DEFAULT_SEED = 20260605


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_split_list(value: str) -> list[str]:
    splits = [item.strip() for item in value.split(",") if item.strip()]
    valid = {"train", "val", "test"}
    invalid = [split for split in splits if split not in valid]
    if invalid:
        raise ValueError(f"Invalid split(s): {invalid}; expected only train,val,test")
    if not splits:
        raise ValueError("Expected at least one split.")
    return splits


def select_device(device_name: str, gpu_id: int) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is False.")
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise RuntimeError(f"Requested --gpu-id {gpu_id}, but only {torch.cuda.device_count()} CUDA device(s) are visible.")
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    if torch.cuda.is_available():
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise RuntimeError(f"Requested --gpu-id {gpu_id}, but only {torch.cuda.device_count()} CUDA device(s) are visible.")
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def split_dataset(collated_root: Path, splits: list[str], pose_style: str) -> torch.utils.data.Dataset:
    datasets = [CollatedBstDataset(collated_root, split, pose_style) for split in splits]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def split_counts(collated_root: Path, splits: list[str], pose_style: str) -> dict[str, int]:
    return {split: len(CollatedBstDataset(collated_root, split, pose_style)) for split in splits}


def labels_from_dataset(dataset: torch.utils.data.Dataset) -> np.ndarray:
    if isinstance(dataset, CollatedBstDataset):
        return np.asarray(dataset.labels)
    if isinstance(dataset, ConcatDataset):
        return np.concatenate([labels_from_dataset(child) for child in dataset.datasets])
    raise TypeError(f"Unsupported dataset type for label extraction: {type(dataset).__name__}")


def class_weights(dataset: torch.utils.data.Dataset, n_classes: int, device: torch.device) -> Tensor:
    labels = labels_from_dataset(dataset)
    counts = np.bincount(labels.astype(np.int64), minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def apply_random_translation(human_pose: Tensor, max_abs_shift: float) -> Tensor:
    if max_abs_shift <= 0:
        return human_pose
    # Shape: [B, T, 2, 36, 2] for JnB_bone. Only the first 17 entries are joints;
    # the remaining 19 are bone vectors and should not receive coordinate shifts.
    joints = human_pose[..., :17, :]
    bones = human_pose[..., 17:, :]
    visible = (joints != 0.0).any(dim=-1, keepdim=True)
    shift = torch.empty(
        (human_pose.shape[0], 1, human_pose.shape[2], 1, 2),
        device=human_pose.device,
        dtype=human_pose.dtype,
    ).uniform_(-max_abs_shift, max_abs_shift)
    joints = torch.where(visible, joints + shift, joints)
    return torch.cat((joints, bones), dim=-2)


def move_batch(
    batch: tuple[tuple[Tensor, Tensor, Tensor], Tensor, Tensor, Tensor],
    device: torch.device,
    random_translation: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    (human_pose, pos, shuttle), video_len, labels, _ = batch
    human_pose = human_pose.to(device=device, dtype=torch.float32, non_blocking=True)
    pos = pos.to(device=device, dtype=torch.float32, non_blocking=True)
    shuttle = shuttle.to(device=device, dtype=torch.float32, non_blocking=True)
    video_len = video_len.to(device=device, dtype=torch.long, non_blocking=True)
    labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
    human_pose = apply_random_translation(human_pose, random_translation)
    human_pose = human_pose.view(*human_pose.shape[:-2], -1)
    return human_pose, pos, shuttle, video_len, labels


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    random_translation: float,
    grad_clip_norm: float,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    correct = 0
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        human_pose, pos, shuttle, video_len, labels = move_batch(batch, device, random_translation)
        optimizer.zero_grad(set_to_none=True)
        logits = model(human_pose, shuttle, pos, video_len)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        preds = torch.argmax(logits.detach(), dim=1)
        total_loss += float(loss.detach().cpu()) * labels.numel()
        total_examples += labels.numel()
        correct += int((preds == labels).sum().item())
    return {
        "loss": total_loss / total_examples if total_examples else 0.0,
        "accuracy": correct / total_examples if total_examples else 0.0,
        "examples": float(total_examples),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    n_classes: int,
    max_batches: int | None,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    correct = 0
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        human_pose, pos, shuttle, video_len, labels = move_batch(batch, device, random_translation=0.0)
        logits = model(human_pose, shuttle, pos, video_len)
        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        total_loss += float(loss.detach().cpu()) * labels.numel()
        total_examples += labels.numel()
        correct += int((preds == labels).sum().item())
        for true_label, pred_label in zip(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), strict=False):
            confusion[int(true_label), int(pred_label)] += 1

    per_class = per_class_metrics(confusion)
    return (
        {
            "loss": total_loss / total_examples if total_examples else 0.0,
            "accuracy": correct / total_examples if total_examples else 0.0,
            "macro_precision": float(np.mean([row["precision"] for row in per_class])),
            "macro_recall": float(np.mean([row["recall"] for row in per_class])),
            "macro_f1": float(np.mean([row["f1"] for row in per_class])),
            "examples": float(total_examples),
        },
        confusion,
    )


def per_class_metrics(confusion: np.ndarray) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for label in range(confusion.shape[0]):
        tp = float(confusion[label, label])
        fp = float(confusion[:, label].sum() - confusion[label, label])
        fn = float(confusion[label, :].sum() - confusion[label, label])
        support = float(confusion[label, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "label": float(label),
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "correct": tp,
            }
        )
    return rows


def confusion_rows(confusion: np.ndarray, class_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for true_idx, true_name in enumerate(class_names):
        row: dict[str, Any] = {"true_label": true_idx, "true_label_name": true_name}
        for pred_idx, pred_name in enumerate(class_names):
            row[f"pred_{pred_idx}_{pred_name}"] = int(confusion[true_idx, pred_idx])
        rows.append(row)
    return rows


def metrics_rows(confusion: np.ndarray, class_names: list[str]) -> list[dict[str, Any]]:
    per_class = per_class_metrics(confusion)
    rows: list[dict[str, Any]] = []
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    rows.append(
        {
            "label": "ALL",
            "label_name": "ALL",
            "support": total,
            "correct": correct,
            "precision": "",
            "recall": "",
            "f1": "",
            "accuracy": correct / total if total else 0.0,
        }
    )
    for row in per_class:
        label = int(row["label"])
        rows.append(
            {
                "label": label,
                "label_name": class_names[label],
                "support": int(row["support"]),
                "correct": int(row["correct"]),
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "accuracy": row["correct"] / row["support"] if row["support"] else 0.0,
            }
        )
    return rows


def load_initial_weights(model: torch.nn.Module, checkpoint: Path | None, device: torch.device) -> dict[str, Any]:
    if checkpoint is None:
        return {"init_checkpoint": None, "missing_keys": [], "unexpected_keys": []}
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={missing}, unexpected={unexpected}")
    return {
        "init_checkpoint": relative(checkpoint),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    columns = [
        "epoch",
        "elapsed_seconds",
        "lr",
        "train_loss",
        "train_accuracy",
        "train_examples",
        "val_loss",
        "val_accuracy",
        "val_macro_precision",
        "val_macro_recall",
        "val_macro_f1",
        "val_examples",
        "is_best",
    ]
    rows = []
    for item in history:
        rows.append(
            {
                "epoch": item["epoch"],
                "elapsed_seconds": item["elapsed_seconds"],
                "lr": item["lr"],
                "train_loss": item["train"]["loss"],
                "train_accuracy": item["train"]["accuracy"],
                "train_examples": item["train"]["examples"],
                "val_loss": item["val"]["loss"],
                "val_accuracy": item["val"]["accuracy"],
                "val_macro_precision": item["val"]["macro_precision"],
                "val_macro_recall": item["val"]["macro_recall"],
                "val_macro_f1": item["val"]["macro_f1"],
                "val_examples": item["val"]["examples"],
                "is_best": item["is_best"],
            }
        )
    write_csv(path, rows, columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BST_CG_AP on Phase 03 collated ShuttleSet features.")
    parser.add_argument("--collated-root", type=Path, default=DEFAULT_COLLATED)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--from-scratch", action="store_true", help="Do not load --init-checkpoint.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--weight-out", type=Path, default=DEFAULT_WEIGHT_OUT)
    parser.add_argument("--train-splits", default="train")
    parser.add_argument("--val-splits", default="val")
    parser.add_argument("--pose-style", default=DEFAULT_POSE_STYLE, choices=["J_only", "JnB_interp", "JnB_bone", "Jn2B"])
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--random-translation", type=float, default=0.02)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seq_len != DEFAULT_SEQ_LEN:
        raise ValueError("This project Phase 04 fine-tuning script expects Phase 03 seq_len=100 collation.")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.early_stop_epochs < 0:
        raise ValueError("--early-stop-epochs must be >= 0")
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    seed_everything(args.seed)
    device = select_device(args.device, args.gpu_id)
    class_names = get_merged_stroke_types()
    n_classes = len(class_names)
    train_splits = parse_split_list(args.train_splits)
    val_splits = parse_split_list(args.val_splits)
    overlap = sorted(set(train_splits) & set(val_splits))

    train_dataset = split_dataset(args.collated_root, train_splits, args.pose_style)
    val_dataset = split_dataset(args.collated_root, val_splits, args.pose_style)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(seq_len=args.seq_len, n_classes=n_classes, device=device)
    init_info = load_initial_weights(
        model,
        checkpoint=None if args.from_scratch else args.init_checkpoint,
        device=device,
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights(train_dataset, n_classes, device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    weight_out = args.weight_out.resolve()
    weight_out.parent.mkdir(parents=True, exist_ok=True)
    best_weight = output_root / "best_bst.pt"
    last_weight = output_root / "last_bst.pt"
    history_csv = output_root / "training_history.csv"
    metrics_csv = output_root / "val_metrics_by_class.csv"
    confusion_csv = output_root / "val_confusion_matrix.csv"
    summary_json = output_root / "training_summary.json"

    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            random_translation=args.random_translation,
            grad_clip_norm=args.grad_clip_norm,
            max_batches=args.max_train_batches,
        )
        val_metrics, val_confusion = evaluate(
            model,
            val_loader,
            criterion,
            device,
            n_classes,
            max_batches=args.max_val_batches,
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        is_best = val_metrics["macro_f1"] > best_macro_f1
        if is_best:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_confusion = val_confusion.copy()
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_weight)
            torch.save(model.state_dict(), weight_out)
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "elapsed_seconds": time.time() - start,
            "lr": current_lr,
            "train": train_metrics,
            "val": val_metrics,
            "is_best": is_best,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if args.early_stop_epochs and epochs_without_improvement >= args.early_stop_epochs:
            print(f"early stopping after {epochs_without_improvement} epoch(s) without validation macro-F1 improvement")
            break

    torch.save(model.state_dict(), last_weight)
    write_history(history_csv, history)
    write_csv(
        metrics_csv,
        metrics_rows(best_confusion, class_names),
        ["label", "label_name", "support", "correct", "precision", "recall", "f1", "accuracy"],
    )
    write_csv(confusion_csv, confusion_rows(best_confusion, class_names))

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "project/tools/phase04_train_bst.py",
        "collated_root": relative(args.collated_root),
        "pose_style": args.pose_style,
        "seq_len": args.seq_len,
        "class_count": n_classes,
        "train_splits": train_splits,
        "val_splits": val_splits,
        "overlapping_train_val_splits": overlap,
        "split_counts_train": split_counts(args.collated_root, train_splits, args.pose_style),
        "split_counts_val": split_counts(args.collated_root, val_splits, args.pose_style),
        "init": init_info,
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "random_translation": args.random_translation,
        "grad_clip_norm": args.grad_clip_norm,
        "early_stop_epochs": args.early_stop_epochs,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "best_weight": relative(best_weight),
        "last_weight": relative(last_weight),
        "phase04_weight": relative(weight_out),
        "history_csv": relative(history_csv),
        "metrics_by_class_csv": relative(metrics_csv),
        "confusion_matrix_csv": relative(confusion_csv),
        "protocol_note": (
            "If train and validation splits overlap, this is a deployment/final-all44 run and not an unbiased validation estimate."
            if overlap
            else "Disjoint train and validation splits for protocol model selection."
        ),
        "history": history,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": relative(summary_json), "best_weight": relative(weight_out), "best_val_macro_f1": best_macro_f1}, indent=2))


if __name__ == "__main__":
    main()
