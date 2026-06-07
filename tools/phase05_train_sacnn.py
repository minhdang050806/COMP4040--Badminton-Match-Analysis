from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[2]
AUTOMATED_SRC = ROOT / "external_repos" / "Automated-Hit-frame-Detection-for-Badminton-Match-Analysis" / "src"
DEFAULT_DATA_ROOT = ROOT / "project" / "dataset" / "sa_cnn_data"
DEFAULT_INIT_WEIGHTS = AUTOMATED_SRC / "models" / "weights" / "sacnn.pt"
DEFAULT_OUTPUT_ROOT = ROOT / "project" / "outputs" / "sacnn_training"
DEFAULT_WEIGHT_OUT = ROOT / "project" / "weights" / "sacnn_shuttleset_finetuned.pt"


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def data_transforms(train: bool) -> transforms.Compose:
    ops: list[Any] = [
        transforms.Resize((216, 384)),
        transforms.CenterCrop((216, 216)),
    ]
    if train:
        ops.extend(
            [
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.10),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)


def load_model(init_weights: Path | None, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(AUTOMATED_SRC))
    from models.sacnn import SACNN

    model = SACNN().to(device)
    if init_weights is not None:
        state_dict = torch.load(init_weights, map_location=device)
        model.load_state_dict(state_dict)
    return model


def class_weights(dataset: datasets.ImageFolder, device: torch.device) -> torch.Tensor:
    counts = np.bincount([label for _, label in dataset.samples], minlength=len(dataset.classes)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def parse_split_list(value: str) -> list[str]:
    splits = [item.strip() for item in value.split(",") if item.strip()]
    valid = {"train", "val", "test"}
    invalid = [split for split in splits if split not in valid]
    if invalid:
        raise ValueError(f"Invalid split(s): {invalid}; expected only train,val,test")
    return splits


def load_split_dataset(data_root: Path, splits: list[str], train: bool) -> torch.utils.data.Dataset:
    if not splits:
        raise RuntimeError("Expected at least one dataset split.")
    datasets_by_split: list[datasets.ImageFolder] = []
    transform = data_transforms(train=train)
    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            raise RuntimeError(f"Expected ImageFolder split under {split_dir}")
        split_dataset = datasets.ImageFolder(split_dir, transform)
        if split_dataset.classes != ["0", "1"]:
            raise RuntimeError(f"Expected class folders ['0', '1'] under {split_dir}; got {split_dataset.classes}")
        datasets_by_split.append(split_dataset)
    if len(datasets_by_split) == 1:
        return datasets_by_split[0]
    return torch.utils.data.ConcatDataset(datasets_by_split)


def dataset_classes(dataset: torch.utils.data.Dataset) -> list[str]:
    if isinstance(dataset, datasets.ImageFolder):
        return dataset.classes
    if isinstance(dataset, torch.utils.data.ConcatDataset) and dataset.datasets:
        first = dataset.datasets[0]
        if isinstance(first, datasets.ImageFolder):
            return first.classes
    return ["0", "1"]


def labels_for_weighting(dataset: torch.utils.data.Dataset) -> list[int]:
    if isinstance(dataset, datasets.ImageFolder):
        return [label for _, label in dataset.samples]
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        labels: list[int] = []
        for child in dataset.datasets:
            labels.extend(labels_for_weighting(child))
        return labels
    raise TypeError(f"Unsupported dataset type for class weights: {type(dataset).__name__}")


def class_weights_from_dataset(dataset: torch.utils.data.Dataset, device: torch.device) -> torch.Tensor:
    labels = labels_for_weighting(dataset)
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def select_device(cpu: bool, gpu_id: int) -> torch.device:
    if cpu:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")

    device_count = torch.cuda.device_count()
    if gpu_id < 0 or gpu_id >= device_count:
        raise RuntimeError(f"Requested --gpu-id {gpu_id}, but only {device_count} CUDA device(s) are visible.")
    torch.cuda.set_device(gpu_id)
    return torch.device(f"cuda:{gpu_id}")


def metrics_from_confusion(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_examples = 0
    tp = fp = fn = tn = 0

    for batch_index, (images, labels) in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits.detach(), dim=1)
        total_loss += float(loss.detach().cpu()) * labels.numel()
        total_examples += labels.numel()
        tp += int(((preds == 1) & (labels == 1)).sum().item())
        fp += int(((preds == 1) & (labels == 0)).sum().item())
        fn += int(((preds == 0) & (labels == 1)).sum().item())
        tn += int(((preds == 0) & (labels == 0)).sum().item())

    metrics = metrics_from_confusion(tp, fp, fn, tn)
    metrics["loss"] = total_loss / total_examples if total_examples else 0.0
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SA-CNN on a ShuttleSet-derived shot-angle dataset.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--init-weights", type=Path, default=DEFAULT_INIT_WEIGHTS)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--weight-out", type=Path, default=DEFAULT_WEIGHT_OUT)
    parser.add_argument("--train-splits", default="train", help="Comma-separated ImageFolder splits to train on.")
    parser.add_argument("--val-splits", default="val", help="Comma-separated ImageFolder splits to validate on.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device index to use when CUDA is available.")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--early-stop-epochs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(cpu=args.cpu, gpu_id=args.gpu_id)

    data_root = Path(args.data_root)
    train_splits = parse_split_list(args.train_splits)
    val_splits = parse_split_list(args.val_splits)
    overlap = sorted(set(train_splits) & set(val_splits))
    train_dataset = load_split_dataset(data_root, train_splits, train=True)
    val_dataset = load_split_dataset(data_root, val_splits, train=False)

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

    init_weights = None if args.from_scratch else Path(args.init_weights).resolve()
    model = load_model(init_weights, device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_from_dataset(train_dataset, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    weight_out = Path(args.weight_out).resolve()
    weight_out.parent.mkdir(parents=True, exist_ok=True)
    best_out = output_root / "best_sacnn.pt"
    last_out = output_root / "last_sacnn.pt"

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, args.max_train_batches)
        val_metrics = run_epoch(model, val_loader, criterion, device, None, args.max_val_batches)
        is_best = val_metrics["f1"] > best_f1
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "is_best": is_best}
        history.append(row)
        print(
            f"epoch {epoch:03d}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_recall={val_metrics['recall']:.4f}"
        )
        if is_best:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_out)
            torch.save(model.state_dict(), weight_out)
        else:
            epochs_without_improvement += 1

        if args.early_stop_epochs and epochs_without_improvement >= args.early_stop_epochs:
            print(f"early stopping after {epochs_without_improvement} epoch(s) without validation F1 improvement")
            break

    torch.save(model.state_dict(), last_out)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "project/tools/phase05_train_sacnn.py",
        "data_root": relative(Path(args.data_root).resolve()),
        "output_root": relative(output_root),
        "best_weight": relative(best_out),
        "last_weight": relative(last_out),
        "phase05_weight": relative(weight_out),
        "init_weights": None if init_weights is None else relative(init_weights),
        "device": str(device),
        "gpu_id": args.gpu_id,
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "seed": args.seed,
        "train_splits": train_splits,
        "val_splits": val_splits,
        "overlapping_train_val_splits": overlap,
        "epochs": args.epochs,
        "epochs_completed": len(history),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "early_stop_epochs": args.early_stop_epochs,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "classes": dataset_classes(train_dataset),
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "protocol_note": (
            "If train and validation splits overlap, this is a deployment/final-all44 run and not an unbiased validation estimate."
            if overlap
            else "Disjoint train and validation splits for protocol model selection."
        ),
        "history": history,
    }
    (output_root / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "16")
    main()
