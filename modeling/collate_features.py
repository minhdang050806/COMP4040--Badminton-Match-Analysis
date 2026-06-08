from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "project" / "dataset" / "ShuttleSet" / "merged_seq100_between_2_hits_with_max_limits"
DEFAULT_OUTPUT = ROOT / "project" / "outputs" / "bst_collated" / "merged_seq100_between_2_hits_with_max_limits"
DEFAULT_SEQ_LEN = 100


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


def get_bone_pairs() -> list[tuple[int, int]]:
    return [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]


def make_seq_len_same(
    target_len: int,
    joints: np.ndarray,
    pos: np.ndarray,
    shuttle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    video_len = len(pos)
    if video_len > target_len:
        need_padding = (video_len % target_len) > (target_len // 2)
        stride = video_len // target_len + int(need_padding)
        joints = joints[::stride][:target_len]
        pos = pos[::stride][:target_len]
        shuttle = shuttle[::stride][:target_len]
        new_video_len = len(pos)
        if need_padding:
            pad_len = target_len - new_video_len
            joints = np.pad(joints, ((0, pad_len), *([(0, 0)] * 3)))
            pos = np.pad(pos, ((0, pad_len), *([(0, 0)] * 2)))
            shuttle = np.pad(shuttle, ((0, pad_len), (0, 0)))
    else:
        new_video_len = video_len
        pad_len = target_len - new_video_len
        joints = np.pad(joints, ((0, pad_len), *([(0, 0)] * 3)))
        pos = np.pad(pos, ((0, pad_len), *([(0, 0)] * 2)))
        shuttle = np.pad(shuttle, ((0, pad_len), (0, 0)))
    return joints, pos, shuttle, new_video_len


def create_bones(joints: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    bones = []
    for start, end in pairs:
        start_j = joints[:, :, start, :]
        end_j = joints[:, :, end, :]
        bone = np.where((start_j != 0.0) & (end_j != 0.0), end_j - start_j, 0.0)
        bones.append(bone)
    return np.stack(bones, axis=-2)


def interpolate_joints(joints: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    mid_joints = []
    for start, end in pairs:
        start_j = joints[:, :, start, :]
        end_j = joints[:, :, end, :]
        mid_j = np.where((start_j != 0.0) & (end_j != 0.0), (start_j + end_j) / 2, 0.0)
        mid_joints.append(mid_j)
    bones_center = np.stack(mid_joints, axis=-2)
    return np.concatenate((joints, bones_center), axis=-2)


def load_and_augment(
    branch: Path,
    seq_len: int,
    bone_pairs: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    joints_raw = np.load(str(branch) + "_joints.npy")
    pos_raw = np.load(str(branch) + "_pos.npy")
    shuttle_raw = np.load(str(branch) + "_shuttle.npy")
    original_len = int(len(pos_raw))

    joints = joints_raw.astype(np.float32, copy=False)
    pos = pos_raw.astype(np.float32, copy=False)
    shuttle = shuttle_raw.astype(np.float32, copy=False)

    joints, pos, shuttle, new_video_len = make_seq_len_same(seq_len, joints, pos, shuttle)
    joints_interpolated = interpolate_joints(joints, bone_pairs)
    bones = create_bones(joints, bone_pairs)
    jnb_bone = np.concatenate((joints, bones), axis=-2)
    jn2b = np.concatenate((joints_interpolated, bones), axis=-2)
    return joints, joints_interpolated, jnb_bone, jn2b, pos, shuttle, new_video_len, original_len


def branches_for_split(source_root: Path, split: str, class_to_id: dict[str, int]) -> list[dict[str, Any]]:
    split_dir = source_root / split
    rows: list[dict[str, Any]] = []
    for class_name in sorted(class_to_id, key=lambda name: class_to_id[name]):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for pos_path in sorted(class_dir.glob("*_pos.npy")):
            branch = Path(str(pos_path).removesuffix("_pos.npy"))
            for suffix in ("_joints.npy", "_pos.npy", "_shuttle.npy"):
                if not Path(str(branch) + suffix).exists():
                    raise FileNotFoundError(f"Missing feature file: {branch}{suffix}")
            rows.append(
                {
                    "branch": branch,
                    "class_name": class_name,
                    "label": class_to_id[class_name],
                    "clip_id": branch.name,
                }
            )
    return rows


def create_output_arrays(split_dir: Path, n: int, seq_len: int) -> dict[str, np.memmap]:
    split_dir.mkdir(parents=True, exist_ok=True)
    return {
        "J_only": open_memmap(split_dir / "J_only.npy", mode="w+", dtype=np.float32, shape=(n, seq_len, 2, 17, 2)),
        "JnB_interp": open_memmap(split_dir / "JnB_interp.npy", mode="w+", dtype=np.float32, shape=(n, seq_len, 2, 36, 2)),
        "JnB_bone": open_memmap(split_dir / "JnB_bone.npy", mode="w+", dtype=np.float32, shape=(n, seq_len, 2, 36, 2)),
        "Jn2B": open_memmap(split_dir / "Jn2B.npy", mode="w+", dtype=np.float32, shape=(n, seq_len, 2, 55, 2)),
        "pos": open_memmap(split_dir / "pos.npy", mode="w+", dtype=np.float32, shape=(n, seq_len, 2, 2)),
        "shuttle": open_memmap(split_dir / "shuttle.npy", mode="w+", dtype=np.float32, shape=(n, seq_len, 2)),
        "videos_len": open_memmap(split_dir / "videos_len.npy", mode="w+", dtype=np.int64, shape=(n,)),
        "labels": open_memmap(split_dir / "labels.npy", mode="w+", dtype=np.int64, shape=(n,)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_split(split_dir: Path, n: int, seq_len: int) -> dict[str, Any]:
    expected_shapes = {
        "J_only": (n, seq_len, 2, 17, 2),
        "JnB_interp": (n, seq_len, 2, 36, 2),
        "JnB_bone": (n, seq_len, 2, 36, 2),
        "Jn2B": (n, seq_len, 2, 55, 2),
        "pos": (n, seq_len, 2, 2),
        "shuttle": (n, seq_len, 2),
        "videos_len": (n,),
        "labels": (n,),
    }
    rows: dict[str, Any] = {}
    for name, shape in expected_shapes.items():
        path = split_dir / f"{name}.npy"
        arr = np.load(path, mmap_mode="r")
        rows[name] = {
            "path": str(path.relative_to(ROOT)),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "shape_ok": tuple(arr.shape) == shape,
            "size_bytes": path.stat().st_size,
        }
    labels = np.load(split_dir / "labels.npy", mmap_mode="r")
    videos_len = np.load(split_dir / "videos_len.npy", mmap_mode="r")
    rows["label_min"] = int(labels.min()) if len(labels) else None
    rows["label_max"] = int(labels.max()) if len(labels) else None
    rows["label_unique_count"] = int(len(np.unique(labels)))
    rows["videos_len_min"] = int(videos_len.min()) if len(videos_len) else None
    rows["videos_len_max"] = int(videos_len.max()) if len(videos_len) else None
    rows["videos_len_over_seq_len"] = int(np.count_nonzero(videos_len > seq_len))
    rows["all_shapes_ok"] = all(v["shape_ok"] for k, v in rows.items() if isinstance(v, dict))
    return rows


def collate_split(
    source_root: Path,
    output_root: Path,
    split: str,
    seq_len: int,
    class_to_id: dict[str, int],
    progress_every: int,
) -> dict[str, Any]:
    branch_rows = branches_for_split(source_root, split, class_to_id)
    split_dir = output_root / split
    arrays = create_output_arrays(split_dir, len(branch_rows), seq_len)
    bone_pairs = get_bone_pairs()
    metadata_rows: list[dict[str, Any]] = []

    for index, item in enumerate(branch_rows):
        j_only, jnb_interp, jnb_bone, jn2b, pos, shuttle, video_len, original_len = load_and_augment(
            item["branch"], seq_len, bone_pairs
        )
        arrays["J_only"][index] = j_only
        arrays["JnB_interp"][index] = jnb_interp
        arrays["JnB_bone"][index] = jnb_bone
        arrays["Jn2B"][index] = jn2b
        arrays["pos"][index] = pos
        arrays["shuttle"][index] = shuttle
        arrays["videos_len"][index] = video_len
        arrays["labels"][index] = item["label"]

        metadata_rows.append(
            {
                "row_index": index,
                "split": split,
                "clip_id": item["clip_id"],
                "source_branch": str(item["branch"].relative_to(ROOT)),
                "class_name": item["class_name"],
                "label": item["label"],
                "original_len": original_len,
                "video_len_after_collation": video_len,
            }
        )
        if progress_every and (index + 1) % progress_every == 0:
            print(f"{split}: collated {index + 1}/{len(branch_rows)}")

    for arr in arrays.values():
        arr.flush()

    metadata_path = output_root / f"{split}_metadata.csv"
    write_csv(
        metadata_path,
        metadata_rows,
        [
            "row_index",
            "split",
            "clip_id",
            "source_branch",
            "class_name",
            "label",
            "original_len",
            "video_len_after_collation",
        ],
    )

    validation = validate_split(split_dir, len(branch_rows), seq_len)
    validation.update(
        {
            "split": split,
            "clip_count": len(branch_rows),
            "metadata_path": str(metadata_path.relative_to(ROOT)),
            "class_count": len({row["class_name"] for row in metadata_rows}),
            "original_len_min": min(row["original_len"] for row in metadata_rows),
            "original_len_max": max(row["original_len"] for row in metadata_rows),
            "collated_video_len_min": min(row["video_len_after_collation"] for row in metadata_rows),
            "collated_video_len_max": max(row["video_len_after_collation"] for row in metadata_rows),
        }
    )
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collate ShuttleSet merged seq100 features for BST.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    class_names = get_merged_stroke_types()
    class_to_id = {name: i for i, name in enumerate(class_names)}

    output_root.mkdir(parents=True, exist_ok=True)
    class_map_rows = [{"label": i, "class_name": name} for i, name in enumerate(class_names)]
    write_csv(output_root / "class_mapping.csv", class_map_rows, ["label", "class_name"])

    validations = []
    for split in ("train", "val", "test"):
        print(f"Collating {split} from {source_root}")
        validations.append(
            collate_split(
                source_root=source_root,
                output_root=output_root,
                split=split,
                seq_len=args.seq_len,
                class_to_id=class_to_id,
                progress_every=args.progress_every,
            )
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root.relative_to(ROOT)),
        "output_root": str(output_root.relative_to(ROOT)),
        "seq_len": args.seq_len,
        "pose_styles_written": ["J_only", "JnB_interp", "JnB_bone", "Jn2B"],
        "class_count": len(class_names),
        "bone_pair_count": len(get_bone_pairs()),
        "splits": validations,
        "total_clips": sum(v["clip_count"] for v in validations),
        "validation_status": "passed"
        if all(v["all_shapes_ok"] and v["videos_len_over_seq_len"] == 0 for v in validations)
        else "review",
    }
    (output_root / "phase03_collation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "output_root": str(output_root.relative_to(ROOT)),
        "total_clips": summary["total_clips"],
        "validation_status": summary["validation_status"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
