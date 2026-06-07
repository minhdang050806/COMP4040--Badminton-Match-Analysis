from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from bst_label_names import translate_side_aware_label, translate_stroke_type
except ModuleNotFoundError:
    from project.tools.bst_label_names import translate_side_aware_label, translate_stroke_type


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "project" / "outputs" / "integration" / "baseline_old_phase09"

SIDE_AWARE_COLUMNS = {
    "true_label_name",
    "predicted_label_name",
    "top2_label_name",
    "top3_label_name",
}
STROKE_COLUMNS = {
    "stroke_type_ground_truth",
    "predicted_stroke_type",
}


def translate_csv(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if fieldnames is None:
        raise RuntimeError(f"Missing CSV header: {path}")

    changed = 0
    for row in rows:
        for column in SIDE_AWARE_COLUMNS:
            if column in row:
                translated = translate_side_aware_label(row[column])
                changed += translated != row[column]
                row[column] = translated
        for column in STROKE_COLUMNS:
            if column in row:
                translated = translate_stroke_type(row[column])
                changed += translated != row[column]
                row[column] = translated

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate Phase 10 display labels to English without rerunning inference.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    paths = [
        args.output_root / "inputs" / "phase10_input_metadata.csv",
        args.output_root / "phase10_predictions.csv",
        args.output_root / "phase10_structured_strokes.csv",
    ]
    for path in paths:
        if path.exists():
            print(f"{path}: translated {translate_csv(path)} values")


if __name__ == "__main__":
    main()
