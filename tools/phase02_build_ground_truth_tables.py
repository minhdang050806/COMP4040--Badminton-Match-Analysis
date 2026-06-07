from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SET_ROOT = ROOT / "project" / "dataset" / "ShuttleSet" / "set"
OUT = ROOT / "project" / "outputs" / "tables"

MATCH_COLUMNS = [
    "id",
    "video",
    "tournament",
    "round",
    "year",
    "month",
    "day",
    "set",
    "duration",
    "winner",
    "loser",
    "downcourt",
    "url",
]

STROKE_COLUMNS = [
    "rally",
    "ball_round",
    "time",
    "frame_num",
    "roundscore_A",
    "roundscore_B",
    "player",
    "server",
    "type",
    "aroundhead",
    "backhand",
    "hit_height",
    "hit_area",
    "hit_x",
    "hit_y",
    "landing_height",
    "landing_area",
    "landing_x",
    "landing_y",
    "lose_reason",
    "win_reason",
    "getpoint_player",
    "flaw",
    "player_location_area",
    "player_location_x",
    "player_location_y",
    "opponent_location_area",
    "opponent_location_x",
    "opponent_location_y",
    "db",
]

OUTPUT_COLUMNS = [
    "stable_key",
    "clip_id",
    "match_id",
    "match_name",
    "set_id",
    "source_set_csv",
    "source_row_index",
    "tournament",
    "match_round",
    "match_year",
    "match_month",
    "match_day",
    "match_date",
    "declared_sets",
    "duration_minutes",
    "winner",
    "loser",
    "downcourt",
    "source_url",
    "has_homography",
    "homography_video",
    "homography_matrix",
    "homo_upleft_x",
    "homo_upright_x",
    "homo_downleft_x",
    "homo_downright_x",
    "homo_upleft_y",
    "homo_upright_y",
    "homo_downleft_y",
    "homo_downright_y",
    "rally",
    "rally_id",
    "ball_round",
    "ball_round_id",
    "time",
    "frame_num",
    "frame_num_int",
    "roundscore_A",
    "roundscore_B",
    "score_state",
    "player",
    "server",
    "stroke_type_ground_truth",
    "type",
    "aroundhead",
    "backhand",
    "hit_height",
    "hit_area",
    "hit_x",
    "hit_y",
    "landing_height",
    "landing_area",
    "landing_x",
    "landing_y",
    "lose_reason",
    "win_reason",
    "getpoint_player",
    "has_outcome",
    "flaw",
    "player_location_area",
    "player_location_x",
    "player_location_y",
    "opponent_location_area",
    "opponent_location_x",
    "opponent_location_y",
    "db",
]

REQUIRED_OUTPUT_COLUMNS = [
    "match_id",
    "match_name",
    "set_id",
    "rally",
    "ball_round",
    "frame_num",
    "player",
    "server",
    "stroke_type_ground_truth",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_key_value(value: str | None) -> str:
    text = (value or "").strip()
    if text == "":
        return ""
    try:
        parsed = float(text)
    except ValueError:
        return text
    if parsed.is_integer():
        return str(int(parsed))
    return text


def maybe_int(value: str | None) -> str:
    return normalize_key_value(value)


def build_match_date(row: dict[str, str]) -> str:
    try:
        return f"{int(row['year']):04d}-{int(row['month']):02d}-{int(row['day']):02d}"
    except (KeyError, TypeError, ValueError):
        return ""


def set_id_from_path(path: Path) -> str:
    match = re.fullmatch(r"set(\d+)\.csv", path.name)
    if not match:
        raise ValueError(f"Unexpected set filename: {path}")
    return match.group(1)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    match_rows = read_csv(SET_ROOT / "match.csv")
    homography_rows = read_csv(SET_ROOT / "homography.csv")

    match_by_video = {row["video"]: row for row in match_rows}
    match_by_id = {row["id"]: row for row in match_rows}
    homography_by_id = {row["id"]: row for row in homography_rows}
    homography_by_video = {row["video"]: row for row in homography_rows}

    set_files = sorted(SET_ROOT.glob("*/set*.csv"))
    output_rows: list[dict[str, Any]] = []
    duplicate_counter: Counter[str] = Counter()
    missing_required_counts: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    player_counter: Counter[str] = Counter()
    server_counter: Counter[str] = Counter()
    outcome_counter: Counter[str] = Counter()
    set_summary: dict[tuple[str, str], dict[str, Any]] = {}
    rally_summary: dict[tuple[str, str, str], dict[str, Any]] = {}
    folder_match_misses: list[str] = []
    homography_video_name_mismatches: list[dict[str, str]] = []

    for folder in sorted([p for p in SET_ROOT.iterdir() if p.is_dir()]):
        if folder.name not in match_by_video:
            folder_match_misses.append(folder.name)

    for h in homography_rows:
        match_row = match_by_id.get(h["id"])
        if match_row and h["video"] != match_row["video"]:
            homography_video_name_mismatches.append(
                {
                    "id": h["id"],
                    "match_video": match_row["video"],
                    "homography_video": h["video"],
                }
            )

    for set_path in set_files:
        match_name = set_path.parent.name
        set_id = set_id_from_path(set_path)
        match_meta = match_by_video.get(match_name)
        if match_meta is None:
            raise RuntimeError(f"No match.csv row for set folder: {match_name}")

        match_id = match_meta["id"]
        homography = homography_by_id.get(match_id, {})

        rows = read_csv(set_path)
        set_key = (match_id, set_id)
        set_summary[set_key] = {
            "match_id": match_id,
            "match_name": match_name,
            "set_id": set_id,
            "source_set_csv": str(set_path.relative_to(ROOT)),
            "stroke_rows": len(rows),
            "rallies": 0,
            "terminal_strokes": 0,
            "unique_stroke_types": 0,
        }
        set_rallies: set[str] = set()
        set_labels: set[str] = set()

        for row_index, stroke in enumerate(rows, start=1):
            rally_id = maybe_int(stroke.get("rally"))
            ball_round_id = maybe_int(stroke.get("ball_round"))
            frame_num_int = maybe_int(stroke.get("frame_num"))
            stable_key = f"{match_id}:{set_id}:{rally_id}:{ball_round_id}"
            clip_id = f"{match_id}_{set_id}_{rally_id}_{ball_round_id}"
            has_outcome = any((stroke.get(c) or "").strip() for c in ("lose_reason", "win_reason", "getpoint_player"))
            stroke_type = (stroke.get("type") or "").strip()

            out = {
                "stable_key": stable_key,
                "clip_id": clip_id,
                "match_id": match_id,
                "match_name": match_name,
                "set_id": set_id,
                "source_set_csv": str(set_path.relative_to(ROOT)),
                "source_row_index": row_index,
                "tournament": match_meta.get("tournament", ""),
                "match_round": match_meta.get("round", ""),
                "match_year": match_meta.get("year", ""),
                "match_month": match_meta.get("month", ""),
                "match_day": match_meta.get("day", ""),
                "match_date": build_match_date(match_meta),
                "declared_sets": match_meta.get("set", ""),
                "duration_minutes": match_meta.get("duration", ""),
                "winner": match_meta.get("winner", ""),
                "loser": match_meta.get("loser", ""),
                "downcourt": match_meta.get("downcourt", ""),
                "source_url": match_meta.get("url", ""),
                "has_homography": bool(homography),
                "homography_video": homography.get("video", ""),
                "homography_matrix": homography.get("homography_matrix", ""),
                "homo_upleft_x": homography.get("upleft_x", ""),
                "homo_upright_x": homography.get("upright_x", ""),
                "homo_downleft_x": homography.get("downleft_x", ""),
                "homo_downright_x": homography.get("downright_x", ""),
                "homo_upleft_y": homography.get("upleft_y", ""),
                "homo_upright_y": homography.get("upright_y", ""),
                "homo_downleft_y": homography.get("downleft_y", ""),
                "homo_downright_y": homography.get("downright_y", ""),
                "rally": stroke.get("rally", ""),
                "rally_id": rally_id,
                "ball_round": stroke.get("ball_round", ""),
                "ball_round_id": ball_round_id,
                "time": stroke.get("time", ""),
                "frame_num": stroke.get("frame_num", ""),
                "frame_num_int": frame_num_int,
                "roundscore_A": stroke.get("roundscore_A", ""),
                "roundscore_B": stroke.get("roundscore_B", ""),
                "score_state": f"{stroke.get('roundscore_A', '')}-{stroke.get('roundscore_B', '')}",
                "player": stroke.get("player", ""),
                "server": stroke.get("server", ""),
                "stroke_type_ground_truth": stroke_type,
                "type": stroke.get("type", ""),
                "aroundhead": stroke.get("aroundhead", ""),
                "backhand": stroke.get("backhand", ""),
                "hit_height": stroke.get("hit_height", ""),
                "hit_area": stroke.get("hit_area", ""),
                "hit_x": stroke.get("hit_x", ""),
                "hit_y": stroke.get("hit_y", ""),
                "landing_height": stroke.get("landing_height", ""),
                "landing_area": stroke.get("landing_area", ""),
                "landing_x": stroke.get("landing_x", ""),
                "landing_y": stroke.get("landing_y", ""),
                "lose_reason": stroke.get("lose_reason", ""),
                "win_reason": stroke.get("win_reason", ""),
                "getpoint_player": stroke.get("getpoint_player", ""),
                "has_outcome": has_outcome,
                "flaw": stroke.get("flaw", ""),
                "player_location_area": stroke.get("player_location_area", ""),
                "player_location_x": stroke.get("player_location_x", ""),
                "player_location_y": stroke.get("player_location_y", ""),
                "opponent_location_area": stroke.get("opponent_location_area", ""),
                "opponent_location_x": stroke.get("opponent_location_x", ""),
                "opponent_location_y": stroke.get("opponent_location_y", ""),
                "db": stroke.get("db", ""),
            }

            for col in REQUIRED_OUTPUT_COLUMNS:
                if not str(out.get(col, "")).strip():
                    missing_required_counts[col] += 1

            duplicate_counter[stable_key] += 1
            label_counter[stroke_type] += 1
            player_counter[(stroke.get("player") or "").strip()] += 1
            server_counter[(stroke.get("server") or "").strip()] += 1
            outcome_counter["rows_with_any_outcome" if has_outcome else "rows_without_outcome"] += 1
            set_rallies.add(rally_id)
            set_labels.add(stroke_type)

            rally_key = (match_id, set_id, rally_id)
            rally = rally_summary.setdefault(
                rally_key,
                {
                    "match_id": match_id,
                    "match_name": match_name,
                    "set_id": set_id,
                    "rally_id": rally_id,
                    "stroke_count": 0,
                    "first_frame_num": frame_num_int,
                    "last_frame_num": frame_num_int,
                    "first_time": stroke.get("time", ""),
                    "last_time": stroke.get("time", ""),
                    "point_winner": "",
                    "terminal_win_reason": "",
                    "terminal_lose_reason": "",
                },
            )
            rally["stroke_count"] += 1
            rally["last_frame_num"] = frame_num_int
            rally["last_time"] = stroke.get("time", "")
            if has_outcome:
                rally["point_winner"] = stroke.get("getpoint_player", "")
                rally["terminal_win_reason"] = stroke.get("win_reason", "")
                rally["terminal_lose_reason"] = stroke.get("lose_reason", "")

            output_rows.append(out)

        set_summary[set_key]["rallies"] = len(set_rallies)
        set_summary[set_key]["terminal_strokes"] = sum(
            1
            for row in rows
            if any((row.get(c) or "").strip() for c in ("lose_reason", "win_reason", "getpoint_player"))
        )
        set_summary[set_key]["unique_stroke_types"] = len(set_labels)

    duplicate_rows = [
        {"stable_key": key, "count": count}
        for key, count in sorted(duplicate_counter.items())
        if count > 1
    ]

    match_summary: dict[str, dict[str, Any]] = {}
    for row in output_rows:
        match_id = row["match_id"]
        item = match_summary.setdefault(
            match_id,
            {
                "match_id": match_id,
                "match_name": row["match_name"],
                "tournament": row["tournament"],
                "match_round": row["match_round"],
                "winner": row["winner"],
                "loser": row["loser"],
                "sets_present": set(),
                "stroke_rows": 0,
                "rallies": set(),
                "terminal_strokes": 0,
            },
        )
        item["sets_present"].add(row["set_id"])
        item["stroke_rows"] += 1
        item["rallies"].add((row["set_id"], row["rally_id"]))
        if row["has_outcome"]:
            item["terminal_strokes"] += 1

    match_summary_rows = []
    for item in match_summary.values():
        match_summary_rows.append(
            {
                **{k: v for k, v in item.items() if k not in {"sets_present", "rallies"}},
                "sets_present": ",".join(sorted(item["sets_present"], key=int)),
                "set_count": len(item["sets_present"]),
                "rally_count": len(item["rallies"]),
            }
        )

    label_rows = [
        {"stroke_type_ground_truth": label, "count": count}
        for label, count in label_counter.most_common()
    ]

    missing_rows = [
        {"column": col, "missing_count": missing_required_counts.get(col, 0)}
        for col in REQUIRED_OUTPUT_COLUMNS
    ]

    validation = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": {
            "match_csv": str((SET_ROOT / "match.csv").relative_to(ROOT)),
            "homography_csv": str((SET_ROOT / "homography.csv").relative_to(ROOT)),
            "set_csv_files": len(set_files),
        },
        "outputs": {
            "ground_truth_table": "project/outputs/tables/shuttleset_ground_truth_strokes.csv",
            "summary_json": "project/outputs/tables/phase02_ground_truth_summary.json",
            "label_distribution": "project/outputs/tables/phase02_label_distribution.csv",
            "match_summary": "project/outputs/tables/phase02_match_summary.csv",
            "set_summary": "project/outputs/tables/phase02_set_summary.csv",
            "rally_summary": "project/outputs/tables/phase02_rally_summary.csv",
            "missing_required_counts": "project/outputs/tables/phase02_missing_required_counts.csv",
            "duplicate_keys": "project/outputs/tables/phase02_duplicate_keys.csv",
            "homography_video_name_mismatches": "project/outputs/tables/phase02_homography_video_name_mismatches.csv",
        },
        "row_count": len(output_rows),
        "match_rows": len(match_rows),
        "homography_rows": len(homography_rows),
        "match_directories": len([p for p in SET_ROOT.iterdir() if p.is_dir()]),
        "set_csv_files": len(set_files),
        "unique_stable_keys": len(duplicate_counter),
        "duplicate_key_count": len(duplicate_rows),
        "missing_required_counts": dict(missing_required_counts),
        "rows_with_any_outcome": outcome_counter["rows_with_any_outcome"],
        "rows_without_outcome": outcome_counter["rows_without_outcome"],
        "unique_ground_truth_stroke_types": len(label_counter),
        "player_counts": dict(player_counter),
        "server_counts": dict(server_counter),
        "folder_match_misses": folder_match_misses,
        "homography_missing_by_match_id": [
            match_id for match_id in sorted(match_by_id, key=int) if match_id not in homography_by_id
        ],
        "homography_video_name_mismatch_count": len(homography_video_name_mismatches),
        "homography_video_name_join_misses_if_joined_by_name": [
            video for video in sorted(match_by_video) if video not in homography_by_video
        ],
        "utf8_non_ascii_label_count": sum(1 for label in label_counter if any(ord(ch) > 127 for ch in label)),
        "validation_status": "passed" if len(output_rows) == len(duplicate_counter) and not folder_match_misses else "review",
    }

    write_csv(OUT / "shuttleset_ground_truth_strokes.csv", output_rows, OUTPUT_COLUMNS)
    write_csv(OUT / "phase02_label_distribution.csv", label_rows, ["stroke_type_ground_truth", "count"])
    write_csv(OUT / "phase02_match_summary.csv", sorted(match_summary_rows, key=lambda x: int(x["match_id"])))
    write_csv(OUT / "phase02_set_summary.csv", sorted(set_summary.values(), key=lambda x: (int(x["match_id"]), int(x["set_id"]))))
    write_csv(OUT / "phase02_rally_summary.csv", sorted(rally_summary.values(), key=lambda x: (int(x["match_id"]), int(x["set_id"]), int(x["rally_id"]))))
    write_csv(OUT / "phase02_missing_required_counts.csv", missing_rows, ["column", "missing_count"])
    write_csv(OUT / "phase02_duplicate_keys.csv", duplicate_rows, ["stable_key", "count"])
    write_csv(
        OUT / "phase02_homography_video_name_mismatches.csv",
        homography_video_name_mismatches,
        ["id", "match_video", "homography_video"],
    )
    (OUT / "phase02_ground_truth_summary.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": str(OUT.relative_to(ROOT)),
                "row_count": validation["row_count"],
                "unique_stable_keys": validation["unique_stable_keys"],
                "duplicate_key_count": validation["duplicate_key_count"],
                "rows_with_any_outcome": validation["rows_with_any_outcome"],
                "unique_ground_truth_stroke_types": validation["unique_ground_truth_stroke_types"],
                "validation_status": validation["validation_status"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
