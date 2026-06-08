from __future__ import annotations

import unittest

from modeling.prepare_inputs import expand_simultaneous_event_aliases, quality_group
from modeling.integrated_inference import evaluation_metrics, evaluation_rows, split_class_name


class Phase10IntegrationTests(unittest.TestCase):
    def test_simultaneous_event_aliases_expand_to_distinct_labels(self) -> None:
        manifest = [
            {"video_id": "41", "rally_id": "80", "event_frame_original": "100", "event_rank": "1", "clip_id": "b", "frame_count": "10"},
            {"video_id": "41", "rally_id": "80", "event_frame_original": "100", "event_rank": "2", "clip_id": "b", "frame_count": "20"},
        ]
        validation = [{"clip_id": "b", "event_rank": "2", "frame_count": "20"}]
        matches = [
            {"video_id": "41", "rally_id": "80", "predicted_frame_original": "100", "clip_id": "a"},
            {"video_id": "41", "rally_id": "80", "predicted_frame_original": "100", "clip_id": "b"},
        ]
        expanded, expanded_validation, repair = expand_simultaneous_event_aliases(manifest, validation, matches)
        self.assertEqual([row["clip_id"] for row in expanded], ["a", "b"])
        self.assertEqual([row["frame_count"] for row in expanded], ["20", "20"])
        self.assertEqual(set(expanded_validation), {"a", "b"})
        self.assertEqual(repair["simultaneous_event_groups_repaired"], 1)

    def test_quality_group_separates_labels_and_pose_dropout(self) -> None:
        self.assertEqual(quality_group(0.0, True), "exact_label_pose_dropout_lt_25")
        self.assertEqual(quality_group(0.3, False), "candidate_only_pose_dropout_25_50")
        self.assertEqual(quality_group(0.7, True), "exact_label_pose_dropout_ge_50")
        self.assertEqual(quality_group(1.0, False), "candidate_only_all_pose_missing")

    def test_class_name_split(self) -> None:
        self.assertEqual(split_class_name("Top_殺球"), ("Top", "殺球"))
        self.assertEqual(split_class_name("未知球種"), ("", "未知球種"))

    def test_decomposed_accuracy_separates_side_and_stroke(self) -> None:
        rows = [
            {
                "true_label": "1",
                "predicted_label": 2,
                "top3_correct": True,
                "side_correct": True,
                "stroke_type_correct": False,
            },
            {
                "true_label": "3",
                "predicted_label": 3,
                "top3_correct": True,
                "side_correct": True,
                "stroke_type_correct": True,
            },
        ]
        metrics = evaluation_metrics(rows, class_count=25)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["top3_accuracy"], 1.0)
        self.assertEqual(metrics["player_side_accuracy"], 1.0)
        self.assertEqual(metrics["stroke_type_accuracy"], 0.5)

    def test_evaluation_keeps_reference_test_separate(self) -> None:
        rows = [
            {
                "video_id": "41",
                "true_label": "1",
                "predicted_label": 1,
                "zero_pose_rate": "0.1",
                "reference_split": "test",
            },
            {
                "video_id": "41",
                "true_label": "2",
                "predicted_label": 3,
                "zero_pose_rate": "0.6",
                "reference_split": "val",
            },
            {
                "video_id": "42",
                "true_label": "-1",
                "predicted_label": 4,
                "zero_pose_rate": "1.0",
                "reference_split": "",
            },
        ]
        metrics = {row["evaluation_group"]: row for row in evaluation_rows(rows, class_count=25)}
        self.assertEqual(metrics["reference_test_primary"]["labeled_rows"], 1)
        self.assertEqual(metrics["reference_test_primary"]["accuracy"], 1.0)
        self.assertEqual(metrics["reference_val_diagnostic"]["labeled_rows"], 1)
        self.assertEqual(metrics["reference_val_diagnostic"]["accuracy"], 0.0)

    def test_primary_evaluation_excludes_shared_feature_aliases(self) -> None:
        rows = [
            {
                "video_id": "42",
                "true_label": "1",
                "predicted_label": 1,
                "pose_missing_rate": "0.1",
                "reference_split": "test",
                "feature_alias_source_clip_id": "",
            },
            {
                "video_id": "42",
                "true_label": "2",
                "predicted_label": 2,
                "pose_missing_rate": "0.1",
                "reference_split": "test",
                "feature_alias_source_clip_id": "shared",
            },
        ]
        metrics = {row["evaluation_group"]: row for row in evaluation_rows(rows, class_count=25)}
        self.assertEqual(metrics["reference_test_primary"]["rows"], 1)
        self.assertEqual(metrics["reference_test_all_including_aliases"]["rows"], 2)
        self.assertEqual(metrics["simultaneous_event_alias_diagnostic"]["rows"], 1)


if __name__ == "__main__":
    unittest.main()
