from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pose_features.extract import (
    Candidate,
    FeatureBundle,
    HomographyInfo,
    Phase06Task,
    StrokeWindow,
    court_candidates,
    rebuild_temporal_window,
    recover_short_gaps,
    save_bundle,
    track_candidates,
)


def candidate(x: float, y: float, pose_valid: bool = True, area: float = 10000.0) -> Candidate:
    keypoints = np.ones((17, 2), dtype=np.float32) if pose_valid else np.zeros((17, 2), dtype=np.float32)
    return Candidate(
        keypoints=keypoints,
        bbox=np.asarray([0.0, 0.0, 100.0, 200.0], dtype=np.float32),
        court_position=np.asarray([x, y], dtype=np.float32),
        pose_valid=pose_valid,
        area=area,
    )


class Phase09TrackingTests(unittest.TestCase):
    def test_bst_compatible_window_matches_reference_contract(self) -> None:
        window = StrokeWindow("40", "2", "2", 12223, 12191, 12230, 12159, 12236)
        task = Phase06Task(Path("video.mp4"), Path("shuttle.csv"), 1280, 720, 30.0, 12060, 12510)
        rebuilt = rebuild_temporal_window(window, task, "bst-compatible")
        self.assertEqual(rebuilt.window_start_original, 12178)
        self.assertEqual(rebuilt.window_end_original, 12242)
        self.assertEqual(rebuilt.window_end_original - rebuilt.window_start_original + 1, 65)

    def test_bst_compatible_window_uses_rally_edge_fallback(self) -> None:
        window = StrokeWindow("40", "2", "1", 12159, 12060, 12191, None, 12223)
        task = Phase06Task(Path("video.mp4"), Path("shuttle.csv"), 1280, 720, 30.0, 12060, 12510)
        rebuilt = rebuild_temporal_window(window, task, "bst-compatible")
        self.assertEqual(rebuilt.window_start_original, 12144)
        self.assertEqual(rebuilt.window_end_original, 12210)

    def test_one_detection_preserves_available_player(self) -> None:
        joints, positions, pose_observed, position_observed = track_candidates(
            [[candidate(0.4, 0.25)]],
            side_split=0.5,
            max_track_distance=0.45,
        )
        self.assertTrue(pose_observed[0, 0])
        self.assertTrue(position_observed[0, 0])
        self.assertFalse(pose_observed[0, 1])
        self.assertFalse(position_observed[0, 1])
        self.assertTrue(np.any(joints[0, 0] != 0.0))
        self.assertTrue(np.all(joints[0, 1] == 0.0))
        np.testing.assert_allclose(positions[0, 0], [0.4, 0.25])

    def test_position_can_be_valid_when_pose_is_missing(self) -> None:
        joints, positions, pose_observed, position_observed = track_candidates(
            [[candidate(0.6, 0.8, pose_valid=False)]],
            side_split=0.5,
            max_track_distance=0.45,
        )
        self.assertFalse(pose_observed[0, 1])
        self.assertTrue(position_observed[0, 1])
        self.assertTrue(np.all(joints[0, 1] == 0.0))
        np.testing.assert_allclose(positions[0, 1], [0.6, 0.8])

    def test_court_filter_rejects_background_person(self) -> None:
        homography = HomographyInfo(np.eye(3), 0.0, 1.0, 0.0, 1.0)
        keypoints = np.zeros((17, 2), dtype=np.float32)
        people = [
            (keypoints, np.asarray([0.2, 0.1, 0.4, 0.8], dtype=np.float32)),
            (keypoints, np.asarray([0.2, 0.1, 0.4, 1.3], dtype=np.float32)),
        ]
        selected = court_candidates(people, homography, court_margin=0.2, min_valid_keypoints=5)
        self.assertEqual(len(selected), 1)
        np.testing.assert_allclose(selected[0].court_position, [0.3, 0.8])

    def test_tracking_prefers_previous_position_over_larger_candidate(self) -> None:
        frames = [
            [candidate(0.2, 0.2)],
            [candidate(0.21, 0.21, area=1000.0), candidate(0.45, 0.2, area=100000.0)],
        ]
        _, positions, _, valid = track_candidates(frames, side_split=0.5, max_track_distance=0.45)
        self.assertTrue(valid[1, 0])
        np.testing.assert_allclose(positions[1, 0], [0.21, 0.21])

    def test_tracking_rejects_implausible_jump(self) -> None:
        frames = [[candidate(0.1, 0.1)], [candidate(0.9, 0.4)]]
        _, _, _, valid = track_candidates(frames, side_split=0.5, max_track_distance=0.45)
        self.assertTrue(valid[0, 0])
        self.assertFalse(valid[1, 0])

    def test_tracking_reacquires_after_stale_track_gap(self) -> None:
        frames = [[candidate(0.1, 0.1)]] + ([[]] * 11) + [[candidate(0.9, 0.4)]]
        _, positions, _, valid = track_candidates(frames, side_split=0.5, max_track_distance=0.45, max_track_gap=10)
        self.assertTrue(valid[-1, 0])
        np.testing.assert_allclose(positions[-1, 0], [0.9, 0.4])

    def test_gap_recovery_rules(self) -> None:
        values = np.zeros((28, 2, 2), dtype=np.float32)
        observed = np.zeros((28, 2), dtype=bool)
        observed[[0, 3, 4, 15], 0] = True
        values[0, 0] = [1.0, 1.0]
        values[3, 0] = [3.0, 3.0]
        values[4, 0] = [4.0, 4.0]
        values[15, 0] = [15.0, 15.0]

        recovered, valid = recover_short_gaps(values, observed)

        np.testing.assert_allclose(recovered[1:3, 0], [[1.0, 1.0], [1.0, 1.0]])
        self.assertTrue(np.all(valid[1:3, 0]))
        np.testing.assert_allclose(recovered[5, 0], [5.0, 5.0])
        np.testing.assert_allclose(recovered[14, 0], [14.0, 14.0])
        self.assertTrue(np.all(valid[5:15, 0]))
        self.assertFalse(valid[16, 0])
        self.assertFalse(valid[27, 0])

    def test_validity_sidecar_has_requested_masks(self) -> None:
        bundle = FeatureBundle(
            joints=np.zeros((2, 2, 17, 2), dtype=np.float32),
            positions=np.zeros((2, 2, 2), dtype=np.float32),
            shuttle=np.zeros((2, 2), dtype=np.float32),
            pose_observed=np.asarray([[True, False], [False, False]]),
            position_observed=np.asarray([[True, True], [False, True]]),
            pose_valid=np.asarray([[True, False], [True, False]]),
            position_valid=np.asarray([[True, True], [True, True]]),
        )
        with tempfile.TemporaryDirectory() as directory:
            validity_path = save_bundle(Path(directory), "41", "clip", bundle)[3]
            validity = np.load(validity_path)
            self.assertEqual(
                set(validity.files),
                {
                    "p1_pose_valid",
                    "p2_pose_valid",
                    "p1_position_valid",
                    "p2_position_valid",
                    "p1_pose_observed",
                    "p2_pose_observed",
                    "p1_position_observed",
                    "p2_position_observed",
                },
            )


if __name__ == "__main__":
    unittest.main()
