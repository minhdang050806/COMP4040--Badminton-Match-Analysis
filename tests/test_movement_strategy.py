import unittest

import numpy as np
import pandas as pd

from data_mining.movement_strategy import build_profiles, player_centric_events


class MovementStrategyTest(unittest.TestCase):
    def test_player_centric_positions_follow_persistent_player_across_sides(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "row_index": 0,
                    "true_label_name": "Top_smash",
                    "player": "A",
                    "event_offset_frames": 0,
                    "video_len_after_collation": 1,
                    "video_id": 41,
                    "set_id": 1,
                    "rally_id": 1,
                    "event_rank": 1,
                    "predicted_stroke_type": "smash",
                    "confidence": 0.9,
                },
                {
                    "row_index": 1,
                    "true_label_name": "Bottom_clear",
                    "player": "A",
                    "event_offset_frames": 0,
                    "video_len_after_collation": 1,
                    "video_id": 41,
                    "set_id": 1,
                    "rally_id": 1,
                    "event_rank": 2,
                    "predicted_stroke_type": "clear",
                    "confidence": 0.8,
                },
                {
                    "row_index": 2,
                    "true_label_name": "Top_smash",
                    "player": "B",
                    "event_offset_frames": 0,
                    "video_len_after_collation": 1,
                    "video_id": 41,
                    "set_id": 1,
                    "rally_id": 1,
                    "event_rank": 3,
                    "predicted_stroke_type": "smash",
                    "confidence": 0.7,
                },
            ]
        )
        positions = np.zeros((3, 1, 2, 2), dtype=np.float32)
        positions[0, 0, 0] = [0.2, 0.25]
        positions[1, 0, 1] = [0.3, 0.75]
        positions[2, 0, 0] = [0.5, -0.1]

        events = player_centric_events(frame, positions)
        profile = build_profiles(events).iloc[0]

        self.assertEqual(len(events), 2)
        self.assertAlmostEqual(events.iloc[0]["lateral_position"], 0.8)
        self.assertAlmostEqual(events.iloc[0]["forward_position"], 0.5)
        self.assertAlmostEqual(events.iloc[1]["lateral_position"], 0.3)
        self.assertAlmostEqual(events.iloc[1]["forward_position"], 0.5)
        self.assertAlmostEqual(events.iloc[1]["interstroke_displacement"], 0.5)
        self.assertEqual(profile["n_displacements"], 1)


if __name__ == "__main__":
    unittest.main()
