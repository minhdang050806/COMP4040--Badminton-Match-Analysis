import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_mining.tactical_mining import cluster_profiles


class ClusterProfilesTest(unittest.TestCase):
    def test_exports_stable_interpretable_clusters(self) -> None:
        units = pd.DataFrame({"unit": [f"u{index}" for index in range(12)]})
        offsets = np.linspace(-0.025, 0.025, 6)
        features = np.asarray(
            [[0.8 + offset, 0.1 - offset, 0.1] for offset in offsets]
            + [[0.1 - offset, 0.8 + offset, 0.1] for offset in offsets],
            dtype=np.float64,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = cluster_profiles(
                units,
                features,
                ["net", "smash", "absent"],
                root / "clusters.csv",
                root / "profiles.csv",
                root / "diagnostics.csv",
                root / "clusters.png",
                "Synthetic profiles",
                bootstrap_repeats=5,
            )
            assignments = pd.read_csv(root / "clusters.csv")
            profiles = pd.read_csv(root / "profiles.csv")
            diagnostics = pd.read_csv(root / "diagnostics.csv")

        self.assertEqual(summary["best_k"], 2)
        self.assertEqual(summary["n_active_stroke_features"], 2)
        self.assertEqual(assignments["cluster"].nunique(), 2)
        self.assertIn("distance_to_centroid", assignments)
        self.assertIn("above_global_average", profiles)
        self.assertIn("bootstrap_stability_mean_ari", diagnostics)


if __name__ == "__main__":
    unittest.main()
