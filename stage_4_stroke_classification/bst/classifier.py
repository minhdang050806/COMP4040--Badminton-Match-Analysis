"""StrokeClassifier — thin inference wrapper around BST_CG_AP.

The training scripts in ``main_on_shuttleset/`` are kept verbatim under the
``bst`` subpackage. This file is the new inference-friendly façade
required by Section 3.1.2 Step 3 of the architecture document.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch

from stage_4_stroke_classification.bst.model.bst import BST_CG_AP


# Default class names for the ShuttleSet 25-class taxonomy. Replace at
# runtime via ``StrokeClassifier.set_class_names`` if you train a model
# with a different label space.
SHUTTLESET_CLASSES = [
    "short_service", "long_service", "smash", "wood_shot", "net_shot",
    "lob", "drop", "drive", "rush", "back_court_drive", "block",
    "drive_lift", "cross_lift", "fast_drop", "passive_drop", "block_lift",
    "push", "cut", "rear_smash", "front_drop", "rush_to_kill",
    "transition", "defensive_lift", "midcourt_push", "other",
]


class StrokeClassifier:
    def __init__(self, weights_path: str, n_classes: int = 25,
                 in_dim: int = 34, seq_len: int = 30, d_model: int = 100,
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BST_CG_AP(
            in_dim=in_dim, seq_len=seq_len, n_class=n_classes,
            d_model=d_model,
        ).to(self.device)
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.class_names = list(SHUTTLESET_CLASSES[:n_classes])

    def set_class_names(self, names):
        self.class_names = list(names)

    @torch.no_grad()
    def predict(self, features: dict) -> Tuple[int, np.ndarray, str]:
        JnB = torch.from_numpy(features["human_pose"]).float().unsqueeze(0).to(self.device)
        shuttle = torch.from_numpy(features["shuttle"]).float().unsqueeze(0).to(self.device)
        pos = torch.from_numpy(features["pos"]).float().unsqueeze(0).to(self.device)
        video_len = torch.tensor([features["video_len"]],
                                 device=self.device, dtype=torch.long)
        logits = self.model(JnB, shuttle, pos, video_len)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        cls = int(np.argmax(probs))
        name = self.class_names[cls] if cls < len(self.class_names) else str(cls)
        return cls, probs, name
