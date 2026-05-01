"""ViT Ensemble adapter — Section 3.1.2 Step 4.

Wraps the legacy ``get_hitframe.py`` + per-attribute ``ViT-pytorch_*/submit.py``
scripts behind a single callable. The original scripts are CLI-style; here
we only need the ``predict`` interface for the orchestrator.

This module is intentionally lightweight — when a ViT weight file is
missing it gracefully returns -1 / empty so the pipeline can still run
end-to-end with partial models.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np


class ViTPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self._models: Dict[str, object] = {}

    # ------------------------------------------------------------------
    def _load_attribute_model(self, attr: str, weight_dir: str):
        """Lazy-load a ViT-B_16 ensemble for an attribute. Caches in memory.

        Returns ``None`` when weights are missing so callers can skip.
        """
        if not weight_dir or not os.path.exists(weight_dir):
            return None
        if attr in self._models:
            return self._models[attr]
        try:
            import torch
            from torchvision.models import vit_b_16
            model = vit_b_16(weights=None)
            ck = sorted(
                f for f in os.listdir(weight_dir)
                if f.endswith((".pt", ".pth", ".bin"))
            )
            if ck:
                state = torch.load(os.path.join(weight_dir, ck[0]),
                                   map_location="cpu")
                if isinstance(state, dict) and "model" in state:
                    state = state["model"]
                model.load_state_dict(state, strict=False)
            model.eval()
            self._models[attr] = model
            return model
        except Exception as e:
            print(f"[ViTPipeline] Could not load {attr}: {e}")
            return None

    # ------------------------------------------------------------------
    def _crop_hit_frame(self, clip_path: str, frame_idx: int) -> np.ndarray:
        import cv2
        cap = cv2.VideoCapture(clip_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        side = min(720, h, w)
        x0, y0 = cx - side // 2, cy - side // 2
        return frame[y0:y0 + side, x0:x0 + side]

    # ------------------------------------------------------------------
    def predict(self, rally, hit_event) -> dict:
        """Returns the ViT attribute dict to splice into a StrokeRecord."""
        out = dict(hitter=-1, backhand=-1, ball_height=-1,
                   ball_type=-1, winner=-1)

        img = self._crop_hit_frame(rally.clip_path, hit_event.hit_frame)
        if img is None:
            return out

        # Each attribute uses a separate weight directory so missing models
        # degrade gracefully.
        attr_to_dir = {
            "hitter":      self.cfg.models.vit_hitter,
            "backhand":    self.cfg.models.vit_backhand,
            "ball_type":   self.cfg.models.vit_ball_type,
            "ball_height": getattr(self.cfg.models, "vit_ball_height", None),
            "winner":      getattr(self.cfg.models, "vit_winner", None),
        }
        for attr, wdir in attr_to_dir.items():
            mdl = self._load_attribute_model(attr, wdir)
            if mdl is None:
                continue
            # Real ensemble would aggregate 5 folds — we run a single forward.
            try:
                import torch
                import torchvision.transforms as T
                tf = T.Compose([
                    T.ToPILImage(), T.Resize((480, 480)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                with torch.no_grad():
                    logit = mdl(tf(img).unsqueeze(0))
                out[attr] = int(logit.argmax(-1).item())
            except Exception:
                continue
        return out
