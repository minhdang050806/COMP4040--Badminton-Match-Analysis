"""Pytest configuration.

Adds the repository root to ``sys.path`` so the pipeline packages
(``modeling``, ``data_mining``, ``pose_features``, ...) are importable from
the test suite without an installable package, e.g.
``from data_mining.tactical_mining import cluster_profiles``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
