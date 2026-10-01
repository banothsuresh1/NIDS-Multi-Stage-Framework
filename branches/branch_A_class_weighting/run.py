#!/usr/bin/env python3
"""Branch A: class weighting only (no resampling), time-blocked split.

Isolates the contribution of cost-sensitive learning: every model receives
``w_c = N / (C * N_c)`` and the training set is left at its natural class
distribution.

Run standalone::

    python branches/branch_A_class_weighting/run.py --data_dir <path> --smoke_test
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from branches._common import run_branch   # noqa: E402

if __name__ == "__main__":
    run_branch(Path(__file__).parent, "Branch A -- class weighting only")
