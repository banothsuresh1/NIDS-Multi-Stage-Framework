#!/usr/bin/env python3
"""Branch C: SMOTE-ENN and class weighting together, time-blocked split.

The MAIN experiment of the methodology; the root pipeline defaults to this
configuration.

Run standalone::

    python branches/branch_C_both/run.py --data_dir <path> --smoke_test
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from branches._common import run_branch   # noqa: E402

if __name__ == "__main__":
    run_branch(Path(__file__).parent, "Branch C -- SMOTE-ENN + class weighting (main)")
