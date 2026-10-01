#!/usr/bin/env python3
"""Branch B: SMOTE-ENN only (no class weights), time-blocked split.

Isolates the contribution of resampling. The LSTM and GNN still train on the
un-resampled flows (a synthetic sample has no session, timestamp or host), so in
this branch they receive neither correction -- which is exactly the comparison
``imbalance.apply_to_sequence_models`` exists to make explicit.

Run standalone::

    python branches/branch_B_smote_enn/run.py --data_dir <path> --smoke_test
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from branches._common import run_branch   # noqa: E402

if __name__ == "__main__":
    run_branch(Path(__file__).parent, "Branch B -- SMOTE-ENN only")
