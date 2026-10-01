#!/usr/bin/env python3
"""Run the root pipeline and all five branches, then aggregate the comparison.

Smoke test (minutes)::

    python run_all.py --data_dir data/synthetic --smoke_test

Full run::

    python run_all.py --data_dir /path/to/MachineLearningCVE --full

Individual branches can also be run standalone; see ``branches/*/run.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import evaluation as ev        # noqa: E402
import pipeline                # noqa: E402
import utils                   # noqa: E402

LOGGER = utils.get_logger()

BRANCHES = [
    ("branch_A_class_weighting", "A: class weighting only"),
    ("branch_B_smote_enn", "B: SMOTE-ENN only"),
    ("branch_C_both", "C: both (main)"),
    ("branch_D_stratified_split", "D: stratified splits / leakage gap"),
    ("branch_E_zero_day", "E: zero-day leave-one-attack-out"),
]


def _branch_summary(branch: str) -> Optional[Dict[str, Any]]:
    path = ROOT / "branches" / branch / "results" / "summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def aggregate(out_dir: Path) -> pd.DataFrame:
    """Build ``results/branch_comparison.csv`` from each branch's summary.json."""
    rows: List[Dict[str, Any]] = []

    root_summary = out_dir / "summary.json"
    if root_summary.exists():
        s = json.loads(root_summary.read_text())
        rows.append({
            "branch": "root (= branch C)",
            "imbalance_mode": s.get("imbalance_mode"),
            "split_strategy": s.get("split_strategy"),
            "n_star": s.get("n_star"),
            "best_model": s.get("best_model"),
            "macro_f1": s.get("fusion_macro_f1") or s.get("best_macro_f1"),
            "best_macro_f1": s.get("best_macro_f1"),
        })

    for branch, label in BRANCHES:
        s = _branch_summary(branch)
        if s is None:
            rows.append({"branch": label, "note": "no summary.json (not run or failed)"})
            continue
        if branch == "branch_D_stratified_split":
            for v in s.get("variants", []):
                rows.append({
                    "branch": f"D: {v['split_strategy']}",
                    "split_strategy": v["split_strategy"],
                    "imbalance_mode": "both",
                    "n_star": v.get("n_star"),
                    "macro_f1": v.get("fusion_macro_f1"),
                    "best_macro_f1": v.get("best_macro_f1"),
                    "accuracy": v.get("best_accuracy"),
                })
        elif branch == "branch_E_zero_day":
            rows.append({
                "branch": label,
                "split_strategy": "loao",
                "imbalance_mode": "both",
                "unseen_detection_recall": s.get("mean_unseen_detection_recall"),
                "false_alarm_rate": s.get("mean_false_alarm_rate"),
                "families_evaluated": s.get("families_evaluated"),
            })
        else:
            rows.append({
                "branch": label,
                "imbalance_mode": s.get("imbalance_mode"),
                "split_strategy": s.get("split_strategy"),
                "n_star": s.get("n_star"),
                "best_model": s.get("best_model"),
                "macro_f1": s.get("fusion_macro_f1") or s.get("best_macro_f1"),
                "best_macro_f1": s.get("best_macro_f1"),
                "accuracy": s.get("accuracy"),
            })

    table = pd.DataFrame(rows)
    utils.save_table(table, out_dir / "branch_comparison.csv")
    return table


def main() -> int:
    ap = pipeline.build_arg_parser("Run the root pipeline and all five branches")
    ap.add_argument("--skip_branches", action="store_true",
                    help="run only the root pipeline")
    ap.add_argument("--only", default=None,
                    help="comma-separated branch directory names to run")
    args = ap.parse_args()

    cfg = pipeline.cfg_from_args(args, str(ROOT / "config.yaml"))
    out_dir = Path(cfg["run"]["output_dir"])
    utils.ensure_dir(out_dir)
    failures: List[str] = []

    # ---------------- root pipeline (= branch C) ----------------
    LOGGER.info("### ROOT PIPELINE ###")
    try:
        pipeline.run_pipeline(
            data_dir=args.data_dir, cfg=cfg, out_dir=str(out_dir),
            tag="root", make_plots=not args.no_plots)
    except Exception:
        traceback.print_exc()
        failures.append("root")

    # ---------------- branches ----------------
    if not args.skip_branches:
        wanted = ([b.strip() for b in args.only.split(",")] if args.only
                  else [b for b, _ in BRANCHES])
        passthrough = []
        if args.smoke_test:
            passthrough.append("--smoke_test")
        if args.full:
            passthrough.append("--full")
        if args.no_plots:
            passthrough.append("--no_plots")
        if args.no_cache:
            passthrough.append("--no_cache")
        if args.seed is not None:
            passthrough += ["--seed", str(args.seed)]

        for branch, label in BRANCHES:
            if branch not in wanted:
                continue
            LOGGER.info("### %s ###", label)
            cmd = [sys.executable, str(ROOT / "branches" / branch / "run.py"),
                   "--data_dir", args.data_dir] + passthrough
            proc = subprocess.run(cmd, cwd=str(ROOT))
            if proc.returncode != 0:
                LOGGER.error("%s exited with code %d", label, proc.returncode)
                failures.append(branch)

    # ---------------- aggregate ----------------
    table = aggregate(out_dir)
    print("\n================ BRANCH COMPARISON ================")
    print(table.to_string(index=False))
    if not args.no_plots and "macro_f1" in table.columns:
        plot_rows = table.dropna(subset=["macro_f1"])
        if len(plot_rows):
            ev.plot_branch_comparison(plot_rows.reset_index(drop=True), cfg,
                                      out_dir / "branch_comparison.png")

    print_checklist(out_dir)
    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1
    print("\nAll stages completed successfully.")
    return 0


def print_checklist(out_dir: Path) -> None:
    """Stage -> file -> status table (Definition of Done item 4)."""
    checks = [
        ("Stage 1: dataset consolidation", "data_preprocessing.py", out_dir / "class_distribution.csv"),
        ("Stage 2: preprocessing", "data_preprocessing.py", out_dir / "preprocessing_report.json"),
        ("Stage 3: splits", "data_preprocessing.py", out_dir / "split_distribution.csv"),
        ("Stage 4: feature selection", "feature_selection.py", out_dir / "feature_votes.csv"),
        ("Stage 5: imbalance handling", "imbalance_handling.py", out_dir / "imbalance_report.csv"),
        ("Stage 6: models", "models.py", out_dir / "model_efficiency.csv"),
        ("Stage 7: evidence fusion", "fusion.py", out_dir / "fusion_weights.csv"),
        ("Stage 8: evaluation", "evaluation.py", out_dir / "metrics.csv"),
        ("Stage 9: temporal mining", "temporal_pattern_mining.py", out_dir / "attack_state_graph.json"),
        ("Branches A-E", "branches/*/run.py", out_dir / "branch_comparison.csv"),
    ]
    rows = [(stage, file, "OK" if Path(art).exists() else "MISSING")
            for stage, file, art in checks]
    print("\n================ STAGE CHECKLIST ================")
    print(utils.format_checklist(rows))


if __name__ == "__main__":
    sys.exit(main())
