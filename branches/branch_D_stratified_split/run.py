#!/usr/bin/env python3
"""Branch D: stratified splits and the leakage gap.

Runs the pipeline three ways on identical data:

  * ``stratified_flow``    -- plain flow-level stratified split. Sessions are
    torn apart, so near-duplicate flows of one session land on both sides. This
    is the protocol most published CIC-IDS2017 numbers use.
  * ``stratified_session`` -- stratified but session-grouped: random, yet no
    session crosses the split.
  * ``time_blocked``       -- the methodology's main protocol (train strictly
    before validation strictly before test).

The *leakage gap* is the drop from the flow-level number to the time-blocked
number: the share of the reported score that comes from the evaluation protocol
rather than from the detector.

Run standalone::

    python branches/branch_D_stratified_split/run.py --data_dir <path> --smoke_test
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import data_preprocessing as dp                     # noqa: E402
import pipeline                                    # noqa: E402
import utils                                       # noqa: E402
from branches._common import resolve_branch_config  # noqa: E402

LOGGER = utils.get_logger()

VARIANTS = ["stratified_flow", "stratified_session", "time_blocked"]


def main() -> None:
    ap = pipeline.build_arg_parser("Branch D -- stratified splits and the leakage gap")
    args = ap.parse_args()
    base_cfg = resolve_branch_config(Path(__file__).parent, args)
    out_root = Path(base_cfg["run"]["output_dir"])

    # Load once: Stages 1-2 do not depend on the split strategy. The cached
    # tuple must carry the Stage 2 CANDIDATE feature columns, not F_corr --
    # F_corr is what survived the redundancy filter, and feeding it back would
    # shrink the candidate pool on every subsequent variant.
    utils.set_global_seed(utils.seeded(base_cfg))
    df0, candidate_features, prep_report = dp.prepare_dataset(
        base_cfg, data_dir=args.data_dir, seed=utils.seeded(base_cfg),
        out_dir=str(out_root), use_cache=bool(base_cfg["run"].get("use_cache", True)))
    df_cache = (df0, candidate_features, prep_report)

    rows = []
    for strategy in VARIANTS:
        cfg = utils.deep_merge(base_cfg, {
            "split": {"strategy": strategy},
            "run": {"name": f"branch_D_{strategy}",
                    "output_dir": str(out_root / strategy)},
        })
        LOGGER.info("Branch D: running split strategy %s", strategy)
        result = pipeline.run_pipeline(
            data_dir=args.data_dir, cfg=cfg, out_dir=cfg["run"]["output_dir"],
            tag=strategy, make_plots=not args.no_plots, df_cache=df_cache,
        )
        # Stages 1-2 are identical across variants, so the cleaned frame is
        # loaded once above and reused; nothing to do per variant.
        best = result.metrics.iloc[0]
        fusion = result.metrics.loc[result.metrics["label"] == "fusion"]
        rows.append({
            "split_strategy": strategy,
            "best_label": best["label"],
            "best_macro_f1": float(best["macro_f1"]),
            "best_accuracy": float(best["accuracy"]),
            "fusion_macro_f1": float(fusion["macro_f1"].iloc[0]) if len(fusion) else None,
            "fusion_accuracy": float(fusion["accuracy"].iloc[0]) if len(fusion) else None,
            "n_star": result.feature_result.n_star,
            **{f"n_{k}": v for k, v in result.split.sizes().items()},
        })

    table = pd.DataFrame(rows)
    # Leakage gap, measured against the time-blocked protocol.
    ref = table.loc[table["split_strategy"] == "time_blocked"]
    if len(ref):
        for col in ("best_macro_f1", "fusion_macro_f1", "best_accuracy"):
            base = ref[col].iloc[0]
            if base is not None:
                table[f"gap_vs_time_blocked_{col}"] = table[col] - base

    utils.save_table(table, out_root / "leakage_gap.csv")
    utils.save_json({"variants": rows,
                     "note": "gap_* columns are the optimism of each protocol "
                             "relative to the time-blocked split"},
                    out_root / "summary.json")
    print("\n=== Branch D -- leakage gap ===")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
