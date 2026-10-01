#!/usr/bin/env python3
"""Branch E: zero-day detection by leave-one-attack-out.

For each attack family in turn the family is removed from training and
validation entirely -- whole sessions, so no fragment of an attack session is
ever seen -- and placed in test. Feature selection, imbalance handling, the five
models, the calibrators and the fusion weights are all refit from scratch for
each held-out family, because every one of them would otherwise have seen the
family it is supposed to be blind to.

Reported per family:
  * ``unseen_detection_recall`` -- share of the unseen family's flows flagged as
    *some* attack (the model cannot name a class it never saw, so "detected"
    means "not classified as benign");
  * ``false_alarm_rate``        -- share of benign test flows wrongly flagged.

Run standalone::

    python branches/branch_E_zero_day/run.py --data_dir <path> --smoke_test
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import data_preprocessing as dp                     # noqa: E402
import pipeline                                     # noqa: E402
import utils                                        # noqa: E402
from branches._common import resolve_branch_config   # noqa: E402

LOGGER = utils.get_logger()


def main() -> None:
    ap = pipeline.build_arg_parser("Branch E -- zero-day leave-one-attack-out")
    ap.add_argument("--families", default=None,
                    help="comma-separated subset of attack families to hold out")
    args = ap.parse_args()
    base_cfg = resolve_branch_config(Path(__file__).parent, args)
    out_root = Path(base_cfg["run"]["output_dir"])
    utils.ensure_dir(out_root)

    # Load once to see which families are actually present.
    utils.set_global_seed(utils.seeded(base_cfg))
    df, feature_columns, prep_report = dp.prepare_dataset(
        base_cfg, data_dir=args.data_dir, seed=utils.seeded(base_cfg),
        out_dir=str(out_root), use_cache=bool(base_cfg["run"].get("use_cache", True)))
    counts = df[dp.COL_LABEL].value_counts().to_dict()

    be = base_cfg.get("branch_e", {}) or {}
    if args.families:
        families = [f.strip() for f in args.families.split(",") if f.strip()]
    else:
        families = be.get("families") or base_cfg["labels"]["attack_families"]
    min_flows = int(be.get("min_family_flows", 5))

    rows = []
    for family in families:
        n = int(counts.get(family, 0))
        if n < min_flows:
            LOGGER.warning("Branch E: skipping %s (%d flows < min_family_flows=%d)",
                           family, n, min_flows)
            rows.append({"held_out_family": family, "n_unseen": n, "status": "skipped"})
            continue

        cfg = utils.deep_merge(base_cfg, {
            "run": {"name": f"branch_E_loao_{family.replace(' ', '_')}",
                    "output_dir": str(out_root / family.replace(" ", "_"))},
        })
        LOGGER.info("Branch E: holding out %s (%d flows)", family, n)
        try:
            result = pipeline.run_pipeline(
                data_dir=args.data_dir, cfg=cfg, out_dir=cfg["run"]["output_dir"],
                tag=f"loao_{family.replace(' ', '_')}",
                make_plots=not args.no_plots, held_out_family=family,
                df_cache=(df, feature_columns, prep_report),
            )
        except Exception as exc:
            LOGGER.error("Branch E: %s failed (%s)", family, exc)
            rows.append({"held_out_family": family, "n_unseen": n,
                         "status": f"failed: {exc}"})
            continue

        zd = result.summary.get("zero_day", {})
        fusion = result.metrics.loc[result.metrics["label"] == "fusion"]
        rows.append({
            "held_out_family": family,
            "status": "ok",
            "n_unseen": zd.get("n_unseen", n),
            "unseen_detection_recall": zd.get("unseen_detection_recall"),
            "false_alarm_rate": zd.get("false_alarm_rate"),
            "n_benign_test": zd.get("n_benign"),
            "fusion_macro_f1_seen_classes": (float(fusion["macro_f1"].iloc[0])
                                             if len(fusion) else None),
            "n_star": result.feature_result.n_star,
        })

    table = pd.DataFrame(rows)
    utils.save_table(table, out_root / "zero_day_results.csv")
    ok = table[table["status"] == "ok"] if "status" in table else table
    summary = {
        "families_evaluated": int(len(ok)),
        "mean_unseen_detection_recall": (float(ok["unseen_detection_recall"].mean())
                                         if len(ok) else None),
        "mean_false_alarm_rate": (float(ok["false_alarm_rate"].mean())
                                  if len(ok) else None),
        "per_family": rows,
    }
    utils.save_json(summary, out_root / "summary.json")
    print("\n=== Branch E -- zero-day (leave-one-attack-out) ===")
    print(table.to_string(index=False))
    if len(ok):
        print(f"\nmean unseen-family detection recall: "
              f"{summary['mean_unseen_detection_recall']:.3f}")
        print(f"mean benign false-alarm rate:        "
              f"{summary['mean_false_alarm_rate']:.3f}")


if __name__ == "__main__":
    main()
