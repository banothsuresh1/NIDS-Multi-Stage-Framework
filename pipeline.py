"""Shared orchestrator wiring Stages 1-9 into one run.

``run_pipeline`` is the single entry point used by ``run_all.py``, by every
``branches/*/run.py`` and by ``pipeline.ipynb``, so no branch ever copies module
code. A branch supplies a config override; everything else is identical.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import data_preprocessing as dp
import evaluation as ev
import feature_selection as fsel
import fusion as fus
import imbalance_handling as ih
import models as M
import temporal_pattern_mining as tpm
import utils
from utils import (
    SplitIndex,
    Timer,
    assert_no_leakage,
    assert_not_resampled,
    ensure_dir,
    get_logger,
    save_json,
    save_table,
)

LOGGER = get_logger()


@dataclass
class PipelineResult:
    """Everything one pipeline run produces."""

    cfg: Dict[str, Any]
    df: pd.DataFrame
    split: SplitIndex
    feature_result: fsel.FeatureSelectionResult
    models: Dict[str, M.BaseModel]
    fusion: Optional[fus.EvidenceFusion]
    fusion_test: Optional[fus.FusionWeights]
    temporal: Optional[tpm.TemporalEvidence]
    test_proba: Dict[str, np.ndarray]
    metrics: pd.DataFrame
    per_class: pd.DataFrame
    ablations: pd.DataFrame
    resample_report: ih.ResampleReport
    leakage_report: Dict[str, bool]
    timings: Dict[str, float] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


def _host_column(df: pd.DataFrame) -> Optional[np.ndarray]:
    """A host identifier for graph construction (never used as a feature)."""
    for col in ("Source IP", "Src IP"):
        if col in df.columns:
            return df[col].to_numpy()
    return None


def _context(df: pd.DataFrame, rows: np.ndarray) -> Dict[str, Any]:
    """Session / time / host context for the sequence and graph models."""
    hosts = _host_column(df)
    return {
        "session_ids": df[dp.COL_SESSION].to_numpy()[rows],
        "times": df[dp.COL_TIME].to_numpy()[rows],
        "hosts": hosts[rows] if hosts is not None else None,
    }


def run_pipeline(
    data_dir: str,
    cfg: Optional[Mapping[str, Any]] = None,
    config_path: str = "config.yaml",
    overrides: Optional[Mapping[str, Any]] = None,
    seed: Optional[int] = None,
    out_dir: Optional[str] = None,
    tag: str = "",
    make_plots: bool = True,
    held_out_family: Optional[str] = None,
    df_cache: Optional[Tuple[pd.DataFrame, List[str], Dict[str, Any]]] = None,
) -> PipelineResult:
    """Run Stages 1-9 end to end and write every artifact into ``out_dir``."""
    cfg = dict(cfg) if cfg is not None else utils.load_config(config_path, overrides)
    seed = int(seed if seed is not None else utils.seeded(cfg))
    out_dir = Path(out_dir or cfg["run"]["output_dir"])
    ensure_dir(out_dir)
    timings: Dict[str, float] = {}

    utils.set_global_seed(seed)
    env = utils.configure_frameworks(cfg)
    LOGGER.info("=" * 78)
    LOGGER.info("RUN %s | seed=%d | smoke=%s | imbalance=%s | split=%s",
                cfg["run"].get("name", tag or "root"), seed,
                cfg["run"].get("smoke_test"), cfg["imbalance"]["mode"],
                cfg["split"]["strategy"])
    LOGGER.info("=" * 78)
    utils.save_config_snapshot(cfg, out_dir)

    classes = list(range(len(cfg["labels"]["class_order"])))
    class_names = {i: n for i, n in enumerate(cfg["labels"]["class_order"])}

    # ---------------------------------------------------------------- 1-2
    with Timer("stage_1_2_preprocessing", timings, LOGGER):
        if df_cache is not None:
            df, feature_columns, prep_report = df_cache
        else:
            df, feature_columns, prep_report = dp.prepare_dataset(
                cfg, data_dir=data_dir, seed=seed, out_dir=str(out_dir),
                use_cache=bool(cfg["run"].get("use_cache", True)))
    dist = dp.class_distribution_table(df, cfg)
    save_table(dist, out_dir / "class_distribution.csv")
    # prepare_dataset returns early on a parquet cache hit, so the Stage 2
    # report is written here rather than there -- otherwise a cached run leaves
    # results/ without it.
    save_json(prep_report, out_dir / "preprocessing_report.json")
    if make_plots:
        ev.plot_class_distribution(dist, cfg, out_dir / "class_distribution.png")

    # ------------------------------------------------------------------ 3
    with Timer("stage_3_split", timings, LOGGER):
        split = dp.make_split(df, cfg, seed, held_out=held_out_family)
    split_table = dp.split_distribution_table(df, split, cfg)
    save_table(split_table, out_dir / "split_distribution.csv")
    LOGGER.info("Split class counts:\n%s", split_table.to_string(index=False))

    leakage_report = assert_no_leakage(
        split, df[dp.COL_SESSION].to_numpy(), df[dp.COL_TIME].to_numpy(),
        df[dp.COL_LABEL].to_numpy(), logger=LOGGER)
    save_json({"boundaries": split.boundaries, "sizes": split.sizes(),
               "gap_dropped": int(len(split.dropped_gap)),
               "leakage_checks": leakage_report,
               "n_exempt_sessions": len(split.meta.get("exempt_sessions", []) or [])},
              out_dir / "split_report.json")

    y = df[dp.COL_Y].to_numpy()
    y_tr, y_va, y_te = y[split.train], y[split.val], y[split.test]

    # ------------------------------------------------------- 2: transform
    with Timer("stage_2_fit_transform", timings, LOGGER):
        pre = dp.TrainFittedPreprocessor(cfg).fit(
            df, feature_columns, rows=split.train, split=split)
        X_tr = pre.transform(df, split.train)
        X_va = pre.transform(df, split.val)
        X_te = pre.transform(df, split.test)

    # ------------------------------------------------------------------ 4
    with Timer("stage_4_feature_selection", timings, LOGGER):
        if cfg["feature_selection"].get("enabled", True):
            fs_result = fsel.run_feature_selection(
                X_tr, y_tr, X_va, y_va, cfg, seed,
                session_ids_train=df[dp.COL_SESSION].to_numpy()[split.train],
                times_train=df[dp.COL_TIME].to_numpy()[split.train],
                out_dir=str(out_dir), cache_tag=tag or cfg["run"].get("name", ""),
            )
        else:
            fs_result = fsel.FeatureSelectionResult(
                f_star=list(X_tr.columns), n_star=X_tr.shape[1],
                pool=list(X_tr.columns), f_corr=list(X_tr.columns),
                votes_table=pd.DataFrame(), n_star_curve=pd.DataFrame(),
                ranker_scores={}, redundancy_report=pd.DataFrame())

    # F* is frozen here and applied unchanged to all three partitions.
    F_tr = fs_result.apply(X_tr, "train")
    F_va = fs_result.apply(X_va, "validation")
    F_te = fs_result.apply(X_te, "test")

    # Stage 8.2 ablations / 8.3 sensitivity (config flags; off by default).
    if cfg["feature_selection"].get("ablation", {}).get("enabled", False):
        with Timer("stage_8_2_fs_ablations", timings, LOGGER):
            fsel.run_feature_ablations(X_tr, y_tr, X_va, y_va, fs_result, cfg,
                                       seed, out_dir=str(out_dir),
                                       X_test=X_te, y_test=y_te)
    if cfg["feature_selection"].get("sensitivity", {}).get("enabled", False):
        with Timer("stage_8_3_fs_sensitivity", timings, LOGGER):
            fsel.run_sensitivity(X_tr, y_tr, X_va, y_va, fs_result, cfg, seed,
                                 out_dir=str(out_dir))

    if make_plots and len(fs_result.votes_table):
        ev.plot_feature_votes(fs_result.votes_table, cfg, out_dir / "feature_votes_heatmap.png")
        ev.plot_rank_scores(fs_result.votes_table, cfg, out_dir / "feature_rank_scores.png")
        ev.plot_n_star_curve(fs_result.n_star_curve, fs_result.n_star, cfg,
                             out_dir / "n_star_curve.png")
        if fs_result.stability is not None:
            ev.plot_stability(fs_result.stability, cfg, out_dir / "feature_stability.png")
        if "shap_lgbm" in fs_result.ranker_scores:
            ev.plot_shap_summary(fs_result.ranker_scores["shap_lgbm"],
                                 fs_result.f_corr, cfg, out_dir / "shap_summary.png")

    # ------------------------------------------------------------------ 5
    n_val_before, n_test_before = len(F_va), len(F_te)
    with Timer("stage_5_imbalance", timings, LOGGER):
        X_bal, y_bal, class_weights, resample_report = ih.apply_imbalance_handling(
            F_tr, y_tr, cfg, seed, classes=np.array(classes))
    # Validation and test must be untouched by Stage 5.
    assert_not_resampled(n_val_before, len(F_va), "validation partition")
    assert_not_resampled(n_test_before, len(F_te), "test partition")
    save_table(resample_report.to_frame(class_names), out_dir / "imbalance_report.csv")

    # ------------------------------------------------------------------ 9
    temporal = None
    sp_tr = tc_tr = None
    if cfg["temporal"].get("enabled", True):
        with Timer("stage_9_temporal_mining", timings, LOGGER):
            temporal = tpm.run_temporal_mining(
                df, split.train, cfg, out_dir=str(out_dir), class_names=class_names)
        if make_plots:
            ev.plot_attack_state_graph(temporal.graph, cfg, out_dir / "attack_state_graph.png")
            ev.plot_top_patterns(temporal.sequences_patterns, cfg, out_dir / "top_patterns.png")

    # ------------------------------------------------------------------ 6
    ctx_tr, ctx_va, ctx_te = (_context(df, split.train), _context(df, split.val),
                              _context(df, split.test))

    # The sequence/graph models cannot consume synthetic rows (no session, no
    # timestamp, no host), so unless imbalance.apply_to_sequence_models is set
    # they train on the original ordered flows with class weighting only.
    seq_models = set(cfg["imbalance"].get("sequence_models", ["lstm", "gnn"]))
    resampled = len(X_bal) != len(F_tr)
    split_training = resampled and not cfg["imbalance"].get("apply_to_sequence_models", False)

    with Timer("stage_6_models", timings, LOGGER):
        flow_cfg = dict(cfg)
        flow_cfg["models"] = {**cfg["models"],
                              "enabled": [m for m in cfg["models"]["enabled"]
                                          if not (split_training and m in seq_models)]}
        trained = M.train_all_models(
            X_bal, y_bal, F_va, y_va, flow_cfg, classes, seed,
            class_weights=class_weights, train_context=ctx_tr, val_context=ctx_va,
            model_dir=cfg["run"].get("model_dir"))

        if split_training:
            seq_cfg = dict(cfg)
            seq_cfg["models"] = {**cfg["models"],
                                 "enabled": [m for m in cfg["models"]["enabled"]
                                             if m in seq_models]}
            LOGGER.info("Stage 5/6: %s train on the UN-resampled flows with class "
                        "weighting (synthetic rows carry no session, time or host)",
                        sorted(seq_cfg["models"]["enabled"]))
            weights_for_seq = ih.compute_class_weights(y_tr, cfg, np.array(classes))
            trained.update(M.train_all_models(
                F_tr, y_tr, F_va, y_va, seq_cfg, classes, seed,
                class_weights=weights_for_seq, train_context=ctx_tr,
                val_context=ctx_va, model_dir=cfg["run"].get("model_dir")))

    if not trained:
        raise RuntimeError("no model trained successfully; cannot continue")

    with Timer("stage_6_predict", timings, LOGGER):
        val_proba = M.predict_all(trained, F_va, ctx_va)
        test_proba = M.predict_all(trained, F_te, ctx_te)
    stats_table = M.model_stats_table(trained)
    save_table(stats_table, out_dir / "model_efficiency.csv")

    # ------------------------------------------------------------------ 7
    fusion_obj: Optional[fus.EvidenceFusion] = None
    fusion_test: Optional[fus.FusionWeights] = None
    ablations = pd.DataFrame()
    proba_for_eval: Dict[str, np.ndarray] = dict(test_proba)

    with Timer("stage_7_fusion", timings, LOGGER):
        y_va_idx = np.array([classes.index(v) if v in classes else 0 for v in y_va])
        calibrators = fus.fit_calibrators(val_proba, y_va_idx, cfg)
        val_cal = fus.apply_calibrators(val_proba, calibrators)
        test_cal = fus.apply_calibrators(test_proba, calibrators)

        use_temporal = (cfg["fusion"]["mode"] == "with_temporal" and temporal is not None)
        ev_va = ev_te = None
        if use_temporal:
            ev_va = temporal.evidence_for(df, split.val, cfg)
            ev_te = temporal.evidence_for(df, split.test, cfg)

        events_va = (temporal.discretiser.transform(df.iloc[split.val])
                     if temporal is not None and temporal.discretiser is not None else None)
        events_te = (temporal.discretiser.transform(df.iloc[split.test])
                     if temporal is not None and temporal.discretiser is not None else None)
        prev_va = fus.previous_states(df, split.val, cfg, events_va) if temporal else None
        prev_te = fus.previous_states(df, split.test, cfg, events_te) if temporal else None

        fusion_obj = fus.EvidenceFusion(
            cfg=cfg, classes=classes, class_names=class_names,
            calibrators=calibrators, use_temporal=use_temporal,
            graph=temporal.graph if temporal else None)
        fusion_obj.fit(
            val_cal, y_va, prev_state=prev_va,
            sp=ev_va["sp"] if ev_va else None, tc=ev_va["tc"] if ev_va else None,
            availability=ev_va["availability"] if ev_va else None,
            graph=temporal.graph if temporal else None)

        fusion_test = fusion_obj.fuse(
            test_cal, prev_state=prev_te,
            sp=ev_te["sp"] if ev_te else None, tc=ev_te["tc"] if ev_te else None,
            availability=ev_te["availability"] if ev_te else None)
        fus.save_fusion_artifacts(fusion_test, fusion_obj, out_dir, tag)
        proba_for_eval["fusion"] = fusion_test.fused_proba
        for name, p in test_cal.items():
            proba_for_eval[f"{name}_calibrated"] = p

        if cfg["fusion"]["ablations"].get("enabled", True):
            ablations = fus.run_fusion_ablations(
                fusion_obj, test_cal, y_te, prev_te,
                ev_te["sp"] if ev_te else None, ev_te["tc"] if ev_te else None,
                ev_te["availability"] if ev_te else None, cfg)
            if len(ablations):
                save_table(ablations, out_dir / "fusion_ablations.csv")
                LOGGER.info("Stage 7.6 ablations:\n%s", ablations.to_string(index=False))

    if make_plots and fusion_test is not None:
        ev.plot_fusion_weights(fusion_test, cfg, out_dir / "fusion_weights.png")
        best_nn = next((m for m in ("cnn", "gnn", "rf") if m in val_proba), None)
        if best_nn:
            ev.plot_reliability_diagram(
                val_proba[best_nn], val_cal.get(best_nn), y_va_idx, cfg,
                out_dir / f"reliability_{best_nn}.png",
                title=f"Reliability diagram -- {best_nn} (validation)")

    # ------------------------------------------------------------------ 8
    with Timer("stage_8_evaluation", timings, LOGGER):
        report = ev.full_evaluation(
            y_te, proba_for_eval, classes, cfg, out_dir, class_names,
            tag=tag, make_plots=make_plots, extra_stats=stats_table)

    summary: Dict[str, Any] = {
        "run_name": cfg["run"].get("name", tag),
        "seed": seed,
        "smoke_test": bool(cfg["run"].get("smoke_test")),
        "split_strategy": cfg["split"]["strategy"],
        "imbalance_mode": cfg["imbalance"]["mode"],
        "fusion_mode": cfg["fusion"]["mode"],
        "use_dst_port": bool(cfg["data"].get("use_dst_port")),
        "n_rows": int(len(df)),
        "n_features_initial": len(feature_columns),
        "n_features_f_corr": len(fs_result.f_corr),
        "n_star": fs_result.n_star,
        "f_star": fs_result.f_star,
        "split_sizes": split.sizes(),
        "gap_dropped": int(len(split.dropped_gap)),
        "train_rows_after_imbalance": int(len(X_bal)),
        "models_trained": sorted(trained),
        "tau_R": float(fusion_obj.tau_R) if fusion_obj else None,
        "lambdas": list(fusion_obj.lambdas) if fusion_obj else None,
        "Q": fusion_obj.Q if fusion_obj else None,
        "leakage_checks": leakage_report,
        "timings_s": timings,
        "environment": env,
        "best_model": report["metrics"].iloc[0]["label"],
        "best_macro_f1": float(report["metrics"].iloc[0]["macro_f1"]),
        "fusion_macro_f1": float(
            report["metrics"].loc[report["metrics"]["label"] == "fusion", "macro_f1"].iloc[0]
        ) if "fusion" in set(report["metrics"]["label"]) else None,
    }

    if held_out_family is not None:
        pred = np.array([classes[i] for i in np.argmax(proba_for_eval["fusion"], axis=1)])
        benign = cfg["labels"]["class_order"].index(cfg["labels"]["benign_class"])
        held_idx = cfg["labels"]["class_order"].index(held_out_family)
        summary["zero_day"] = ev.zero_day_metrics(y_te, pred, held_idx, benign)

    save_json(summary, out_dir / "summary.json")
    LOGGER.info("Run complete. Best: %s (macro-F1 %.4f). Artifacts -> %s",
                summary["best_model"], summary["best_macro_f1"], out_dir)

    return PipelineResult(
        cfg=cfg, df=df, split=split, feature_result=fs_result, models=trained,
        fusion=fusion_obj, fusion_test=fusion_test, temporal=temporal,
        test_proba=proba_for_eval, metrics=report["metrics"],
        per_class=report["per_class"], ablations=ablations,
        resample_report=resample_report, leakage_report=leakage_report,
        timings=timings, summary=summary,
    )


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """The CLI shared by run_all.py and every branch runner."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--data_dir", required=True,
                    help="path to the CIC-IDS2017 directory of daily CSVs")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--out_dir", default=None, help="override the output directory")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", "--smoke_test", dest="smoke_test", action="store_true",
                    help="force smoke mode: stratified subsample + tiny epochs, "
                         "written under its own cache signature (quick test only)")
    ap.add_argument("--full", action="store_true",
                    help="force full mode (the default; use to override a config "
                         "that sets run.smoke_test)")
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--no_cache", action="store_true")
    return ap


def cfg_from_args(args: argparse.Namespace, config_path: str,
                  overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Resolve the config from CLI flags plus a branch override."""
    over: Dict[str, Any] = dict(overrides or {})
    run_over: Dict[str, Any] = dict(over.get("run", {}))
    if args.smoke_test:
        run_over["smoke_test"] = True
    if args.full:
        run_over["smoke_test"] = False
    if args.no_cache:
        run_over["use_cache"] = False
    if args.out_dir:
        run_over["output_dir"] = args.out_dir
    if run_over:
        over["run"] = run_over
    if args.seed is not None:
        over["seed"] = {**over.get("seed", {}), "primary": args.seed}
    return utils.load_config(args.config or config_path, over)
