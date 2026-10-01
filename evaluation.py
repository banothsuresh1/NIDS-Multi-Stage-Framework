"""Stage 8: metrics, calibration quality, plots and significance testing.

Primary metrics are macro-F1 and per-class recall, because the benign class
dominates accuracy (Stage 8.1). Everything is computed on the TEST partition
with models, calibrators, weights and thresholds already frozen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils import ensure_dir, get_logger, save_fig, save_json, save_table

LOGGER = get_logger()

# Headless by default: the notebook re-enables inline rendering itself.
import matplotlib

if matplotlib.get_backend().lower() not in ("module://matplotlib_inline.backend_inline",):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ===========================================================================
# Core metrics
# ===========================================================================
def expected_calibration_error(proba: np.ndarray, y_idx: np.ndarray,
                               n_bins: int = 15) -> float:
    """ECE over the predicted-class confidence, with equal-width bins."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_idx).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def brier_score(proba: np.ndarray, y_idx: np.ndarray) -> float:
    """Multi-class Brier score (mean squared error against the one-hot target)."""
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def evaluate_predictions(y_true: np.ndarray, proba: np.ndarray,
                         classes: Sequence[Any], cfg: Mapping[str, Any],
                         class_names: Optional[Mapping[Any, str]] = None,
                         label: str = "model") -> Dict[str, Any]:
    """Every Stage 8.1 metric for one probability matrix."""
    from sklearn.metrics import (
        accuracy_score, average_precision_score, classification_report,
        f1_score, precision_score, recall_score, roc_auc_score,
    )

    classes = list(classes)
    names = [str((class_names or {}).get(c, c)) for c in classes]
    index = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([index.get(v, 0) for v in y_true])
    pred_idx = np.argmax(proba, axis=1)
    pred = np.array([classes[i] for i in pred_idx])

    present = sorted(set(y_idx.tolist()) | set(pred_idx.tolist()))
    onehot = np.zeros((len(y_idx), len(classes)))
    onehot[np.arange(len(y_idx)), y_idx] = 1.0

    def safe(fn, *args, **kwargs):
        try:
            return float(fn(*args, **kwargs))
        except Exception:
            return float("nan")

    out: Dict[str, Any] = {
        "label": label,
        "n_samples": int(len(y_true)),
        "accuracy": safe(accuracy_score, y_true, pred),
        "precision_macro": safe(precision_score, y_true, pred, average="macro", zero_division=0),
        "recall_macro": safe(recall_score, y_true, pred, average="macro", zero_division=0),
        "macro_f1": safe(f1_score, y_true, pred, average="macro", zero_division=0),
        "weighted_f1": safe(f1_score, y_true, pred, average="weighted", zero_division=0),
        "precision_weighted": safe(precision_score, y_true, pred, average="weighted", zero_division=0),
        "recall_weighted": safe(recall_score, y_true, pred, average="weighted", zero_division=0),
        "ece": expected_calibration_error(proba, y_idx, int(cfg["evaluation"].get("ece_bins", 15))),
        "brier": brier_score(proba, y_idx),
    }
    # Macro one-vs-rest AUC / PR-AUC over the classes actually present.
    cols = [i for i in range(len(classes)) if onehot[:, i].sum() > 0]
    if len(cols) >= 2:
        out["roc_auc_ovr_macro"] = safe(roc_auc_score, onehot[:, cols], proba[:, cols],
                                        average="macro", multi_class="ovr")
        out["pr_auc_macro"] = safe(average_precision_score, onehot[:, cols],
                                   proba[:, cols], average="macro")
    else:
        out["roc_auc_ovr_macro"] = float("nan")
        out["pr_auc_macro"] = float("nan")

    # -- per-class block, with the rare classes flagged -------------------
    rare = set(cfg["evaluation"].get("rare_classes", []))
    report = classification_report(y_true, pred, labels=classes, target_names=names,
                                   output_dict=True, zero_division=0)
    per_class = []
    for c, n in zip(classes, names):
        entry = report.get(n, {})
        per_class.append({
            "class": n,
            "support": int(entry.get("support", 0)),
            "precision": float(entry.get("precision", 0.0)),
            "recall": float(entry.get("recall", 0.0)),
            "f1": float(entry.get("f1-score", 0.0)),
            "is_rare": n in rare,
        })
    out["per_class"] = per_class
    rare_recalls = [r["recall"] for r in per_class if r["is_rare"] and r["support"] > 0]
    out["rare_class_recall_macro"] = float(np.mean(rare_recalls)) if rare_recalls else float("nan")
    return out


def metrics_to_frame(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Flatten a list of :func:`evaluate_predictions` outputs into one table."""
    rows = []
    for r in results:
        rows.append({k: v for k, v in r.items() if k != "per_class"})
    return pd.DataFrame(rows)


def per_class_frame(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in results:
        for pc in r.get("per_class", []):
            rows.append({"label": r["label"], **pc})
    return pd.DataFrame(rows)


# ===========================================================================
# Multi-seed aggregation and significance (Stage 8.3)
# ===========================================================================
def aggregate_seeds(frames: Sequence[pd.DataFrame],
                    key: str = "label") -> pd.DataFrame:
    """Mean +/- std across seeds for every numeric metric."""
    if not frames:
        return pd.DataFrame()
    allf = pd.concat(frames, ignore_index=True)
    numeric = [c for c in allf.columns
               if c != key and pd.api.types.is_numeric_dtype(allf[c])]
    grouped = allf.groupby(key)[numeric]
    mean, std = grouped.mean(), grouped.std(ddof=1).fillna(0.0)
    out = mean.copy()
    for c in numeric:
        out[f"{c}_std"] = std[c]
    out["n_seeds"] = allf.groupby(key).size()
    return out.reset_index()


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray
                 ) -> Dict[str, float]:
    """Paired McNemar test on the two models' correct/incorrect disagreements."""
    from scipy import stats

    a_ok = (pred_a == y_true)
    b_ok = (pred_b == y_true)
    n01 = int(np.sum(a_ok & ~b_ok))
    n10 = int(np.sum(~a_ok & b_ok))
    if n01 + n10 == 0:
        return {"n01": n01, "n10": n10, "statistic": 0.0, "p_value": 1.0}
    # Exact binomial for small discordant counts, chi-square with continuity
    # correction otherwise.
    if n01 + n10 < 25:
        p = float(stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)
        stat = float(min(n01, n10))
    else:
        stat = (abs(n01 - n10) - 1.0) ** 2 / (n01 + n10)
        p = float(stats.chi2.sf(stat, df=1))
    return {"n01": n01, "n10": n10, "statistic": float(stat), "p_value": p}


def bootstrap_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                   n_iters: int = 1000, seed: int = 42) -> Dict[str, float]:
    """Paired bootstrap on the macro-F1 difference (a - b)."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    n = len(y_true)
    observed = (f1_score(y_true, pred_a, average="macro", zero_division=0)
                - f1_score(y_true, pred_b, average="macro", zero_division=0))
    diffs = np.empty(n_iters)
    for i in range(n_iters):
        idx = rng.integers(0, n, n)
        diffs[i] = (f1_score(y_true[idx], pred_a[idx], average="macro", zero_division=0)
                    - f1_score(y_true[idx], pred_b[idx], average="macro", zero_division=0))
    p = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"observed_diff": float(observed),
            "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)),
            "p_value": min(1.0, p)}


def compare_models(y_true: np.ndarray, predictions: Mapping[str, np.ndarray],
                   cfg: Mapping[str, Any], seed: int = 42) -> pd.DataFrame:
    """Pairwise significance tests between every pair of prediction vectors."""
    sig = cfg["evaluation"].get("significance", {})
    if not sig.get("enabled", True):
        return pd.DataFrame()
    test = str(sig.get("test", "mcnemar"))
    names = list(predictions)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if test == "bootstrap":
                res = bootstrap_test(y_true, predictions[a], predictions[b],
                                     int(sig.get("bootstrap_iters", 1000)), seed)
            else:
                res = mcnemar_test(y_true, predictions[a], predictions[b])
            rows.append({"model_a": a, "model_b": b, "test": test, **res,
                         "significant": res["p_value"] < float(sig.get("alpha", 0.05))})
    return pd.DataFrame(rows)


# ===========================================================================
# Plots (all saved at >= 200 dpi)
# ===========================================================================
def _dpi(cfg: Mapping[str, Any]) -> int:
    return int(cfg["run"].get("figure_dpi", 220))


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          classes: Sequence[Any], cfg: Mapping[str, Any],
                          out_path: str | Path, normalize: bool = False,
                          class_names: Optional[Mapping[Any, str]] = None,
                          title: str = "Confusion matrix"):
    from sklearn.metrics import confusion_matrix

    names = [str((class_names or {}).get(c, c)) for c in classes]
    cm = confusion_matrix(y_true, y_pred, labels=list(classes))
    data = cm.astype(np.float64)
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            data = data / np.maximum(data.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(1.0 + 0.72 * len(names), 0.9 + 0.62 * len(names)))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max() if data.max() > 0 else 1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("Predicted"), ax.set_ylabel("True")
    ax.set_title(title + (" (row-normalised)" if normalize else ""))
    thresh = data.max() / 2 if data.max() > 0 else 0.5
    for i in range(len(names)):
        for j in range(len(names)):
            txt = f"{data[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if data[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return save_fig(fig, out_path, _dpi(cfg))


def plot_roc_curves(y_true: np.ndarray, proba: np.ndarray, classes: Sequence[Any],
                    cfg: Mapping[str, Any], out_path: str | Path,
                    class_names: Optional[Mapping[Any, str]] = None):
    from sklearn.metrics import auc, roc_curve

    names = [str((class_names or {}).get(c, c)) for c in classes]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, c in enumerate(classes):
        target = (y_true == c).astype(int)
        if target.sum() == 0 or target.sum() == len(target):
            continue
        fpr, tpr, _ = roc_curve(target, proba[:, i])
        ax.plot(fpr, tpr, lw=1.4, label=f"{names[i]} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("False positive rate"), ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    return save_fig(fig, out_path, _dpi(cfg))


def plot_pr_curves(y_true: np.ndarray, proba: np.ndarray, classes: Sequence[Any],
                   cfg: Mapping[str, Any], out_path: str | Path,
                   class_names: Optional[Mapping[Any, str]] = None):
    from sklearn.metrics import average_precision_score, precision_recall_curve

    names = [str((class_names or {}).get(c, c)) for c in classes]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, c in enumerate(classes):
        target = (y_true == c).astype(int)
        if target.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(target, proba[:, i])
        ap = average_precision_score(target, proba[:, i])
        ax.plot(rec, prec, lw=1.4, label=f"{names[i]} (AP {ap:.3f})")
    ax.set_xlabel("Recall"), ax.set_ylabel("Precision")
    ax.set_title("One-vs-rest precision-recall curves")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    return save_fig(fig, out_path, _dpi(cfg))


def plot_reliability_diagram(proba_before: np.ndarray, proba_after: Optional[np.ndarray],
                             y_idx: np.ndarray, cfg: Mapping[str, Any],
                             out_path: str | Path, n_bins: int = 15,
                             title: str = "Reliability diagram"):
    """Confidence vs. accuracy, before and after calibration (Stage 7.1)."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    edges = np.linspace(0, 1, n_bins + 1)
    for proba, name, style in ((proba_before, "before calibration", "o-"),
                               (proba_after, "after calibration", "s-")):
        if proba is None:
            continue
        conf, correct = proba.max(axis=1), (proba.argmax(axis=1) == y_idx).astype(float)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() == 0:
                continue
            xs.append(conf[m].mean()), ys.append(correct[m].mean())
        ece = expected_calibration_error(proba, y_idx, n_bins)
        ax.plot(xs, ys, style, lw=1.4, ms=4, label=f"{name} (ECE {ece:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    ax.set_xlabel("Mean predicted confidence"), ax.set_ylabel("Empirical accuracy")
    ax.set_title(title), ax.legend(fontsize=8), ax.grid(alpha=0.3)
    return save_fig(fig, out_path, _dpi(cfg))


def plot_feature_votes(votes: pd.DataFrame, cfg: Mapping[str, Any],
                       out_path: str | Path, top_n: int = 30):
    """Ranker-membership heatmap: which FeS Set contains which feature."""
    from feature_selection import FES_SET_LABELS

    cols = [c for c in votes.columns if c.startswith("in_")]
    sub = votes.head(top_n)
    mat = sub[cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.6 + 0.55 * len(cols), 1.2 + 0.26 * len(sub)))
    ax.imshow(mat, cmap="Greens", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(sub)), sub["feature"], fontsize=7)
    ax.set_xticks(range(len(cols)),
                  [FES_SET_LABELS.get(c[3:], c[3:]).split(" (")[-1].rstrip(")")
                   for c in cols], rotation=45, ha="right", fontsize=8)
    for i in range(len(sub)):
        ax.text(len(cols) - 0.3, i, f"psi={int(sub['psi_votes'].iloc[i])}",
                va="center", fontsize=7)
    ax.set_title(f"Feature membership across the {len(cols)} rankers "
                 f"(top {len(sub)} by fused Score)")
    return save_fig(fig, out_path, _dpi(cfg))


def plot_rank_scores(votes: pd.DataFrame, cfg: Mapping[str, Any],
                     out_path: str | Path, top_n: int = 30):
    sub = votes.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 1.0 + 0.26 * len(sub)))
    colors = ["#2a7f3f" if p else "#9aa0a6" for p in sub["in_pool"]]
    ax.barh(range(len(sub)), sub["score"], color=colors)
    ax.set_yticks(range(len(sub)), sub["feature"], fontsize=7)
    ax.set_xlabel("Fused Score(f) = mean normalized rank over the rankers")
    ax.set_title("Rank-aggregation score (green = in the voted pool P)")
    ax.grid(axis="x", alpha=0.3)
    return save_fig(fig, out_path, _dpi(cfg))


def plot_n_star_curve(curve: pd.DataFrame, n_star: int, cfg: Mapping[str, Any],
                      out_path: str | Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(curve["N"], curve["macro_f1"], "o-", label="validation macro-F1")
    if "weighted_f1" in curve:
        ax.plot(curve["N"], curve["weighted_f1"], "s--", lw=1, label="validation weighted F1")
    if "minority_recall" in curve:
        ax.plot(curve["N"], curve["minority_recall"], "^:", lw=1, label="minority recall")
    ax.axvline(n_star, color="crimson", ls="--", lw=1.2, label=f"N* = {n_star}")
    ax.set_xlabel("Number of selected features N"), ax.set_ylabel("Score")
    ax.set_title("Subset-size selection on the validation partition")
    ax.legend(fontsize=8), ax.grid(alpha=0.3)
    return save_fig(fig, out_path, _dpi(cfg))


def plot_stability(stability: pd.DataFrame, cfg: Mapping[str, Any],
                   out_path: str | Path, top_n: int = 30):
    sub = stability.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 1.0 + 0.26 * len(sub)))
    ax.barh(range(len(sub)), sub["stability"], color="#3b6fb6")
    ax.set_yticks(range(len(sub)), sub["feature"], fontsize=7)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Stability(f) = selected runs / total runs")
    ax.set_title("Selection stability over repeated stratified subsamples")
    ax.grid(axis="x", alpha=0.3)
    return save_fig(fig, out_path, _dpi(cfg))


def plot_class_distribution(dist: pd.DataFrame, cfg: Mapping[str, Any],
                            out_path: str | Path):
    agg = dist.groupby("grouped_class")["records"].sum().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(agg.index, agg.values, color="#4a6fa5")
    ax.set_xscale("log")
    ax.set_xlabel("Flow records (log scale)")
    ax.set_title("Class distribution after label grouping")
    for i, v in enumerate(agg.values):
        ax.text(v, i, f" {int(v):,}", va="center", fontsize=7)
    ax.grid(axis="x", alpha=0.3)
    return save_fig(fig, out_path, _dpi(cfg))


def plot_attack_state_graph(graph: Any, cfg: Mapping[str, Any], out_path: str | Path,
                            min_weight: float = 0.05):
    """Draw G_T with edge weights (Stage 9)."""
    import networkx as nx

    g = graph.to_networkx(min_weight=min_weight)
    if g.number_of_nodes() == 0:
        return None
    fig, ax = plt.subplots(figsize=(9, 7))
    pos = nx.spring_layout(g, seed=42, k=1.4)
    weights = [g[u][v]["weight"] for u, v in g.edges()]
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color="#cfe3f7",
                           edgecolors="#2d5f91", node_size=1500)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=7)
    if weights:
        nx.draw_networkx_edges(g, pos, ax=ax, width=[0.5 + 3 * w for w in weights],
                               edge_color=weights, edge_cmap=plt.cm.viridis,
                               arrowsize=12, connectionstyle="arc3,rad=0.08")
        nx.draw_networkx_edge_labels(
            g, pos, ax=ax, font_size=6,
            edge_labels={(u, v): f"{d['weight']:.2f}" for u, v, d in g.edges(data=True)})
    ax.set_title("Attack-state graph $G_T$: transition probabilities (train)")
    ax.axis("off")
    return save_fig(fig, out_path, _dpi(cfg))


def plot_top_patterns(patterns: pd.DataFrame, cfg: Mapping[str, Any],
                      out_path: str | Path, top_n: int = 20):
    if patterns is None or patterns.empty:
        return None
    sub = patterns.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 1.0 + 0.3 * len(sub)))
    ax.barh(range(len(sub)), sub["support"], color="#7a5195")
    ax.set_yticks(range(len(sub)), sub["pattern"], fontsize=6)
    ax.set_xlabel("Support on the training sequences")
    ax.set_title("Top mined sequential patterns (PrefixSpan)")
    ax.grid(axis="x", alpha=0.3)
    return save_fig(fig, out_path, _dpi(cfg))


def plot_fusion_weights(fw: Any, cfg: Mapping[str, Any], out_path: str | Path,
                        max_rows: int = 200):
    """Stacked bars of the per-session weights, showing the adaptivity."""
    n = min(max_rows, fw.weights.shape[0])
    w = fw.weights[:n]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7),
                                   gridspec_kw={"height_ratios": [2, 1]})
    bottom = np.zeros(n)
    for j, src in enumerate(fw.sources):
        ax1.bar(range(n), w[:, j], bottom=bottom, label=src, width=1.0)
        bottom += w[:, j]
    ax1.set_xlim(-0.5, n - 0.5), ax1.set_ylim(0, 1)
    ax1.set_ylabel("weight $w_i(t)$")
    ax1.set_title(f"Per-session adaptive fusion weights (first {n} sessions)")
    ax1.legend(ncol=min(len(fw.sources), 7), fontsize=7, loc="upper center",
               bbox_to_anchor=(0.5, -0.06))

    means = fw.weights.mean(axis=0)
    stds = fw.weights.std(axis=0)
    ax2.bar(fw.sources, means, yerr=stds, capsize=4, color="#4a6fa5")
    ax2.set_ylabel("mean weight")
    ax2.set_title("Mean +/- std weight per evidence source (whole partition)")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return save_fig(fig, out_path, _dpi(cfg))


def plot_branch_comparison(comparison: pd.DataFrame, cfg: Mapping[str, Any],
                           out_path: str | Path,
                           metrics: Sequence[str] = ("macro_f1", "weighted_f1", "accuracy")):
    metrics = [m for m in metrics if m in comparison.columns]
    if comparison.empty or not metrics:
        return None
    fig, ax = plt.subplots(figsize=(1.5 + 1.5 * len(comparison), 4.6))
    width = 0.8 / max(len(metrics), 1)
    x = np.arange(len(comparison))
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, comparison[m], width, label=m)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2,
                  comparison["branch"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Score"), ax.set_ylim(0, 1.02)
    ax.set_title("Branch comparison on the test partition")
    ax.legend(fontsize=8), ax.grid(axis="y", alpha=0.3)
    return save_fig(fig, out_path, _dpi(cfg))


def plot_shap_summary(model_scores: np.ndarray, features: Sequence[str],
                      cfg: Mapping[str, Any], out_path: str | Path, top_n: int = 25):
    """Bar chart of the SHAP-LGBM ranker's mean |SHAP| per feature."""
    order = np.argsort(-np.asarray(model_scores))[:top_n][::-1]
    fig, ax = plt.subplots(figsize=(7, 1.0 + 0.26 * len(order)))
    ax.barh(range(len(order)), np.asarray(model_scores)[order], color="#d1495b")
    ax.set_yticks(range(len(order)), [features[i] for i in order], fontsize=7)
    ax.set_xlabel("mean |SHAP| over rows and classes")
    ax.set_title("SHAP-LGBM feature importance (FeS Set-4)")
    ax.grid(axis="x", alpha=0.3)
    return save_fig(fig, out_path, _dpi(cfg))


# ===========================================================================
# Full report
# ===========================================================================
def full_evaluation(y_true: np.ndarray, proba_by_model: Mapping[str, np.ndarray],
                    classes: Sequence[Any], cfg: Mapping[str, Any],
                    out_dir: str | Path, class_names: Optional[Mapping[Any, str]] = None,
                    tag: str = "", make_plots: bool = True,
                    extra_stats: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Evaluate every model (and the fusion), save tables and plots."""
    out_dir = ensure_dir(out_dir)
    suffix = f"_{tag}" if tag else ""
    plots = cfg["evaluation"].get("plots", {})

    results = [evaluate_predictions(y_true, p, classes, cfg, class_names, label=name)
               for name, p in proba_by_model.items()]
    table = metrics_to_frame(results).sort_values("macro_f1", ascending=False)
    per_class = per_class_frame(results)

    save_table(table, Path(out_dir) / f"metrics{suffix}.csv")
    save_table(per_class, Path(out_dir) / f"metrics_per_class{suffix}.csv")
    save_json({"results": results}, Path(out_dir) / f"metrics{suffix}.json")
    if extra_stats is not None and len(extra_stats):
        save_table(extra_stats, Path(out_dir) / f"model_efficiency{suffix}.csv")

    predictions = {
        name: np.array([classes[i] for i in np.argmax(p, axis=1)])
        for name, p in proba_by_model.items()
    }
    significance = compare_models(y_true, predictions, cfg)
    if len(significance):
        save_table(significance, Path(out_dir) / f"significance{suffix}.csv")

    if make_plots:
        best = table.iloc[0]["label"]
        for name, p in proba_by_model.items():
            if plots.get("confusion_matrix", True):
                plot_confusion_matrix(y_true, predictions[name], classes, cfg,
                                      Path(out_dir) / f"confusion_{name}{suffix}.png",
                                      normalize=False, class_names=class_names,
                                      title=f"Confusion matrix -- {name}")
            if plots.get("confusion_matrix_normalized", True):
                plot_confusion_matrix(y_true, predictions[name], classes, cfg,
                                      Path(out_dir) / f"confusion_{name}_norm{suffix}.png",
                                      normalize=True, class_names=class_names,
                                      title=f"Confusion matrix -- {name}")
        if plots.get("roc_curves", True):
            plot_roc_curves(y_true, proba_by_model[best], classes, cfg,
                            Path(out_dir) / f"roc_curves{suffix}.png", class_names)
        if plots.get("pr_curves", True):
            plot_pr_curves(y_true, proba_by_model[best], classes, cfg,
                           Path(out_dir) / f"pr_curves{suffix}.png", class_names)

    LOGGER.info("Stage 8: evaluated %d prediction set(s); best macro-F1 %.4f (%s)",
                len(results), table.iloc[0]["macro_f1"], table.iloc[0]["label"])
    return {"metrics": table, "per_class": per_class, "results": results,
            "significance": significance}


def zero_day_metrics(y_true: np.ndarray, y_pred: np.ndarray, held_out: Any,
                     benign: Any) -> Dict[str, float]:
    """Branch E: recall on the unseen family and the benign false-alarm rate.

    A zero-day detection counts when an unseen-family flow is predicted as
    *any* attack class -- the model cannot name a class it never saw, so
    "detected" means "not classified as benign".
    """
    unseen = (y_true == held_out)
    benign_rows = (y_true == benign)
    detected = unseen & (y_pred != benign)
    false_alarm = benign_rows & (y_pred != benign)
    return {
        "held_out_family": str(held_out),
        "n_unseen": int(unseen.sum()),
        "unseen_detection_recall": float(detected.sum() / max(unseen.sum(), 1)),
        "n_benign": int(benign_rows.sum()),
        "false_alarm_rate": float(false_alarm.sum() / max(benign_rows.sum(), 1)),
    }
