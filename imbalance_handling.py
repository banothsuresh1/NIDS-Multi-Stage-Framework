"""Stage 5: imbalance handling -- class weighting and SMOTE-ENN.

Both techniques apply to the TRAINING partition only, and only *after* feature
selection, so synthetic samples can never influence which features were chosen
(Stage 5.1). Validation and test are never resampled; the callers assert this
with :func:`utils.assert_not_resampled`.

Modes (``imbalance.mode``):
  ``none`` | ``class_weight`` | ``smote_enn`` | ``both``
corresponding to branches A (class weighting), B (SMOTE-ENN), C (both).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from utils import assert_not_resampled, get_logger

LOGGER = get_logger()


# ===========================================================================
# Class weighting
# ===========================================================================
def compute_class_weights(y: np.ndarray, cfg: Optional[Mapping[str, Any]] = None,
                          classes: Optional[np.ndarray] = None) -> Dict[int, float]:
    """``w_c = N / (C * N_c)`` (Stage 5.2).

    Classes present in ``classes`` but absent from ``y`` receive weight 0.0, so
    the mapping is always complete for a fixed class order.
    """
    y = np.asarray(y)
    if classes is None:
        classes = np.unique(y)
    classes = np.asarray(classes)
    n = len(y)
    c = len(classes)
    out: Dict[int, float] = {}
    for cls in classes:
        n_c = int((y == cls).sum())
        out[int(cls)] = (n / (c * n_c)) if n_c > 0 else 0.0
    return out


def sample_weights_from_class_weights(y: np.ndarray,
                                      weights: Mapping[int, float]) -> np.ndarray:
    """Broadcast class weights to per-row sample weights (XGBoost, Keras)."""
    return np.array([weights.get(int(v), 1.0) for v in np.asarray(y)], dtype=np.float64)


# ===========================================================================
# SMOTE-ENN
# ===========================================================================
@dataclass
class ResampleReport:
    """Per-class counts before/after plus the policy decisions that were taken."""

    before: Dict[int, int] = field(default_factory=dict)
    after: Dict[int, int] = field(default_factory=dict)
    k_neighbors_used: Dict[int, int] = field(default_factory=dict)
    fallback_classes: Dict[int, str] = field(default_factory=dict)
    majority_capped: Optional[int] = None
    enn_restored: Dict[int, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_frame(self, class_names: Optional[Mapping[int, str]] = None) -> pd.DataFrame:
        rows = []
        for cls in sorted(set(self.before) | set(self.after)):
            rows.append({
                "class": (class_names or {}).get(cls, cls),
                "y": cls,
                "before": self.before.get(cls, 0),
                "after": self.after.get(cls, 0),
                "k_neighbors": self.k_neighbors_used.get(cls, ""),
                "policy": self.fallback_classes.get(cls, "smote"),
                "enn_restored": self.enn_restored.get(cls, 0),
            })
        return pd.DataFrame(rows)


def apply_smote_enn(X_train: pd.DataFrame, y_train: np.ndarray,
                    cfg: Mapping[str, Any], seed: int = 42
                    ) -> Tuple[pd.DataFrame, np.ndarray, ResampleReport]:
    """SMOTE-ENN on the TRAINING partition only, after F* has been applied.

    Documented minimum-sample policy (Stage 5.2, "very small classes need a
    documented minimum-sample policy"):

    * ``k_neighbors = min(cfg_k, n_c - 1)`` -- SMOTE interpolates between a
      sample and its k nearest same-class neighbours, so k cannot exceed
      ``n_c - 1``;
    * a class with fewer than ``min_samples_for_smote`` training samples cannot
      support meaningful interpolation (its neighbours are essentially the whole
      class, and synthetic points would merely duplicate the convex hull of 2-3
      records). Such classes use ``small_class_fallback``:
      ``random_oversample`` (duplicate real records) or ``skip`` (leave as is).
      Heartbleed and Infiltration hit this path on CIC-IDS2017;
    * ENN can delete *every* member of a rare class. With
      ``protect_rare_classes`` the deleted originals of any class that ENN
      emptied are restored, so no class disappears from training.

    Every decision is logged and returned in :class:`ResampleReport`.
    """
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import EditedNearestNeighbours

    p = cfg["imbalance"]["smote_enn"]
    report = ResampleReport()
    y_train = np.asarray(y_train)
    report.before = {int(c): int(n) for c, n in zip(*np.unique(y_train, return_counts=True))}

    X = X_train.reset_index(drop=True)
    y = y_train.copy()

    # -- cap the majority class to bound memory ---------------------------
    cap = p.get("majority_cap")
    if cap:
        cap = int(cap)
        counts = pd.Series(y).value_counts()
        majority = int(counts.idxmax())
        if counts.max() > cap:
            rng = np.random.default_rng(seed)
            maj_idx = np.where(y == majority)[0]
            keep = rng.choice(maj_idx, size=cap, replace=False)
            others = np.where(y != majority)[0]
            sel = np.sort(np.concatenate([keep, others]))
            X, y = X.iloc[sel].reset_index(drop=True), y[sel]
            report.majority_capped = cap
            report.notes.append(
                f"majority class {majority} capped from {int(counts.max())} to {cap} "
                f"before resampling (imbalance.smote_enn.majority_cap)")
            LOGGER.info("Stage 5: capped majority class %d to %d rows", majority, cap)

    # -- partition classes by whether SMOTE can be applied ----------------
    cfg_k = int(p.get("k_neighbors", 5))
    min_for_smote = int(p.get("min_samples_for_smote", 6))
    fallback = str(p.get("small_class_fallback", "random_oversample"))
    counts = pd.Series(y).value_counts().to_dict()
    majority = int(max(counts.values()))
    strategy_mode = str(p.get("target_strategy", "capped"))
    max_ratio = float(p.get("max_oversample_ratio", 20))

    def target_for(c: int) -> int:
        """Oversampling target for class ``c`` under the configured strategy."""
        if strategy_mode == "full":
            return majority
        return int(min(majority, max(counts[c], np.ceil(counts[c] * max_ratio))))

    if strategy_mode == "capped":
        report.notes.append(
            f"oversampling capped at {max_ratio:g}x each class's real count "
            f"(majority = {majority}); residual imbalance is left to class weighting")

    smote_classes = {c: n for c, n in counts.items() if n >= min_for_smote}
    small_classes = {c: n for c, n in counts.items() if n < min_for_smote}

    for c in small_classes:
        report.fallback_classes[int(c)] = fallback
        LOGGER.info("Stage 5: class %d has %d samples (< %d) -> fallback '%s'",
                    c, small_classes[c], min_for_smote, fallback)

    # Effective k per class, bounded by the class's own size.
    k_eff = cfg_k
    for c, n in smote_classes.items():
        k_c = min(cfg_k, n - 1)
        report.k_neighbors_used[int(c)] = k_c
        k_eff = min(k_eff, k_c)
    k_eff = max(1, k_eff)
    if k_eff != cfg_k:
        report.notes.append(
            f"k_neighbors reduced from {cfg_k} to {k_eff} = min(k, n_c - 1) over the "
            f"classes SMOTE was applied to")

    X_np = X.to_numpy(dtype=np.float32)

    # -- oversample --------------------------------------------------------
    if len(smote_classes) >= 2:
        strategy = {int(c): target_for(c) for c in smote_classes
                    if counts[c] < target_for(c)}
        if strategy:
            sm = SMOTE(sampling_strategy=strategy, k_neighbors=k_eff, random_state=seed)
            # SMOTE needs every class present; small classes ride along untouched.
            X_np, y = sm.fit_resample(X_np, y)
        else:
            report.notes.append("no class required oversampling (already balanced)")
    else:
        report.notes.append("fewer than two classes large enough for SMOTE; skipped")

    if small_classes and fallback == "random_oversample":
        strategy = {int(c): target_for(c) for c in small_classes
                    if counts[c] < target_for(c)}
        if strategy:
            ros = RandomOverSampler(sampling_strategy=strategy, random_state=seed)
            X_np, y = ros.fit_resample(X_np, y)

    # -- ENN cleaning ------------------------------------------------------
    pre_enn_X, pre_enn_y = X_np, y
    try:
        enn = EditedNearestNeighbours(n_neighbors=int(p.get("enn_n_neighbors", 3)))
        X_np, y = enn.fit_resample(X_np, y)
    except ValueError as exc:
        LOGGER.warning("Stage 5: ENN cleaning skipped (%s)", exc)
        report.notes.append(f"ENN skipped: {exc}")
        X_np, y = pre_enn_X, pre_enn_y

    # -- restore any class ENN wiped out ----------------------------------
    if p.get("protect_rare_classes", True):
        after_counts = pd.Series(y).value_counts().to_dict()
        restored_X, restored_y = [], []
        for c in np.unique(pre_enn_y):
            if after_counts.get(c, 0) == 0:
                rows = np.where(pre_enn_y == c)[0]
                restored_X.append(pre_enn_X[rows])
                restored_y.append(pre_enn_y[rows])
                report.enn_restored[int(c)] = len(rows)
                LOGGER.warning("Stage 5: ENN removed every sample of class %d; "
                               "restoring %d original row(s)", c, len(rows))
        if restored_X:
            X_np = np.vstack([X_np] + restored_X)
            y = np.concatenate([y] + restored_y)
            report.notes.append(
                "ENN emptied " + ", ".join(f"class {c} ({n} rows)"
                                           for c, n in report.enn_restored.items())
                + "; originals restored (imbalance.smote_enn.protect_rare_classes)")

    X_out = pd.DataFrame(X_np, columns=list(X_train.columns))
    report.after = {int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))}
    LOGGER.info("Stage 5: SMOTE-ENN %d -> %d training rows (%d classes)",
                len(y_train), len(y), len(report.after))
    return X_out, y, report


# ===========================================================================
# Dispatcher
# ===========================================================================
def apply_imbalance_handling(
    X_train: pd.DataFrame, y_train: np.ndarray,
    cfg: Mapping[str, Any], seed: int = 42,
    classes: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, np.ndarray, Optional[Dict[int, float]], ResampleReport]:
    """Apply the configured mode and return ``(X, y, class_weights, report)``.

    ``class_weights`` is ``None`` when the mode does not use weighting, so a
    caller can pass it straight to a model without re-reading the config.
    """
    mode = str(cfg["imbalance"].get("mode", "both"))
    if mode not in {"none", "class_weight", "smote_enn", "both"}:
        raise ValueError(f"unknown imbalance.mode {mode!r}")

    report = ResampleReport()
    report.before = {int(c): int(n) for c, n in zip(*np.unique(y_train, return_counts=True))}
    X_out, y_out = X_train, np.asarray(y_train)

    if mode in ("smote_enn", "both"):
        X_out, y_out, report = apply_smote_enn(X_train, y_train, cfg, seed)
    else:
        report.after = dict(report.before)
        report.notes.append(f"mode={mode}: training data not resampled")

    weights = None
    if mode in ("class_weight", "both"):
        weights = compute_class_weights(y_out, cfg, classes)

    LOGGER.info("Stage 5 [%s]: train %d -> %d rows; class weighting %s",
                mode, len(y_train), len(y_out), "on" if weights else "off")
    return X_out, y_out, weights, report


def assert_eval_sets_untouched(n_val_before: int, n_val_after: int,
                               n_test_before: int, n_test_after: int) -> None:
    """Validation and test must come out of Stage 5 with identical row counts."""
    assert_not_resampled(n_val_before, n_val_after, "validation partition")
    assert_not_resampled(n_test_before, n_test_after, "test partition")
