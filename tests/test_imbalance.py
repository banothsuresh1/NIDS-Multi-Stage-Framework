"""Stage 5 unit tests: class weights, SMOTE-ENN policy, mode dispatch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import imbalance_handling as ih
import utils


@pytest.fixture
def imbalanced():
    """A 9-class frame shaped like CIC-IDS2017, with two classes below the floor."""
    rng = np.random.default_rng(0)
    counts = {0: 1200, 1: 300, 2: 250, 3: 200, 4: 80, 5: 40, 6: 12, 7: 4, 8: 2}
    X, y = [], []
    for c, n in counts.items():
        X.append(rng.normal(c, 1.0, size=(n, 6)))
        y.append(np.full(n, c))
    return (pd.DataFrame(np.vstack(X), columns=[f"f{i}" for i in range(6)]),
            np.concatenate(y), counts)


def test_class_weight_formula(imbalanced):
    _X, y, counts = imbalanced
    w = ih.compute_class_weights(y)
    n, c = len(y), len(counts)
    for cls, n_c in counts.items():
        assert w[cls] == pytest.approx(n / (c * n_c))
    # Rarer class -> larger weight.
    assert w[8] > w[7] > w[6] > w[0]


def test_class_weights_cover_absent_classes():
    y = np.array([0, 0, 1, 1])
    w = ih.compute_class_weights(y, classes=np.arange(4))
    assert set(w) == {0, 1, 2, 3}
    assert w[2] == 0.0 and w[3] == 0.0


def test_smote_enn_min_samples_policy(imbalanced, cfg):
    """Classes below min_samples_for_smote take the documented fallback."""
    X, y, _ = imbalanced
    X2, y2, report = ih.apply_smote_enn(X, y, cfg, seed=42)
    floor = cfg["imbalance"]["smote_enn"]["min_samples_for_smote"]
    for cls, n in report.before.items():
        if n < floor:
            assert report.fallback_classes[cls] == \
                cfg["imbalance"]["smote_enn"]["small_class_fallback"]
        else:
            # k_neighbors can never exceed n_c - 1.
            assert report.k_neighbors_used[cls] <= n - 1


def test_smote_enn_loses_no_class(imbalanced, cfg):
    X, y, _ = imbalanced
    _X2, y2, report = ih.apply_smote_enn(X, y, cfg, seed=42)
    assert set(np.unique(y2)) == set(np.unique(y)), "no class may vanish"
    for cls in np.unique(y):
        assert report.after[int(cls)] >= 1


def test_oversampling_is_capped(imbalanced, cfg):
    """A 2-sample class must not be inflated to the majority count."""
    X, y, _ = imbalanced
    ratio = cfg["imbalance"]["smote_enn"]["max_oversample_ratio"]
    _X2, _y2, report = ih.apply_smote_enn(X, y, cfg, seed=42)
    assert cfg["imbalance"]["smote_enn"]["target_strategy"] == "capped"
    for cls, before in report.before.items():
        after = report.after.get(cls, 0)
        assert after <= max(before * ratio, max(report.before.values())) + 1, \
            f"class {cls} grew {before} -> {after}, beyond the {ratio}x cap"


def test_modes_dispatch_correctly(imbalanced, cfg):
    X, y, _ = imbalanced
    expectations = {
        "none": (False, False),
        "class_weight": (False, True),
        "smote_enn": (True, False),
        "both": (True, True),
    }
    for mode, (resampled, weighted) in expectations.items():
        c = utils.deep_merge(cfg, {"imbalance": {"mode": mode}})
        X2, y2, w, _rep = ih.apply_imbalance_handling(X, y, c, 42)
        assert (len(y2) != len(y)) is resampled, f"mode {mode}: resampling mismatch"
        assert (w is not None) is weighted, f"mode {mode}: weighting mismatch"


def test_unknown_mode_raises(imbalanced, cfg):
    X, y, _ = imbalanced
    c = utils.deep_merge(cfg, {"imbalance": {"mode": "magic"}})
    with pytest.raises(ValueError, match="unknown imbalance.mode"):
        ih.apply_imbalance_handling(X, y, c, 42)


def test_eval_sets_are_never_resampled():
    ih.assert_eval_sets_untouched(100, 100, 50, 50)
    with pytest.raises(utils.LeakageError):
        ih.assert_eval_sets_untouched(100, 150, 50, 50)
    with pytest.raises(utils.LeakageError):
        ih.assert_eval_sets_untouched(100, 100, 50, 90)


def test_sample_weights_broadcast():
    y = np.array([0, 1, 1, 2])
    w = ih.sample_weights_from_class_weights(y, {0: 2.0, 1: 0.5, 2: 4.0})
    assert np.allclose(w, [2.0, 0.5, 0.5, 4.0])
