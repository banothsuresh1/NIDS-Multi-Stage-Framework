"""Leakage controls (methodology Stage 8.4), enforced as executable checks.

Covers: split disjointness, whole sessions, time ordering, fit-on-train-only,
no resampling of validation/test, F* frozen, and the probability / fusion-weight
invariants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import data_preprocessing as dp
import utils
from utils import LeakageError, SplitIndex


# --------------------------------------------------------------------------
# Split structure
# --------------------------------------------------------------------------
def test_splits_are_disjoint(split):
    utils.assert_disjoint_splits(split)
    allidx = np.concatenate([split.train, split.val, split.test])
    assert len(allidx) == len(np.unique(allidx))


def test_splits_partition_every_row(prepared, split):
    df = prepared[0]
    covered = np.concatenate([split.train, split.val, split.test, split.dropped_gap])
    assert len(np.unique(covered)) == len(df), "every row is assigned or in the gap"


def test_disjointness_guard_catches_overlap():
    bad = SplitIndex(np.array([0, 1, 2]), np.array([2, 3]), np.array([4]))
    with pytest.raises(LeakageError, match="overlap"):
        utils.assert_disjoint_splits(bad)


def test_sessions_stay_whole(prepared, split):
    df = prepared[0]
    utils.assert_sessions_not_split(split, df[dp.COL_SESSION].values,
                                    exempt=split.meta.get("exempt_sessions"))


def test_session_guard_catches_a_torn_session():
    bad = SplitIndex(np.array([0, 1]), np.array([2]), np.array([3]))
    sessions = np.array(["s1", "s1", "s1", "s2"])     # s1 spans train and val
    with pytest.raises(LeakageError, match="appears in both"):
        utils.assert_sessions_not_split(bad, sessions)


def test_time_ordering_train_before_val_before_test(prepared, split):
    """No validation/test flow may predate the preceding block (Stage 3.2)."""
    df = prepared[0]
    utils.assert_no_time_leakage(split, df[dp.COL_TIME].values, df[dp.COL_LABEL].values)

    # Explicit per-assignment-group restatement of the same invariant.
    ts = df[dp.COL_TIME].to_numpy()
    groups = split.meta["assign_group"]
    for g in np.unique(groups):
        rows = np.where(groups == g)[0]
        rs = set(rows.tolist())
        tr = [i for i in split.train if i in rs]
        va = [i for i in split.val if i in rs]
        te = [i for i in split.test if i in rs]
        if tr and va:
            assert ts[tr].max() <= ts[va].min(), f"group {g}: train overlaps val"
        if va and te:
            assert ts[va].max() <= ts[te].min(), f"group {g}: val overlaps test"
        if tr and te:
            assert ts[tr].max() <= ts[te].min(), f"group {g}: train overlaps test"


def test_time_guard_catches_reversed_blocks():
    bad = SplitIndex(np.array([2, 3]), np.array([]), np.array([0, 1]),
                     strategy="time_blocked")
    ts = pd.to_datetime(["2017-07-03 09:00", "2017-07-03 09:01",
                         "2017-07-03 10:00", "2017-07-03 10:01"]).values
    with pytest.raises(LeakageError, match="predates"):
        utils.assert_no_time_leakage(bad, ts, np.array(["A"] * 4))


def test_every_class_present_in_every_partition(prepared, split, cfg):
    df = prepared[0]
    table = dp.split_distribution_table(df, split, cfg)
    present = df[dp.COL_LABEL].unique()
    missing = []
    for _, row in table.iterrows():
        if row["class"] not in present:
            continue
        for part in ("train", "val", "test"):
            if row[part] == 0:
                missing.append((row["class"], part))
    assert not missing, f"classes absent from a partition: {missing}"


def test_loao_hides_the_family_from_train_and_val(prepared, cfg):
    df = prepared[0]
    present = set(df[dp.COL_LABEL].unique())
    for family in cfg["labels"]["attack_families"]:
        if family not in present:
            continue
        sp = dp.leave_one_attack_out_split(df, cfg, family, seed=42)
        assert (df.iloc[sp.train][dp.COL_LABEL] == family).sum() == 0
        assert (df.iloc[sp.val][dp.COL_LABEL] == family).sum() == 0
        assert (df.iloc[sp.test][dp.COL_LABEL] == family).sum() > 0
        utils.assert_disjoint_splits(sp)
        utils.assert_sessions_not_split(sp, df[dp.COL_SESSION].values,
                                        exempt=sp.meta.get("exempt_sessions"))


# --------------------------------------------------------------------------
# Transformers fitted on train only
# --------------------------------------------------------------------------
def test_preprocessor_fit_rejects_non_train_rows(prepared, cfg, split):
    df, feature_columns, _ = prepared
    pre = dp.TrainFittedPreprocessor(cfg)
    bad_rows = np.concatenate([split.train[:10], split.test[:1]])
    with pytest.raises(LeakageError, match="non-train rows"):
        pre.fit(df, feature_columns, rows=bad_rows, split=split)


def test_preprocessor_statistics_come_only_from_train(prepared, cfg, split):
    """Fitting on train, then on train+test, must give different statistics.

    If they matched, the transform would be ignoring its fit rows -- which is
    how a leak hides.
    """
    df, feature_columns, _ = prepared
    a = dp.TrainFittedPreprocessor(cfg).fit(df, feature_columns, rows=split.train)
    b = dp.TrainFittedPreprocessor(cfg).fit(
        df, feature_columns, rows=np.concatenate([split.train, split.test]))
    assert not np.allclose(a.mins_, b.mins_) or not np.allclose(a.ranges_, b.ranges_)
    # And the train-fitted min/max must equal the train partition's own min/max.
    Xtr = df.iloc[split.train][a.numeric_columns_].to_numpy(dtype=np.float64, na_value=np.nan)
    Xtr = np.where(np.isnan(Xtr), a.medians_[None, :], Xtr)
    assert np.allclose(a.mins_, Xtr.min(axis=0))


def test_transform_does_not_refit(prepared, cfg, split):
    df, feature_columns, _ = prepared
    pre = dp.TrainFittedPreprocessor(cfg).fit(df, feature_columns, rows=split.train)
    before = (pre.medians_.copy(), pre.mins_.copy(), pre.ranges_.copy())
    pre.transform(df, rows=split.test)
    for x, y in zip(before, (pre.medians_, pre.mins_, pre.ranges_)):
        assert np.array_equal(x, y), "transform must not update fitted statistics"


def test_fit_guard_accepts_train_subset(split):
    utils.assert_fitted_on_train_only(split.train[:50], split, "probe")


# --------------------------------------------------------------------------
# Frozen feature mask, resampling, probability invariants
# --------------------------------------------------------------------------
def test_frozen_feature_mask_guard():
    utils.assert_feature_mask_frozen(["a", "b"], ["a", "b"], "test")
    with pytest.raises(LeakageError, match="frozen feature set"):
        utils.assert_feature_mask_frozen(["a", "b"], ["b", "a"], "reordered")
    with pytest.raises(LeakageError):
        utils.assert_feature_mask_frozen(["a", "b"], ["a"], "truncated")


def test_no_resampling_guard():
    utils.assert_not_resampled(100, 100, "validation")
    with pytest.raises(LeakageError, match="never be resampled"):
        utils.assert_not_resampled(100, 180, "validation")


def test_probability_invariants():
    good = np.array([[0.2, 0.8], [0.5, 0.5]])
    utils.assert_probabilities_valid(good)
    with pytest.raises(LeakageError, match="sum to 1"):
        utils.assert_probabilities_valid(np.array([[0.2, 0.2]]))
    with pytest.raises(LeakageError, match="negative"):
        utils.assert_probabilities_valid(np.array([[-0.1, 1.1]]))


def test_fusion_weight_invariants():
    utils.assert_weights_sum_to_one(np.array([[0.25, 0.25, 0.5], [0.1, 0.1, 0.8]]))
    with pytest.raises(LeakageError, match="sum to 1"):
        utils.assert_weights_sum_to_one(np.array([[0.3, 0.3, 0.3]]))


def test_stratified_flow_split_is_the_leaky_baseline(prepared, cfg):
    """The flow-level split must actually tear sessions -- that is its purpose."""
    df = prepared[0]
    sp = dp.stratified_flow_split(df, cfg, seed=42)
    utils.assert_disjoint_splits(sp)
    sess = df[dp.COL_SESSION].to_numpy()
    torn = 0
    train_sessions = set(sess[sp.train])
    for s in set(sess[sp.test]):
        if s in train_sessions:
            torn += 1
    assert torn > 0, "the leaky baseline should share sessions across train/test"
