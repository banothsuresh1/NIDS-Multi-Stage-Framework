"""Stage 4 unit tests: redundancy filter, the eight rankers, vote + rank fusion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import feature_selection as fsel
import utils


def test_redundancy_filter_drops_constant_duplicate_and_correlated(cfg):
    rng = np.random.default_rng(0)
    base = rng.normal(size=400)
    X = pd.DataFrame({
        "informative": base,
        "constant": np.full(400, 3.0),             # near-zero variance
        "duplicate": base,                          # exact duplicate column
        "correlated": base * 2.0 + 1e-9 * rng.normal(size=400),  # |rho| ~ 1
        "independent": rng.normal(size=400),
    })
    filt = fsel.RedundancyFilter(cfg).fit(X, seed=0)
    assert "constant" in filt.dropped_
    assert "near-zero variance" in filt.dropped_["constant"]
    assert "duplicate" in filt.dropped_
    assert "correlated" in filt.dropped_
    assert "informative" in filt.kept_ and "independent" in filt.kept_


def test_topk_discards_non_positive_scores():
    scores = np.array([0.5, 0.0, -0.2, 0.9, 0.1])
    feats = ["a", "b", "c", "d", "e"]
    out = fsel.topk_set(scores, feats, k=10)
    assert out == ["d", "a", "e"]            # b (0) and c (<0) discarded
    assert fsel.topk_set(scores, feats, k=2) == ["d", "a"]


def test_normalized_rank_formula():
    """NR_m(f) = 1 - (rank - 1) / (K - 1), so the best scores 1 and the Kth 0."""
    nr = fsel.normalized_rank(["a", "b", "c", "d", "e"], k=5)
    assert nr["a"] == pytest.approx(1.0)
    assert nr["c"] == pytest.approx(0.5)
    assert nr["e"] == pytest.approx(0.0)


def test_fuse_rankings_vote_and_score():
    feats = ["f1", "f2", "f3", "f4"]
    scores = {
        "m1": np.array([1.0, 0.9, 0.0, 0.0]),    # top set: f1, f2
        "m2": np.array([0.8, 0.0, 0.7, 0.0]),    # top set: f1, f3
        "m3": np.array([0.5, 0.4, 0.0, 0.3]),    # top set: f1, f2, f4
    }
    table = fsel.fuse_rankings(scores, feats, k=3, eta=2)
    row = table.set_index("feature")
    assert row.loc["f1", "psi_votes"] == 3        # in every set
    assert row.loc["f2", "psi_votes"] == 2
    assert row.loc["f3", "psi_votes"] == 1
    assert row.loc["f1", "in_pool"] == 1          # psi >= eta
    assert row.loc["f3", "in_pool"] == 0
    # f1 is ranked first everywhere -> Score = mean of 1.0 = 1.0
    assert row.loc["f1", "score"] == pytest.approx(1.0)
    # Pool members sort ahead of non-members.
    assert table.iloc[0]["feature"] == "f1"


def test_every_ranker_returns_one_score_per_feature(prepared, cfg, split):
    """All eight rankers must return a finite score vector of width |F_corr|."""
    import data_preprocessing as dp

    df, feature_columns, _ = prepared
    pre = dp.TrainFittedPreprocessor(cfg).fit(df, feature_columns, rows=split.train)
    X = pre.transform(df, split.train)
    y = df[dp.COL_Y].to_numpy()[split.train]
    filt = fsel.RedundancyFilter(cfg).fit(X, seed=42)
    Xc = filt.transform(X)

    sessions = df[dp.COL_SESSION].to_numpy()[split.train]
    times = df[dp.COL_TIME].to_numpy()[split.train]

    for name in cfg["feature_selection"]["rankers"]:
        kwargs = {"session_ids": sessions, "times": times} if name == "lstm" else {}
        scores = fsel.RANKERS[name](Xc, y, cfg, 42, **kwargs)
        scores = np.asarray(scores).ravel()
        assert len(scores) == Xc.shape[1], f"{name} returned {len(scores)} scores"
        assert np.isfinite(scores).all(), f"{name} produced non-finite scores"


def test_session_windows_shape_padding_and_alignment():
    X = np.arange(24, dtype=np.float32).reshape(12, 2)
    y = np.arange(12)
    sessions = np.array(["a"] * 7 + ["b"] * 5)
    times = np.arange(12)
    Xw, yw, rows = fsel.build_session_windows(X, y, sessions, times, window=4,
                                              return_index=True)
    assert Xw.shape == (12, 4, 2)          # one window per flow
    assert len(np.unique(rows)) == 12      # every flow gets exactly one window
    # The first flow of a session is left-padded with zeros.
    first_a = list(rows).index(0)
    assert np.allclose(Xw[first_a, :3], 0.0)
    assert np.allclose(Xw[first_a, 3], X[0])
    # A window never reaches across a session boundary.
    first_b = list(rows).index(7)
    assert np.allclose(Xw[first_b, :3], 0.0)


def test_n_star_is_smallest_within_delta(prepared, cfg, split):
    import data_preprocessing as dp

    df, feature_columns, _ = prepared
    pre = dp.TrainFittedPreprocessor(cfg).fit(df, feature_columns, rows=split.train)
    X_tr, X_va = pre.transform(df, split.train), pre.transform(df, split.val)
    y = df[dp.COL_Y].to_numpy()
    res = fsel.run_feature_selection(
        X_tr, y[split.train], X_va, y[split.val], cfg, 42,
        session_ids_train=df[dp.COL_SESSION].to_numpy()[split.train],
        times_train=df[dp.COL_TIME].to_numpy()[split.train],
        out_dir=cfg["run"]["output_dir"], cache_tag="test_nstar")

    curve = res.n_star_curve
    best = curve["macro_f1"].max()
    delta = cfg["feature_selection"]["delta"]
    eligible = curve.loc[curve["macro_f1"] >= best - delta, "N"]
    assert res.n_star == eligible.min()
    assert len(res.f_star) == res.n_star


def test_f_star_is_frozen_and_applied_unchanged(prepared, cfg, split):
    """F* must come back in the same order from every partition (Stage 4.7 step 8)."""
    import data_preprocessing as dp

    df, feature_columns, _ = prepared
    pre = dp.TrainFittedPreprocessor(cfg).fit(df, feature_columns, rows=split.train)
    X_tr = pre.transform(df, split.train)
    res = fsel.FeatureSelectionResult(
        f_star=list(X_tr.columns[:5]), n_star=5, pool=list(X_tr.columns),
        f_corr=list(X_tr.columns), votes_table=pd.DataFrame(),
        n_star_curve=pd.DataFrame(), ranker_scores={}, redundancy_report=pd.DataFrame())

    for part in (split.train, split.val, split.test):
        out = res.apply(pre.transform(df, part), "part")
        assert list(out.columns) == res.f_star

    # A reordered frame must still come back in F* order, not the frame's.
    shuffled = X_tr[list(X_tr.columns[::-1])]
    assert list(res.apply(shuffled, "shuffled").columns) == res.f_star

    with pytest.raises(KeyError):
        res.apply(X_tr.drop(columns=[res.f_star[0]]), "missing")


def test_stratified_subsample_keeps_rare_classes():
    y = np.array([0] * 5000 + [1] * 50 + [2] * 3)
    idx = utils.stratified_subsample(y, 500, seed=1, min_per_class=3)
    kept = np.unique(y[idx])
    assert set(kept) == {0, 1, 2}, "no class may be dropped by the subsample"
    assert (y[idx] == 2).sum() == 3
