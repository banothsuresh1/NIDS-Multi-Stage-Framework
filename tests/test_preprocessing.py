"""Stage 1-2 unit tests: consolidation, label grouping, sessions, transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import data_preprocessing as dp
import utils


def test_label_grouping_covers_every_raw_label(prepared, cfg):
    df = prepared[0]
    assert df[dp.COL_LABEL].isin(cfg["labels"]["class_order"]).all()
    # The raw label is preserved alongside the grouped one (Stage 1.3).
    assert dp.COL_LABEL_RAW in df.columns
    assert df[dp.COL_LABEL_RAW].nunique() >= df[dp.COL_LABEL].nunique()
    # Label encoding matches the canonical order.
    code = {c: i for i, c in enumerate(cfg["labels"]["class_order"])}
    assert (df[dp.COL_Y].to_numpy() == df[dp.COL_LABEL].map(code).to_numpy()).all()


def test_web_attack_variants_merge():
    cfg = {"labels": {"grouping": {"web attack brute force": "Web Attack",
                                   "web attack xss": "Web Attack",
                                   "benign": "Benign"},
                      "class_order": ["Benign", "Web Attack"]},
           "data": {"label_column": "Label"}}
    raw = pd.DataFrame({"Label": ["Web Attack \x96 Brute Force", "Web Attack - XSS", "BENIGN"]})
    out = dp.group_labels(raw, cfg)
    assert out[dp.COL_LABEL].tolist() == ["Web Attack", "Web Attack", "Benign"]


def test_unmapped_label_raises():
    cfg = {"labels": {"grouping": {"benign": "Benign"}, "class_order": ["Benign"]},
           "data": {"label_column": "Label"}}
    with pytest.raises(ValueError, match="unmapped raw labels"):
        dp.group_labels(pd.DataFrame({"Label": ["BENIGN", "Martian Attack"]}), cfg)


def test_identifiers_excluded_from_features(prepared, cfg):
    _df, feature_columns, report = prepared
    forbidden = {"Flow ID", "Source IP", "Destination IP", "Timestamp", "Source Port",
                 "Label", dp.COL_LABEL, dp.COL_LABEL_RAW, dp.COL_Y, dp.COL_SESSION}
    assert not (set(feature_columns) & forbidden)
    # Destination Port is excluded unless explicitly enabled (Stage 8.4).
    if not cfg["data"]["use_dst_port"]:
        assert "Destination Port" not in feature_columns


def test_inf_and_negative_handling(prepared):
    df, feature_columns, report = prepared
    vals = df[feature_columns].to_numpy(dtype=np.float64, na_value=np.nan)
    assert not np.isinf(vals).any(), "no infinities may survive cleaning"
    assert report["inf_values_to_nan"] > 0, "the fixture contains infinities to convert"


def test_negative_sentinels_are_preserved(prepared):
    df, _feature_columns, report = prepared
    preserved = report["sentinel_features_preserved"]
    assert "Init_Win_bytes_forward" in preserved
    # -1 must still be present: it is a documented sentinel, not an invalid value.
    assert (df["Init_Win_bytes_forward"] == -1).any()


def test_session_key_is_symmetric(cfg):
    """Opposite directions of the same conversation map to one session (eq. 22)."""
    c = {**cfg, "preprocessing": {**cfg["preprocessing"], "session_key_mode": "five_tuple",
                                  "session_timeout_s": 0, "session_max_flows": 0}}
    df = pd.DataFrame({
        "Source IP": ["10.0.0.1", "10.0.0.2"],
        "Destination IP": ["10.0.0.2", "10.0.0.1"],
        "Source Port": [1234, 80],
        "Destination Port": [80, 1234],
        "Protocol": [6, 6],
        dp.COL_TIME: pd.to_datetime(["2017-07-03 09:00:00", "2017-07-03 09:00:01"]),
        dp.COL_SRC_FILE: ["a.csv", "a.csv"],
    })
    keys = dp.build_session_ids(df, c)
    assert keys.nunique() == 1, "A->B and B->A must share one session id"


def test_session_timeout_splits_long_gaps(cfg):
    c = {**cfg, "preprocessing": {**cfg["preprocessing"], "session_key_mode": "five_tuple",
                                  "session_timeout_s": 60, "session_max_flows": 0}}
    t = pd.to_datetime(["2017-07-03 09:00:00", "2017-07-03 09:00:30",
                        "2017-07-03 10:00:00"])          # 3rd flow is an hour later
    df = pd.DataFrame({
        "Source IP": ["10.0.0.1"] * 3, "Destination IP": ["10.0.0.2"] * 3,
        "Source Port": [1234] * 3, "Destination Port": [80] * 3, "Protocol": [6] * 3,
        dp.COL_TIME: t, dp.COL_SRC_FILE: ["a.csv"] * 3,
    })
    keys = dp.build_session_ids(df, c)
    assert keys.nunique() == 2
    assert keys.iloc[0] == keys.iloc[1] != keys.iloc[2]


def test_minmax_scaling_and_constant_features(prepared, cfg, split):
    df, feature_columns, _ = prepared
    pre = dp.TrainFittedPreprocessor(cfg)
    pre.fit(df, feature_columns, rows=split.train, split=split)
    Xtr = pre.transform(df, rows=split.train)
    Xte = pre.transform(df, rows=split.test)

    assert Xtr.to_numpy().min() >= 0.0 and Xtr.to_numpy().max() <= 1.0
    assert not Xtr.isna().any().any(), "imputation must leave no NaN"
    assert not Xte.isna().any().any()
    # Clipping policy: test values outside the training range stay in [0, 1].
    if cfg["preprocessing"]["minmax_out_of_range"] == "clip":
        assert Xte.to_numpy().min() >= 0.0 and Xte.to_numpy().max() <= 1.0
    # Constant features must not produce NaN/inf from a zero range.
    assert np.isfinite(Xtr.to_numpy()).all()


def test_transform_is_deterministic_and_column_stable(prepared, cfg, split):
    df, feature_columns, _ = prepared
    pre = dp.TrainFittedPreprocessor(cfg).fit(df, feature_columns, rows=split.train, split=split)
    a = pre.transform(df, rows=split.val)
    b = pre.transform(df, rows=split.val)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == pre.output_columns_


def test_unseen_category_maps_to_all_zeros(cfg):
    c = {**cfg}
    c["data"] = {**cfg["data"], "protocol_encoding": "onehot", "protocol_column": "Protocol"}
    df = pd.DataFrame({"Protocol": [6, 6, 17, 99], "x": [1.0, 2.0, 3.0, 4.0]})
    pre = dp.TrainFittedPreprocessor(c)
    pre.fit(df, ["x", "Protocol"], rows=[0, 1, 2])      # protocol 99 unseen in train
    out = pre.transform(df, rows=[3])
    onehot = [col for col in out.columns if col.startswith("Protocol=")]
    assert len(onehot) == 2                              # only 6 and 17 were seen
    assert out[onehot].to_numpy().sum() == 0.0           # unseen -> all zeros


def test_fixture_guard_rejects_small_data(cfg, data_dir, tmp_path):
    """Full mode must refuse an obviously-too-small folder (data.expect_min_rows).

    This is the guard that stops a run silently training on a synthetic fixture
    when --data_dir still points at test data.
    """
    import utils

    guarded = utils.deep_merge(cfg, {
        "run": {"smoke_test": False, "interim_dir": str(tmp_path / "interim")},
        "data": {"expect_min_rows": 2_000_000},
    })
    with pytest.raises(ValueError, match="expect_min_rows"):
        dp.prepare_dataset(guarded, data_dir=str(data_dir), use_cache=False)

    # ... and allows it once the guard is cleared.
    cleared = utils.deep_merge(guarded, {"data": {"expect_min_rows": None}})
    df, _feats, _rep = dp.prepare_dataset(cleared, data_dir=str(data_dir), use_cache=False)
    assert len(df) > 0


def test_cache_signature_separates_modes_and_sources(cfg):
    """A smoke cache and a full cache must never share a filename."""
    import utils

    full = utils.deep_merge(cfg, {"run": {"smoke_test": False},
                                  "data": {"data_dir": "/real/MachineLearningCVE"}})
    smoke = utils.deep_merge(cfg, {"run": {"smoke_test": True},
                                   "data": {"data_dir": "data/synthetic",
                                            "subsample_frac": 0.05}})
    other = utils.deep_merge(full, {"data": {"use_dst_port": True}})

    sigs = {utils.data_signature(c) for c in (full, smoke, other)}
    assert len(sigs) == 3, f"signatures collide: {sigs}"
    assert utils.data_signature(full).startswith("full__")
    assert utils.data_signature(smoke).startswith("smoke")
    # The signature is carried into every cached artifact name.
    for c in (full, smoke, other):
        assert utils.cache_path(c, "ranker_mi.joblib").name.startswith(utils.data_signature(c))
