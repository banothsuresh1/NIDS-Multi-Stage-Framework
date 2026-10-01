"""Stage 6, 7 and 9 unit tests: model interface, fusion algebra, temporal evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fusion as fus
import models as M
import temporal_pattern_mining as tpm
import utils


# ===========================================================================
# Stage 6
# ===========================================================================
@pytest.fixture(scope="module")
def tiny_cfg():
    import utils as u
    from pathlib import Path

    c = u.load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    c["models"]["rf"]["n_estimators"] = 20
    c["models"]["xgb"]["n_estimators"] = 20
    c["models"]["lstm"].update(epochs=2, patience=1)
    c["models"]["cnn"].update(epochs=2, patience=1)
    c["models"]["gnn"].update(epochs=10)
    return c


@pytest.fixture(scope="module")
def tiny_data():
    rng = np.random.default_rng(3)
    n, k, C = 300, 6, 4
    y = rng.integers(0, C, n)
    X = pd.DataFrame(rng.normal(y[:, None] * 1.5, 1.0, size=(n, k)),
                     columns=[f"f{i}" for i in range(k)])
    sessions = np.array([f"s{i // 5}" for i in range(n)])
    times = pd.to_datetime("2017-07-03") + pd.to_timedelta(np.arange(n), unit="s")
    hosts = np.array([f"h{i % 7}" for i in range(n)])
    return X, y, sessions, times.to_numpy(), hosts


@pytest.mark.parametrize("name", ["rf", "xgb", "lstm", "cnn", "gnn"])
def test_model_probabilities_are_aligned_and_valid(name, tiny_cfg, tiny_data):
    """Every model emits an (n, C) simplex over the SAME canonical class order."""
    X, y, sessions, times, hosts = tiny_data
    classes = list(range(len(tiny_cfg["labels"]["class_order"])))   # 9 canonical
    model = M.build_model(name, tiny_cfg, classes, seed=0)
    ctx = {"session_ids": sessions, "times": times, "hosts": hosts}
    model.fit(X, y, X, y, class_weights=None, **ctx)
    proba = model.predict_proba(X, **ctx)

    assert proba.shape == (len(X), len(classes))
    utils.assert_probabilities_valid(proba, name)
    # y only contains classes 0-3; the columns for 4-8 must exist and be ~0.
    assert np.allclose(proba[:, 4:].sum(), 0.0, atol=1e-6)


@pytest.mark.parametrize("name", ["rf", "xgb", "lstm", "cnn", "gnn"])
def test_model_save_load_roundtrip(name, tiny_cfg, tiny_data, tmp_path):
    X, y, sessions, times, hosts = tiny_data
    classes = list(range(len(tiny_cfg["labels"]["class_order"])))
    ctx = {"session_ids": sessions, "times": times, "hosts": hosts}
    model = M.build_model(name, tiny_cfg, classes, seed=0)
    model.fit(X, y, X, y, **ctx)
    before = model.predict_proba(X, **ctx)

    path = tmp_path / f"{name}_model"
    model.save(path)
    restored = type(model).load(path, tiny_cfg, classes)
    after = restored.predict_proba(X, **ctx)
    assert np.allclose(before, after, atol=1e-5)


def test_session_graph_is_time_causal(tiny_cfg, tiny_data):
    """Every edge must run from an earlier session to a later one."""
    X, y, sessions, times, hosts = tiny_data
    g = M.build_session_graph(sessions, times, hosts, X.to_numpy(np.float32), y, tiny_cfg)
    ei = g["edge_index"]
    if ei.shape[1] == 0:
        pytest.skip("no edges in this fixture")
    # Recover each node's first-flow time and check ordering.
    order = np.argsort(times, kind="mergesort")
    t_num = pd.to_datetime(pd.Series(times)).astype("int64").to_numpy() / 1e9
    node_time = {}
    for row in order:
        i = g["node_index"][sessions[row]]
        node_time[i] = min(node_time.get(i, np.inf), t_num[row])
    for s, d in zip(ei[0], ei[1]):
        assert node_time[s] <= node_time[d], "edge runs backwards in time"


def test_model_stats_recorded(tiny_cfg, tiny_data):
    X, y, sessions, times, hosts = tiny_data
    classes = list(range(len(tiny_cfg["labels"]["class_order"])))
    m = M.build_model("rf", tiny_cfg, classes, 0).fit(X, y, X, y)
    m.predict_proba(X)
    assert m.stats.train_time_s > 0
    assert m.stats.n_parameters > 0
    assert m.stats.inference_ms_per_flow >= 0
    table = M.model_stats_table({"rf": m})
    assert {"train_time_s", "inference_ms_per_flow", "n_parameters"} <= set(table.columns)


# ===========================================================================
# Stage 7
# ===========================================================================
def test_confidence_term_bounds():
    """C = 1 - H/log C is 1 for a one-hot row and 0 for a uniform row."""
    onehot = np.array([[1.0, 0.0, 0.0, 0.0]])
    uniform = np.full((1, 4), 0.25)
    assert fus.confidence_term(onehot)[0] == pytest.approx(1.0, abs=1e-6)
    assert fus.confidence_term(uniform)[0] == pytest.approx(0.0, abs=1e-6)


def test_temperature_scaler_roundtrip():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(500, 3)) * 3
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = proba.argmax(axis=1)
    cal = fus.TemperatureScaler(60).fit(proba, y)
    out = cal.transform(proba)
    utils.assert_probabilities_valid(out, "temperature")
    assert cal.temperature_ > 0


def test_per_class_calibrator_outputs_simplex():
    rng = np.random.default_rng(1)
    proba = rng.dirichlet(np.ones(4), size=400)
    y = rng.integers(0, 4, 400)
    for method in ("isotonic", "platt"):
        out = fus.PerClassCalibrator(method).fit(proba, y).transform(proba)
        utils.assert_probabilities_valid(out, method)


@pytest.fixture
def fusion_setup(tiny_cfg):
    rng = np.random.default_rng(7)
    n, C = 200, 9
    classes = list(range(C))
    y = rng.integers(0, 3, n)
    def make():
        p = rng.dirichlet(np.ones(C) * 0.4, size=n)
        p[np.arange(n), y] += 1.2
        return p / p.sum(axis=1, keepdims=True)
    proba = {m: make() for m in ("rf", "xgb", "lstm", "cnn", "gnn")}
    return tiny_cfg, classes, y, proba, n


def test_fusion_weights_sum_to_one_and_probabilities_valid(fusion_setup):
    cfg, classes, y, proba, n = fusion_setup
    cfg = utils.deep_merge(cfg, {"fusion": {"tune_lambdas": {"enabled": False}}})
    f = fus.EvidenceFusion(cfg=cfg, classes=classes, use_temporal=True)
    sp, tc = np.random.default_rng(0).random(n), np.random.default_rng(1).random(n)
    avail = np.ones(n)
    f.fit(proba, y, sp=sp, tc=tc, availability=avail)
    out = f.fuse(proba, sp=sp, tc=tc, availability=avail)

    utils.assert_weights_sum_to_one(out.weights, "fusion weights")
    utils.assert_probabilities_valid(out.fused_proba, "fused")
    assert out.weights.shape == (n, len(out.sources))
    assert set(out.sources) == set(proba) | {"sp", "tc"}
    assert np.all(out.risk >= 0) and np.all(out.risk <= 1.0001)


def test_unavailable_temporal_evidence_gets_zero_weight(fusion_setup):
    """a(t) = 0 must zero the SP/TC weights for that row (Stage 7.3)."""
    cfg, classes, y, proba, n = fusion_setup
    cfg = utils.deep_merge(cfg, {"fusion": {"tune_lambdas": {"enabled": False}}})
    f = fus.EvidenceFusion(cfg=cfg, classes=classes, use_temporal=True)
    sp, tc = np.full(n, 0.8), np.full(n, 0.7)
    avail = np.zeros(n)
    avail[: n // 2] = 1.0
    f.fit(proba, y, sp=sp, tc=tc, availability=avail)
    out = f.fuse(proba, sp=sp, tc=tc, availability=avail)

    i_sp = out.sources.index("sp")
    i_tc = out.sources.index("tc")
    assert np.allclose(out.weights[n // 2:, i_sp], 0.0)
    assert np.allclose(out.weights[n // 2:, i_tc], 0.0)
    assert out.weights[: n // 2, i_sp].sum() > 0
    # Rows with no temporal evidence still form a valid convex combination.
    utils.assert_weights_sum_to_one(out.weights)


def test_models_only_mode_excludes_temporal(fusion_setup):
    cfg, classes, y, proba, n = fusion_setup
    cfg = utils.deep_merge(cfg, {"fusion": {"tune_lambdas": {"enabled": False}}})
    f = fus.EvidenceFusion(cfg=cfg, classes=classes, use_temporal=False)
    f.fit(proba, y)
    out = f.fuse(proba)
    assert set(out.sources) == set(proba)
    assert "sp" not in out.sources and "tc" not in out.sources
    utils.assert_weights_sum_to_one(out.weights)


def test_equal_weights_ablation_is_uniform(fusion_setup):
    cfg, classes, y, proba, n = fusion_setup
    cfg = utils.deep_merge(cfg, {"fusion": {"tune_lambdas": {"enabled": False}}})
    f = fus.EvidenceFusion(cfg=cfg, classes=classes, use_temporal=False)
    f.fit(proba, y)
    out = f.fuse(proba, equal_weights=True)
    assert np.allclose(out.weights, 1.0 / len(out.sources))


def test_static_weights_do_not_vary_per_row(fusion_setup):
    """lambda = (1, 0, 0) must give every row the same weights (Stage 7.6)."""
    cfg, classes, y, proba, n = fusion_setup
    cfg = utils.deep_merge(cfg, {"fusion": {"tune_lambdas": {"enabled": False}}})
    f = fus.EvidenceFusion(cfg=cfg, classes=classes, use_temporal=False)
    f.fit(proba, y)
    out = f.fuse(proba, lambdas=(1.0, 0.0, 0.0))
    assert np.allclose(out.weights, out.weights[0], atol=1e-9)
    # ... whereas the adaptive weights do vary.
    adaptive = f.fuse(proba, lambdas=(0.5, 0.3, 0.2))
    assert adaptive.weights.std(axis=0).max() > 0


def test_threshold_is_fitted_on_validation(fusion_setup):
    cfg, classes, y, proba, n = fusion_setup
    cfg = utils.deep_merge(cfg, {"fusion": {"tune_lambdas": {"enabled": False}}})
    f = fus.EvidenceFusion(cfg=cfg, classes=classes, use_temporal=False)
    f.fit(proba, y)
    assert 0.0 <= f.tau_R <= 1.0


# ===========================================================================
# Stage 9
# ===========================================================================
def test_attack_state_graph_rows_are_distributions(cfg):
    seqs = [["Benign", "PortScan", "DoS"], ["Benign", "Benign", "PortScan"],
            ["PortScan", "DoS", "DoS"]]
    g = tpm.AttackStateGraph.fit(seqs, cfg, states=["Benign", "PortScan", "DoS"])
    assert np.allclose(g.matrix.sum(axis=1), 1.0)
    assert (g.matrix > 0).all(), "Laplace smoothing keeps every transition positive"
    # An observed transition must outrank an unobserved one.
    assert g.transition("Benign", "PortScan") > g.transition("DoS", "Benign")


def test_tc_in_unit_interval_and_needs_two_events(cfg):
    seqs = [["Benign", "PortScan", "DoS"], ["Benign"], []]
    g = tpm.AttackStateGraph.fit([["Benign", "PortScan"]], cfg,
                                 states=["Benign", "PortScan", "DoS"])
    tc, avail = tpm.compute_tc(seqs, g, cfg)
    assert np.all((tc >= 0) & (tc <= 1))
    assert avail[0] == 1.0 and avail[1] == 0.0 and avail[2] == 0.0
    assert tc[1] == 0.0


def test_prefixspan_finds_ordered_patterns():
    seqs = [["a", "b", "c"], ["a", "b", "d"], ["a", "b", "e"], ["x", "y"]]
    found = tpm._prefixspan_bundled(seqs, min_count=3, max_len=3, top_k=50)
    patterns = [tuple(p) for _c, p in found]
    assert ("a",) in patterns and ("b",) in patterns
    assert ("a", "b") in patterns, "ordered pair a->b occurs in 3 of 4 sequences"
    assert ("b", "a") not in patterns, "reverse order must not be reported"


def test_fpgrowth_bundled_supports():
    seqs = [["a", "b"], ["a", "b"], ["a", "c"], ["d"]]
    out = tpm._fpgrowth_bundled(seqs, min_support=0.5, max_len=2)
    sets = {frozenset(s): v for s, v in zip(out["itemsets"], out["support"])}
    assert sets[frozenset({"a"})] == pytest.approx(0.75)
    assert sets[frozenset({"a", "b"})] == pytest.approx(0.5)


def test_is_subsequence():
    assert tpm._is_subsequence(["a", "c"], ["a", "b", "c"])        # gaps allowed
    assert not tpm._is_subsequence(["c", "a"], ["a", "b", "c"])    # order matters
    assert tpm._is_subsequence([], ["a"])


def test_sp_is_in_unit_interval(cfg):
    patterns = pd.DataFrame({
        "pattern_items": [["a", "b"], ["c"]],
        "support": [0.8, 0.4],
        "pattern": ["a -> b", "c"],
    })
    seqs = [["a", "x", "b"], ["c"], ["z"]]
    sp = tpm.compute_sp(seqs, patterns, cfg)
    assert np.all((sp >= 0) & (sp <= 1))
    assert sp[0] == pytest.approx(1.0)      # matches the strongest pattern
    assert sp[2] == 0.0                     # matches nothing


def test_worked_example_reproduces_the_risk_score(fusion_setup):
    """The w x score column must sum to R_t (methodology section 7.4)."""
    cfg, classes, y, proba, n = fusion_setup
    cfg = utils.deep_merge(cfg, {"fusion": {"tune_lambdas": {"enabled": False}}})
    sp, tc = np.random.default_rng(2).random(n), np.random.default_rng(3).random(n)
    avail = np.ones(n)
    f = fus.EvidenceFusion(cfg=cfg, classes=classes, use_temporal=True)
    f.fit(proba, y, sp=sp, tc=tc, availability=avail)
    fw = f.fuse(proba, sp=sp, tc=tc, availability=avail)

    table = fus.worked_example(fw, f, row=0, proba=proba, sp=sp, tc=tc)
    total = table.loc[table["source"] == "TOTAL", "w_x_score"].iloc[0]
    assert total == pytest.approx(float(fw.risk[0]), abs=1e-3)
    # Weights in the table sum to 1.
    assert table.loc[table["source"] == "TOTAL", "weight_w"].iloc[0] == pytest.approx(1.0, abs=1e-3)
