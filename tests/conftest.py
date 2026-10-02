"""Shared pytest fixtures: a small synthetic dataset prepared once per session."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils                      # noqa: E402
import data_preprocessing as dp   # noqa: E402
from tests.synthetic_data import generate   # noqa: E402


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    """A small synthetic CIC-IDS2017-shaped dataset."""
    out = tmp_path_factory.mktemp("cicids")
    generate(out, total_rows=9000, seed=11)
    return out


@pytest.fixture(scope="session")
def cfg(tmp_path_factory) -> dict:
    """Root config in smoke mode, with outputs redirected into a temp dir."""
    c = utils.load_config(ROOT / "config.yaml")
    tmp = tmp_path_factory.mktemp("run")
    c["run"]["output_dir"] = str(tmp / "results")
    c["run"]["interim_dir"] = str(tmp / "interim")
    c["run"]["cache_dir"] = str(tmp / "cache")
    c["run"]["model_dir"] = str(tmp / "models")
    c["run"]["use_cache"] = False
    # The suite runs against a deliberately tiny generated fixture, so the
    # real-dataset size guard must not fire here. (That the guard DOES fire on
    # small data in full mode is covered by test_fixture_guard_rejects_small_data.)
    c["run"]["smoke_test"] = True
    c["data"]["expect_min_rows"] = None
    c["data"]["subsample_frac"] = 0.25
    # The stability loop re-runs all eight rankers n_runs times; the dedicated
    # ranker tests already cover them, so keep it out of the shared fixture.
    c["feature_selection"]["stability"]["enabled"] = False
    c["feature_selection"]["pi_svm"].update(max_rows=600, n_repeats=1)
    c["feature_selection"]["lstm"].update(epochs=1, max_windows=1200, grad_batches=3)
    return c


@pytest.fixture(scope="session")
def prepared(cfg, data_dir):
    """``(df, feature_columns, report)`` from Stages 1-2."""
    utils.set_global_seed(utils.seeded(cfg))
    return dp.prepare_dataset(cfg, data_dir=str(data_dir), seed=utils.seeded(cfg),
                              use_cache=False)


@pytest.fixture(scope="session")
def split(cfg, prepared):
    """The MAIN time-blocked split."""
    df = prepared[0]
    return dp.time_blocked_split(df, cfg)
