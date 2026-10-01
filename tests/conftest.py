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
    generate(out, total_rows=12000, seed=11)
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
    c["data"]["subsample_frac"] = 0.25
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
