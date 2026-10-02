"""Shared utilities: config loading, seeding, logging, IO, timers, leakage guards.

Every module in this project reads its hyperparameters from the config dict
returned by :func:`load_config`; nothing is hard-coded at the call site.

The leakage guards in this module are *assertions executed at run time*, not
documentation. They are exercised by ``tests/test_leakage.py``.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # pandas is a hard dependency but keep utils importable for config-only use
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

LOGGER_NAME = "nids"
_LOG_CONFIGURED = False


# ===========================================================================
# Configuration
# ===========================================================================
def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` and return a new dict.

    Mappings are merged key-by-key; every other type (including lists) is
    replaced wholesale. Used for both the smoke-test block and branch configs.
    """
    out: Dict[str, Any] = dict(copy.deepcopy(dict(base)))
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(
    path: str | os.PathLike = "config.yaml",
    overrides: Optional[Mapping[str, Any]] = None,
    apply_smoke: bool = True,
) -> Dict[str, Any]:
    """Load ``config.yaml``, apply branch overrides, then the smoke-test block.

    Parameters
    ----------
    path:
        Path to the root YAML config.
    overrides:
        Optional mapping deep-merged onto the root config (branch configs,
        CLI flags). Applied *before* the smoke block so that a branch can turn
        smoke mode on or off.
    apply_smoke:
        When True and ``run.smoke_test`` is truthy, the ``smoke:`` sub-tree is
        deep-merged onto the top level so the full pipeline runs in minutes.

    Returns
    -------
    dict
        The fully resolved configuration. The unmerged smoke block is retained
        under ``_smoke_raw`` for reference and ``run.smoke_test`` is preserved.
    """
    import yaml

    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh) or {}

    if overrides:
        cfg = deep_merge(cfg, overrides)

    smoke_block = cfg.pop("smoke", {}) or {}
    cfg["_smoke_raw"] = smoke_block
    if apply_smoke and cfg.get("run", {}).get("smoke_test", False):
        smoke_test_flag = cfg["run"]["smoke_test"]
        cfg = deep_merge(cfg, smoke_block)
        cfg.setdefault("run", {})["smoke_test"] = smoke_test_flag
    cfg["_config_path"] = str(path.resolve())
    return cfg


def save_config_snapshot(cfg: Mapping[str, Any], out_dir: str | os.PathLike) -> Path:
    """Persist the resolved config next to the results so a run is reproducible."""
    out = Path(out_dir) / "resolved_config.json"
    ensure_dir(out.parent)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(cfg), fh, indent=2, sort_keys=True)
    return out


# ===========================================================================
# Seeding & framework configuration
# ===========================================================================
def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed python, numpy, torch, tensorflow and the hashing of dict ordering.

    xgboost/lightgbm/sklearn take ``random_state`` explicitly at construction
    time (see :func:`seeded`), because seeding globally is not enough for them.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def seeded(cfg: Mapping[str, Any]) -> int:
    """Return the primary seed from config."""
    return int(cfg.get("seed", {}).get("primary", 42))


def seed_list(cfg: Mapping[str, Any]) -> List[int]:
    """Seeds for multi-seed aggregation; ``[primary]`` when multi-seed is off."""
    sd = cfg.get("seed", {})
    if sd.get("enable_multi_seed", False):
        return [int(s) for s in sd.get("multi_seed", [sd.get("primary", 42)])]
    return [int(sd.get("primary", 42))]


def configure_frameworks(cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Configure TF/torch so the two can coexist in one process.

    TensorFlow allocates the whole GPU on first use, which starves torch. We
    enable memory growth *before* any TF op is created, and cap intra/inter-op
    threads so TF and torch do not oversubscribe the CPU.

    Returns a dict describing the detected devices (logged into the report).
    """
    info: Dict[str, Any] = {"torch": None, "tensorflow": None, "gpu": False}

    # Keep TF quiet unless the user asked for debug output.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # joblib/loky probes for physical cores and warns on Windows when it cannot
    # ("Could not find the number of physical cores"). Pin it to the logical
    # count (or run.loky_max_cpu_count) before any joblib pool is created.
    n_cpu = (cfg or {}).get("run", {}).get("loky_max_cpu_count") or os.cpu_count() or 1
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(int(n_cpu)))

    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = bool(torch.cuda.is_available())
        info["gpu"] = info["gpu"] or info["torch_cuda"]
        n_threads = max(1, (os.cpu_count() or 2) // 2)
        torch.set_num_threads(n_threads)
    except Exception as exc:  # pragma: no cover
        info["torch_error"] = str(exc)

    try:
        import tensorflow as tf

        info["tensorflow"] = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            with contextlib.suppress(RuntimeError):
                tf.config.experimental.set_memory_growth(gpu, True)
        info["tf_gpu"] = len(gpus)
        info["gpu"] = info["gpu"] or bool(gpus)
    except Exception as exc:  # pragma: no cover
        info["tf_error"] = str(exc)

    return info


# ===========================================================================
# Logging & timing
# ===========================================================================
def get_logger(name: str = LOGGER_NAME, level: str = "INFO",
               log_file: Optional[str | os.PathLike] = None) -> logging.Logger:
    """Return the project logger, configuring handlers exactly once."""
    global _LOG_CONFIGURED
    logger = logging.getLogger(name)
    if not _LOG_CONFIGURED:
        logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if log_file is not None:
            ensure_dir(Path(log_file).parent)
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        logger.propagate = False
        _LOG_CONFIGURED = True
    return logger


@dataclass
class Timer:
    """Context manager that records wall-clock seconds into ``registry``."""

    label: str
    registry: Optional[Dict[str, float]] = None
    logger: Optional[logging.Logger] = None
    elapsed: float = field(default=0.0, init=False)
    _t0: float = field(default=0.0, init=False)

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._t0
        if self.registry is not None:
            self.registry[self.label] = self.elapsed
        if self.logger is not None:
            self.logger.info("[timing] %s: %.2fs", self.label, self.elapsed)


def progress(iterable: Iterable, desc: str = "", total: Optional[int] = None,
             disable: bool = False) -> Iterator:
    """tqdm wrapper that degrades gracefully when tqdm is missing."""
    try:
        # Plain tqdm, never tqdm.auto: the notebook flavour needs ipywidgets and
        # prints "Error displaying widget: model not found" when it is missing or
        # the kernel is driven by nbconvert. Text bars work everywhere.
        from tqdm import tqdm

        return tqdm(iterable, desc=desc, total=total, disable=disable, leave=False)
    except Exception:  # pragma: no cover
        return iter(iterable)


# ===========================================================================
# IO helpers
# ===========================================================================
def ensure_dir(path: str | os.PathLike) -> Path:
    """Create ``path`` (and parents) if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _jsonable(obj: Any) -> Any:
    """Recursively coerce numpy/pandas scalars and arrays into JSON types."""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if pd is not None and isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if pd is not None and isinstance(obj, pd.Series):
        return _jsonable(obj.to_dict())
    if pd is not None and isinstance(obj, pd.DataFrame):
        return _jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Any, path: str | os.PathLike, indent: int = 2) -> Path:
    """Write ``obj`` as JSON, coercing numpy types."""
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(obj), fh, indent=indent, sort_keys=False)
    return p


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_table(df: "pd.DataFrame", path: str | os.PathLike, index: bool = False) -> Path:
    """Write a DataFrame to CSV, creating parents."""
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=index)
    return p


def save_fig(fig, path: str | os.PathLike, dpi: int = 220, close: bool = True) -> Path:
    """Save a matplotlib figure at >= 200 dpi (Stage 8 requirement)."""
    p = Path(path)
    ensure_dir(p.parent)
    fig.savefig(p, dpi=max(int(dpi), 200), bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return p


def data_signature(cfg: Mapping[str, Any]) -> str:
    """A short tag identifying WHICH data and WHICH mode produced an artifact.

    Every cache name embeds this, so a cache written by a smoke run over the
    synthetic fixture can never be silently reused by a full run over the real
    CIC-IDS2017 CSVs -- the filenames simply do not collide. Pointing
    ``--data_dir`` at a different folder also changes the signature.

    Example: ``full__MachineLearningCVE__noport`` vs
    ``smoke5pct__synthetic__noport``.
    """
    run = cfg.get("run", {})
    data = cfg.get("data", {})
    if run.get("smoke_test"):
        frac = data.get("subsample_frac")
        mode = f"smoke{round(float(frac) * 100)}pct" if frac else "smoke"
    else:
        mode = "full"
    source = Path(str(data.get("data_dir", "unknown"))).name or "unknown"
    source = re.sub(r"[^A-Za-z0-9._-]+", "-", source)[:48]
    port = "withport" if data.get("use_dst_port") else "noport"
    return f"{mode}__{source}__{port}"


def cache_path(cfg: Mapping[str, Any], name: str) -> Path:
    """Resolve a cache filename inside the configured cache directory.

    The data signature is prefixed automatically unless ``name`` already
    carries one, so callers cannot forget it.
    """
    sig = data_signature(cfg)
    if not name.startswith(sig):
        name = f"{sig}__{name}"
    return ensure_dir(cfg.get("run", {}).get("cache_dir", "data/interim/cache")) / name


def joblib_cache(cfg: Mapping[str, Any], name: str, producer, force: bool = False):
    """Return a cached artifact, computing and storing it on a miss.

    ``producer`` is a zero-argument callable. Caching is skipped entirely when
    ``run.use_cache`` is false or ``force`` is set.
    """
    import joblib

    use_cache = bool(cfg.get("run", {}).get("use_cache", True)) and not force
    path = cache_path(cfg, name)
    if use_cache and path.exists():
        try:
            return joblib.load(path), True
        except Exception:
            pass
    value = producer()
    if bool(cfg.get("run", {}).get("use_cache", True)):
        with contextlib.suppress(Exception):
            joblib.dump(value, path, compress=3)
    return value, False


def memory_mb() -> float:
    """Resident set size of this process in MB (0.0 if unavailable)."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


# ===========================================================================
# Split bookkeeping
# ===========================================================================
@dataclass
class SplitIndex:
    """Row indices (positional, into the cleaned frame) for one split.

    ``boundaries`` carries the per-class timestamp block boundaries of the
    time-blocked split so :func:`assert_no_time_leakage` can verify them.
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    strategy: str = "time_blocked"
    boundaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dropped_gap: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def sizes(self) -> Dict[str, int]:
        return {k: int(len(v)) for k, v in self.as_dict().items()}


# ===========================================================================
# LEAKAGE GUARDS  (enforced in code, see Stage 8.4)
# ===========================================================================
class LeakageError(AssertionError):
    """Raised when a train/val/test contamination check fails."""


def assert_disjoint_splits(split: SplitIndex) -> None:
    """Train, validation and test index sets must be pairwise disjoint."""
    tr, va, te = set(split.train.tolist()), set(split.val.tolist()), set(split.test.tolist())
    for a_name, a, b_name, b in (
        ("train", tr, "val", va),
        ("train", tr, "test", te),
        ("val", va, "test", te),
    ):
        overlap = a & b
        if overlap:
            raise LeakageError(
                f"{a_name}/{b_name} splits overlap on {len(overlap)} rows "
                f"(e.g. {sorted(list(overlap))[:5]})"
            )


def assert_sessions_not_split(split: SplitIndex, session_ids: Sequence[Any],
                              exempt: Optional[Iterable[Any]] = None) -> None:
    """A session must live entirely in one partition (Stage 8.4).

    ``exempt`` lists sessions excused from the rule. These are the sessions of
    *tiny* classes (Heartbleed, Infiltration) split at flow level so that the
    methodology's "every class appears in all three partitions" requirement can
    be met at all -- the two requirements are mutually unsatisfiable for a class
    with fewer sessions than partitions. The exemption is recorded in
    ``split.meta['exempt_sessions']`` and reported in the split table.
    """
    if split.strategy == "stratified_flow":
        # This split exists precisely to tear sessions apart and quantify the
        # resulting optimism (the "leakage gap", branch D). Checking session
        # integrity here would assert that the baseline is not what it is.
        return
    sess = np.asarray(session_ids)
    skip = set(exempt or split.meta.get("exempt_sessions", set()) or set())
    seen: Dict[Any, str] = {}
    for part, idx in split.as_dict().items():
        for s in np.unique(sess[idx]):
            if s in skip:
                continue
            if s in seen and seen[s] != part:
                raise LeakageError(
                    f"session {s!r} appears in both {seen[s]} and {part}"
                )
            seen[s] = part


def assert_no_time_leakage(split: SplitIndex, timestamps: Sequence[Any],
                           labels: Optional[Sequence[Any]] = None) -> None:
    """No test flow may predate the train block boundary of its class.

    For the time-blocked split the per-class ordering is train < val < test, and
    blocks are cut per class (Stage 3.2), so the check runs per class.

    The grouping used is the *assignment group* -- the primary class of the
    session, recorded in ``split.meta['assign_group']`` -- not the raw flow
    label. The two differ only for a session carrying more than one class: such
    a session is placed by its rarest class and its benign flows travel with it,
    because sessions are kept whole. The invariant the construction guarantees
    (and the one that matters for leakage) is per assignment group. When no
    assignment group is recorded the check falls back to ``labels``.
    """
    if split.strategy not in ("time_blocked", "loao"):
        return
    ts = pd.to_datetime(pd.Series(np.asarray(timestamps)), errors="coerce").values
    group_src = split.meta.get("assign_group")
    if group_src is None:
        group_src = labels
    if group_src is None:
        groups = {"__all__": np.arange(len(ts))}
    else:
        lab = np.asarray(group_src)
        groups = {c: np.where(lab == c)[0] for c in np.unique(lab)}

    for cls, rows in groups.items():
        rowset = set(rows.tolist())
        tr = np.array([i for i in split.train if i in rowset], dtype=np.int64)
        va = np.array([i for i in split.val if i in rowset], dtype=np.int64)
        te = np.array([i for i in split.test if i in rowset], dtype=np.int64)
        if len(tr) == 0 or len(te) == 0:
            continue
        tr_max = np.nanmax(ts[tr])
        te_min = np.nanmin(ts[te])
        if te_min < tr_max:
            raise LeakageError(
                f"class {cls!r}: test flow at {te_min} predates the last train "
                f"flow at {tr_max} -- time-blocked ordering violated"
            )
        if len(va):
            va_min, va_max = np.nanmin(ts[va]), np.nanmax(ts[va])
            if va_min < tr_max:
                raise LeakageError(
                    f"class {cls!r}: validation flow at {va_min} predates the "
                    f"last train flow at {tr_max}"
                )
            if te_min < va_max:
                raise LeakageError(
                    f"class {cls!r}: test flow at {te_min} predates the last "
                    f"validation flow at {va_max}"
                )


def assert_fitted_on_train_only(fitted_rows: Sequence[int], split: SplitIndex,
                                what: str = "transformer") -> None:
    """Every row used to ``fit`` must be a training row."""
    train = set(split.train.tolist())
    bad = [int(i) for i in fitted_rows if int(i) not in train]
    if bad:
        raise LeakageError(
            f"{what} was fitted on {len(bad)} non-train rows "
            f"(e.g. {bad[:5]}); fit must use the train partition only"
        )


def assert_not_resampled(n_before: int, n_after: int, what: str) -> None:
    """Validation/test row counts must be unchanged by imbalance handling."""
    if n_before != n_after:
        raise LeakageError(
            f"{what} changed from {n_before} to {n_after} rows -- validation "
            f"and test partitions must never be resampled (Stage 5.2)"
        )


def assert_feature_mask_frozen(expected: Sequence[str], actual: Sequence[str],
                               where: str) -> None:
    """F* must be applied unchanged (same features, same order) everywhere."""
    if list(expected) != list(actual):
        raise LeakageError(
            f"frozen feature set F* differs at {where}: expected {list(expected)[:5]}... "
            f"({len(expected)} features) got {list(actual)[:5]}... ({len(actual)} features)"
        )


def assert_probabilities_valid(proba: np.ndarray, what: str = "probabilities",
                               tol: float = 1e-4) -> None:
    """Probability rows must be non-negative and sum to one."""
    arr = np.asarray(proba, dtype=np.float64)
    if arr.ndim != 2:
        raise LeakageError(f"{what}: expected a 2-D (n, C) matrix, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise LeakageError(f"{what}: contains non-finite values")
    if arr.min() < -tol:
        raise LeakageError(f"{what}: negative probability {arr.min():.6f}")
    sums = arr.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-3):
        worst = float(np.abs(sums - 1.0).max())
        raise LeakageError(f"{what}: rows do not sum to 1 (max deviation {worst:.6f})")


def assert_weights_sum_to_one(weights: np.ndarray, what: str = "fusion weights",
                              tol: float = 1e-6) -> None:
    """Per-session fusion weights must form a convex combination (eq. 24/26)."""
    arr = np.asarray(weights, dtype=np.float64)
    sums = arr.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=1e-5):
        worst = float(np.abs(sums - 1.0).max())
        raise LeakageError(f"{what}: do not sum to 1 (max deviation {worst:.6f})")
    if arr.min() < -tol:
        raise LeakageError(f"{what}: negative weight {arr.min():.6f}")


def assert_no_leakage(
    split: SplitIndex,
    session_ids: Optional[Sequence[Any]] = None,
    timestamps: Optional[Sequence[Any]] = None,
    labels: Optional[Sequence[Any]] = None,
    fitted_rows: Optional[Sequence[int]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, bool]:
    """Run every applicable leakage check and return a pass/fail report.

    Raises :class:`LeakageError` on the first violation; the returned dict is
    for the preprocessing report when all checks pass.
    """
    report: Dict[str, bool] = {}
    assert_disjoint_splits(split)
    report["splits_disjoint"] = True

    if session_ids is not None:
        assert_sessions_not_split(split, session_ids,
                                  exempt=split.meta.get("exempt_sessions"))
        report["sessions_whole"] = True
    if timestamps is not None:
        assert_no_time_leakage(split, timestamps, labels)
        report["time_ordering"] = True
    if fitted_rows is not None:
        assert_fitted_on_train_only(fitted_rows, split, "preprocessor")
        report["fit_on_train_only"] = True

    if logger is not None:
        logger.info("[leakage] checks passed: %s", ", ".join(sorted(report)))
    return report


# ===========================================================================
# Misc
# ===========================================================================
def stratified_subsample(y: np.ndarray, max_rows: int, seed: int,
                         min_per_class: int = 1) -> np.ndarray:
    """Per-class capped stratified row sample (Stage 4.4).

    Rare classes are never dropped: each class contributes at least
    ``min_per_class`` rows (or all of its rows, if it has fewer).
    """
    y = np.asarray(y)
    n = len(y)
    if max_rows is None or n <= max_rows:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    # Proportional allocation with a floor, then trim the largest classes.
    alloc = np.maximum(
        np.minimum(counts, min_per_class),
        np.floor(counts / n * max_rows).astype(int),
    )
    alloc = np.minimum(alloc, counts)
    # Trim proportionally if the floor pushed us over budget.
    while alloc.sum() > max_rows:
        over = alloc.sum() - max_rows
        shrinkable = alloc > min_per_class
        if not shrinkable.any():
            break
        order = np.argsort(-alloc)
        for i in order:
            if over <= 0:
                break
            if alloc[i] > min_per_class:
                take = min(over, alloc[i] - min_per_class)
                alloc[i] -= take
                over -= take

    picks: List[np.ndarray] = []
    for cls, k in zip(classes, alloc):
        idx = np.where(y == cls)[0]
        if k >= len(idx):
            picks.append(idx)
        else:
            picks.append(rng.choice(idx, size=int(k), replace=False))
    out = np.concatenate(picks) if picks else np.arange(0)
    return np.sort(out)


def chunked(n: int, size: int) -> Iterator[Tuple[int, int]]:
    """Yield ``(start, stop)`` slices of length ``size`` covering ``range(n)``."""
    size = max(1, int(size))
    for start in range(0, n, size):
        yield start, min(start + size, n)


def class_order(cfg: Mapping[str, Any]) -> List[str]:
    """Canonical class name order (fixes the probability-matrix column order)."""
    return list(cfg["labels"]["class_order"])


def label_to_index(cfg: Mapping[str, Any]) -> Dict[str, int]:
    return {c: i for i, c in enumerate(class_order(cfg))}


def format_checklist(rows: Sequence[Tuple[str, str, str]]) -> str:
    """Render a stage -> file -> status table as fixed-width text."""
    header = ("Stage", "File", "Status")
    widths = [max(len(header[i]), max((len(r[i]) for r in rows), default=0)) for i in range(3)]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [line, "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |", line]
    for r in rows:
        out.append("| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(3)) + " |")
    out.append(line)
    return "\n".join(out)
