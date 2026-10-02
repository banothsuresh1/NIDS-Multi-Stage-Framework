"""Stages 1-3: dataset consolidation, preprocessing and train/val/test splitting.

Stage 1  -- consolidate the daily CIC-IDS2017 CSVs, normalise labels, build
            session identifiers from the symmetric communication key (eq. 22).
Stage 2  -- :class:`ColumnCleaner` (split-independent cleaning) and
            :class:`TrainFittedPreprocessor` (median imputer + Min-Max scaler +
            one-hot encoder, all fitted on TRAIN only, Stage 2.3 - 2.5).
Stage 3  -- time-blocked (main), stratified (session-grouped and flow-level)
            and leave-one-attack-out splits.

Leakage rule enforced here: ``ColumnCleaner`` performs only operations that
learn nothing from the data (schema alignment, dedup, inf->NaN, sentinel
preservation) and may therefore run before the split.
``TrainFittedPreprocessor.fit`` records the row indices it saw and refuses to
fit on anything but training rows.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils import (
    LeakageError,
    SplitIndex,
    data_signature,
    assert_fitted_on_train_only,
    ensure_dir,
    get_logger,
    progress,
    save_json,
    save_table,
)

LOGGER = get_logger()

# Internal metadata columns added by this module. Never part of X.
COL_LABEL = "Label_grouped"
COL_LABEL_RAW = "Label_raw"
COL_Y = "y"
COL_SESSION = "session_id"
COL_TIME = "__timestamp"
COL_ORDER = "__order"
COL_SRC_FILE = "__source_file"
META_COLUMNS = [COL_LABEL, COL_LABEL_RAW, COL_Y, COL_SESSION, COL_TIME, COL_ORDER, COL_SRC_FILE]


# ===========================================================================
# Stage 1: consolidation
# ===========================================================================
def _normalise_colname(name: str) -> str:
    """Strip whitespace and collapse internal runs of spaces."""
    return re.sub(r"\s+", " ", str(name)).strip()


def _normalise_label(label: str) -> str:
    """Canonicalise a raw label for the Stage 1.3 grouping lookup.

    Handles the several spellings in the public release, e.g.
    ``'Web Attack \\x96 Brute Force'``, ``'Web Attack - XSS'``,
    ``'Web Attack: SQL Injection'``.
    """
    s = str(label)
    s = s.replace("\x96", "-").replace("–", "-").replace("—", "-")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    # 'web attack brute force' etc. already collapse to the config keys.
    return s


def _is_afternoon_file(filename: str, tokens: Sequence[str]) -> bool:
    low = filename.lower()
    return any(t.lower() in low for t in tokens)


def _parse_day_from_filename(filename: str) -> Optional[int]:
    """Map a CIC-IDS2017 file name to its weekday index (Mon=0 ... Fri=4)."""
    low = filename.lower()
    for i, day in enumerate(["monday", "tuesday", "wednesday", "thursday", "friday"]):
        if day in low:
            return i
    return None


def _fix_timestamps(df: pd.DataFrame, filename: str, cfg: Mapping[str, Any]) -> pd.Series:
    """Parse the Timestamp column, correcting the 12-hour ambiguity.

    CIC-IDS2017 CSVs store times in 12-hour format without AM/PM. Afternoon
    capture files are identified by name; any parsed hour below
    ``afternoon_shift_hour_max`` in such a file is shifted by +12 hours
    (Stage 3.2).
    """
    data_cfg = cfg["data"]
    raw = df[data_cfg["timestamp_column"]]
    ts = pd.to_datetime(raw, errors="coerce", dayfirst=True)
    if ts.isna().all():
        ts = pd.to_datetime(raw, errors="coerce")

    if _is_afternoon_file(filename, data_cfg.get("afternoon_file_tokens", [])):
        shift_below = int(data_cfg.get("afternoon_shift_hour_max", 12))
        shift_by = int(data_cfg.get("afternoon_hour_shift", 12))
        mask = ts.notna() & (ts.dt.hour < shift_below)
        if mask.any():
            ts.loc[mask] = ts.loc[mask] + pd.Timedelta(hours=shift_by)
            LOGGER.info(
                "  [%s] shifted %d afternoon timestamps by +%dh",
                Path(filename).name, int(mask.sum()), shift_by,
            )
    return ts


def _synthesise_timestamps(df: pd.DataFrame, filename: str, start: pd.Timestamp) -> pd.Series:
    """Fabricate a monotone surrogate time axis when no Timestamp column exists.

    The ``MachineLearningCVE`` release of CIC-IDS2017 ships only the 78 numeric
    features and the label -- no Flow ID, IPs or Timestamp. CICFlowMeter writes
    rows in (approximately) flow-start order, so row position within a file,
    offset by the file's weekday, is used as the ordering key. This is a
    documented deviation (see README) and is only ever used for *ordering*.
    """
    day = _parse_day_from_filename(filename)
    if day is None:
        day = 0
    offset = pd.Timedelta(days=day)
    if _is_afternoon_file(filename, ["afternoon"]):
        offset += pd.Timedelta(hours=12)
    return pd.Series(
        start + offset + pd.to_timedelta(np.arange(len(df), dtype=np.int64), unit="s"),
        index=df.index,
    )


def discover_csv_files(data_dir: str | os.PathLike, pattern: str = "*.csv") -> List[Path]:
    """Return every CSV under ``data_dir`` (recursively), sorted by weekday.

    The methodology says 7 files in Stage 2.2 and 8 in Stage 1.2; we simply
    glob whatever is present and report the count (see README "Deviations").
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data directory not found: {root}")
    files = sorted(set(glob.glob(str(root / pattern)) + glob.glob(str(root / "**" / pattern), recursive=True)))
    paths = [Path(f) for f in files]
    if not paths:
        raise FileNotFoundError(f"no CSV files matching {pattern!r} under {root}")

    def sort_key(p: Path) -> Tuple[int, int, str]:
        day = _parse_day_from_filename(p.name)
        return (day if day is not None else 99,
                1 if _is_afternoon_file(p.name, ["afternoon"]) else 0,
                p.name)

    return sorted(paths, key=sort_key)


def load_raw_dataset(cfg: Mapping[str, Any], data_dir: Optional[str] = None) -> pd.DataFrame:
    """Stage 1: concatenate the daily CSVs into one schema-aligned frame.

    Column names are stripped of whitespace, the schema is aligned across files
    (union of columns; missing columns become NaN), timestamps are corrected,
    and a monotone ``__order`` key is attached for downstream sorting.
    """
    data_cfg = cfg["data"]
    data_dir = data_dir or data_cfg["data_dir"]
    files = discover_csv_files(data_dir, data_cfg.get("csv_glob", "*.csv"))
    LOGGER.info("Stage 1: found %d CSV file(s) under %s", len(files), data_dir)

    float_dtype = data_cfg.get("float_dtype", "float32")
    frames: List[pd.DataFrame] = []
    synth_start = pd.Timestamp("2017-07-03 09:00:00")

    for path in progress(files, desc="reading CSVs", total=len(files)):
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")
        df.columns = [_normalise_colname(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]

        ts_col = data_cfg["timestamp_column"]
        if ts_col in df.columns:
            df[COL_TIME] = _fix_timestamps(df, path.name, cfg)
            if df[COL_TIME].isna().all():
                LOGGER.warning("  [%s] timestamps unparseable; synthesising order",
                               path.name)
                df[COL_TIME] = _synthesise_timestamps(df, path.name, synth_start)
        else:
            df[COL_TIME] = _synthesise_timestamps(df, path.name, synth_start)

        df[COL_SRC_FILE] = path.name
        # Downcast numerics early: 2.8M x 78 float64 is ~1.7 GB, float32 halves it.
        for col in df.columns:
            if col in (COL_TIME, COL_SRC_FILE):
                continue
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype(float_dtype)
        frames.append(df)
        LOGGER.info("  [%s] %d rows x %d cols", path.name, len(df), df.shape[1])

    merged = pd.concat(frames, ignore_index=True, sort=False)
    del frames

    # Deterministic global ordering: time first, then original file order.
    merged = merged.sort_values(
        [COL_TIME, COL_SRC_FILE], kind="mergesort"
    ).reset_index(drop=True)
    merged[COL_ORDER] = np.arange(len(merged), dtype=np.int64)
    LOGGER.info("Stage 1: consolidated frame %d rows x %d cols", *merged.shape)
    return merged


# ===========================================================================
# Stage 1/2: label grouping and session construction
# ===========================================================================
def group_labels(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Attach ``Label_raw``, ``Label_grouped`` and the integer code ``y``.

    Grouping follows the Stage 1.3 table exactly. Unmapped raw labels raise,
    rather than being silently dropped into a catch-all class.
    """
    label_col = cfg["data"]["label_column"]
    if label_col not in df.columns:
        candidates = [c for c in df.columns if c.lower() == label_col.lower()]
        if not candidates:
            raise KeyError(f"label column {label_col!r} not found in {list(df.columns)[:10]}...")
        label_col = candidates[0]

    grouping = {_normalise_label(k): v for k, v in cfg["labels"]["grouping"].items()}
    raw = df[label_col].astype(str)
    norm = raw.map(_normalise_label)
    grouped = norm.map(grouping)

    # Rows with no usable label. The public CIC-IDS2017 CSVs contain some blank
    # and ragged rows, and a file that was concatenated from two others carries
    # the second file's HEADER row as data (its Label cell reads "Label").
    # Such a row carries no supervision: it can be neither trained on nor scored,
    # so it is dropped rather than mapped to a class. The count is logged and
    # kept in the preprocessing report.
    unlabeled_tokens = {"", "nan", "none", "null", "na",
                        _normalise_label(label_col), "label"}
    unlabeled = grouped.isna() & norm.isin(unlabeled_tokens)
    n_unlabeled = int(unlabeled.sum())

    if n_unlabeled and cfg["data"].get("drop_unlabeled_rows", True):
        LOGGER.warning(
            "Stage 1: dropping %d row(s) (%.4f%%) with no usable label "
            "(blank/NaN, or an embedded header row). Set "
            "data.drop_unlabeled_rows: false to raise instead.",
            n_unlabeled, 100.0 * n_unlabeled / max(len(df), 1))
        df = df.loc[~unlabeled.to_numpy()].reset_index(drop=True)
        raw = raw.loc[~unlabeled.to_numpy()].reset_index(drop=True)
        norm = norm.loc[~unlabeled.to_numpy()].reset_index(drop=True)
        grouped = grouped.loc[~unlabeled.to_numpy()].reset_index(drop=True)

    # Anything still unmapped is a real label this config does not know about --
    # that must be fixed in labels.grouping, never silently discarded.
    unmapped = sorted(set(norm[grouped.isna()].unique()))
    if unmapped:
        raise ValueError(
            "unmapped raw labels (add them to labels.grouping in config.yaml): "
            f"{unmapped}"
        )

    order = cfg["labels"]["class_order"]
    code = {c: i for i, c in enumerate(order)}
    missing = sorted(set(grouped.unique()) - set(order))
    if missing:
        raise ValueError(f"grouped classes missing from labels.class_order: {missing}")

    out = df.copy()
    out[COL_LABEL_RAW] = raw
    out[COL_LABEL] = grouped.astype(str)
    out[COL_Y] = grouped.map(code).astype(np.int16)
    return out


def _first_present(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def _apply_session_timeout(key: pd.Series, times: pd.Series,
                           cfg: Mapping[str, Any]) -> pd.Series:
    """Split each communication key into time-bounded sub-sessions.

    A new sub-session starts when the inter-flow gap within a key exceeds
    ``preprocessing.session_timeout_s``, or when the sub-session reaches
    ``preprocessing.session_max_flows`` flows. This is the standard NetFlow
    idle-timeout rule; without it a long-lived host pair forms one session
    spanning the entire capture, which breaks both the time-blocked split and
    the fixed-length LSTM windows.
    """
    timeout = float(cfg["preprocessing"].get("session_timeout_s", 0) or 0)
    max_flows = int(cfg["preprocessing"].get("session_max_flows", 0) or 0)
    if timeout <= 0 and max_flows <= 0:
        return key

    order = np.argsort(times.to_numpy(), kind="mergesort")
    k = key.to_numpy()[order]
    t = pd.to_datetime(pd.Series(times.to_numpy()[order])).astype("int64").to_numpy() / 1e9

    frame = pd.DataFrame({"k": k, "t": t})
    grp = frame.groupby("k", sort=False)
    gap_break = grp["t"].diff().to_numpy()
    new_sess = ~np.isfinite(gap_break)
    if timeout > 0:
        new_sess |= np.nan_to_num(gap_break, nan=np.inf) > timeout
    if max_flows > 0:
        # Position within the current (timeout-delimited) run of this key.
        pos = grp.cumcount().to_numpy()
        base = np.where(new_sess, pos, 0)
        base = pd.Series(base).groupby(pd.Series(k)).cummax().to_numpy()
        new_sess |= ((pos - base) >= max_flows)

    sub = pd.Series(new_sess.astype(np.int64)).groupby(pd.Series(k)).cumsum().to_numpy()
    out_sorted = np.array([f"{a}#{b}" for a, b in zip(k, sub)], dtype=object)

    out = np.empty(len(key), dtype=object)
    out[order] = out_sorted
    return pd.Series(out, index=key.index, name=COL_SESSION)


def build_session_ids(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    """Build the symmetric session key {IPmin, IPmax, Portmin, Portmax, Proto}.

    Equation (22): communication in opposite directions maps to the same
    session, so the key is order-invariant in (IP, port) pairs.

    When the release lacks IP columns (the ``MachineLearningCVE`` subset), a
    documented surrogate is used: consecutive flows sharing
    (source file, Destination Port, Protocol) form a session, capped at
    ``temporal.max_sequence_length`` flows. See README "Deviations".
    """
    src_ip = _first_present(df, ["Source IP", "Src IP"])
    dst_ip = _first_present(df, ["Destination IP", "Dst IP"])
    src_pt = _first_present(df, ["Source Port", "Src Port"])
    dst_pt = _first_present(df, ["Destination Port", "Dst Port"])
    proto = _first_present(df, [cfg["data"].get("protocol_column", "Protocol")])

    mode = cfg["preprocessing"].get("session_key_mode", "service_port")

    if src_ip and dst_ip:
        a = df[src_ip].astype(str).values
        b = df[dst_ip].astype(str).values
        ip_min = np.where(a <= b, a, b)
        ip_max = np.where(a <= b, b, a)
        if src_pt and dst_pt:
            pa = pd.to_numeric(df[src_pt], errors="coerce").fillna(-1).astype(np.int64).values
            pb = pd.to_numeric(df[dst_pt], errors="coerce").fillna(-1).astype(np.int64).values
            pt_min = np.minimum(pa, pb)
            pt_max = np.maximum(pa, pb)
        else:
            pt_min = pt_max = np.zeros(len(df), dtype=np.int64)
        pr = (pd.to_numeric(df[proto], errors="coerce").fillna(-1).astype(np.int64).values
              if proto else np.zeros(len(df), dtype=np.int64))

        if mode == "five_tuple":
            # Equation (22) verbatim.
            parts = [ip_min, ip_max, pt_min, pt_max, pr]
        elif mode == "host_pair":
            parts = [ip_min, ip_max]
        else:  # "service_port" (default)
            # Keep the service (lower) port, drop the ephemeral (higher) one.
            parts = [ip_min, ip_max, pt_min, pr]

        key = pd.Series(
            ["|".join(str(v) for v in row) for row in zip(*parts)],
            index=df.index, name=COL_SESSION,
        )
        key = _apply_session_timeout(key, df[COL_TIME], cfg)
        n_sess = key.nunique()
        LOGGER.info("Session key [%s] from eq.22 fields: %d sessions, %.2f flows/session",
                    mode, n_sess, len(df) / max(n_sess, 1))
        if n_sess > 0.9 * len(df) and mode == "five_tuple":
            LOGGER.warning(
                "Equation (22) yields ~1 flow per session on this data (ephemeral "
                "client ports). The LSTM/GNN/temporal stages will be degenerate; "
                "consider preprocessing.session_key_mode: service_port."
            )
        return key

    # --- surrogate session key (no identifier columns in this release) ---
    LOGGER.warning(
        "No IP columns present; using the documented surrogate session key "
        "(source file + Destination Port + Protocol, run-length grouped)."
    )
    parts = [df[COL_SRC_FILE].astype(str)]
    if dst_pt:
        parts.append(pd.to_numeric(df[dst_pt], errors="coerce").fillna(-1).astype(np.int64).astype(str))
    if proto:
        parts.append(pd.to_numeric(df[proto], errors="coerce").fillna(-1).astype(np.int64).astype(str))
    base = parts[0].str.cat(parts[1:], sep="|") if len(parts) > 1 else parts[0]

    max_len = int(cfg.get("temporal", {}).get("max_sequence_length", 50))
    # New session whenever the key changes OR the run exceeds max_len flows.
    changed = base.ne(base.shift()).to_numpy()
    run_id = np.cumsum(changed) - 1
    pos_in_run = np.arange(len(base)) - np.maximum.accumulate(
        np.where(changed, np.arange(len(base)), 0)
    )
    block = pos_in_run // max(1, max_len)
    key = pd.Series(
        [f"S{r}_{b}" for r, b in zip(run_id, block)], index=df.index, name=COL_SESSION
    )
    LOGGER.info("Surrogate sessions: %d", key.nunique())
    return key


# ===========================================================================
# Stage 2: cleaning (split-independent) and the train-fitted transformer
# ===========================================================================
@dataclass
class ColumnCleaner:
    """Split-independent cleaning (Stage 2.2 steps 1-3).

    These operations learn no parameters from the data, so the methodology
    permits running them before the split. Anything that *does* learn
    parameters lives in :class:`TrainFittedPreprocessor`.
    """

    cfg: Mapping[str, Any]
    feature_columns_: List[str] = field(default_factory=list)
    report_: Dict[str, Any] = field(default_factory=dict)

    def identifier_columns(self, df: pd.DataFrame) -> List[str]:
        """Columns excluded from X: identifiers, time, and (optionally) Dst Port."""
        data_cfg = self.cfg["data"]
        drop = [c for c in data_cfg["identifier_columns"] if c in df.columns]
        if not data_cfg.get("use_dst_port", False):
            drop += [c for c in data_cfg.get("dst_port_columns", []) if c in df.columns]
        drop += [c for c in META_COLUMNS if c in df.columns]
        drop += [c for c in [data_cfg["label_column"]] if c in df.columns]
        return sorted(set(drop))

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply duplicate removal, inf->NaN and the negative-value policy."""
        data_cfg = self.cfg["data"]
        rep: Dict[str, Any] = {"rows_in": int(len(df))}

        if data_cfg.get("drop_exact_duplicates", True):
            # Duplicates are judged on the real feature + label content, not on
            # the surrogate ordering columns we added.
            subset = [c for c in df.columns if c not in (COL_ORDER, COL_TIME, COL_SRC_FILE)]
            before = len(df)
            df = df.drop_duplicates(subset=subset, keep="first")
            rep["duplicates_removed"] = int(before - len(df))
            LOGGER.info("Stage 2: removed %d exact duplicate rows", rep["duplicates_removed"])

        df = df.reset_index(drop=True)
        df[COL_ORDER] = np.arange(len(df), dtype=np.int64)

        ident = self.identifier_columns(df)
        numeric = [
            c for c in df.columns
            if c not in ident and c not in META_COLUMNS
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        non_numeric = [
            c for c in df.columns
            if c not in ident and c not in META_COLUMNS and c not in numeric
        ]
        rep["non_numeric_feature_columns"] = non_numeric

        if data_cfg.get("inf_to_nan", True):
            n_inf = 0
            for col in numeric:
                mask = np.isinf(df[col].to_numpy(dtype=np.float64, na_value=np.nan))
                if mask.any():
                    n_inf += int(mask.sum())
                    df.loc[mask, col] = np.nan
            rep["inf_values_to_nan"] = n_inf
            LOGGER.info("Stage 2: converted %d +/-inf values to NaN", n_inf)

        # Stage 2.3: negatives are invalid measurements EXCEPT in the documented
        # sentinel features, where -1 means "not observed" and is preserved.
        if data_cfg.get("negatives_to_nan", True):
            sentinels = set(data_cfg.get("negative_sentinel_features", []))
            n_neg = 0
            for col in numeric:
                if col in sentinels:
                    continue
                vals = df[col].to_numpy(dtype=np.float64, na_value=np.nan)
                mask = vals < 0
                if mask.any():
                    n_neg += int(mask.sum())
                    df.loc[mask, col] = np.nan
            rep["negative_values_to_nan"] = n_neg
            rep["sentinel_features_preserved"] = sorted(sentinels & set(numeric))
            LOGGER.info("Stage 2: converted %d negative values to NaN "
                        "(%d sentinel feature(s) preserved)",
                        n_neg, len(rep["sentinel_features_preserved"]))

        self.feature_columns_ = numeric + [
            c for c in non_numeric if c == data_cfg.get("protocol_column")
        ]
        rep["n_feature_columns"] = len(self.feature_columns_)
        rep["identifier_columns_excluded"] = ident
        rep["use_dst_port"] = bool(data_cfg.get("use_dst_port", False))
        rep["rows_out"] = int(len(df))
        self.report_ = rep
        return df


@dataclass
class TrainFittedPreprocessor:
    """Median imputer + Min-Max scaler + one-hot encoder, fitted on TRAIN only.

    ``fit`` records the positional row indices it was given; the caller passes
    them to :func:`utils.assert_fitted_on_train_only` (and ``fit`` itself
    refuses a ``SplitIndex`` whose train set does not contain them).

    Stage 2.4 notes:
      * constant features (max == min) map to ``constant_feature_value``
        instead of dividing by zero;
      * values outside the training range are clipped to [0, 1] when
        ``minmax_out_of_range == 'clip'``, otherwise left outside the unit
        interval. The TEST minimum/maximum is never consulted either way.
    """

    cfg: Mapping[str, Any]
    numeric_columns_: List[str] = field(default_factory=list)
    categorical_columns_: List[str] = field(default_factory=list)
    output_columns_: List[str] = field(default_factory=list)
    medians_: Optional[np.ndarray] = None
    mins_: Optional[np.ndarray] = None
    ranges_: Optional[np.ndarray] = None
    constant_mask_: Optional[np.ndarray] = None
    categories_: Dict[str, List[Any]] = field(default_factory=dict)
    fitted_rows_: Optional[np.ndarray] = None
    is_fitted_: bool = False

    # -- fit ---------------------------------------------------------------
    def fit(self, df: pd.DataFrame, feature_columns: Sequence[str],
            rows: Optional[Sequence[int]] = None,
            split: Optional[SplitIndex] = None) -> "TrainFittedPreprocessor":
        """Fit on the training rows only.

        Parameters
        ----------
        df:
            The full cleaned frame (positional index == split indices).
        feature_columns:
            Candidate feature columns from :class:`ColumnCleaner`.
        rows:
            Positional indices to fit on. Must be a subset of ``split.train``.
        split:
            When given, the train-only membership is asserted here, so a
            mis-wired caller fails loudly instead of leaking silently.
        """
        rows = np.arange(len(df)) if rows is None else np.asarray(rows, dtype=np.int64)
        if split is not None:
            assert_fitted_on_train_only(rows, split, "TrainFittedPreprocessor")

        proto = self.cfg["data"].get("protocol_column")
        encode_proto = self.cfg["data"].get("protocol_encoding", "numeric") == "onehot"

        self.categorical_columns_ = [
            c for c in feature_columns if c == proto and encode_proto
        ]
        self.numeric_columns_ = [c for c in feature_columns if c not in self.categorical_columns_]

        sub = df.iloc[rows]
        X = sub[self.numeric_columns_].to_numpy(dtype=np.float64, na_value=np.nan)

        # --- median imputation (Stage 2.3), train statistics only ---
        with np.errstate(all="ignore"):
            med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)   # all-NaN column -> 0
        self.medians_ = med
        X = np.where(np.isnan(X), med[None, :], X)

        # --- Min-Max scaling (Stage 2.4), train statistics only ---
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        rng = maxs - mins
        self.constant_mask_ = rng <= 0
        self.mins_ = mins
        self.ranges_ = np.where(self.constant_mask_, 1.0, rng)

        # --- one-hot categories (Stage 2.5), train categories only ---
        self.categories_ = {}
        for col in self.categorical_columns_:
            cats = sorted(pd.unique(sub[col].dropna()))
            self.categories_[col] = list(cats)

        self.output_columns_ = list(self.numeric_columns_) + [
            f"{col}={cat}" for col in self.categorical_columns_
            for cat in self.categories_[col]
        ]
        self.fitted_rows_ = rows
        self.is_fitted_ = True
        LOGGER.info(
            "Stage 2: preprocessor fitted on %d TRAIN rows -> %d output features "
            "(%d numeric, %d one-hot, %d constant)",
            len(rows), len(self.output_columns_), len(self.numeric_columns_),
            len(self.output_columns_) - len(self.numeric_columns_),
            int(self.constant_mask_.sum()),
        )
        return self

    # -- transform ---------------------------------------------------------
    def transform(self, df: pd.DataFrame, rows: Optional[Sequence[int]] = None) -> pd.DataFrame:
        """Apply the frozen imputer/scaler/encoder. Learns nothing."""
        if not self.is_fitted_:
            raise RuntimeError("TrainFittedPreprocessor.transform called before fit")
        sub = df if rows is None else df.iloc[np.asarray(rows, dtype=np.int64)]

        X = sub[self.numeric_columns_].to_numpy(dtype=np.float64, na_value=np.nan)
        X = np.where(np.isnan(X), self.medians_[None, :], X)
        X = (X - self.mins_[None, :]) / self.ranges_[None, :]
        if self.constant_mask_ is not None and self.constant_mask_.any():
            X[:, self.constant_mask_] = float(
                self.cfg["preprocessing"].get("constant_feature_value", 0.0)
            )
        if self.cfg["preprocessing"].get("minmax_out_of_range", "clip") == "clip":
            np.clip(X, 0.0, 1.0, out=X)

        blocks = [pd.DataFrame(X, columns=self.numeric_columns_, index=sub.index)]
        for col in self.categorical_columns_:
            cats = self.categories_[col]
            vals = sub[col].to_numpy()
            # Unseen categories -> all zeros (Stage 2.5).
            oh = np.zeros((len(sub), len(cats)), dtype=np.float64)
            for j, cat in enumerate(cats):
                oh[:, j] = (vals == cat).astype(np.float64)
            blocks.append(pd.DataFrame(
                oh, columns=[f"{col}={c}" for c in cats], index=sub.index
            ))

        out = pd.concat(blocks, axis=1) if len(blocks) > 1 else blocks[0]
        out = out[self.output_columns_]
        dtype = self.cfg["data"].get("float_dtype", "float32")
        return out.astype(dtype)

    def fit_transform(self, df: pd.DataFrame, feature_columns: Sequence[str],
                      rows: Optional[Sequence[int]] = None,
                      split: Optional[SplitIndex] = None) -> pd.DataFrame:
        self.fit(df, feature_columns, rows=rows, split=split)
        return self.transform(df, rows=rows)


# ===========================================================================
# Stage 3: splits
# ===========================================================================
def _session_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per session: start/end time, flow count and primary class.

    The *primary class* of a session is its rarest class, so that a session
    containing a handful of attack flows is placed by the attack, not by the
    benign majority. This keeps rare classes present in every partition.
    """
    g = df.groupby(COL_SESSION, sort=False)
    tab = g.agg(
        start=(COL_TIME, "min"),
        end=(COL_TIME, "max"),
        n_flows=(COL_TIME, "size"),
    )
    global_counts = df[COL_LABEL].value_counts()
    rarity = {c: i for i, c in enumerate(global_counts.index)}  # 0 = most common

    def primary(labels: pd.Series) -> str:
        uniq = labels.unique()
        return max(uniq, key=lambda c: rarity.get(c, 0))

    tab["primary_class"] = g[COL_LABEL].agg(primary)
    return tab.reset_index()


def _cut_positions(n_total: int, train_frac: float, val_frac: float) -> Tuple[float, float]:
    return n_total * train_frac, n_total * (train_frac + val_frac)


def _assign_units_in_time(units: List[Tuple[Any, np.ndarray, Any, Any, int]],
                          train_frac: float, val_frac: float,
                          gap_td: "pd.Timedelta") -> Tuple[List[List[int]], List[int]]:
    """Cut time-ordered units into train/val/test blocks at time boundaries.

    Two boundary *times* are chosen as flow-count quantiles of the class's own
    timeline: ``T1`` at ``train_frac`` and ``T2`` at ``train_frac + val_frac``.
    A unit (a session, or a single flow) is then placed by where it lies
    relative to those boundaries:

      * ends at or before ``T1``                      -> train
      * starts after ``T1 + gap`` and ends by ``T2``  -> validation
      * starts after ``T2 + gap``                     -> test
      * otherwise it *crosses* a boundary             -> gap (discarded)

    Only units genuinely straddling a boundary are discarded -- a long session
    that ended well before ``T1`` does not disqualify later ones. Because a
    train unit ends by ``T1`` and a val unit starts after ``T1``, every
    validation flow is strictly later than every training flow, and likewise
    for validation vs. test: the strict Stage 3.2 ordering.

    Returns ``(blocks, gap_unit_indices)``, ``blocks`` holding three lists of
    positions into ``units``.
    """
    # Flow-weighted time quantiles over this class's own flows.
    starts = [u[2] for u in units]
    counts = np.array([u[4] for u in units], dtype=np.int64)
    order = np.argsort(np.asarray(starts), kind="mergesort")
    cum = np.cumsum(counts[order])
    total = int(cum[-1]) if len(cum) else 0
    if total == 0:
        return [[], [], []], []

    def boundary(frac: float) -> Any:
        pos = int(np.searchsorted(cum, max(1.0, total * frac)))
        pos = min(pos, len(units) - 1)
        return units[order[pos]][3]          # end time of the unit at that quantile

    t1 = boundary(train_frac)
    t2 = boundary(train_frac + val_frac)

    blocks: List[List[int]] = [[], [], []]
    gap: List[int] = []
    for i, (_sid, _rows, u_start, u_end, _n) in enumerate(units):
        if u_end <= t1:
            blocks[0].append(i)
        elif u_start > t1 + gap_td and u_end <= t2:
            blocks[1].append(i)
        elif u_start > t2 + gap_td:
            blocks[2].append(i)
        else:
            gap.append(i)
    return blocks, gap


def time_blocked_split(df: pd.DataFrame, cfg: Mapping[str, Any]) -> SplitIndex:
    """MAIN split (Stage 3.2): per class, 70/15/15 in time order.

    Per class (the class of the *session*, since sessions are kept whole):

      1. order that class's sessions by start time;
      2. cut at 70% / 85% of the cumulative flow count;
      3. a session joins a later block only if it starts after the previous
         block's last flow plus the gap -- straddlers are dropped (see
         :func:`_assign_units_in_time`).

    Sessions that overlap in time cannot all be assigned without straddling, so
    some flows are necessarily discarded into the gap; the count is logged and
    saved in the split report.

    When this leaves a class absent from a partition -- unavoidable for a class
    with fewer separable sessions than partitions (Heartbleed, Infiltration) --
    that class alone falls back to flow-level time blocking, and its sessions
    are recorded in ``meta['exempt_sessions']`` as excused from the
    session-wholeness rule. The two methodology requirements ("every class in
    all three partitions" and "sessions kept whole") are mutually unsatisfiable
    for such classes; this is the documented policy for choosing between them
    (``split.tiny_class_policy``).
    """
    sp = cfg["split"]
    train_frac = float(sp["train_frac"])
    val_frac = float(sp["val_frac"])
    gap_frac = float(sp.get("gap_frac", 0.0))
    tiny_min = int(sp.get("tiny_class_min_flows", 0))
    tiny_policy = str(sp.get("tiny_class_policy", "flow_level"))
    require_all = bool(sp.get("require_all_classes_in_all_splits", True))

    sessions = _session_table(df)
    sess_rows: Dict[Any, np.ndarray] = {
        s: np.asarray(idx) for s, idx in df.groupby(COL_SESSION, sort=False).groups.items()
    }

    train, val, test, gap_rows = [], [], [], []
    boundaries: Dict[str, Dict[str, Any]] = {}
    exempt_sessions: set = set()
    class_counts = df[COL_LABEL].value_counts().to_dict()
    assign_group = np.empty(len(df), dtype=object)
    assign_group[:] = ""

    labels_arr = df[COL_LABEL].to_numpy()
    times_arr = df[COL_TIME].to_numpy()

    def flow_units(rows: np.ndarray) -> List[Tuple[Any, np.ndarray, Any, Any, int]]:
        """Each flow as its own unit, ordered in time (always separable)."""
        order = np.argsort(times_arr[rows], kind="mergesort")
        rows = rows[order]
        ts = times_arr[rows]
        return [(f"flow:{r}", np.array([r]), t, t, 1) for r, t in zip(rows, ts)]

    def label_counts(block: List[int],
                     units: List[Tuple[Any, np.ndarray, Any, Any, int]], cls: str) -> int:
        """Flows actually *labelled* ``cls`` inside a block of units."""
        if not block:
            return 0
        rows = np.concatenate([units[i][1] for i in block])
        return int((labels_arr[rows] == cls).sum())

    for cls in sorted(class_counts):
        n_flows_cls = int(class_counts[cls])

        sub = sessions[sessions["primary_class"] == cls].sort_values(
            ["start", COL_SESSION], kind="mergesort"
        )
        session_units = [
            (r[COL_SESSION], sess_rows[r[COL_SESSION]], r["start"], r["end"], int(r["n_flows"]))
            for _, r in sub.iterrows()
        ]
        if not session_units:
            continue
        # Rows OWNED by this class: every row of every session whose primary
        # class is `cls`. Each row belongs to exactly one session and each
        # session to exactly one primary class, so iterating the classes
        # partitions the frame -- no row can be assigned twice.
        owned_rows = np.concatenate([u[1] for u in session_units])

        force_flow_level = (n_flows_cls < tiny_min and tiny_policy == "flow_level")
        units = flow_units(owned_rows) if force_flow_level else session_units

        span = units[-1][3] - units[0][2]
        gap_td = span * gap_frac if isinstance(span, pd.Timedelta) else pd.Timedelta(0)
        blocks, gap_idx = _assign_units_in_time(units, train_frac, val_frac, gap_td)

        # Fall back to flow-level blocking for this class if the session-level
        # assignment cannot populate every partition.
        used_flow_level = force_flow_level
        missing_now = [i for i in range(3) if label_counts(blocks[i], units, cls) == 0]
        if require_all and not used_flow_level and missing_now:
            missing = [["train", "val", "test"][i] for i in missing_now]
            units = flow_units(owned_rows)
            span = units[-1][3] - units[0][2]
            gap_td = span * gap_frac if isinstance(span, pd.Timedelta) else pd.Timedelta(0)
            blocks, gap_idx = _assign_units_in_time(units, train_frac, val_frac, gap_td)
            if any(label_counts(blocks[i], units, cls) == 0 for i in range(3)):
                # Still empty -> drop the gap entirely for this class.
                blocks, gap_idx = _assign_units_in_time(
                    units, train_frac, val_frac, pd.Timedelta(0))
            used_flow_level = True
            LOGGER.warning(
                "class %s: session-level time blocking left %s empty; falling back "
                "to flow-level blocking for this class (sessions exempted from the "
                "whole-session rule)", cls, "/".join(missing),
            )

        if used_flow_level:
            exempt_sessions.update(
                pd.unique(df[COL_SESSION].to_numpy()[owned_rows]).tolist())

        for b, bucket in zip(range(3), (train, val, test)):
            for i in blocks[b]:
                bucket.append(units[i][1])
                assign_group[units[i][1]] = cls
        for i in gap_idx:
            gap_rows.append(units[i][1])

        def _end(idxs: List[int]) -> Optional[Any]:
            return max((units[i][3] for i in idxs), default=None)

        still_missing = [["train", "val", "test"][i] for i in range(3)
                         if label_counts(blocks[i], units, cls) == 0]
        if require_all and still_missing:
            LOGGER.warning("class %s is absent from %s even after fallback "
                           "(only %d flow(s) of this class available)",
                           cls, still_missing, n_flows_cls)

        boundaries[cls] = {
            "n_flows": n_flows_cls,
            "n_units": len(units),
            "train_end": str(_end(blocks[0])),
            "val_end": str(_end(blocks[1])),
            "test_end": str(_end(blocks[2])),
            "gap": str(gap_td),
            "gap_units_dropped": len(gap_idx),
            "flow_level_policy": bool(used_flow_level),
            "absent_from": still_missing,
        }

    def _cat(parts: List[np.ndarray]) -> np.ndarray:
        return np.sort(np.concatenate(parts)).astype(np.int64) if parts else np.array([], dtype=np.int64)

    split = SplitIndex(
        train=_cat(train), val=_cat(val), test=_cat(test),
        strategy="time_blocked", boundaries=boundaries,
        dropped_gap=_cat(gap_rows),
        meta={"exempt_sessions": exempt_sessions,
              "tiny_class_policy": tiny_policy,
              "gap_frac": gap_frac,
              "assign_group": assign_group},
    )
    n_total = len(split.train) + len(split.val) + len(split.test) + len(split.dropped_gap)
    LOGGER.info("Stage 3 [time_blocked]: train=%d val=%d test=%d "
                "(gap dropped %d = %.1f%%)",
                len(split.train), len(split.val), len(split.test),
                len(split.dropped_gap), 100.0 * len(split.dropped_gap) / max(n_total, 1))
    return split


def stratified_split(df: pd.DataFrame, cfg: Mapping[str, Any], seed: int = 42) -> SplitIndex:
    """Session-grouped stratified random split (Stage 3.2 'Secondary')."""
    sp = cfg["split"]
    rng = np.random.default_rng(seed)
    sessions = _session_table(df)
    sess_rows = {s: idx.to_numpy() for s, idx in df.groupby(COL_SESSION, sort=False).groups.items()}

    train, val, test = [], [], []
    for cls, sub in sessions.groupby("primary_class", sort=True):
        order = rng.permutation(len(sub))
        sub = sub.iloc[order]
        total = int(sub["n_flows"].sum())
        cut1, cut2 = _cut_positions(total, sp["train_frac"], sp["val_frac"])
        cum = 0
        for _, r in sub.iterrows():
            bucket = train if cum < cut1 else (val if cum < cut2 else test)
            bucket.append(sess_rows[r[COL_SESSION]])
            cum += int(r["n_flows"])

    def _cat(parts):
        return np.sort(np.concatenate(parts)).astype(np.int64) if parts else np.array([], dtype=np.int64)

    split = SplitIndex(_cat(train), _cat(val), _cat(test), strategy="stratified_session")
    LOGGER.info("Stage 3 [stratified_session]: train=%d val=%d test=%d",
                len(split.train), len(split.val), len(split.test))
    return split


def stratified_flow_split(df: pd.DataFrame, cfg: Mapping[str, Any], seed: int = 42) -> SplitIndex:
    """Plain flow-level stratified split -- deliberately leaky.

    Sessions are torn across partitions, so near-duplicate flows of the same
    session appear in both train and test. Reported only to quantify the
    "leakage gap" against the time-blocked results (branch D).
    """
    from sklearn.model_selection import train_test_split

    sp = cfg["split"]
    y = df[COL_Y].to_numpy()
    idx = np.arange(len(df))
    # Classes with a single member cannot be stratified; handle them manually.
    counts = pd.Series(y).value_counts()
    rare = set(counts[counts < 3].index.tolist())
    rare_idx = idx[np.isin(y, list(rare))] if rare else np.array([], dtype=np.int64)
    main_idx = idx[~np.isin(y, list(rare))] if rare else idx

    tr, rest = train_test_split(
        main_idx, train_size=sp["train_frac"], random_state=seed,
        stratify=y[main_idx], shuffle=sp.get("stratified_flow_shuffle", True),
    )
    rel_val = sp["val_frac"] / (sp["val_frac"] + sp["test_frac"])
    va, te = train_test_split(
        rest, train_size=rel_val, random_state=seed, stratify=y[rest], shuffle=True,
    )
    if len(rare_idx):
        tr = np.concatenate([tr, rare_idx])

    split = SplitIndex(np.sort(tr), np.sort(va), np.sort(te), strategy="stratified_flow")
    LOGGER.info("Stage 3 [stratified_flow]: train=%d val=%d test=%d (LEAKY baseline)",
                len(split.train), len(split.val), len(split.test))
    return split


def leave_one_attack_out_split(df: pd.DataFrame, cfg: Mapping[str, Any],
                               held_out: str, seed: int = 42) -> SplitIndex:
    """Zero-day split (branch E): ``held_out`` family is absent from train/val.

    The held-out family's flows all go to test; every other class is split
    time-blocked as usual. Detection recall on the unseen family and the
    benign false-alarm rate are the reported metrics.
    """
    flow_mask = (df[COL_LABEL] == held_out).to_numpy()
    if not flow_mask.any():
        raise ValueError(f"attack family {held_out!r} is absent from the data")
    # Hold out whole SESSIONS, not individual flows: any session containing even
    # one flow of the unseen family goes entirely to test. Holding out flows
    # alone would tear mixed sessions across partitions (violating the
    # whole-session rule) and would let the model see the benign half of an
    # attack session during training, which is not a zero-day setting.
    held_sessions = set(pd.unique(df.loc[flow_mask, COL_SESSION]))
    mask_held = df[COL_SESSION].isin(held_sessions).to_numpy()
    # Positions in `df` that survive into train/val; `rest` is re-indexed 0..n-1
    # so that time_blocked_split's positional indices are well defined.
    back = np.where(~mask_held)[0]
    rest = df.loc[~mask_held].reset_index(drop=True)
    if rest.empty:
        raise ValueError(f"holding out {held_out!r} leaves no data")

    base = time_blocked_split(rest, cfg)
    train = back[base.train]
    val = back[base.val]
    test = np.concatenate([back[base.test], np.where(mask_held)[0]])

    assign_group = np.empty(len(df), dtype=object)
    assign_group[:] = held_out                       # held-out rows form their own group
    base_groups = base.meta.get("assign_group")
    if base_groups is not None:
        assign_group[back] = np.asarray(base_groups, dtype=object)

    split = SplitIndex(
        np.sort(train), np.sort(val), np.sort(test), strategy="loao",
        boundaries=base.boundaries,
        meta={"held_out_family": held_out,
              "exempt_sessions": base.meta.get("exempt_sessions", set()),
              "assign_group": assign_group,
              "n_held_out_flows": int(flow_mask.sum()),
              "n_held_out_sessions": len(held_sessions),
              "n_held_out_rows": int(mask_held.sum())},
    )
    LOGGER.info("Stage 3 [loao:%s]: train=%d val=%d test=%d "
                "(%d flows of the unseen family in %d held-out session(s), "
                "%d carried-along flows)",
                held_out, len(split.train), len(split.val), len(split.test),
                int(flow_mask.sum()), len(held_sessions),
                int(mask_held.sum() - flow_mask.sum()))
    return split


def make_split(df: pd.DataFrame, cfg: Mapping[str, Any], seed: int = 42,
               strategy: Optional[str] = None, held_out: Optional[str] = None) -> SplitIndex:
    """Dispatch to the split named by ``split.strategy`` (or the override)."""
    strategy = strategy or cfg["split"]["strategy"]
    if strategy == "time_blocked":
        return time_blocked_split(df, cfg)
    if strategy == "stratified_session":
        return stratified_split(df, cfg, seed)
    if strategy == "stratified_flow":
        return stratified_flow_split(df, cfg, seed)
    if strategy == "loao":
        if not held_out:
            raise ValueError("strategy 'loao' requires held_out=<attack family>")
        return leave_one_attack_out_split(df, cfg, held_out, seed)
    raise ValueError(f"unknown split strategy {strategy!r}")


# ===========================================================================
# Reporting
# ===========================================================================
def class_distribution_table(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Stage 1.3 style table: raw label -> count -> grouped class."""
    tab = (
        df.groupby([COL_LABEL_RAW, COL_LABEL], sort=False)
        .size().reset_index(name="records")
        .sort_values("records", ascending=False)
    )
    tab.columns = ["raw_label", "grouped_class", "records"]
    tab["pct"] = (tab["records"] / len(df) * 100).round(4)
    return tab.reset_index(drop=True)


def split_distribution_table(df: pd.DataFrame, split: SplitIndex,
                             cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Per-split class counts (Stage 3: 'Print and save the per-split counts')."""
    rows = []
    for part, idx in split.as_dict().items():
        counts = df.iloc[idx][COL_LABEL].value_counts()
        for cls in cfg["labels"]["class_order"]:
            rows.append({"split": part, "class": cls, "count": int(counts.get(cls, 0))})
    tab = pd.DataFrame(rows)
    wide = tab.pivot(index="class", columns="split", values="count").fillna(0).astype(int)
    wide = wide.reindex(cfg["labels"]["class_order"]).fillna(0).astype(int)
    for col in ("train", "val", "test"):
        if col not in wide.columns:
            wide[col] = 0
    wide["total"] = wide[["train", "val", "test"]].sum(axis=1)
    return wide.reset_index()


def subsample_for_smoke(df: pd.DataFrame, cfg: Mapping[str, Any], seed: int = 42) -> pd.DataFrame:
    """Stratified subsample for smoke-test mode, keeping rare classes intact."""
    smoke = cfg.get("data", {})
    frac = float(smoke.get("subsample_frac", 0) or 0)
    if frac <= 0 or frac >= 1:
        return df
    min_rows = int(smoke.get("min_rows_per_class", 40))
    rng = np.random.default_rng(seed)
    keep: List[np.ndarray] = []
    # Sample whole sessions so session integrity survives the subsample.
    sessions = _session_table(df)
    sess_rows = {s: idx.to_numpy() for s, idx in df.groupby(COL_SESSION, sort=False).groups.items()}
    for cls, sub in sessions.groupby("primary_class", sort=True):
        total = int(sub["n_flows"].sum())
        target = max(min_rows, int(total * frac))
        order = rng.permutation(len(sub))
        sub = sub.iloc[order].sort_values("start", kind="mergesort")
        cum, picked = 0, []
        for _, r in sub.iterrows():
            picked.append(sess_rows[r[COL_SESSION]])
            cum += int(r["n_flows"])
            if cum >= target:
                break
        keep.extend(picked)
    idx = np.sort(np.concatenate(keep))
    out = df.iloc[idx].reset_index(drop=True)
    out[COL_ORDER] = np.arange(len(out), dtype=np.int64)
    LOGGER.info("Smoke test: subsampled %d -> %d rows (frac=%.3f)", len(df), len(out), frac)
    return out


def prepare_dataset(cfg: Mapping[str, Any], data_dir: Optional[str] = None,
                    seed: int = 42, out_dir: Optional[str] = None,
                    use_cache: bool = True) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Run Stages 1-2 end to end and return ``(clean_df, feature_columns, report)``.

    The cleaned frame is cached as parquet under ``run.interim_dir`` so reruns
    skip the (slow) CSV merge.
    """
    out_dir = out_dir or cfg["run"]["output_dir"]
    interim = ensure_dir(cfg["run"]["interim_dir"])
    # The signature carries mode (full/smoke+frac), the data folder name and the
    # Dst-Port flag, so a smoke cache over the fixture and a full cache over the
    # real CSVs are different files and can never be confused for each other.
    sig = data_signature({**cfg, "data": {**cfg["data"], "data_dir": data_dir or cfg["data"]["data_dir"]}})
    cache_file = interim / f"clean__{sig}.parquet"
    meta_file = interim / f"clean__{sig}.meta.json"

    if use_cache and cache_file.exists() and meta_file.exists():
        LOGGER.info("Stage 1-2: loading cached clean frame from %s", cache_file)
        df = pd.read_parquet(cache_file)
        import json
        meta = json.loads(meta_file.read_text())
        return df, meta["feature_columns"], meta["report"]

    LOGGER.info("Stage 1-2: cache signature %r -> %s", sig, cache_file.name)
    raw = load_raw_dataset(cfg, data_dir)

    # Fail loudly rather than quietly training on a fixture. CIC-IDS2017 is
    # ~2.83M rows; data.expect_min_rows (null disables) catches the case where
    # --data_dir still points at a small synthetic folder.
    expect = cfg["data"].get("expect_min_rows")
    if expect and not cfg["run"].get("smoke_test") and len(raw) < int(expect):
        raise ValueError(
            f"only {len(raw):,} rows were read from {data_dir or cfg['data']['data_dir']!r}, "
            f"but data.expect_min_rows is {int(expect):,}. This looks like a "
            f"synthetic/test folder rather than the real CIC-IDS2017 CSVs. Point "
            f"--data_dir at the real folder, or lower/clear data.expect_min_rows."
        )
    n_before_labels = len(raw)
    raw = group_labels(raw, cfg)
    n_unlabeled_dropped = n_before_labels - len(raw)
    raw[COL_SESSION] = build_session_ids(raw, cfg)

    cleaner = ColumnCleaner(cfg)
    df = cleaner.run(raw)
    del raw

    if cfg["run"].get("smoke_test"):
        df = subsample_for_smoke(df, cfg, seed)

    feature_columns = [c for c in cleaner.feature_columns_ if c in df.columns]
    report = dict(cleaner.report_)
    report["unlabeled_rows_dropped"] = int(n_unlabeled_dropped)
    report["n_sessions"] = int(df[COL_SESSION].nunique())
    report["time_min"] = str(df[COL_TIME].min())
    report["time_max"] = str(df[COL_TIME].max())
    report["classes"] = df[COL_LABEL].value_counts().to_dict()

    df.to_parquet(cache_file, index=False)
    save_json({"feature_columns": feature_columns, "report": report}, meta_file)

    dist = class_distribution_table(df, cfg)
    save_table(dist, Path(out_dir) / "class_distribution.csv")
    save_json(report, Path(out_dir) / "preprocessing_report.json")
    LOGGER.info("Stage 1-2 complete: %d rows, %d candidate features, %d sessions",
                len(df), len(feature_columns), report["n_sessions"])
    return df, feature_columns, report
