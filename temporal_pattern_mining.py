"""Stage 9 (extension): sequential pattern mining and the attack-state graph.

The methodology names this stage and the quantities it must produce -- ``SP_t``
(sequential-pattern support), ``TC_t`` (temporal transition consistency) and
``K_i(t)`` (the attack-state-graph lookup used by the Stage 7 reliability term)
-- but does not specify how to compute them. The reading implemented here is
documented in README "Deviations / Assumptions"; every choice is configurable.

  * an **event sequence** per session (or host), ordered in time, where an
    event is a discretised flow state, the true label, or a predicted class
    (``temporal.event_definition``);
  * **FP-Growth** (itemsets) and **PrefixSpan** (ordered subsequences) mined on
    TRAIN sequences only; both have a bundled implementation so the pipeline
    runs without the optional ``mlxtend`` / ``prefixspan`` packages;
  * ``SP_t`` = normalised support of the mined patterns matching session ``t``;
  * an **attack-state graph** ``G_T`` whose nodes are behavioural states and
    whose edge weights are transition probabilities estimated on TRAIN;
  * ``TC_t`` = mean transition probability along the session's own path, and
    ``K_i(t)`` = P(state predicted by model i | previous state).

Everything is estimated on the TRAIN partition and then applied unchanged.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils import ensure_dir, get_logger, progress, save_json, save_table

LOGGER = get_logger()


# ===========================================================================
# Event sequences
# ===========================================================================
@dataclass
class FlowStateDiscretiser:
    """Map a flow to a categorical behavioural state.

    Bin edges are quantiles of duration, byte volume and packet count, fitted on
    TRAIN only. The resulting state string (e.g. ``"D0|B2|P1"``) is the event
    alphabet when ``temporal.event_definition == 'flow_state'``, which keeps the
    sequence model independent of the labels.
    """

    cfg: Mapping[str, Any]
    edges_: Dict[str, np.ndarray] = field(default_factory=dict)
    columns_: Dict[str, Optional[str]] = field(default_factory=dict)

    _CANDIDATES = {
        "D": ["Flow Duration", "Flow IAT Mean"],
        "B": ["Flow Bytes/s", "Total Length of Fwd Packets"],
        "P": ["Flow Packets/s", "Total Fwd Packets"],
    }

    def fit(self, df: pd.DataFrame) -> "FlowStateDiscretiser":
        p = self.cfg["temporal"]["flow_state"]
        n_bins = {"D": int(p.get("n_duration_bins", 3)),
                  "B": int(p.get("n_bytes_bins", 3)),
                  "P": int(p.get("n_packet_bins", 3))}
        for key, candidates in self._CANDIDATES.items():
            col = next((c for c in candidates if c in df.columns), None)
            self.columns_[key] = col
            if col is None:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            qs = np.linspace(0, 1, n_bins[key] + 1)[1:-1]
            self.edges_[key] = np.unique(np.quantile(vals, qs)) if len(vals) else np.array([])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts: List[np.ndarray] = []
        for key in self._CANDIDATES:
            col = self.columns_.get(key)
            if col is None or col not in df.columns:
                parts.append(np.zeros(len(df), dtype=np.int64))
                continue
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(np.searchsorted(self.edges_.get(key, np.array([])), vals))
        return np.array([f"D{a}|B{b}|P{c}" for a, b, c in zip(*parts)], dtype=object)


def build_event_sequences(df: pd.DataFrame, rows: np.ndarray, cfg: Mapping[str, Any],
                          events: Optional[np.ndarray] = None,
                          unit_column: Optional[str] = None,
                          time_column: str = "__timestamp",
                          ) -> Tuple[List[List[str]], List[Any]]:
    """Ordered event sequence per session (or host).

    Returns ``(sequences, unit_ids)`` with the sequences truncated to
    ``temporal.max_sequence_length`` most recent events.
    """
    t = cfg["temporal"]
    unit_column = unit_column or ("session_id" if t.get("sequence_unit", "session") == "session"
                                  else "__host")
    sub = df.iloc[rows]
    if unit_column not in sub.columns:
        raise KeyError(f"sequence unit column {unit_column!r} not in the frame")

    ev = events if events is not None else sub[unit_column].astype(str).to_numpy()
    order = np.argsort(sub[time_column].to_numpy(), kind="mergesort")
    units = sub[unit_column].to_numpy()[order]
    ev = np.asarray(ev)[order]

    max_len = int(t.get("max_sequence_length", 50))
    buckets: Dict[Any, List[str]] = defaultdict(list)
    for u, e in zip(units, ev):
        buckets[u].append(str(e))
    ids = list(buckets)
    seqs = [buckets[u][-max_len:] for u in ids]
    return seqs, ids


# ===========================================================================
# FP-Growth
# ===========================================================================
def _fpgrowth_bundled(sequences: Sequence[Sequence[str]], min_support: float,
                      max_len: int) -> pd.DataFrame:
    """Frequent itemsets by breadth-first Apriori-style counting.

    Equivalent output to FP-Growth (the same frequent itemsets with the same
    supports); used when ``mlxtend`` is unavailable. Transaction sets are small
    (the event alphabet is a few dozen symbols), so the simpler enumeration is
    adequate.
    """
    n = len(sequences)
    if n == 0:
        return pd.DataFrame(columns=["support", "itemsets"])
    transactions = [frozenset(s) for s in sequences]
    min_count = max(1, int(np.ceil(min_support * n)))

    counts = Counter()
    for tr in transactions:
        counts.update(tr)
    current = {frozenset([item]): c for item, c in counts.items() if c >= min_count}
    all_sets = dict(current)

    k = 1
    while current and k < max_len:
        candidates: Dict[frozenset, int] = {}
        items = sorted({i for s in current for i in s})
        for s in current:
            for item in items:
                if item in s:
                    continue
                cand = s | {item}
                if cand in candidates or cand in all_sets:
                    continue
                c = sum(1 for tr in transactions if cand <= tr)
                if c >= min_count:
                    candidates[cand] = c
        all_sets.update(candidates)
        current = candidates
        k += 1

    rows = [{"support": c / n, "itemsets": frozenset(s)} for s, c in all_sets.items()]
    return pd.DataFrame(rows).sort_values("support", ascending=False).reset_index(drop=True)


def mine_fp_growth(sequences: Sequence[Sequence[str]],
                   cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Frequent itemsets over the TRAIN event sequences."""
    p = cfg["temporal"]["fp_growth"]
    min_support = float(p.get("min_support", 0.02))
    max_len = int(p.get("max_len", 4))

    if p.get("use_mlxtend", True):
        try:
            from mlxtend.frequent_patterns import fpgrowth
            from mlxtend.preprocessing import TransactionEncoder

            te = TransactionEncoder()
            arr = te.fit_transform([list(set(s)) for s in sequences])
            frame = pd.DataFrame(arr, columns=te.columns_)
            out = fpgrowth(frame, min_support=min_support, use_colnames=True,
                           max_len=max_len)
            if len(out):
                out = out.sort_values("support", ascending=False).reset_index(drop=True)
            LOGGER.info("Stage 9: FP-Growth (mlxtend) found %d itemsets "
                        "at min_support=%.3f", len(out), min_support)
            return out
        except Exception as exc:
            LOGGER.warning("Stage 9: mlxtend FP-Growth unavailable (%s); "
                           "using the bundled implementation", exc)

    out = _fpgrowth_bundled(sequences, min_support, max_len)
    LOGGER.info("Stage 9: FP-Growth (bundled) found %d itemsets at min_support=%.3f",
                len(out), min_support)
    return out


# ===========================================================================
# PrefixSpan
# ===========================================================================
def _prefixspan_bundled(sequences: Sequence[Sequence[str]], min_count: int,
                        max_len: int, top_k: int) -> List[Tuple[int, List[str]]]:
    """Textbook PrefixSpan: recursive projected-database pattern growth.

    Mines *ordered* subsequences (not necessarily contiguous), which is what
    distinguishes this from FP-Growth's unordered itemsets.
    """
    results: List[Tuple[int, List[str]]] = []

    def project(db: List[Tuple[int, int]], prefix: List[str]) -> None:
        if len(prefix) >= max_len:
            return
        # Count each item's first occurrence after the current position.
        counts: Dict[str, int] = defaultdict(int)
        for sid, pos in db:
            for item in set(sequences[sid][pos:]):
                counts[item] += 1
        for item, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if c < min_count:
                continue
            new_prefix = prefix + [item]
            results.append((c, new_prefix))
            if len(results) >= top_k * 10:      # bound the search
                return
            new_db: List[Tuple[int, int]] = []
            for sid, pos in db:
                seq = sequences[sid]
                for j in range(pos, len(seq)):
                    if seq[j] == item:
                        new_db.append((sid, j + 1))
                        break
            if new_db:
                project(new_db, new_prefix)

    project([(i, 0) for i in range(len(sequences))], [])
    results.sort(key=lambda kv: (-kv[0], len(kv[1])))
    return results[:top_k]


def mine_prefixspan(sequences: Sequence[Sequence[str]],
                    cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Frequent ordered subsequences over the TRAIN event sequences."""
    p = cfg["temporal"]["prefixspan"]
    n = len(sequences)
    min_frac = float(p.get("min_support_frac", 0.02))
    min_count = max(1, int(np.ceil(min_frac * n)))
    max_len = int(p.get("max_pattern_len", 5))
    top_k = int(p.get("top_k_patterns", 200))

    patterns: List[Tuple[int, List[str]]] = []
    if p.get("use_package", True):
        try:
            from prefixspan import PrefixSpan

            ps = PrefixSpan(list(sequences))
            ps.maxlen = max_len
            patterns = [(c, list(pat)) for c, pat in ps.frequent(min_count)][:top_k]
            LOGGER.info("Stage 9: PrefixSpan (package) found %d patterns", len(patterns))
        except Exception as exc:
            LOGGER.warning("Stage 9: prefixspan package unavailable (%s); "
                           "using the bundled implementation", exc)
            patterns = []

    if not patterns:
        patterns = _prefixspan_bundled(list(sequences), min_count, max_len, top_k)
        LOGGER.info("Stage 9: PrefixSpan (bundled) found %d patterns", len(patterns))

    rows = [{"pattern": " -> ".join(pat), "pattern_items": list(pat),
             "length": len(pat), "count": c, "support": c / max(n, 1)}
            for c, pat in patterns]
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("support", ascending=False).reset_index(drop=True)
    return out


# ===========================================================================
# SP_t -- sequential-pattern support
# ===========================================================================
def _is_subsequence(pattern: Sequence[str], sequence: Sequence[str]) -> bool:
    """True when ``pattern`` appears in ``sequence`` in order (gaps allowed)."""
    it = iter(sequence)
    return all(any(p == s for s in it) for p in pattern)


def compute_sp(sequences: Sequence[Sequence[str]], patterns: pd.DataFrame,
               cfg: Mapping[str, Any]) -> np.ndarray:
    """``SP_t`` in [0, 1]: normalised support of the patterns matching session t.

    A session's score is the maximum support among the mined TRAIN patterns it
    contains, divided by the largest train support (``sp_normalisation: max``),
    so a session matching the most common attack pattern scores near 1 and a
    session matching nothing scores 0.
    """
    if patterns is None or patterns.empty:
        return np.zeros(len(sequences))
    pats = list(patterns["pattern_items"]) if "pattern_items" in patterns.columns else []
    sups = patterns["support"].to_numpy(dtype=np.float64)
    if not pats:
        return np.zeros(len(sequences))

    denom = float(sups.max()) if cfg["temporal"].get("sp_normalisation", "max") == "max" else 1.0
    denom = denom if denom > 0 else 1.0

    out = np.zeros(len(sequences), dtype=np.float64)
    for i, seq in enumerate(progress(sequences, desc="SP_t", total=len(sequences))):
        best = 0.0
        for pat, sup in zip(pats, sups):
            if sup <= best:
                continue                      # cannot improve the running max
            if _is_subsequence(pat, seq):
                best = sup
        out[i] = min(1.0, best / denom)
    return out


# ===========================================================================
# Attack-state graph, TC_t and K_i(t)
# ===========================================================================
@dataclass
class AttackStateGraph:
    """Transition probabilities between behavioural states, fitted on TRAIN.

    ``P[a][b]`` is the Laplace-smoothed probability of moving to state ``b``
    given state ``a``. The smoothing keeps ``K_i(t)`` strictly positive, so an
    unseen transition lowers a model's reliability without zeroing its weight.
    """

    states: List[str]
    matrix: np.ndarray
    counts: np.ndarray
    alpha: float = 0.1

    @classmethod
    def fit(cls, sequences: Sequence[Sequence[str]], cfg: Mapping[str, Any],
            states: Optional[Sequence[str]] = None) -> "AttackStateGraph":
        p = cfg["temporal"]["attack_state_graph"]
        alpha = float(p.get("laplace_alpha", 0.1))
        observed = sorted({e for s in sequences for e in s})
        states = list(states if states is not None else p.get("states") or observed)
        for e in observed:                     # keep states the data actually shows
            if e not in states:
                states.append(e)
        index = {s: i for i, s in enumerate(states)}

        counts = np.zeros((len(states), len(states)), dtype=np.float64)
        for seq in sequences:
            for a, b in zip(seq, seq[1:]):
                if a in index and b in index:
                    if a == b and not p.get("self_loops", True):
                        continue
                    counts[index[a], index[b]] += 1

        smoothed = counts + alpha
        matrix = smoothed / smoothed.sum(axis=1, keepdims=True)
        LOGGER.info("Stage 9: attack-state graph over %d states, %d transitions observed",
                    len(states), int(counts.sum()))
        return cls(states=states, matrix=matrix, counts=counts, alpha=alpha)

    @property
    def index(self) -> Dict[str, int]:
        return {s: i for i, s in enumerate(self.states)}

    def transition(self, prev_state: Any, next_state: Any) -> float:
        """``K`` lookup: P(next | prev); 1/|S| when either state is unknown."""
        idx = self.index
        if prev_state not in idx or next_state not in idx:
            return 1.0 / max(len(self.states), 1)
        return float(self.matrix[idx[prev_state], idx[next_state]])

    def consistency(self, sequence: Sequence[Any]) -> float:
        """``TC_t`` in [0, 1]: mean transition probability along the session path.

        Each step is scaled by the number of states, so a transition that is
        merely as likely as chance scores 1/|S| * |S| = 1 before clipping, and a
        strongly expected transition approaches 1. A session with fewer than two
        events has no transition and the caller marks it unavailable
        (``a(t) = 0`` in Stage 7.3).
        """
        if len(sequence) < 2:
            return 0.0
        probs = [self.transition(a, b) for a, b in zip(sequence, sequence[1:])]
        if not probs:
            return 0.0
        return float(np.clip(np.mean(probs) * len(self.states), 0.0, 1.0))

    def to_networkx(self, min_weight: float = 0.0):
        import networkx as nx

        g = nx.DiGraph()
        for s in self.states:
            g.add_node(s)
        for i, a in enumerate(self.states):
            for j, b in enumerate(self.states):
                w = float(self.matrix[i, j])
                if w > min_weight and self.counts[i, j] > 0:
                    g.add_edge(a, b, weight=round(w, 5), count=int(self.counts[i, j]))
        return g

    def to_dict(self) -> Dict[str, Any]:
        return {"states": self.states, "alpha": self.alpha,
                "transition_matrix": self.matrix.tolist(),
                "counts": self.counts.tolist()}


def compute_tc(sequences: Sequence[Sequence[str]], graph: AttackStateGraph,
               cfg: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """``(TC_t, availability a(t))`` per session (Stage 7.3)."""
    min_events = int(cfg["fusion"].get("min_events_for_temporal", 2))
    tc = np.array([graph.consistency(s) for s in sequences], dtype=np.float64)
    avail = np.array([1.0 if len(s) >= min_events else 0.0 for s in sequences])
    return tc, avail


# ===========================================================================
# Orchestration
# ===========================================================================
@dataclass
class TemporalEvidence:
    """Stage 9 output, consumed by the Stage 7 fusion."""

    graph: AttackStateGraph
    fp_itemsets: pd.DataFrame
    sequences_patterns: pd.DataFrame
    discretiser: Optional[FlowStateDiscretiser] = None
    train_support_max: float = 0.0

    def evidence_for(self, df: pd.DataFrame, rows: np.ndarray, cfg: Mapping[str, Any],
                     events: Optional[np.ndarray] = None
                     ) -> Dict[str, np.ndarray]:
        """Per-flow SP_t, TC_t and availability for the given rows.

        Session-level values are broadcast to each flow of the session, so the
        arrays align with the models' per-flow probability matrices.
        """
        if events is None and self.discretiser is not None:
            events = self.discretiser.transform(df.iloc[rows])
        seqs, ids = build_event_sequences(df, rows, cfg, events=events)
        sp = compute_sp(seqs, self.sequences_patterns, cfg)
        tc, avail = compute_tc(seqs, self.graph, cfg)

        unit_col = ("session_id" if cfg["temporal"].get("sequence_unit", "session") == "session"
                    else "__host")
        pos = {u: i for i, u in enumerate(ids)}
        units = df.iloc[rows][unit_col].to_numpy()
        take = np.array([pos.get(u, -1) for u in units])
        out_sp = np.where(take >= 0, sp[np.clip(take, 0, None)], 0.0)
        out_tc = np.where(take >= 0, tc[np.clip(take, 0, None)], 0.0)
        out_av = np.where(take >= 0, avail[np.clip(take, 0, None)], 0.0)
        return {"sp": out_sp, "tc": out_tc, "availability": out_av,
                "sequences": seqs, "unit_ids": ids}


def run_temporal_mining(df: pd.DataFrame, train_rows: np.ndarray,
                        cfg: Mapping[str, Any], out_dir: Optional[str] = None,
                        class_names: Optional[Mapping[int, str]] = None,
                        tag: str = "") -> TemporalEvidence:
    """Fit the Stage 9 extension on TRAIN rows only and save its artifacts."""
    t = cfg["temporal"]
    out_dir = Path(out_dir or cfg["run"]["output_dir"])
    ensure_dir(out_dir)
    suffix = f"_{tag}" if tag else ""

    discretiser = None
    event_def = str(t.get("event_definition", "flow_state"))
    if event_def == "flow_state":
        discretiser = FlowStateDiscretiser(cfg).fit(df.iloc[train_rows])
        events = discretiser.transform(df.iloc[train_rows])
        states = None
    elif event_def == "true_label":
        events = df.iloc[train_rows]["Label_grouped"].to_numpy()
        states = t["attack_state_graph"].get("states")
    else:
        raise ValueError(f"temporal.event_definition {event_def!r} must be "
                         f"'flow_state' or 'true_label' at fit time "
                         f"('predicted_class' is applied at inference)")

    seqs, _ids = build_event_sequences(df, train_rows, cfg, events=events)
    LOGGER.info("Stage 9: %d TRAIN sequences, mean length %.2f, alphabet %d",
                len(seqs), float(np.mean([len(s) for s in seqs])) if seqs else 0.0,
                len({e for s in seqs for e in s}))

    fp = mine_fp_growth(seqs, cfg)
    ps = mine_prefixspan(seqs, cfg)
    graph = AttackStateGraph.fit(seqs, cfg, states=states)

    evidence = TemporalEvidence(
        graph=graph, fp_itemsets=fp, sequences_patterns=ps, discretiser=discretiser,
        train_support_max=float(ps["support"].max()) if len(ps) else 0.0,
    )

    if t.get("save_patterns", True):
        if len(fp):
            fp_out = fp.copy()
            fp_out["itemsets"] = fp_out["itemsets"].apply(lambda s: " & ".join(sorted(s)))
            save_table(fp_out, out_dir / f"fp_growth_itemsets{suffix}.csv")
        if len(ps):
            save_table(ps.drop(columns=["pattern_items"]),
                       out_dir / f"prefixspan_patterns{suffix}.csv")
        save_json(graph.to_dict(), out_dir / f"attack_state_graph{suffix}.json")
        try:
            import networkx as nx

            nx.write_graphml(graph.to_networkx(), out_dir / f"attack_state_graph{suffix}.graphml")
        except Exception as exc:
            LOGGER.warning("Stage 9: could not write GraphML (%s)", exc)
    return evidence
