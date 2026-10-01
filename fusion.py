"""Stage 7: reliability-aware adaptive evidence fusion with dynamic weighting.

Pipeline (Stage 7.3):

  Step 1  collect the evidence on VALIDATION;
  Step 2  baseline reliability ``Q_i`` = validation macro-F1 per model, and
          validation ROC-AUC for SP/TC used alone as an attack score;
  Step 3  context reliability
            ``q_i(t) = l1*Q_i + l2*C_i(t) + l3*K_i(t)``
          with ``C_i(t) = 1 - H(P_i(t)) / log C`` (normalised entropy) and
          ``K_i(t)`` the attack-state-graph probability of the state model i
          predicts given the previous state. For SP/TC,
          ``q_S(t) = Q_S * a_S(t)`` with availability ``a(t) = 0`` when the
          session has fewer than two events;
  Step 4  ``w_i(t) = q_i(t) / sum_j q_j(t)`` (eq. 26);
  Step 5  ``R_t = sum_i w_i P_i + w_S SP + w_T TC`` (eq. 24) and
          ``P_fused(c|x) = sum_i w_i(x) P_i(c|x)``, ``y = argmax`` (eq. 27);
          the alert threshold ``tau_R`` is chosen on validation.

Probabilities are calibrated first (isotonic/Platt for the tree and LSTM models,
temperature scaling for CNN and GNN) so the weights compare like with like.
Every calibrator, lambda, weight and threshold is fitted on the VALIDATION
partition only -- never on test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils import (
    assert_probabilities_valid,
    assert_weights_sum_to_one,
    ensure_dir,
    get_logger,
    save_json,
    save_table,
)

LOGGER = get_logger()

EPS = 1e-12


# ===========================================================================
# Calibration (Stage 7.1)
# ===========================================================================
class TemperatureScaler:
    """Single-parameter temperature scaling for the CNN and GNN.

    Fits ``T`` minimising the validation NLL of ``softmax(log p / T)``. ``T > 1``
    softens an over-confident network; ``T < 1`` sharpens an under-confident one.
    """

    def __init__(self, max_iter: int = 200):
        self.max_iter = int(max_iter)
        self.temperature_ = 1.0

    def fit(self, proba: np.ndarray, y_idx: np.ndarray) -> "TemperatureScaler":
        logits = np.log(np.clip(proba, EPS, 1.0))
        grid = np.geomspace(0.25, 8.0, self.max_iter)
        best_t, best_nll = 1.0, np.inf
        for t in grid:
            scaled = logits / t
            scaled -= scaled.max(axis=1, keepdims=True)
            probs = np.exp(scaled)
            probs /= probs.sum(axis=1, keepdims=True)
            nll = -np.mean(np.log(np.clip(probs[np.arange(len(y_idx)), y_idx], EPS, 1.0)))
            if nll < best_nll:
                best_nll, best_t = nll, float(t)
        self.temperature_ = best_t
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(proba, EPS, 1.0)) / self.temperature_
        logits -= logits.max(axis=1, keepdims=True)
        out = np.exp(logits)
        return out / out.sum(axis=1, keepdims=True)


class PerClassCalibrator:
    """One-vs-rest isotonic or Platt calibration, renormalised to a simplex.

    sklearn's ``CalibratedClassifierCV`` refits the estimator; here the models
    are already trained and we only have their probability matrices, so the
    calibration is applied directly to each class column and the rows are
    renormalised.
    """

    def __init__(self, method: str = "isotonic", out_of_bounds: str = "clip"):
        self.method = method
        self.out_of_bounds = out_of_bounds
        self.models_: List[Any] = []

    def fit(self, proba: np.ndarray, y_idx: np.ndarray) -> "PerClassCalibrator":
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        self.models_ = []
        for j in range(proba.shape[1]):
            target = (y_idx == j).astype(int)
            col = proba[:, j]
            if target.sum() == 0 or target.sum() == len(target) or np.ptp(col) < 1e-9:
                self.models_.append(None)       # not calibratable -> identity
                continue
            if self.method == "isotonic":
                m = IsotonicRegression(out_of_bounds=self.out_of_bounds,
                                       y_min=0.0, y_max=1.0).fit(col, target)
            else:                                # Platt scaling
                m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
                m.fit(col.reshape(-1, 1), target)
            self.models_.append(m)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        out = np.array(proba, dtype=np.float64, copy=True)
        for j, m in enumerate(self.models_):
            if m is None:
                continue
            col = proba[:, j]
            if hasattr(m, "predict_proba"):
                out[:, j] = m.predict_proba(col.reshape(-1, 1))[:, 1]
            else:
                out[:, j] = m.predict(col)
        out = np.clip(out, EPS, 1.0)
        return out / out.sum(axis=1, keepdims=True)


def fit_calibrators(val_proba: Mapping[str, np.ndarray], y_val_idx: np.ndarray,
                    cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Fit one calibrator per model on the VALIDATION partition (Stage 7.1)."""
    cal_cfg = cfg["fusion"]["calibration"]
    if not cal_cfg.get("enabled", True):
        return {}
    out: Dict[str, Any] = {}
    for name, proba in val_proba.items():
        method = str(cal_cfg.get(name, "isotonic"))
        try:
            if method == "temperature":
                cal = TemperatureScaler(int(cal_cfg.get("temperature_max_iter", 200)))
                cal.fit(proba, y_val_idx)
                LOGGER.info("Stage 7.1: %s temperature scaling T=%.3f",
                            name, cal.temperature_)
            elif method in ("isotonic", "platt", "sigmoid"):
                cal = PerClassCalibrator(
                    "isotonic" if method == "isotonic" else "platt",
                    str(cal_cfg.get("isotonic_out_of_bounds", "clip")))
                cal.fit(proba, y_val_idx)
                LOGGER.info("Stage 7.1: %s calibrated (%s)", name, method)
            else:
                continue
            out[name] = cal
        except Exception as exc:
            LOGGER.warning("Stage 7.1: calibration of %s failed (%s); using raw "
                           "probabilities", name, exc)
    return out


def apply_calibrators(proba: Mapping[str, np.ndarray],
                      calibrators: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Apply the frozen validation-fitted calibrators. Learns nothing."""
    out: Dict[str, np.ndarray] = {}
    for name, p in proba.items():
        cal = calibrators.get(name)
        q = cal.transform(p) if cal is not None else np.array(p, dtype=np.float64)
        assert_probabilities_valid(q, f"calibrated {name}")
        out[name] = q
    return out


# ===========================================================================
# Reliability terms (Stage 7.3)
# ===========================================================================
def confidence_term(proba: np.ndarray) -> np.ndarray:
    """``C_i(t) = 1 - H(P_i(t)) / log C`` -- normalised-entropy confidence."""
    p = np.clip(np.asarray(proba, dtype=np.float64), EPS, 1.0)
    h = -(p * np.log(p)).sum(axis=1)
    c = p.shape[1]
    return 1.0 - h / np.log(max(c, 2))


def context_term(proba: np.ndarray, classes: Sequence[Any],
                 prev_state: Sequence[Any], graph: Any,
                 class_names: Optional[Mapping[Any, str]] = None) -> np.ndarray:
    """``K_i(t)``: P(state predicted by model i | previous state) under ``G_T``.

    Returns a flat 1/|S| when no attack-state graph is available, which makes
    the term constant and reduces eq. 25 to the confidence-only form.
    """
    if graph is None:
        return np.full(len(proba), 1.0 / max(len(classes), 2))
    pred_idx = np.argmax(proba, axis=1)
    names = [(class_names or {}).get(classes[i], str(classes[i])) for i in pred_idx]
    return np.array([graph.transition(prev, nxt)
                     for prev, nxt in zip(prev_state, names)], dtype=np.float64)


def previous_states(df: pd.DataFrame, rows: np.ndarray, cfg: Mapping[str, Any],
                    events: Optional[np.ndarray] = None) -> np.ndarray:
    """The preceding event in the same session, for the ``K_i(t)`` lookup."""
    unit_col = ("session_id" if cfg["temporal"].get("sequence_unit", "session") == "session"
                else "__host")
    sub = df.iloc[rows]
    order = np.argsort(sub["__timestamp"].to_numpy(), kind="mergesort")
    units = sub[unit_col].to_numpy()
    ev = (np.asarray(events) if events is not None
          else sub["Label_grouped"].to_numpy())

    out = np.empty(len(sub), dtype=object)
    out[:] = None
    last: Dict[Any, Any] = {}
    for pos in order:
        u = units[pos]
        out[pos] = last.get(u)
        last[u] = ev[pos]
    return out


# ===========================================================================
# The fusion itself
# ===========================================================================
@dataclass
class FusionWeights:
    """Per-row reliabilities, weights and the fused outputs."""

    sources: List[str]
    weights: np.ndarray                 # (n, n_sources), rows sum to 1
    reliabilities: np.ndarray           # (n, n_sources), pre-normalisation
    fused_proba: np.ndarray             # (n, C)
    risk: np.ndarray                    # (n,) R_t
    Q: Dict[str, float] = field(default_factory=dict)
    lambdas: Tuple[float, float, float] = (0.5, 0.3, 0.2)
    tau_R: float = 0.5

    def to_frame(self, unit_ids: Optional[Sequence[Any]] = None) -> pd.DataFrame:
        out = pd.DataFrame(self.weights, columns=[f"w_{s}" for s in self.sources])
        out["R_t"] = self.risk
        if unit_ids is not None:
            out.insert(0, "unit_id", list(unit_ids))
        return out


@dataclass
class EvidenceFusion:
    """Reliability-aware adaptive fusion fitted on VALIDATION (Stage 7)."""

    cfg: Mapping[str, Any]
    classes: List[Any]
    class_names: Dict[Any, str] = field(default_factory=dict)
    calibrators: Dict[str, Any] = field(default_factory=dict)
    Q: Dict[str, float] = field(default_factory=dict)
    lambdas: Tuple[float, float, float] = (0.5, 0.3, 0.2)
    tau_R: float = 0.5
    graph: Any = None
    model_names: List[str] = field(default_factory=list)
    use_temporal: bool = True

    # -- Step 2: baseline reliability -------------------------------------
    def _baseline_Q(self, val_proba: Mapping[str, np.ndarray], y_val: np.ndarray,
                    sp: Optional[np.ndarray], tc: Optional[np.ndarray]) -> Dict[str, float]:
        from sklearn.metrics import f1_score, roc_auc_score

        Q: Dict[str, float] = {}
        for name, proba in val_proba.items():
            pred = np.array([self.classes[i] for i in np.argmax(proba, axis=1)])
            Q[name] = float(f1_score(y_val, pred, average="macro", zero_division=0))

        # SP and TC are scalar attack scores: their reliability is the validation
        # ROC-AUC of each used ALONE to separate attack from benign (Stage 7.3
        # step 2) -- a discriminative measure, not their mean value.
        benign = self._benign_class()
        is_attack = (y_val != benign).astype(int)
        for key, arr in (("sp", sp), ("tc", tc)):
            if arr is None:
                continue
            if is_attack.min() == is_attack.max():
                Q[key] = 0.5
            else:
                try:
                    Q[key] = float(roc_auc_score(is_attack, arr))
                except ValueError:
                    Q[key] = 0.5
        return Q

    def _benign_class(self) -> Any:
        name = self.cfg["labels"].get("benign_class", "Benign")
        order = self.cfg["labels"]["class_order"]
        return order.index(name) if name in order else 0

    # -- Steps 3-4: reliabilities and weights -----------------------------
    def _reliabilities(self, proba: Mapping[str, np.ndarray],
                       prev_state: Optional[np.ndarray],
                       sp: Optional[np.ndarray], tc: Optional[np.ndarray],
                       availability: Optional[np.ndarray],
                       lambdas: Optional[Tuple[float, float, float]] = None,
                       ) -> Tuple[List[str], np.ndarray]:
        l1, l2, l3 = lambdas or self.lambdas
        n = len(next(iter(proba.values())))
        sources: List[str] = []
        cols: List[np.ndarray] = []

        for name in self.model_names:
            if name not in proba:
                continue
            p = proba[name]
            C = confidence_term(p)
            if l3 > 0 and prev_state is not None and self.graph is not None:
                K = context_term(p, self.classes, prev_state, self.graph, self.class_names)
            else:
                K = np.full(n, 1.0 / max(len(self.classes), 2))
            q = l1 * self.Q.get(name, 0.0) + l2 * C + l3 * K
            sources.append(name)
            cols.append(np.clip(q, 0.0, None))

        if self.use_temporal:
            avail = availability if availability is not None else np.ones(n)
            if sp is not None:
                sources.append("sp")
                cols.append(self.Q.get("sp", 0.0) * avail)
            if tc is not None:
                sources.append("tc")
                cols.append(self.Q.get("tc", 0.0) * avail)

        rel = np.column_stack(cols) if cols else np.zeros((n, 0))
        return sources, rel

    @staticmethod
    def _normalise(rel: np.ndarray) -> np.ndarray:
        total = rel.sum(axis=1, keepdims=True)
        flat = (total <= EPS).ravel()
        w = np.where(total > EPS, rel / np.maximum(total, EPS), 0.0)
        if flat.any():            # every source unreliable -> fall back to equal
            w[flat] = 1.0 / max(rel.shape[1], 1)
        return w

    # -- Step 5: fuse ------------------------------------------------------
    def fuse(self, proba: Mapping[str, np.ndarray],
             prev_state: Optional[np.ndarray] = None,
             sp: Optional[np.ndarray] = None, tc: Optional[np.ndarray] = None,
             availability: Optional[np.ndarray] = None,
             lambdas: Optional[Tuple[float, float, float]] = None,
             equal_weights: bool = False) -> FusionWeights:
        """Compute w(t), the fused class probabilities and the risk score R_t."""
        sources, rel = self._reliabilities(proba, prev_state, sp, tc, availability, lambdas)
        if rel.shape[1] == 0:
            raise ValueError("no evidence sources available for fusion")
        w = (np.full_like(rel, 1.0 / rel.shape[1]) if equal_weights
             else self._normalise(rel))
        assert_weights_sum_to_one(w, "fusion weights")

        n = rel.shape[0]
        fused = np.zeros((n, len(self.classes)), dtype=np.float64)
        risk = np.zeros(n, dtype=np.float64)
        benign = self._benign_class()

        for j, src in enumerate(sources):
            if src in proba:
                p = proba[src]
                fused += w[:, j:j + 1] * p
                # Scalar attack evidence of a model = 1 - P(Benign).
                risk += w[:, j] * (1.0 - p[:, benign])
            elif src == "sp" and sp is not None:
                risk += w[:, j] * sp
            elif src == "tc" and tc is not None:
                risk += w[:, j] * tc

        # Renormalise: the SP/TC weights carry no class distribution, so the
        # model weights alone do not sum to 1 over the class axis.
        total = fused.sum(axis=1, keepdims=True)
        fused = np.where(total > EPS, fused / np.maximum(total, EPS),
                         1.0 / len(self.classes))
        assert_probabilities_valid(fused, "fused probabilities")

        return FusionWeights(sources=sources, weights=w, reliabilities=rel,
                             fused_proba=fused, risk=risk, Q=dict(self.Q),
                             lambdas=lambdas or self.lambdas, tau_R=self.tau_R)

    # -- fitting on validation --------------------------------------------
    def fit(self, val_proba: Mapping[str, np.ndarray], y_val: np.ndarray,
            prev_state: Optional[np.ndarray] = None,
            sp: Optional[np.ndarray] = None, tc: Optional[np.ndarray] = None,
            availability: Optional[np.ndarray] = None,
            graph: Any = None) -> "EvidenceFusion":
        """Fit Q, the lambdas and tau_R -- on VALIDATION data only."""
        from sklearn.metrics import f1_score

        self.graph = graph if graph is not None else self.graph
        self.model_names = [n for n in val_proba]
        y_idx = np.array([self.classes.index(v) if v in self.classes else 0
                          for v in y_val])

        self.Q = self._baseline_Q(val_proba, y_val, sp, tc)
        LOGGER.info("Stage 7.3 step 2: baseline reliabilities Q = %s",
                    {k: round(v, 4) for k, v in self.Q.items()})

        # -- tune the lambdas on a simplex grid (validation macro-F1) ------
        tune = self.cfg["fusion"].get("tune_lambdas", {})
        if tune.get("enabled", True):
            step = float(tune.get("grid_step", 0.1))
            best, best_score = tuple(self.cfg["fusion"]["lambdas"]), -np.inf
            grid = np.arange(0.0, 1.0 + 1e-9, step)
            for l1 in grid:
                for l2 in grid:
                    l3 = 1.0 - l1 - l2
                    if l3 < -1e-9 or l3 > 1.0 + 1e-9:
                        continue
                    cand = (float(l1), float(l2), float(max(l3, 0.0)))
                    out = self.fuse(val_proba, prev_state, sp, tc, availability,
                                    lambdas=cand)
                    pred = np.array([self.classes[i]
                                     for i in np.argmax(out.fused_proba, axis=1)])
                    score = f1_score(y_val, pred, average="macro", zero_division=0)
                    if score > best_score:
                        best_score, best = score, cand
            self.lambdas = best
            LOGGER.info("Stage 7.3 step 3: lambdas tuned on validation to "
                        "(%.1f, %.1f, %.1f), macro-F1 %.4f", *best, best_score)
        else:
            self.lambdas = tuple(self.cfg["fusion"]["lambdas"])

        # -- tau_R on validation ------------------------------------------
        out = self.fuse(val_proba, prev_state, sp, tc, availability)
        self.tau_R = self._fit_threshold(out.risk, y_val)
        LOGGER.info("Stage 7.3 step 5: tau_R = %.4f (objective=%s)",
                    self.tau_R, self.cfg["fusion"]["threshold"]["objective"])
        return self

    def _fit_threshold(self, risk: np.ndarray, y_val: np.ndarray) -> float:
        """Choose tau_R on validation: maximise macro-F1 or hit a target FAR."""
        from sklearn.metrics import f1_score

        th_cfg = self.cfg["fusion"]["threshold"]
        benign = self._benign_class()
        is_attack = (y_val != benign).astype(int)
        grid = np.linspace(0.0, 1.0, int(th_cfg.get("grid_points", 101)))

        if str(th_cfg.get("objective", "macro_f1")) == "target_far":
            target = float(th_cfg.get("target_false_alarm_rate", 0.01))
            benign_rows = is_attack == 0
            best = 1.0
            for t in grid:
                far = float((risk[benign_rows] >= t).mean()) if benign_rows.any() else 0.0
                if far <= target:
                    best = float(t)
                    break
            return best

        best_t, best_score = 0.5, -np.inf
        for t in grid:
            pred = (risk >= t).astype(int)
            score = f1_score(is_attack, pred, average="macro", zero_division=0)
            if score > best_score:
                best_score, best_t = score, float(t)
        return best_t


# ===========================================================================
# Ablations (Stage 7.6)
# ===========================================================================
def run_fusion_ablations(fusion: EvidenceFusion, proba: Mapping[str, np.ndarray],
                         y_true: np.ndarray, prev_state: Optional[np.ndarray],
                         sp: Optional[np.ndarray], tc: Optional[np.ndarray],
                         availability: Optional[np.ndarray],
                         cfg: Mapping[str, Any]) -> pd.DataFrame:
    """The Stage 7.6 ablation baselines, reported as baselines -- not methods."""
    from sklearn.metrics import accuracy_score, f1_score

    variants = cfg["fusion"]["ablations"].get("variants", [])
    rows: List[Dict[str, Any]] = []

    def score(name: str, pred: np.ndarray, note: str = "") -> None:
        rows.append({
            "variant": name,
            "accuracy": accuracy_score(y_true, pred),
            "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, pred, average="weighted", zero_division=0),
            "note": note,
        })

    def argmax_pred(fw: FusionWeights) -> np.ndarray:
        return np.array([fusion.classes[i] for i in np.argmax(fw.fused_proba, axis=1)])

    for variant in variants:
        try:
            if variant == "single_best":
                best = max(fusion.Q, key=lambda k: fusion.Q[k] if k in proba else -1)
                pred = np.array([fusion.classes[i]
                                 for i in np.argmax(proba[best], axis=1)])
                score("single_best", pred, f"best validation Q: {best}")
            elif variant == "equal_weights":
                fw = fusion.fuse(proba, prev_state, sp, tc, availability, equal_weights=True)
                score("equal_weights", argmax_pred(fw), "w_i = 1/m")
            elif variant == "static_weights":
                fw = fusion.fuse(proba, prev_state, sp, tc, availability, lambdas=(1.0, 0.0, 0.0))
                score("static_weights", argmax_pred(fw), "lambda = (1, 0, 0)")
            elif variant == "no_confidence":
                l1, _l2, l3 = fusion.lambdas
                tot = l1 + l3
                lam = ((l1 / tot, 0.0, l3 / tot) if tot > 0 else (1.0, 0.0, 0.0))
                fw = fusion.fuse(proba, prev_state, sp, tc, availability, lambdas=lam)
                score("no_confidence", argmax_pred(fw), "lambda2 = 0")
            elif variant == "no_context":
                l1, l2, _l3 = fusion.lambdas
                tot = l1 + l2
                lam = ((l1 / tot, l2 / tot, 0.0) if tot > 0 else (1.0, 0.0, 0.0))
                fw = fusion.fuse(proba, prev_state, sp, tc, availability, lambdas=lam)
                score("no_context", argmax_pred(fw), "lambda3 = 0")
            elif variant == "no_temporal":
                saved = fusion.use_temporal
                fusion.use_temporal = False
                fw = fusion.fuse(proba, prev_state, None, None, availability)
                fusion.use_temporal = saved
                score("no_temporal", argmax_pred(fw), "SP and TC removed (5-model fusion)")
            elif variant == "proposed":
                fw = fusion.fuse(proba, prev_state, sp, tc, availability)
                score("proposed", argmax_pred(fw), "reliability-aware adaptive fusion")
        except Exception as exc:
            LOGGER.warning("Stage 7.6: ablation %s failed (%s)", variant, exc)

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return out


def worked_example(fw: FusionWeights, fusion: EvidenceFusion, row: int = 0,
                   class_names: Optional[Mapping[Any, str]] = None) -> pd.DataFrame:
    """Reproduce the Stage 7.4 per-source table for one session."""
    benign = fusion._benign_class()
    rows = []
    for j, src in enumerate(fw.sources):
        entry: Dict[str, Any] = {
            "source": src,
            "baseline_Q": round(fw.Q.get(src, float("nan")), 4),
            "reliability_q": round(float(fw.reliabilities[row, j]), 4),
            "weight_w": round(float(fw.weights[row, j]), 4),
        }
        entry["score"] = round(float(fw.risk[row]), 4) if src in ("sp", "tc") else None
        entry["w_x_score"] = None
        rows.append(entry)
    out = pd.DataFrame(rows)
    out.loc[len(out)] = {"source": "TOTAL", "baseline_Q": None,
                         "reliability_q": round(float(fw.reliabilities[row].sum()), 4),
                         "weight_w": round(float(fw.weights[row].sum()), 4),
                         "score": None, "w_x_score": round(float(fw.risk[row]), 4)}
    return out


def save_fusion_artifacts(fw: FusionWeights, fusion: EvidenceFusion,
                          out_dir: str | Path, tag: str = "",
                          unit_ids: Optional[Sequence[Any]] = None) -> None:
    """Persist the per-session weight matrix and the fitted fusion parameters."""
    out_dir = ensure_dir(out_dir)
    suffix = f"_{tag}" if tag else ""
    if fusion.cfg["fusion"].get("save_weights", True):
        save_table(fw.to_frame(unit_ids), Path(out_dir) / f"fusion_weights{suffix}.csv")
    save_json({
        "sources": fw.sources,
        "Q": fusion.Q,
        "lambdas": list(fusion.lambdas),
        "tau_R": fusion.tau_R,
        "mode": fusion.cfg["fusion"]["mode"],
        "mean_weights": {s: float(fw.weights[:, j].mean())
                         for j, s in enumerate(fw.sources)},
        "calibration": {k: type(v).__name__ for k, v in fusion.calibrators.items()},
    }, Path(out_dir) / f"fusion_weights{suffix}.json")
