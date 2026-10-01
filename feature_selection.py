"""Stage 4: feature selection by fusion of eight independent rankers.

Pipeline (Stage 4.2 / 4.7), fitted on TRAIN only and frozen thereafter:

  1. redundancy removal -> ``F_corr``
     near-zero variance (eps), duplicate columns, Spearman |rho| > 0.95;
  2. eight rankers, each scoring every feature of ``F_corr`` and contributing
     its top-K set ``FeS Set-m`` (K = 30, scores <= 0 discarded);
  3. frequency vote psi(f) >= eta (eta = 4 of 8) giving the pool ``P``;
  4. normalized-rank aggregation Score(f) = mean_m NR_m(f), ordering ``P``;
  5. subset size N* chosen on VALIDATION macro-F1 with tolerance delta;
  6. ``F* = Top_{N*}(P)``, frozen and applied unchanged to val and test.

Class imbalance inside the rankers is handled with class weights and per-class
capped stratified sampling (Stage 4.4), never by resampling -- SMOTE-ENN comes
later (Stage 5) so synthetic rows cannot influence which features are chosen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils import (
    Timer,
    assert_feature_mask_frozen,
    ensure_dir,
    get_logger,
    joblib_cache,
    progress,
    save_json,
    save_table,
    stratified_subsample,
)

LOGGER = get_logger()

#: Ranker key -> the "FeS Set-m" label used in the methodology.
FES_SET_LABELS = {
    "mi": "FeS Set-1 (MI)",
    "rfi": "FeS Set-2 (RFI)",
    "pi_svm": "FeS Set-3 (PI-SVM)",
    "shap_lgbm": "FeS Set-4 (SHAP-LGBM)",
    "dmm": "FeS Set-5 (Mean-Median)",
    "sd": "FeS Set-6 (SD)",
    "xgb": "FeS Set-7 (XGB)",
    "lstm": "FeS Set-8 (LSTM)",
}


# ===========================================================================
# Step 1: redundancy removal -> F_corr
# ===========================================================================
@dataclass
class RedundancyFilter:
    """Near-zero variance, duplicate columns and Spearman correlation filter.

    All three statistics are computed on the TRAIN partition only; the surviving
    column list is then applied unchanged to validation and test.
    """

    cfg: Mapping[str, Any]
    kept_: List[str] = field(default_factory=list)
    dropped_: Dict[str, str] = field(default_factory=dict)

    def fit(self, X_train: pd.DataFrame, seed: int = 42) -> "RedundancyFilter":
        fs = self.cfg["feature_selection"]
        eps = float(fs["variance_epsilon"])
        rho = float(fs["spearman_threshold"])
        cols = list(X_train.columns)

        # -- near-zero variance --------------------------------------------
        variances = X_train.var(axis=0, ddof=0)
        for c in cols:
            if not np.isfinite(variances[c]) or variances[c] <= eps:
                self.dropped_[c] = f"near-zero variance ({variances[c]:.3g} <= {eps:g})"
        cols = [c for c in cols if c not in self.dropped_]

        # -- duplicate columns ---------------------------------------------
        if fs.get("drop_duplicate_columns", True) and cols:
            seen: Dict[bytes, str] = {}
            for c in cols:
                digest = pd.util.hash_pandas_object(X_train[c], index=False).values.tobytes()
                if digest in seen:
                    self.dropped_[c] = f"duplicate of {seen[digest]}"
                else:
                    seen[digest] = c
            cols = [c for c in cols if c not in self.dropped_]

        # -- Spearman rank correlation -------------------------------------
        if cols and rho < 1.0:
            n_rows = int(fs.get("spearman_sample_rows", 100000))
            sub = X_train[cols]
            if len(sub) > n_rows:
                rng = np.random.default_rng(seed)
                sub = sub.iloc[rng.choice(len(sub), size=n_rows, replace=False)]
            # Spearman == Pearson on ranks; ranking once is far cheaper than
            # scipy.stats.spearmanr on a wide matrix.
            ranks = sub.rank(axis=0, method="average").to_numpy(dtype=np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(ranks, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0)
            if corr.ndim == 0:
                corr = corr.reshape(1, 1)
            # Keep the earlier column of every over-correlated pair.
            upper = np.triu(np.abs(corr), k=1)
            for j in range(1, len(cols)):
                partner = np.argmax(upper[:j, j])
                if upper[partner, j] > rho and cols[j] not in self.dropped_:
                    self.dropped_[cols[j]] = (
                        f"|spearman| {upper[partner, j]:.3f} > {rho} with {cols[partner]}"
                    )
            cols = [c for c in cols if c not in self.dropped_]

        self.kept_ = cols
        LOGGER.info("Stage 4.1: redundancy filter kept %d of %d features "
                    "(F_corr); dropped %d", len(self.kept_),
                    len(X_train.columns), len(self.dropped_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.kept_]

    def report(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"feature": f, "reason": r} for f, r in sorted(self.dropped_.items())]
        )


# ===========================================================================
# Step 2: the eight rankers
# ===========================================================================
def _sample(X: pd.DataFrame, y: np.ndarray, max_rows: Optional[int], seed: int,
            min_per_class: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Per-class capped stratified sample (Stage 4.4)."""
    if max_rows is None or len(X) <= max_rows:
        return X.to_numpy(dtype=np.float32), y
    idx = stratified_subsample(y, int(max_rows), seed, min_per_class=min_per_class)
    return X.iloc[idx].to_numpy(dtype=np.float32), y[idx]


def _balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    """w_c = N / (C * N_c), broadcast to each row (Stage 5.2)."""
    classes, counts = np.unique(y, return_counts=True)
    w = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
    return np.array([w[v] for v in y], dtype=np.float64)


def rank_mi(X: pd.DataFrame, y: np.ndarray, cfg: Mapping[str, Any],
            seed: int) -> np.ndarray:
    """FeS Set-1: mutual information, k-NN estimator (k=3), stratified sample."""
    from sklearn.feature_selection import mutual_info_classif

    p = cfg["feature_selection"]["mi"]
    Xs, ys = _sample(X, y, p.get("max_rows"), seed)
    return mutual_info_classif(
        Xs, ys, discrete_features=False,
        n_neighbors=int(p.get("n_neighbors", 3)), random_state=seed,
    ).astype(np.float64)


def rank_rfi(X: pd.DataFrame, y: np.ndarray, cfg: Mapping[str, Any],
             seed: int) -> np.ndarray:
    """FeS Set-2: Random Forest Gini importance, 200 trees, balanced weights."""
    from sklearn.ensemble import RandomForestClassifier

    p = cfg["feature_selection"]["rfi"]
    Xs, ys = _sample(X, y, p.get("max_rows"), seed)
    rf = RandomForestClassifier(
        n_estimators=int(p.get("n_estimators", 200)),
        criterion=str(p.get("criterion", "gini")),
        class_weight=p.get("class_weight", "balanced"),
        n_jobs=int(p.get("n_jobs", -1)), random_state=seed,
    ).fit(Xs, ys)
    return rf.feature_importances_.astype(np.float64)


def rank_pi_svm(X: pd.DataFrame, y: np.ndarray, cfg: Mapping[str, Any],
                seed: int) -> np.ndarray:
    """FeS Set-3: RBF-SVM permutation importance (drop in macro-F1).

    The SVM is fitted on one slice of TRAIN and the permutation score is
    measured on a *different, held-out slice of TRAIN* -- never on validation or
    test (Stage 4.5, "Why a held-out slice").
    """
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    p = cfg["feature_selection"]["pi_svm"]
    Xs, ys = _sample(X, y, p.get("max_rows"), seed, min_per_class=4)

    # Stratify only over classes with at least 2 members in the sample.
    counts = pd.Series(ys).value_counts()
    strat = ys if (counts >= 2).all() else None
    Xa, Xb, ya, yb = train_test_split(
        Xs, ys, test_size=float(p.get("holdout_frac", 0.30)),
        random_state=seed, stratify=strat,
    )
    svm = SVC(
        kernel=str(p.get("kernel", "rbf")), C=float(p.get("C", 1.0)),
        gamma=p.get("gamma", "auto"), class_weight=p.get("class_weight", "balanced"),
        random_state=seed,
    ).fit(Xa, ya)

    # A plain (estimator, X, y) callable rather than make_scorer: a scorer built
    # from f1_score inherits that function's binary default pos_label=1, which
    # sklearn then validates against the estimator's classes_. When class 1 is
    # absent from the SVM's training sample -- which is exactly what branch E
    # does when it holds out PortScan -- that validation raises
    # "pos_label=1 is not a valid label". A direct callable sidesteps the
    # binary-metric machinery entirely.
    scoring = str(p.get("scoring", "macro_f1"))
    average = {"macro_f1": "macro", "weighted_f1": "weighted",
               "micro_f1": "micro"}.get(scoring, "macro")

    def macro_f1_scorer(estimator, X_eval, y_eval) -> float:
        return float(f1_score(y_eval, estimator.predict(X_eval),
                              average=average, zero_division=0))

    result = permutation_importance(
        svm, Xb, yb, scoring=macro_f1_scorer, n_repeats=int(p.get("n_repeats", 3)),
        random_state=seed, n_jobs=1,
    )
    return result.importances_mean.astype(np.float64)


def rank_shap_lgbm(X: pd.DataFrame, y: np.ndarray, cfg: Mapping[str, Any],
                   seed: int) -> np.ndarray:
    """FeS Set-4: mean |SHAP| of a class-balanced LightGBM, over rows AND classes."""
    import lightgbm as lgb
    import shap

    p = cfg["feature_selection"]["shap_lgbm"]
    Xs, ys = _sample(X, y, p.get("fit_max_rows"), seed)
    model = lgb.LGBMClassifier(
        n_estimators=int(p.get("n_estimators", 200)),
        learning_rate=float(p.get("learning_rate", 0.1)),
        num_leaves=int(p.get("num_leaves", 31)),
        class_weight=p.get("class_weight", "balanced"),
        random_state=seed, n_jobs=-1, verbose=-1,
    ).fit(Xs, ys)

    n_explain = min(int(p.get("explain_rows", 5000)), len(Xs))
    idx = stratified_subsample(ys, n_explain, seed, min_per_class=5)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(Xs[idx], check_additivity=False)

    # Normalise the several shapes shap returns for multi-class tree models:
    # list of (n, p) per class, or (n, p, C), or (n, p) for binary.
    if isinstance(values, list):
        arr = np.stack([np.abs(v) for v in values], axis=0)   # (C, n, p)
        imp = arr.mean(axis=(0, 1))
    else:
        values = np.asarray(values)
        if values.ndim == 3:
            imp = np.abs(values).mean(axis=(0, 2))            # (n, p, C) -> p
        else:
            imp = np.abs(values).mean(axis=0)
    return np.asarray(imp, dtype=np.float64).ravel()[:X.shape[1]]


def rank_dmm(X: pd.DataFrame, y: np.ndarray, cfg: Mapping[str, Any],
             seed: int) -> np.ndarray:
    """FeS Set-5: |mean - median| of the Min-Max scaled train feature (eq. 7)."""
    arr = X.to_numpy(dtype=np.float64)
    return np.abs(arr.mean(axis=0) - np.median(arr, axis=0))


def rank_sd(X: pd.DataFrame, y: np.ndarray, cfg: Mapping[str, Any],
            seed: int) -> np.ndarray:
    """FeS Set-6: standard deviation of the scaled train feature (eq. 8)."""
    return X.to_numpy(dtype=np.float64).std(axis=0, ddof=0)


def rank_xgb(X: pd.DataFrame, y: np.ndarray, cfg: Mapping[str, Any],
             seed: int) -> np.ndarray:
    """FeS Set-7: XGBoost gain importance with balanced sample weights."""
    import xgboost as xgb

    p = cfg["feature_selection"]["xgb"]
    Xs, ys = _sample(X, y, p.get("max_rows"), seed)
    classes = np.unique(ys)
    remap = {c: i for i, c in enumerate(classes)}
    ys_mapped = np.array([remap[v] for v in ys])

    model = xgb.XGBClassifier(
        n_estimators=int(p.get("n_estimators", 200)),
        max_depth=int(p.get("max_depth", 6)),
        learning_rate=float(p.get("learning_rate", 0.1)),
        subsample=float(p.get("subsample", 0.9)),
        colsample_bytree=float(p.get("colsample_bytree", 0.9)),
        objective="multi:softprob" if len(classes) > 2 else "binary:logistic",
        num_class=len(classes) if len(classes) > 2 else None,
        tree_method="hist", random_state=seed, n_jobs=-1,
        importance_type=str(p.get("importance_type", "gain")),
        verbosity=0,
    )
    model.fit(Xs, ys_mapped, sample_weight=_balanced_sample_weights(ys))

    booster = model.get_booster()
    gains = booster.get_score(importance_type=str(p.get("importance_type", "gain")))
    out = np.zeros(X.shape[1], dtype=np.float64)
    for key, value in gains.items():
        # Booster feature names are f0, f1, ... in column order.
        out[int(key[1:])] = value
    return out


def rank_lstm(X: pd.DataFrame, y: np.ndarray, cfg: Mapping[str, Any], seed: int,
              session_ids: Optional[np.ndarray] = None,
              times: Optional[np.ndarray] = None) -> np.ndarray:
    """FeS Set-8: gradient x input importance of a class-weighted LSTM (eq. 10).

    Input windows are L consecutive flows of one session over the ``F_corr``
    candidates. The model early-stops on a held-out slice of TRAIN, and the
    gradients are taken on that same held-out slice -- never on validation or
    test.
    """
    import tensorflow as tf

    p = cfg["feature_selection"]["lstm"]
    L = int(p.get("window", 10))
    Xw, yw = build_session_windows(
        X.to_numpy(dtype=np.float32), y, session_ids, times, L,
        max_windows=int(p.get("max_windows", 40000)), seed=seed,
    )
    if len(Xw) < 8:
        LOGGER.warning("LSTM ranker: only %d windows available; returning zeros", len(Xw))
        return np.zeros(X.shape[1], dtype=np.float64)

    classes = np.unique(yw)
    remap = {c: i for i, c in enumerate(classes)}
    yw_m = np.array([remap[v] for v in yw], dtype=np.int32)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xw))
    n_hold = max(1, int(len(Xw) * float(p.get("holdout_frac", 0.20))))
    hold, fit = perm[:n_hold], perm[n_hold:]
    if len(fit) < 4:
        fit, hold = perm, perm

    counts = np.bincount(yw_m, minlength=len(classes)).astype(np.float64)
    weights = {i: (len(yw_m) / (len(classes) * c) if c > 0 else 0.0)
               for i, c in enumerate(counts)}

    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(L, X.shape[1])),
        tf.keras.layers.Masking(mask_value=0.0),
        tf.keras.layers.LSTM(int(p.get("units", 64))),
        tf.keras.layers.Dropout(float(p.get("dropout", 0.3))),
        tf.keras.layers.Dense(len(classes), activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(float(p.get("learning_rate", 1e-3))),
        loss="sparse_categorical_crossentropy",
    )
    model.fit(
        Xw[fit], yw_m[fit],
        validation_data=(Xw[hold], yw_m[hold]),
        epochs=int(p.get("epochs", 10)),
        batch_size=int(p.get("batch_size", 256)),
        class_weight=weights, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=int(p.get("patience", 3)),
            restore_best_weights=True)],
    )

    # importance = mean |x * dL/dx| over windows, timesteps and classes
    batch = int(p.get("batch_size", 256))
    n_batches = int(p.get("grad_batches", 40))
    total = np.zeros(X.shape[1], dtype=np.float64)
    seen = 0
    for b in range(min(n_batches, max(1, len(hold) // batch + 1))):
        rows = hold[b * batch:(b + 1) * batch]
        if len(rows) == 0:
            break
        xb = tf.constant(Xw[rows], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(xb)
            out = model(xb, training=False)
            # Sum over classes: importance aggregates across all class outputs.
            loss = tf.reduce_sum(out, axis=-1)
        grads = tape.gradient(loss, xb)
        if grads is None:
            continue
        contrib = tf.abs(grads * xb).numpy()        # (batch, L, F)
        total += contrib.sum(axis=(0, 1))
        seen += contrib.shape[0] * contrib.shape[1]
    return total / max(seen, 1)


RANKERS: Dict[str, Callable[..., np.ndarray]] = {
    "mi": rank_mi,
    "rfi": rank_rfi,
    "pi_svm": rank_pi_svm,
    "shap_lgbm": rank_shap_lgbm,
    "dmm": rank_dmm,
    "sd": rank_sd,
    "xgb": rank_xgb,
    "lstm": rank_lstm,
}


# ===========================================================================
# Session windows (shared with the Stage 6 LSTM)
# ===========================================================================
def build_session_windows(X: np.ndarray, y: np.ndarray,
                          session_ids: Optional[np.ndarray],
                          times: Optional[np.ndarray], window: int,
                          max_windows: Optional[int] = None,
                          seed: int = 42,
                          return_index: bool = False):
    """Build (n, L, F) windows of up to ``window`` consecutive flows per session.

    Short sessions are left-padded with zeros (the Keras ``Masking`` layer with
    ``mask_value=0.0`` then ignores the padding). One window is emitted per
    flow, ending at that flow, so the label of the window is the label of its
    final flow and every flow receives a prediction.

    Returns ``(Xw, yw)``, or ``(Xw, yw, row_index)`` when ``return_index``.
    """
    n, n_feat = X.shape
    if session_ids is None:
        session_ids = np.zeros(n, dtype=np.int64)
    order = (np.argsort(times, kind="mergesort") if times is not None
             else np.arange(n))

    rows_by_session: Dict[Any, List[int]] = {}
    for i in order:
        rows_by_session.setdefault(session_ids[i], []).append(int(i))

    targets = np.arange(n)
    if max_windows is not None and n > max_windows:
        targets = stratified_subsample(y, int(max_windows), seed, min_per_class=5)
    wanted = set(targets.tolist())

    Xw = np.zeros((len(targets), window, n_feat), dtype=np.float32)
    yw = np.empty(len(targets), dtype=y.dtype)
    idx_out = np.empty(len(targets), dtype=np.int64)

    k = 0
    for _sid, rows in rows_by_session.items():
        for pos, row in enumerate(rows):
            if row not in wanted:
                continue
            start = max(0, pos - window + 1)
            hist = rows[start:pos + 1]
            Xw[k, window - len(hist):, :] = X[hist]     # left-pad with zeros
            yw[k] = y[row]
            idx_out[k] = row
            k += 1
    Xw, yw, idx_out = Xw[:k], yw[:k], idx_out[:k]
    if return_index:
        return Xw, yw, idx_out
    return Xw, yw


# ===========================================================================
# Steps 3-5: vote, rank aggregation, N* on validation
# ===========================================================================
def topk_set(scores: np.ndarray, features: Sequence[str], k: int) -> List[str]:
    """Top-K features by score, discarding scores <= 0 (eq. 11)."""
    order = np.argsort(-scores, kind="mergesort")
    out = [features[i] for i in order if scores[i] > 0]
    return out[:k]


def normalized_rank(members: Sequence[str], k: int) -> Dict[str, float]:
    """NR_m(f) = 1 - (rank - 1) / (K - 1) for members, 0 otherwise (eq. 13)."""
    if k <= 1:
        return {f: 1.0 for f in members}
    return {f: 1.0 - (r / (k - 1)) for r, f in enumerate(members)}


def fuse_rankings(scores: Mapping[str, np.ndarray], features: Sequence[str],
                  k: int, eta: int) -> pd.DataFrame:
    """Frequency vote + normalized-rank aggregation (eqs. 11-14).

    Returns one row per feature with its per-ranker membership, the vote count
    psi, the fused Score, and whether it entered the pool P.
    """
    sets = {m: topk_set(s, features, k) for m, s in scores.items()}
    nrs = {m: normalized_rank(members, k) for m, members in sets.items()}
    n_rankers = len(scores)

    rows = []
    for f in features:
        votes = sum(1 for m in sets if f in nrs[m])
        score = sum(nrs[m].get(f, 0.0) for m in sets) / max(n_rankers, 1)
        row: Dict[str, Any] = {"feature": f, "psi_votes": votes, "score": score}
        for m in sets:
            row[f"in_{m}"] = int(f in nrs[m])
            row[f"nr_{m}"] = round(nrs[m].get(f, 0.0), 6)
        rows.append(row)

    table = pd.DataFrame(rows)
    table["in_pool"] = (table["psi_votes"] >= eta).astype(int)
    table = table.sort_values(["in_pool", "score", "psi_votes"],
                              ascending=[False, False, False]).reset_index(drop=True)
    return table


def select_n_star(table: pd.DataFrame, X_train: pd.DataFrame, y_train: np.ndarray,
                  X_val: pd.DataFrame, y_val: np.ndarray, cfg: Mapping[str, Any],
                  seed: int) -> Tuple[int, pd.DataFrame]:
    """Choose N* on VALIDATION macro-F1 with tolerance delta (eq. 15).

    The probe is a class-balanced LightGBM, and every candidate size is scored
    on the validation partition only. N* is the *smallest* N whose macro-F1 is
    within ``delta`` of the best.
    """
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, f1_score, recall_score

    fs = cfg["feature_selection"]
    probe = fs["n_star_probe"]
    pool = table.loc[table["in_pool"] == 1, "feature"].tolist()
    if not pool:        # vote threshold too strict -> fall back to the full order
        pool = table["feature"].tolist()
        LOGGER.warning("Stage 4.6: vote threshold eta=%s emptied the pool; "
                       "falling back to the rank order over all features",
                       fs["vote_threshold"])

    grid = [n for n in fs["n_grid"] if n <= len(pool)]
    if not grid:
        grid = [len(pool)]
    classes = np.unique(y_train)

    rows = []
    for n in progress(grid, desc="N* search", total=len(grid)):
        feats = pool[:n]
        tr_idx = stratified_subsample(
            y_train, int(probe.get("max_train_rows", 200000)), seed, min_per_class=5)
        model = lgb.LGBMClassifier(
            n_estimators=int(probe.get("n_estimators", 150)),
            learning_rate=float(probe.get("learning_rate", 0.1)),
            num_leaves=int(probe.get("num_leaves", 31)),
            class_weight=probe.get("class_weight", "balanced"),
            random_state=seed, n_jobs=-1, verbose=-1,
        )
        model.fit(X_train.iloc[tr_idx][feats], y_train[tr_idx])

        with Timer(f"n_star_predict_{n}") as t:
            proba = model.predict_proba(X_val[feats])
        pred = model.classes_[np.argmax(proba, axis=1)]

        minority = [c for c in classes if (y_train == c).sum() < 0.05 * len(y_train)]
        rec = recall_score(y_val, pred, labels=minority, average="macro",
                           zero_division=0) if minority else float("nan")
        onehot = np.zeros((len(y_val), len(model.classes_)))
        for j, c in enumerate(model.classes_):
            onehot[:, j] = (y_val == c).astype(float)
        try:
            pr_auc = average_precision_score(onehot, proba, average="macro")
        except ValueError:
            pr_auc = float("nan")

        rows.append({
            "N": n,
            "macro_f1": f1_score(y_val, pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_val, pred, average="weighted", zero_division=0),
            "minority_recall": rec,
            "pr_auc": pr_auc,
            "inference_time_s": t.elapsed,
            "inference_ms_per_flow": 1000.0 * t.elapsed / max(len(X_val), 1),
        })

    curve = pd.DataFrame(rows)
    best = curve["macro_f1"].max()
    delta = float(fs.get("delta", 0.005))
    eligible = curve.loc[curve["macro_f1"] >= best - delta, "N"]
    n_star = int(eligible.min())
    LOGGER.info("Stage 4.6: best validation macro-F1 %.4f; N* = %d "
                "(smallest N within delta=%.3f)", best, n_star, delta)
    return n_star, curve


# ===========================================================================
# Orchestration
# ===========================================================================
@dataclass
class FeatureSelectionResult:
    """Everything Stage 4 produces, including the frozen mask F*."""

    f_star: List[str]
    n_star: int
    pool: List[str]
    f_corr: List[str]
    votes_table: pd.DataFrame
    n_star_curve: pd.DataFrame
    ranker_scores: Dict[str, np.ndarray]
    redundancy_report: pd.DataFrame
    timings: Dict[str, float] = field(default_factory=dict)
    stability: Optional[pd.DataFrame] = None

    def apply(self, X: pd.DataFrame, where: str = "") -> pd.DataFrame:
        """Apply F* unchanged, asserting the mask has not drifted."""
        missing = [f for f in self.f_star if f not in X.columns]
        if missing:
            raise KeyError(f"F* features missing from {where or 'frame'}: {missing[:5]}")
        out = X[self.f_star]
        assert_feature_mask_frozen(self.f_star, list(out.columns), where or "apply")
        return out


def run_feature_selection(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
    cfg: Mapping[str, Any], seed: int = 42,
    session_ids_train: Optional[np.ndarray] = None,
    times_train: Optional[np.ndarray] = None,
    out_dir: Optional[str] = None,
    cache_tag: str = "",
) -> FeatureSelectionResult:
    """Run Stage 4 end to end on TRAIN (+ VALIDATION for N* only)."""
    fs = cfg["feature_selection"]
    out_dir = Path(out_dir or cfg["run"]["output_dir"])
    ensure_dir(out_dir)
    timings: Dict[str, float] = {}

    # -- step 1: redundancy removal ---------------------------------------
    with Timer("redundancy_filter", timings, LOGGER):
        filt = RedundancyFilter(cfg).fit(X_train, seed)
    Xc_train = filt.transform(X_train)
    features = list(Xc_train.columns)

    # -- step 2: the eight rankers ----------------------------------------
    scores: Dict[str, np.ndarray] = {}
    for name in fs["rankers"]:
        fn = RANKERS[name]
        key = f"ranker_{name}_{cache_tag}_s{seed}_{len(features)}f.joblib"

        def produce(fn=fn, name=name):
            kwargs = {}
            if name == "lstm":
                kwargs = {"session_ids": session_ids_train, "times": times_train}
            with Timer(f"ranker_{name}", timings, LOGGER):
                return fn(Xc_train, y_train, cfg, seed, **kwargs)

        value, hit = joblib_cache(cfg, key, produce) if fs.get("cache_ranker_scores", True) \
            else (produce(), False)
        if hit:
            LOGGER.info("Stage 4.5: %s loaded from cache", FES_SET_LABELS.get(name, name))
            timings.setdefault(f"ranker_{name}", 0.0)
        arr = np.asarray(value, dtype=np.float64).ravel()
        if len(arr) != len(features):       # cache from a different feature count
            arr = np.resize(arr, len(features))
        scores[name] = arr
        LOGGER.info("  %s: %d non-zero scores, top=%s",
                    FES_SET_LABELS.get(name, name), int((arr > 0).sum()),
                    [features[i] for i in np.argsort(-arr)[:3]])

    # -- steps 3-4: vote + rank aggregation -------------------------------
    k = int(fs["top_k"])
    eta = int(fs["vote_threshold"])
    votes = fuse_rankings(scores, features, k, eta)
    pool = votes.loc[votes["in_pool"] == 1, "feature"].tolist()
    LOGGER.info("Stage 4.6: pool P has %d features with psi >= %d (of %d rankers)",
                len(pool), eta, len(scores))

    # -- step 5: N* on validation -----------------------------------------
    with Timer("n_star_selection", timings, LOGGER):
        n_star, curve = select_n_star(votes, Xc_train, y_train,
                                      filt.transform(X_val), y_val, cfg, seed)
    pool_or_all = pool if pool else votes["feature"].tolist()
    f_star = pool_or_all[:n_star]

    result = FeatureSelectionResult(
        f_star=f_star, n_star=n_star, pool=pool_or_all, f_corr=features,
        votes_table=votes, n_star_curve=curve, ranker_scores=scores,
        redundancy_report=filt.report(), timings=timings,
    )

    # -- stability over repeated stratified subsamples (Stage 8.3) --------
    if fs.get("stability", {}).get("enabled", False):
        with Timer("stability", timings, LOGGER):
            result.stability = compute_stability(Xc_train, y_train, cfg, seed,
                                                 session_ids_train, times_train,
                                                 n_star=n_star)

    _save_artifacts(result, cfg, out_dir, cache_tag)
    LOGGER.info("Stage 4 complete: F* = %d features %s", len(f_star), f_star[:8])
    return result


def compute_stability(X: pd.DataFrame, y: np.ndarray, cfg: Mapping[str, Any],
                      seed: int, session_ids: Optional[np.ndarray],
                      times: Optional[np.ndarray], n_star: int) -> pd.DataFrame:
    """Stability(f) = selected runs / total runs over stratified subsamples (eq. 28)."""
    fs = cfg["feature_selection"]
    st = fs["stability"]
    n_runs = int(st.get("n_runs", 5))
    frac = float(st.get("subsample_frac", 0.6))
    k, eta = int(fs["top_k"]), int(fs["vote_threshold"])

    counts: Dict[str, int] = {f: 0 for f in X.columns}
    for run in progress(range(n_runs), desc="stability", total=n_runs):
        rs = seed + 1000 * (run + 1)
        idx = stratified_subsample(y, max(50, int(len(y) * frac)), rs, min_per_class=5)
        Xi, yi = X.iloc[idx], y[idx]
        si = session_ids[idx] if session_ids is not None else None
        ti = times[idx] if times is not None else None
        sub_scores = {}
        for name in fs["rankers"]:
            kwargs = {"session_ids": si, "times": ti} if name == "lstm" else {}
            try:
                sub_scores[name] = np.asarray(
                    RANKERS[name](Xi, yi, cfg, rs, **kwargs), dtype=np.float64).ravel()
            except Exception as exc:
                LOGGER.warning("stability run %d: ranker %s failed (%s)", run, name, exc)
                sub_scores[name] = np.zeros(Xi.shape[1])
        tbl = fuse_rankings(sub_scores, list(Xi.columns), k, eta)
        chosen = tbl.loc[tbl["in_pool"] == 1, "feature"].tolist()[:n_star]
        for f in chosen:
            counts[f] += 1

    out = pd.DataFrame({"feature": list(counts), "runs_selected": list(counts.values())})
    out["stability"] = out["runs_selected"] / max(n_runs, 1)
    return out.sort_values("stability", ascending=False).reset_index(drop=True)


def _save_artifacts(res: FeatureSelectionResult, cfg: Mapping[str, Any],
                    out_dir: Path, tag: str = "") -> None:
    """Persist votes, scores, membership, N* curve, stability and F*."""
    suffix = f"_{tag}" if tag else ""
    save_table(res.votes_table, out_dir / f"feature_votes{suffix}.csv")
    save_table(res.n_star_curve, out_dir / f"n_star_curve{suffix}.csv")
    save_table(res.redundancy_report, out_dir / f"redundancy_dropped{suffix}.csv")
    if res.stability is not None:
        save_table(res.stability, out_dir / f"feature_stability{suffix}.csv")

    per_ranker = pd.DataFrame({"feature": res.f_corr})
    for name, arr in res.ranker_scores.items():
        per_ranker[f"score_{name}"] = arr
    save_table(per_ranker, out_dir / f"ranker_scores{suffix}.csv")

    save_json({
        "f_star": res.f_star,
        "n_star": res.n_star,
        "pool_size": len(res.pool),
        "f_corr_size": len(res.f_corr),
        "top_k": cfg["feature_selection"]["top_k"],
        "vote_threshold": cfg["feature_selection"]["vote_threshold"],
        "fes_sets": {FES_SET_LABELS.get(m, m): topk_set(
            s, res.f_corr, int(cfg["feature_selection"]["top_k"]))
            for m, s in res.ranker_scores.items()},
        "timings_s": res.timings,
    }, out_dir / f"feature_selection{suffix}.json")


# ===========================================================================
# Stage 8.2 ablations and Stage 8.3 sensitivity grids
# ===========================================================================
def _equal_size_baseline(name: str, X: pd.DataFrame, y: np.ndarray, size: int,
                         cfg: Mapping[str, Any], seed: int) -> List[str]:
    """PCC / IG / PCA selectors at the same subset size as F* (Stage 8.2 exp. 9)."""
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import mutual_info_classif

    cols = list(X.columns)
    if name == "pcc":
        # |Pearson| against the one-vs-rest indicator, maxed over classes.
        arr = X.to_numpy(dtype=np.float64)
        best = np.zeros(arr.shape[1])
        for c in np.unique(y):
            target = (y == c).astype(np.float64)
            if target.std() == 0:
                continue
            centred = arr - arr.mean(axis=0)
            denom = arr.std(axis=0) * target.std() * len(target)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.abs((centred * (target - target.mean())[:, None]).sum(axis=0) / denom)
            best = np.maximum(best, np.nan_to_num(r))
        order = np.argsort(-best)
    elif name == "ig":
        # Information gain == mutual information with a discrete estimator.
        idx = stratified_subsample(y, 20000, seed, min_per_class=5)
        scores = mutual_info_classif(X.iloc[idx], y[idx], discrete_features=False,
                                     random_state=seed)
        order = np.argsort(-scores)
    elif name == "pca":
        # PCA is a projection, not a selector; the comparable subset is the
        # features with the largest loadings on the leading components.
        n_comp = min(size, X.shape[1], len(X))
        pca = PCA(n_components=n_comp, random_state=seed).fit(X.to_numpy(dtype=np.float64))
        loading = np.abs(pca.components_).T @ pca.explained_variance_ratio_
        order = np.argsort(-loading)
    else:
        raise ValueError(f"unknown equal-size baseline {name!r}")
    return [cols[i] for i in order[:size]]


def run_feature_ablations(X_train: pd.DataFrame, y_train: np.ndarray,
                          X_val: pd.DataFrame, y_val: np.ndarray,
                          result: FeatureSelectionResult, cfg: Mapping[str, Any],
                          seed: int = 42, out_dir: Optional[str] = None,
                          X_test: Optional[pd.DataFrame] = None,
                          y_test: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Stage 8.2 feature-selection ablations, every subset matched in size to F*.

    Each experiment is scored with the same class-balanced LightGBM probe used
    for N*, so the only thing that varies is which features it receives.
    """
    import lightgbm as lgb
    from sklearn.metrics import f1_score, recall_score

    fs = cfg["feature_selection"]
    experiments = fs["ablation"].get("experiments", [])
    size = len(result.f_star)
    probe = fs["n_star_probe"]
    Xc_train = X_train[result.f_corr]
    Xc_val = X_val[result.f_corr]
    Xc_test = X_test[result.f_corr] if X_test is not None else None
    k = int(fs["top_k"])

    def subsets() -> Dict[str, List[str]]:
        sets: Dict[str, List[str]] = {}
        scores = result.ranker_scores
        feats = result.f_corr
        votes = result.votes_table

        for exp in experiments:
            if exp == "all_features":
                sets["1. all features"] = list(feats)
            elif exp == "mi_only" and "mi" in scores:
                sets["2. MI only"] = topk_set(scores["mi"], feats, size)
            elif exp == "rfi_xgb":
                combo = {m: scores[m] for m in ("rfi", "xgb") if m in scores}
                if combo:
                    tbl = fuse_rankings(combo, feats, k, 1)
                    sets["3. RFI + XGB"] = tbl["feature"].tolist()[:size]
            elif exp == "sd_dmm_mi":
                combo = {m: scores[m] for m in ("sd", "dmm", "mi") if m in scores}
                if combo:
                    tbl = fuse_rankings(combo, feats, k, 1)
                    sets["4. SD + DMM + MI"] = tbl["feature"].tolist()[:size]
            elif exp == "pi_svm_only" and "pi_svm" in scores:
                sets["5. PI-SVM only"] = topk_set(scores["pi_svm"], feats, size)
            elif exp == "vote_only" and len(votes):
                # Frequency vote WITHOUT rank aggregation: order by psi alone.
                pool = votes[votes["in_pool"] == 1].sort_values(
                    "psi_votes", ascending=False)
                sets["6. vote only (psi >= eta)"] = pool["feature"].tolist()[:size]
            elif exp == "vote_rank":
                sets["7. vote + rank (proposed F*)"] = list(result.f_star)
            elif exp == "leave_one_ranker_out":
                for drop in scores:
                    rest = {m: s for m, s in scores.items() if m != drop}
                    tbl = fuse_rankings(rest, feats, k, max(1, int(fs["vote_threshold"]) - 1))
                    pool = tbl[tbl["in_pool"] == 1]["feature"].tolist() or tbl["feature"].tolist()
                    sets[f"8. leave-one-out: no {drop}"] = pool[:size]
            elif exp == "base_paper_six":
                six = {m: s for m, s in scores.items()
                       if m in ("mi", "rfi", "pi_svm", "shap_lgbm", "dmm", "sd")}
                if six:
                    tbl = fuse_rankings(six, feats, k, 4)
                    pool = tbl[tbl["in_pool"] == 1]["feature"].tolist() or tbl["feature"].tolist()
                    sets["9. base-paper 6 rankers"] = pool[:size]
            elif exp in ("pcc_equal_size", "ig_equal_size", "pca_equal_size"):
                which = exp.split("_")[0]
                try:
                    sets[f"9. {which.upper()} (equal size)"] = _equal_size_baseline(
                        which, Xc_train, y_train, size, cfg, seed)
                except Exception as exc:
                    LOGGER.warning("ablation %s failed: %s", exp, exc)
        return sets

    rows: List[Dict[str, Any]] = []
    all_sets = subsets()
    for label, feats_i in progress(list(all_sets.items()), desc="FS ablations",
                                   total=len(all_sets)):
        if not feats_i:
            continue
        tr_idx = stratified_subsample(
            y_train, int(probe.get("max_train_rows", 200000)), seed, min_per_class=5)
        model = lgb.LGBMClassifier(
            n_estimators=int(probe.get("n_estimators", 150)),
            learning_rate=float(probe.get("learning_rate", 0.1)),
            num_leaves=int(probe.get("num_leaves", 31)),
            class_weight=probe.get("class_weight", "balanced"),
            random_state=seed, n_jobs=-1, verbose=-1,
        ).fit(Xc_train.iloc[tr_idx][feats_i], y_train[tr_idx])

        def score_on(X, y):
            if X is None or y is None:
                return {}
            pred = model.predict(X[feats_i])
            minority = [c for c in np.unique(y_train)
                        if (y_train == c).sum() < 0.05 * len(y_train)]
            return {
                "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
                "minority_recall": (recall_score(y, pred, labels=minority,
                                                 average="macro", zero_division=0)
                                    if minority else float("nan")),
            }

        row: Dict[str, Any] = {"experiment": label, "n_features": len(feats_i)}
        row.update({f"val_{k2}": v for k2, v in score_on(Xc_val, y_val).items()})
        row.update({f"test_{k2}": v for k2, v in score_on(Xc_test, y_test).items()})
        row["features"] = ", ".join(feats_i[:10]) + ("..." if len(feats_i) > 10 else "")
        rows.append(row)

    table = pd.DataFrame(rows)
    if len(table):
        table = table.sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
        if out_dir:
            save_table(table, Path(out_dir) / "feature_selection_ablations.csv")
    LOGGER.info("Stage 8.2: ran %d feature-selection ablation(s)", len(table))
    return table


def run_sensitivity(X_train: pd.DataFrame, y_train: np.ndarray,
                    X_val: pd.DataFrame, y_val: np.ndarray,
                    result: FeatureSelectionResult, cfg: Mapping[str, Any],
                    seed: int = 42, out_dir: Optional[str] = None) -> pd.DataFrame:
    """Stage 8.3 sensitivity grids over N, the vote threshold eta and the list size K."""
    import lightgbm as lgb
    from sklearn.metrics import f1_score

    fs = cfg["feature_selection"]
    sens = fs["sensitivity"]
    probe = fs["n_star_probe"]
    Xc_train, Xc_val = X_train[result.f_corr], X_val[result.f_corr]
    rows: List[Dict[str, Any]] = []

    def evaluate(feats: Sequence[str]) -> float:
        if not feats:
            return float("nan")
        idx = stratified_subsample(y_train, int(probe.get("max_train_rows", 200000)),
                                   seed, min_per_class=5)
        m = lgb.LGBMClassifier(
            n_estimators=int(probe.get("n_estimators", 150)),
            learning_rate=float(probe.get("learning_rate", 0.1)),
            class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1,
        ).fit(Xc_train.iloc[idx][list(feats)], y_train[idx])
        return float(f1_score(y_val, m.predict(Xc_val[list(feats)]),
                              average="macro", zero_division=0))

    combos = [(k, eta, n)
              for k in sens.get("k_values", [fs["top_k"]])
              for eta in sens.get("eta_values", [fs["vote_threshold"]])
              for n in sens.get("n_values", fs["n_grid"])]
    for k, eta, n in progress(combos, desc="sensitivity", total=len(combos)):
        tbl = fuse_rankings(result.ranker_scores, result.f_corr, int(k), int(eta))
        pool = tbl[tbl["in_pool"] == 1]["feature"].tolist()
        if len(pool) < n:
            rows.append({"K": k, "eta": eta, "N": n, "pool_size": len(pool),
                         "val_macro_f1": float("nan"),
                         "note": "pool smaller than N"})
            continue
        rows.append({"K": k, "eta": eta, "N": n, "pool_size": len(pool),
                     "val_macro_f1": evaluate(pool[:n]), "note": ""})

    table = pd.DataFrame(rows)
    if out_dir and len(table):
        save_table(table, Path(out_dir) / "feature_selection_sensitivity.csv")
    LOGGER.info("Stage 8.3: evaluated %d (K, eta, N) combination(s)", len(table))
    return table
