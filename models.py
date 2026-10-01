"""Stage 6: the five heterogeneous classifiers (RF, XGBoost, LSTM, 1-D CNN, GNN).

Every model implements the same interface::

    model.fit(X_tr, y_tr, X_val, y_val, **context)
    model.predict_proba(X, **context) -> (n, C) aligned to ``cfg.labels.class_order``
    model.save(path) / Model.load(path)

The probability matrices of all five models share one column order (the
canonical class order from the config), which is what makes the Stage 7 fusion
well defined. A model that never saw a class still emits a column for it,
filled with zeros and renormalised.

Sequence/graph context (session ids, timestamps) is passed through ``**context``
rather than hidden in the feature matrix, because identifiers must never become
features (Stage 1.4).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from feature_selection import build_session_windows
from imbalance_handling import sample_weights_from_class_weights
from utils import (
    Timer,
    assert_probabilities_valid,
    chunked,
    ensure_dir,
    get_logger,
    memory_mb,
)

LOGGER = get_logger()


# ===========================================================================
# Base class
# ===========================================================================
@dataclass
class ModelStats:
    """Training cost and inference cost, reported in Stage 8."""

    train_time_s: float = 0.0
    inference_time_s: float = 0.0
    inference_ms_per_flow: float = 0.0
    n_parameters: int = 0
    peak_memory_mb: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    """Common interface for the five Stage 6 classifiers."""

    name: str = "base"

    def __init__(self, cfg: Mapping[str, Any], classes: Sequence[Any], seed: int = 42):
        self.cfg = cfg
        self.classes = list(classes)          # canonical, fixed order
        self.seed = int(seed)
        self.stats = ModelStats()
        self._class_index = {c: i for i, c in enumerate(self.classes)}

    # -- helpers -----------------------------------------------------------
    def _align(self, proba: np.ndarray, model_classes: Sequence[Any]) -> np.ndarray:
        """Map a model's own class order onto the canonical order.

        Classes the model never saw get a zero column; rows are renormalised so
        every row still sums to one.
        """
        out = np.zeros((proba.shape[0], len(self.classes)), dtype=np.float64)
        for j, c in enumerate(model_classes):
            if c in self._class_index:
                out[:, self._class_index[c]] = proba[:, j]
        total = out.sum(axis=1, keepdims=True)
        bad = (total <= 0).ravel()
        if bad.any():                          # degenerate row -> uniform
            out[bad] = 1.0 / len(self.classes)
            total = out.sum(axis=1, keepdims=True)
        return out / total

    def _finish_predict(self, proba: np.ndarray, n_rows: int, elapsed: float) -> np.ndarray:
        self.stats.inference_time_s = elapsed
        self.stats.inference_ms_per_flow = 1000.0 * elapsed / max(n_rows, 1)
        assert_probabilities_valid(proba, f"{self.name} predict_proba")
        return proba

    # -- interface ---------------------------------------------------------
    @abstractmethod
    def fit(self, X_tr: pd.DataFrame, y_tr: np.ndarray,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None,
            class_weights: Optional[Mapping[int, float]] = None, **context: Any) -> "BaseModel":
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame, **context: Any) -> np.ndarray:
        ...

    def predict(self, X: pd.DataFrame, **context: Any) -> np.ndarray:
        proba = self.predict_proba(X, **context)
        return np.array([self.classes[i] for i in np.argmax(proba, axis=1)])

    @abstractmethod
    def save(self, path: str | os.PathLike) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | os.PathLike, cfg: Mapping[str, Any],
             classes: Sequence[Any]) -> "BaseModel":
        ...


# ===========================================================================
# Random Forest (Stage 6.3)
# ===========================================================================
class RandomForestModel(BaseModel):
    """200 trees, max_features=sqrt(k), gini, class weights from Stage 5."""

    name = "rf"

    def fit(self, X_tr, y_tr, X_val=None, y_val=None, class_weights=None, **context):
        from sklearn.ensemble import RandomForestClassifier

        p = self.cfg["models"]["rf"]
        weight_arg = ({int(k): float(v) for k, v in class_weights.items()}
                      if class_weights else None)
        self.model = RandomForestClassifier(
            n_estimators=int(p.get("n_estimators", 200)),
            max_features=p.get("max_features", "sqrt"),
            criterion=str(p.get("criterion", "gini")),
            max_depth=p.get("max_depth"),
            min_samples_leaf=int(p.get("min_samples_leaf", 1)),
            class_weight=weight_arg,
            n_jobs=int(p.get("n_jobs", -1)),
            random_state=self.seed,
        )
        with Timer("fit_rf") as t:
            self.model.fit(X_tr.to_numpy(dtype=np.float32), y_tr)
        self.stats.train_time_s = t.elapsed
        self.stats.n_parameters = int(sum(e.tree_.node_count for e in self.model.estimators_))
        self.stats.peak_memory_mb = memory_mb()
        self.feature_names_ = list(X_tr.columns)
        return self

    def predict_proba(self, X, **context):
        chunk = int(self.cfg["models"].get("predict_chunk_size", 200000))
        parts = []
        with Timer("predict_rf") as t:
            for a, b in chunked(len(X), chunk):
                parts.append(self.model.predict_proba(X.iloc[a:b].to_numpy(dtype=np.float32)))
        proba = self._align(np.vstack(parts), list(self.model.classes_))
        return self._finish_predict(proba, len(X), t.elapsed)

    def save(self, path):
        import joblib

        ensure_dir(Path(path).parent)
        joblib.dump({"model": self.model, "features": self.feature_names_,
                     "stats": self.stats}, path, compress=3)

    @classmethod
    def load(cls, path, cfg, classes):
        import joblib

        obj = cls(cfg, classes)
        blob = joblib.load(path)
        obj.model, obj.feature_names_, obj.stats = blob["model"], blob["features"], blob["stats"]
        return obj


# ===========================================================================
# XGBoost (Stage 6.4)
# ===========================================================================
class XGBoostModel(BaseModel):
    """multi:softprob with L1/L2 regularisation and validation early stopping."""

    name = "xgb"

    def fit(self, X_tr, y_tr, X_val=None, y_val=None, class_weights=None, **context):
        import xgboost as xgb

        p = self.cfg["models"]["xgb"]
        self._seen = np.unique(y_tr)
        self._remap = {c: i for i, c in enumerate(self._seen)}
        y_mapped = np.array([self._remap[v] for v in y_tr])

        kwargs: Dict[str, Any] = dict(
            n_estimators=int(p.get("n_estimators", 300)),
            max_depth=int(p.get("max_depth", 6)),
            learning_rate=float(p.get("learning_rate", 0.1)),
            subsample=float(p.get("subsample", 0.9)),
            colsample_bytree=float(p.get("colsample_bytree", 0.9)),
            reg_lambda=float(p.get("reg_lambda", 1.0)),
            reg_alpha=float(p.get("reg_alpha", 0.0)),
            tree_method=str(p.get("tree_method", "hist")),
            random_state=self.seed, n_jobs=-1, verbosity=0,
        )
        if len(self._seen) > 2:
            kwargs.update(objective=str(p.get("objective", "multi:softprob")),
                          num_class=len(self._seen))
        else:
            kwargs.update(objective="binary:logistic")

        # Early stopping on VALIDATION only (Stage 6.10).
        eval_set = None
        if X_val is not None and y_val is not None:
            keep = np.isin(y_val, self._seen)
            if keep.sum() > 0:
                eval_set = [(X_val[keep].to_numpy(dtype=np.float32),
                             np.array([self._remap[v] for v in np.asarray(y_val)[keep]]))]
                kwargs["early_stopping_rounds"] = int(p.get("early_stopping_rounds", 20))

        self.model = xgb.XGBClassifier(**kwargs)
        weights = (sample_weights_from_class_weights(y_tr, class_weights)
                   if class_weights else None)
        with Timer("fit_xgb") as t:
            self.model.fit(X_tr.to_numpy(dtype=np.float32), y_mapped,
                           sample_weight=weights, eval_set=eval_set, verbose=False)
        self.stats.train_time_s = t.elapsed
        self.stats.n_parameters = int(self.model.get_booster().num_boosted_rounds()
                                      * max(len(self._seen), 1))
        self.stats.peak_memory_mb = memory_mb()
        self.feature_names_ = list(X_tr.columns)
        return self

    def predict_proba(self, X, **context):
        chunk = int(self.cfg["models"].get("predict_chunk_size", 200000))
        parts = []
        with Timer("predict_xgb") as t:
            for a, b in chunked(len(X), chunk):
                pr = self.model.predict_proba(X.iloc[a:b].to_numpy(dtype=np.float32))
                if pr.ndim == 1 or pr.shape[1] == 1:       # binary edge case
                    pr = np.column_stack([1 - pr.ravel(), pr.ravel()])
                parts.append(pr)
        proba = self._align(np.vstack(parts), list(self._seen))
        return self._finish_predict(proba, len(X), t.elapsed)

    def save(self, path):
        import joblib

        ensure_dir(Path(path).parent)
        joblib.dump({"model": self.model, "seen": self._seen,
                     "features": self.feature_names_, "stats": self.stats}, path, compress=3)

    @classmethod
    def load(cls, path, cfg, classes):
        import joblib

        obj = cls(cfg, classes)
        blob = joblib.load(path)
        obj.model, obj._seen = blob["model"], blob["seen"]
        obj._remap = {c: i for i, c in enumerate(obj._seen)}
        obj.feature_names_, obj.stats = blob["features"], blob["stats"]
        return obj


# ===========================================================================
# Keras helpers shared by the LSTM and the CNN
# ===========================================================================
def _keras_class_weight(y: np.ndarray, n_classes: int) -> Dict[int, float]:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    present = (counts > 0).sum()
    return {i: (len(y) / (present * c) if c > 0 else 0.0) for i, c in enumerate(counts)}



class KerasModelIO:
    """Save/load for the Keras-backed models.

    This is a mixin rather than an assignment such as ``save = LSTMModel.save``:
    a classmethod accessed through another class is still bound to the class
    that defined it, so ``CNN1DModel.load = LSTMModel.load`` would rebuild an
    ``LSTMModel`` from a CNN checkpoint and then feed it sequence-shaped input.
    Inheriting keeps ``cls`` pointing at the real class.
    """

    def save(self, path: str | os.PathLike) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        self.model.save(path.with_suffix(".keras"))
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as fh:
            json.dump({"features": self.feature_names_,
                       "seen_classes": [int(c) for c in self._seen],
                       "model": self.name,
                       "train_time_s": self.stats.train_time_s,
                       "n_parameters": self.stats.n_parameters}, fh)

    @classmethod
    def load(cls, path: str | os.PathLike, cfg: Mapping[str, Any],
             classes: Sequence[Any]) -> "BaseModel":
        import tensorflow as tf

        obj = cls(cfg, classes)
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        saved_as = meta.get("model")
        if saved_as and saved_as != cls.name:
            raise ValueError(
                f"{path} holds a {saved_as!r} model but {cls.__name__} "
                f"({cls.name!r}) was asked to load it")
        obj.model = tf.keras.models.load_model(path.with_suffix(".keras"))
        obj.feature_names_ = meta["features"]
        obj._seen = np.array(meta.get("seen_classes", list(classes)))
        obj.stats.train_time_s = meta.get("train_time_s", 0.0)
        obj.stats.n_parameters = meta.get("n_parameters", 0)
        return obj


# ===========================================================================
# LSTM (Stage 6.5)
# ===========================================================================
class LSTMModel(KerasModelIO, BaseModel):
    """Sequence model over windows of L consecutive flows of one session.

    Input is ``(L, k)`` -- up to ``L`` consecutive flows of the session ending
    at the flow being classified. Shorter histories are left-padded with zeros
    and ignored via a ``Masking`` layer. One window is produced per flow, so the
    output aligns one-to-one with the flow-level models.
    """

    name = "lstm"

    def _build(self, n_features: int, n_classes: int):
        import tensorflow as tf

        p = self.cfg["models"]["lstm"]
        tf.keras.utils.set_random_seed(self.seed)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(int(p.get("window", 10)), n_features)),
            tf.keras.layers.Masking(mask_value=float(p.get("masking_value", 0.0))),
            tf.keras.layers.LSTM(int(p.get("units", 64))),
            tf.keras.layers.Dropout(float(p.get("dropout", 0.3))),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(float(p.get("learning_rate", 1e-3))),
            loss="sparse_categorical_crossentropy", metrics=["accuracy"],
        )
        return model

    def fit(self, X_tr, y_tr, X_val=None, y_val=None, class_weights=None,
            session_ids=None, times=None, val_session_ids=None, val_times=None, **context):
        import tensorflow as tf

        p = self.cfg["models"]["lstm"]
        L = int(p.get("window", 10))
        self.feature_names_ = list(X_tr.columns)

        Xw, yw = build_session_windows(
            X_tr.to_numpy(dtype=np.float32), np.asarray(y_tr), session_ids, times, L)
        # Output units cover only the classes actually present in training, so a
        # class the model never saw receives exactly zero probability rather than
        # a share of the softmax. _align then places them in the canonical order.
        # This matters for branch E, where the held-out family is absent by
        # design, and it makes the neural models behave like the tree models.
        self._seen = np.unique(y_tr)
        seen_index = {c: i for i, c in enumerate(self._seen)}
        n_out = len(self._seen)
        y_idx = np.array([seen_index.get(v, 0) for v in yw], dtype=np.int32)

        val_data = None
        if X_val is not None and y_val is not None:
            Xv, yv = build_session_windows(
                X_val.to_numpy(dtype=np.float32), np.asarray(y_val),
                val_session_ids, val_times, L)
            keep = np.isin(yv, self._seen)
            if len(Xv) and keep.any():
                val_data = (Xv[keep], np.array([seen_index[v] for v in yv[keep]],
                                               dtype=np.int32))

        self.model = self._build(X_tr.shape[1], n_out)
        weights = ({int(seen_index[k]): float(v) for k, v in class_weights.items()
                    if k in seen_index} if class_weights
                   else _keras_class_weight(y_idx, n_out))
        callbacks = []
        if val_data is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=int(p.get("patience", 4)),
                restore_best_weights=True))

        with Timer("fit_lstm") as t:
            self.history_ = self.model.fit(
                Xw, y_idx, validation_data=val_data,
                epochs=int(p.get("epochs", 20)),
                batch_size=int(p.get("batch_size", 256)),
                class_weight=weights, callbacks=callbacks, verbose=0,
            ).history
        self.stats.train_time_s = t.elapsed
        self.stats.n_parameters = int(self.model.count_params())
        self.stats.peak_memory_mb = memory_mb()
        return self

    def predict_proba(self, X, session_ids=None, times=None, **context):
        L = int(self.cfg["models"]["lstm"].get("window", 10))
        n = len(X)
        Xw, _dummy, rows = build_session_windows(
            X.to_numpy(dtype=np.float32), np.zeros(n, dtype=np.int64),
            session_ids, times, L, return_index=True)
        chunk = int(self.cfg["models"].get("predict_chunk_size", 200000))
        raw = np.full((n, len(self._seen)), 1.0 / len(self._seen), dtype=np.float64)
        with Timer("predict_lstm") as t:
            for a, b in chunked(len(Xw), chunk):
                raw[rows[a:b]] = self.model.predict(Xw[a:b], verbose=0)
        return self._finish_predict(self._align(raw, list(self._seen)), n, t.elapsed)


# ===========================================================================
# 1-D CNN (Stage 6.6)
# ===========================================================================
class CNN1DModel(KerasModelIO, BaseModel):
    """Exactly the Stage 6.6 architecture table.

    Conv1D(64, k=3, same) -> ReLU -> BatchNorm -> MaxPool(2) ->
    Conv1D(128, k=3)      -> ReLU -> MaxPool(2) ->
    Flatten -> Dense(128) -> Dropout(0.3) -> Softmax(C)

    Features arrive already ordered by descending fusion Score (eq. 19); the
    caller passes ``X`` with that column order, which this model records and
    re-imposes at predict time.
    """

    name = "cnn"

    def _build(self, k: int, n_classes: int):
        import tensorflow as tf

        p = self.cfg["models"]["cnn"]
        tf.keras.utils.set_random_seed(self.seed)
        layers = [tf.keras.layers.Input(shape=(k, 1))]
        layers += [
            tf.keras.layers.Conv1D(int(p.get("conv1_filters", 64)),
                                   int(p.get("conv1_kernel", 3)),
                                   padding=str(p.get("conv1_padding", "same"))),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.BatchNormalization(
                momentum=float(p.get("batchnorm_momentum", 0.9))),
        ]
        if k >= int(p.get("pool1", 2)):
            layers.append(tf.keras.layers.MaxPooling1D(int(p.get("pool1", 2))))
        layers += [
            tf.keras.layers.Conv1D(int(p.get("conv2_filters", 128)),
                                   int(p.get("conv2_kernel", 3)),
                                   padding=str(p.get("conv2_padding", "valid"))),
            tf.keras.layers.Activation("relu"),
        ]
        # Pooled lengths are rounded down; skip a pool that would empty the map.
        model = tf.keras.Sequential(layers)
        if model.output_shape[1] and model.output_shape[1] >= int(p.get("pool2", 2)):
            model.add(tf.keras.layers.MaxPooling1D(int(p.get("pool2", 2))))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(int(p.get("dense_units", 128))))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.Dropout(float(p.get("dropout", 0.3))))
        model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(float(p.get("learning_rate", 1e-3))),
            loss="sparse_categorical_crossentropy", metrics=["accuracy"],
        )
        return model

    def fit(self, X_tr, y_tr, X_val=None, y_val=None, class_weights=None, **context):
        import tensorflow as tf

        p = self.cfg["models"]["cnn"]
        self.feature_names_ = list(X_tr.columns)
        # Output units cover only the classes present in training (see LSTMModel).
        self._seen = np.unique(y_tr)
        seen_index = {c: i for i, c in enumerate(self._seen)}
        n_out = len(self._seen)
        y_idx = np.array([seen_index.get(v, 0) for v in np.asarray(y_tr)], dtype=np.int32)
        Xa = X_tr.to_numpy(dtype=np.float32)[..., None]

        val_data = None
        if X_val is not None and y_val is not None and len(X_val):
            yv = np.asarray(y_val)
            keep = np.isin(yv, self._seen)
            if keep.any():
                val_data = (X_val[self.feature_names_].to_numpy(
                                dtype=np.float32)[keep][..., None],
                            np.array([seen_index[v] for v in yv[keep]], dtype=np.int32))

        self.model = self._build(X_tr.shape[1], n_out)
        weights = ({int(seen_index[k]): float(v) for k, v in class_weights.items()
                    if k in seen_index} if class_weights
                   else _keras_class_weight(y_idx, n_out))
        callbacks = []
        if val_data is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=int(p.get("patience", 4)),
                restore_best_weights=True))

        with Timer("fit_cnn") as t:
            self.history_ = self.model.fit(
                Xa, y_idx, validation_data=val_data,
                epochs=int(p.get("epochs", 20)),
                batch_size=int(p.get("batch_size", 256)),
                class_weight=weights, callbacks=callbacks, verbose=0,
            ).history
        self.stats.train_time_s = t.elapsed
        self.stats.n_parameters = int(self.model.count_params())
        self.stats.peak_memory_mb = memory_mb()
        return self

    def predict_proba(self, X, **context):
        X = X[self.feature_names_]            # re-impose the Score ordering
        chunk = int(self.cfg["models"].get("predict_chunk_size", 200000))
        parts = []
        with Timer("predict_cnn") as t:
            for a, b in chunked(len(X), chunk):
                arr = X.iloc[a:b].to_numpy(dtype=np.float32)[..., None]
                parts.append(self.model.predict(arr, verbose=0))
        proba = self._align(np.vstack(parts).astype(np.float64), list(self._seen))
        return self._finish_predict(proba, len(X), t.elapsed)


# ===========================================================================
# Temporal GNN (Stage 6.7)
# ===========================================================================
def build_session_graph(session_ids: np.ndarray, times: np.ndarray,
                        hosts: Optional[np.ndarray], X: np.ndarray,
                        y: Optional[np.ndarray], cfg: Mapping[str, Any]
                        ) -> Dict[str, Any]:
    """Build a time-causal communication graph whose nodes are sessions.

    Node choice (README "Deviations / Assumptions"): the methodology allows
    "hosts or endpoints" as nodes. We use **sessions** as nodes, because the
    label and the selected features F* are defined per flow and aggregate
    naturally to a session, whereas a host carries many unrelated labels over
    a capture. Node attributes are the mean of F* over the session's flows;
    the node label is the session's primary (rarest) class.

    Edges connect two sessions that share a host endpoint, directed from the
    earlier to the later session -- so information only ever flows forward in
    time ("time-causal"). Each node keeps at most ``max_in_degree`` of its most
    recent predecessors, which bounds the edge count on 2.8M flows.

    The graph is built from the rows of ONE split, so no edge can cross a split
    boundary.
    """
    p = cfg["models"]["gnn"]
    max_in = int(p.get("max_in_degree", 10))
    max_edges = int(p.get("max_edges", 2_000_000))
    max_gap = p.get("edge_max_time_gap_s")
    max_gap = float(max_gap) if max_gap else None

    order = np.argsort(times, kind="mergesort")
    sess_of_row = session_ids
    uniq, first = np.unique(sess_of_row[order], return_index=True)
    # Node id in time order of first appearance.
    node_order = uniq[np.argsort(first)]
    node_index = {s: i for i, s in enumerate(node_order)}
    n_nodes = len(node_order)

    # -- node features: mean of F* over the session's flows ---------------
    feat = np.zeros((n_nodes, X.shape[1]), dtype=np.float32)
    counts = np.zeros(n_nodes, dtype=np.float64)
    for row in range(len(X)):
        i = node_index[sess_of_row[row]]
        feat[i] += X[row]
        counts[i] += 1
    feat /= np.maximum(counts, 1)[:, None]

    # -- node label: MAJORITY class of the session ------------------------
    # The node's prediction is broadcast back to every flow of the session, so
    # the training target must be what most of those flows actually are. (The
    # *rarest* class is used elsewhere, to decide which side of a split a
    # session falls on, where the goal is the opposite: not to bury a rare
    # class inside a mostly-benign session.)
    node_y = None
    if y is not None:
        tally: Dict[int, Counter] = defaultdict(Counter)
        for row in range(len(y)):
            tally[node_index[sess_of_row[row]]][y[row]] += 1
        node_y = np.array([tally[i].most_common(1)[0][0] if tally[i] else 0
                           for i in range(n_nodes)])

    # -- node time (first flow) -------------------------------------------
    node_time = np.full(n_nodes, np.inf)
    t_num = pd.to_datetime(pd.Series(times)).astype("int64").to_numpy() / 1e9
    for row in range(len(t_num)):
        i = node_index[sess_of_row[row]]
        node_time[i] = min(node_time[i], t_num[row])

    # -- edges: share a host, earlier -> later ----------------------------
    src: List[int] = []
    dst: List[int] = []
    if hosts is not None:
        by_host: Dict[Any, List[int]] = {}
        for row in range(len(hosts)):
            i = node_index[sess_of_row[row]]
            by_host.setdefault(hosts[row], []).append(i)
        for _h, nodes in by_host.items():
            nodes = sorted(set(nodes), key=lambda i: node_time[i])
            for pos, j in enumerate(nodes):
                lo = max(0, pos - max_in)
                for i in nodes[lo:pos]:
                    if max_gap is not None and (node_time[j] - node_time[i]) > max_gap:
                        continue            # too far apart in time to be related
                    src.append(i)
                    dst.append(j)
                    if len(src) >= max_edges:
                        break
                if len(src) >= max_edges:
                    break
            if len(src) >= max_edges:
                LOGGER.warning("GNN: edge budget %d reached; graph truncated", max_edges)
                break

    edge_index = (np.array([src, dst], dtype=np.int64) if src
                  else np.zeros((2, 0), dtype=np.int64))
    return {"x": feat, "y": node_y, "edge_index": edge_index,
            "node_index": node_index, "n_nodes": n_nodes,
            "row_to_node": np.array([node_index[s] for s in sess_of_row], dtype=np.int64)}


class GNNModel(BaseModel):
    """2-layer GCN over the time-causal session graph (eq. 23).

    Uses ``torch_geometric.nn.GCNConv`` when available, and otherwise an
    equivalent dense implementation of
    ``H' = sigma(D^-1/2 (A + I) D^-1/2 H W)`` in plain torch (the fallback is
    recorded in the run report).

    Predictions are produced per *session node* and then broadcast to every flow
    of that session, so the output aligns with the flow-level models.
    """

    name = "gnn"

    def __init__(self, cfg, classes, seed: int = 42):
        super().__init__(cfg, classes, seed)
        self.use_pyg = False

    def _build(self, n_features: int, n_classes: int):
        import torch
        import torch.nn as nn

        p = self.cfg["models"]["gnn"]
        hidden = int(p.get("hidden_dim", 64))
        dropout = float(p.get("dropout", 0.3))
        residual = bool(p.get("residual", True))
        torch.manual_seed(self.seed)

        try:
            from torch_geometric.nn import GCNConv

            class PyGGCN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.c1 = GCNConv(n_features, hidden)
                    self.c2 = GCNConv(hidden, n_classes)
                    self.drop = nn.Dropout(dropout)
                    self.skip = nn.Linear(n_features, n_classes) if residual else None

                def forward(self, x, edge_index, adj=None):
                    h = torch.relu(self.c1(x, edge_index))
                    out = self.c2(self.drop(h), edge_index)
                    return out + self.skip(x) if self.skip is not None else out

            self.use_pyg = True
            return PyGGCN()
        except Exception as exc:
            if not p.get("allow_torch_fallback", True):
                raise
            LOGGER.warning("torch_geometric unavailable (%s); using the dense "
                           "normalised-adjacency GCN fallback", exc)

        class DenseGCN(nn.Module):
            """H' = sigma(D^-1/2 (A+I) D^-1/2 H W), eq. 23, as a sparse matmul."""

            def __init__(self):
                super().__init__()
                self.w1 = nn.Linear(n_features, hidden)
                self.w2 = nn.Linear(hidden, n_classes)
                self.drop = nn.Dropout(dropout)
                self.skip = nn.Linear(n_features, n_classes) if residual else None

            def forward(self, x, edge_index=None, adj=None):
                h = torch.relu(torch.sparse.mm(adj, self.w1(x)))
                out = self.w2(torch.sparse.mm(adj, self.drop(h)))
                return out + self.skip(x) if self.skip is not None else out

        self.use_pyg = False
        return DenseGCN()

    @staticmethod
    def _normalised_adj(edge_index: np.ndarray, n: int):
        """Sparse D^-1/2 (A + I) D^-1/2 for the fallback path."""
        import torch

        src = np.concatenate([edge_index[0], edge_index[1], np.arange(n)])
        dst = np.concatenate([edge_index[1], edge_index[0], np.arange(n)])
        deg = np.bincount(dst, minlength=n).astype(np.float64)
        norm = 1.0 / np.sqrt(np.maximum(deg[src], 1) * np.maximum(deg[dst], 1))
        idx = torch.tensor(np.vstack([dst, src]), dtype=torch.long)
        val = torch.tensor(norm, dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()

    def fit(self, X_tr, y_tr, X_val=None, y_val=None, class_weights=None,
            session_ids=None, times=None, hosts=None,
            val_session_ids=None, val_times=None, val_hosts=None, **context):
        import torch
        import torch.nn.functional as F

        p = self.cfg["models"]["gnn"]
        self.feature_names_ = list(X_tr.columns)
        graph = build_session_graph(session_ids, times, hosts,
                                    X_tr.to_numpy(dtype=np.float32), np.asarray(y_tr),
                                    self.cfg)
        # Output units cover only the classes present among the NODE labels
        # (see LSTMModel); _align places them in the canonical order.
        self._seen = np.unique(graph["y"])
        seen_index = {c: i for i, c in enumerate(self._seen)}
        self.model = self._build(X_tr.shape[1], len(self._seen))

        x = torch.tensor(graph["x"], dtype=torch.float32)
        ei = torch.tensor(graph["edge_index"], dtype=torch.long)
        adj = self._normalised_adj(graph["edge_index"], graph["n_nodes"])
        y_node = torch.tensor([seen_index.get(v, 0) for v in graph["y"]],
                              dtype=torch.long)

        # Class weights must be computed at the granularity of the LOSS. This
        # loss is over session NODES, whose class distribution differs from the
        # flow distribution the caller's weights were derived from (a weight of
        # N_flows/(C*N_c_flows) applied to node targets biases the model toward
        # rare classes). So when weighting is requested, re-derive w_c =
        # N_nodes/(C*N_c_nodes) from the node labels.
        if class_weights:
            counts = torch.bincount(y_node, minlength=len(self._seen)).double()
            present = max(int((counts > 0).sum()), 1)
            w = torch.tensor(
                [(len(y_node) / (present * c) if c > 0 else 0.0) for c in counts],
                dtype=torch.float32)
        else:
            w = None

        opt = torch.optim.Adam(self.model.parameters(),
                               lr=float(p.get("learning_rate", 0.01)),
                               weight_decay=float(p.get("weight_decay", 5e-4)))
        self.history_ = {"loss": []}
        with Timer("fit_gnn") as t:
            self.model.train()
            for _epoch in range(int(p.get("epochs", 30))):
                opt.zero_grad()
                out = self.model(x, ei, adj)
                loss = F.cross_entropy(out, y_node, weight=w)
                loss.backward()
                opt.step()
                self.history_["loss"].append(float(loss.item()))
        self.stats.train_time_s = t.elapsed
        self.stats.n_parameters = int(sum(q.numel() for q in self.model.parameters()))
        self.stats.peak_memory_mb = memory_mb()
        self.stats.extra["backend"] = "torch_geometric" if self.use_pyg else "dense_torch_fallback"
        self.stats.extra["n_nodes"] = graph["n_nodes"]
        self.stats.extra["n_edges"] = int(graph["edge_index"].shape[1])
        return self

    def predict_proba(self, X, session_ids=None, times=None, hosts=None, **context):
        import torch

        X = X[self.feature_names_]
        graph = build_session_graph(session_ids, times, hosts,
                                    X.to_numpy(dtype=np.float32), None, self.cfg)
        x = torch.tensor(graph["x"], dtype=torch.float32)
        ei = torch.tensor(graph["edge_index"], dtype=torch.long)
        adj = self._normalised_adj(graph["edge_index"], graph["n_nodes"])
        with Timer("predict_gnn") as t:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(x, ei, adj)
            node_proba = torch.softmax(logits, dim=-1).numpy().astype(np.float64)
        # Broadcast each session node's distribution to its flows.
        proba = self._align(node_proba[graph["row_to_node"]], list(self._seen))
        return self._finish_predict(proba, len(X), t.elapsed)

    def save(self, path):
        import torch

        path = Path(path)
        ensure_dir(path.parent)
        torch.save({"state_dict": self.model.state_dict(),
                    "features": self.feature_names_,
                    "n_features": len(self.feature_names_),
                    "seen_classes": [int(c) for c in self._seen],
                    "use_pyg": self.use_pyg,
                    "stats_extra": self.stats.extra}, path.with_suffix(".pt"))

    @classmethod
    def load(cls, path, cfg, classes):
        import torch

        obj = cls(cfg, classes)
        blob = torch.load(Path(path).with_suffix(".pt"), weights_only=False)
        obj.feature_names_ = blob["features"]
        obj._seen = np.array(blob.get("seen_classes", list(classes)))
        obj.model = obj._build(blob["n_features"], len(obj._seen))
        obj.model.load_state_dict(blob["state_dict"])
        obj.stats.extra = blob.get("stats_extra", {})
        return obj


# ===========================================================================
# Registry & orchestration
# ===========================================================================
MODEL_REGISTRY: Dict[str, type] = {
    "rf": RandomForestModel,
    "xgb": XGBoostModel,
    "lstm": LSTMModel,
    "cnn": CNN1DModel,
    "gnn": GNNModel,
}


def build_model(name: str, cfg: Mapping[str, Any], classes: Sequence[Any],
                seed: int = 42) -> BaseModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](cfg, classes, seed)


def train_all_models(
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
    cfg: Mapping[str, Any], classes: Sequence[Any], seed: int = 42,
    class_weights: Optional[Mapping[int, float]] = None,
    train_context: Optional[Mapping[str, Any]] = None,
    val_context: Optional[Mapping[str, Any]] = None,
    model_dir: Optional[str] = None,
) -> Dict[str, BaseModel]:
    """Fit every enabled model and return them keyed by name."""
    train_context = dict(train_context or {})
    val_context = dict(val_context or {})
    models: Dict[str, BaseModel] = {}

    for name in cfg["models"]["enabled"]:
        LOGGER.info("Stage 6: training %s ...", name.upper())
        model = build_model(name, cfg, classes, seed)
        ctx = dict(train_context)
        ctx.update({f"val_{k}": v for k, v in val_context.items()})
        try:
            model.fit(X_tr, y_tr, X_val, y_val, class_weights=class_weights, **ctx)
        except Exception as exc:
            LOGGER.error("Stage 6: %s failed to train (%s); skipping", name, exc)
            continue
        models[name] = model
        LOGGER.info("  %s trained in %.1fs (%d params)",
                    name, model.stats.train_time_s, model.stats.n_parameters)
        if model_dir:
            try:
                model.save(Path(model_dir) / f"{name}_seed{seed}")
            except Exception as exc:
                LOGGER.warning("  could not save %s: %s", name, exc)
    return models


def predict_all(models: Mapping[str, BaseModel], X: pd.DataFrame,
                context: Optional[Mapping[str, Any]] = None) -> Dict[str, np.ndarray]:
    """Probability matrix per model, all sharing the canonical class order."""
    context = dict(context or {})
    out: Dict[str, np.ndarray] = {}
    for name, model in models.items():
        out[name] = model.predict_proba(X, **context)
    return out


def model_stats_table(models: Mapping[str, BaseModel]) -> pd.DataFrame:
    """Training time, inference latency, parameter count and memory per model."""
    rows = []
    for name, m in models.items():
        rows.append({
            "model": name,
            "train_time_s": round(m.stats.train_time_s, 3),
            "inference_time_s": round(m.stats.inference_time_s, 4),
            "inference_ms_per_flow": round(m.stats.inference_ms_per_flow, 5),
            "n_parameters": m.stats.n_parameters,
            "peak_memory_mb": round(m.stats.peak_memory_mb, 1),
            **{k: v for k, v in m.stats.extra.items()},
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Optional hyperparameter search (VALIDATION only, Stage 6.10)
# ===========================================================================
def tune_hyperparameters(X_tr: pd.DataFrame, y_tr: np.ndarray,
                         X_val: pd.DataFrame, y_val: np.ndarray,
                         cfg: Mapping[str, Any], classes: Sequence[Any],
                         seed: int = 42,
                         class_weights: Optional[Mapping[int, float]] = None
                         ) -> Dict[str, Dict[str, Any]]:
    """Small Optuna search, scored on VALIDATION macro-F1. Off by default."""
    import optuna
    from sklearn.metrics import f1_score

    tuning = cfg["models"].get("tuning", {})
    if not tuning.get("enabled", False):
        return {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    best: Dict[str, Dict[str, Any]] = {}

    for name in tuning.get("models", ["xgb"]):
        def objective(trial: "optuna.Trial") -> float:
            if name == "xgb":
                params = {"max_depth": trial.suggest_int("max_depth", 3, 10),
                          "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                          "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50)}
            elif name == "rf":
                params = {"n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                          "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8)}
            else:
                return 0.0
            trial_cfg = {**cfg, "models": {**cfg["models"],
                                           name: {**cfg["models"][name], **params}}}
            model = build_model(name, trial_cfg, classes, seed)
            model.fit(X_tr, y_tr, X_val, y_val, class_weights=class_weights)
            pred = model.predict(X_val)
            return float(f1_score(y_val, pred, average="macro", zero_division=0))

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=int(tuning.get("n_trials", 15)),
                       timeout=tuning.get("timeout_s"))
        best[name] = {"params": study.best_params, "val_macro_f1": study.best_value}
        LOGGER.info("Stage 6.10: %s best validation macro-F1 %.4f with %s",
                    name, study.best_value, study.best_params)
    return best
