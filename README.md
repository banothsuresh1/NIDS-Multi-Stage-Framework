# Flow-Aware Temporal Pattern Mining for Multi-Stage Network Intrusion Detection

A complete, runnable, config-driven implementation of the methodology in
`Methodology_NIDS_.docx`, on **CIC-IDS2017**.

Eight feature rankers are fused into one optimal feature set; five heterogeneous
classifiers (RF, XGBoost, LSTM, 1-D CNN, GNN) are combined by **reliability-aware
adaptive evidence fusion** whose weights change per session; and a Stage 9
extension mines sequential patterns and an attack-state graph that feed the
fusion as additional evidence.

The project's organising concern is **leakage**. CIC-IDS2017 results in the
literature are frequently inflated by random flow-level splits that tear a
session apart and put near-duplicate flows on both sides. Here the main protocol
is a per-class **time-blocked** split, every fitted object is fitted on training
data only, and the controls are enforced by assertions in `utils.py` that the
test-suite exercises directly. Branch D measures the resulting "leakage gap"
against the usual protocol.

---

## Contents

1. [Install](#install)
2. [How to run](#how-to-run)
3. [Folder tree](#folder-tree)
4. [Stage-to-file mapping](#stage-to-file-mapping)
5. [Branches](#branches)
6. [Expected outputs](#expected-outputs)
7. [Leakage-controls checklist](#leakage-controls-checklist)
8. [Deviations / Assumptions](#deviations--assumptions)
9. [Known limitations](#known-limitations)

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10 or 3.11
pip install -r requirements.txt
```

**numpy is pinned to the 1.26 line** because TensorFlow 2.16.x requires
`numpy < 2.0`. Do not bump numpy without also bumping `tensorflow-cpu` to
`>= 2.17`.

### TensorFlow + torch coexistence

Both frameworks run in one process (Keras for the LSTM and CNN, torch +
PyTorch Geometric for the GNN). Both ship their own CUDA/BLAS runtimes and both
grab GPU memory greedily on import, so `utils.configure_frameworks()` runs
before any TF op is built and:

* enables **memory growth** on every TF GPU, so TF does not pre-allocate the
  whole device and starve torch;
* caps torch's thread count at half the CPUs, so the two do not oversubscribe;
* sets `TF_CPP_MIN_LOG_LEVEL=2` to silence TF's startup chatter.

Import order is torch first, TF second. On a GPU box, install the CUDA wheels
for torch from the PyTorch index (see the comment in `requirements.txt`);
everything falls back to CPU automatically when no GPU is present.

`prefixspan` is **optional**. Its `extratools` dependency fails to build on
recent toolchains; `temporal_pattern_mining.py` ships an equivalent PrefixSpan
implementation and falls back to it automatically, logging which one ran. The
same is true of `mlxtend` for FP-Growth.

### Getting the data

Download CIC-IDS2017 from the
[Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
and point `--data_dir` at the directory of daily CSVs. Both public layouts work:

* `MachineLearningCVE/` — 78 numeric features + label, **no** Flow ID, IPs or
  Timestamp. The pipeline detects this and uses documented surrogates (see
  [Deviations](#deviations--assumptions) D4).
* `TrafficLabelling/` (a.k.a. `GeneratedLabelledFlows`) — includes the
  identifier and timestamp columns. **Preferred**, because the session key, the
  time-blocked split and the communication graph are then built from real
  identifiers rather than surrogates.

No dataset to hand? A faithful synthetic stand-in — same 78 column names
(including the duplicated `Fwd Header Length.1`), same raw label spellings, same
12-hour timestamps, same `inf`/NaN/`-1`-sentinel warts — ships with the tests:

```bash
python tests/synthetic_data.py --out_dir data/synthetic --rows 60000
python run_all.py --data_dir data/synthetic --smoke_test
```

---

## How to run

### Smoke test (minutes — the whole pipeline plus all five branches)

```bash
python run_all.py --data_dir /path/to/MachineLearningCVE --smoke_test
```

`run.smoke_test: true` in `config.yaml` deep-merges the `smoke:` block over the
top level: a 5 % stratified subsample (rare classes kept whole), tiny epoch
budgets, and reduced MI / SHAP / SVM sample sizes. **Every stage still runs** —
it is a scaled-down run, not a partial one.

### Full run

```bash
python run_all.py --data_dir /path/to/MachineLearningCVE --full
```

Handles the full ~2.8 M rows: `float32` throughout, parquet caching of the
cleaned frame in `data/interim/`, chunked prediction, cached ranker scores, and
a GPU when one is available.

### One branch at a time

```bash
python branches/branch_A_class_weighting/run.py --data_dir <path> --smoke_test
python branches/branch_B_smote_enn/run.py       --data_dir <path> --smoke_test
python branches/branch_C_both/run.py            --data_dir <path> --smoke_test
python branches/branch_D_stratified_split/run.py --data_dir <path> --smoke_test
python branches/branch_E_zero_day/run.py        --data_dir <path> --smoke_test
```

Each branch reads `branch_config.yaml` — which overrides **only** what differs
from the root config — and calls the same `pipeline.run_pipeline`. No module
code is duplicated into a branch.

### Notebook

```bash
jupyter lab pipeline.ipynb
# or, non-interactively:
jupyter nbconvert --to notebook --execute pipeline.ipynb --output executed.ipynb
```

Set `DATA_DIR` in the notebook's first cell.

### Tests

```bash
pytest tests/ -q
```

### Useful flags

| Flag | Effect |
|---|---|
| `--data_dir` | **required** — directory of CIC-IDS2017 daily CSVs |
| `--smoke_test` / `--full` | force smoke or full mode regardless of config |
| `--seed N` | override `seed.primary` |
| `--out_dir DIR` | redirect `results/` |
| `--no_plots` | skip figure rendering (faster) |
| `--no_cache` | ignore cached parquet / ranker scores / models |
| `--only A,B` | (run_all) run a subset of branches |
| `--skip_branches` | (run_all) root pipeline only |
| `--families X,Y` | (branch E) hold out only these attack families |

Other knobs worth knowing, all in `config.yaml`:

* `data.use_dst_port` — Destination Port can act as a shortcut on CIC-IDS2017,
  so it is **excluded by default**; flip to `true` and re-run to report both ways
  (Stage 8.4).
* `feature_selection.ablation.enabled` — the Stage 8.2 ablation battery.
* `feature_selection.sensitivity.enabled` — the Stage 8.3 (N, η, K) grids.
* `models.tuning.enabled` — a small Optuna search on validation (off by default).
* `fusion.mode` — `models_only` (the figure's Stages 1-8) or `with_temporal`.
* `seed.enable_multi_seed` — repeat over `seed.multi_seed` and report mean ± std.

---

## Folder tree

```
.
├── config.yaml                 # EVERY hyperparameter, path, split ratio and seed
├── requirements.txt
├── README.md
├── run_all.py                  # root pipeline + all five branches + comparison
├── pipeline.py                 # the shared orchestrator (Stages 1-9)
├── utils.py                    # config, seeding, logging, IO, timers, LEAKAGE GUARDS
├── data_preprocessing.py       # Stages 1-3
├── feature_selection.py        # Stage 4 (+ 8.2 ablations, 8.3 sensitivity)
├── imbalance_handling.py       # Stage 5
├── models.py                   # Stage 6
├── fusion.py                   # Stage 7
├── evaluation.py               # Stage 8
├── temporal_pattern_mining.py  # Stage 9
├── pipeline.ipynb              # all stages with visible outputs
├── branches/
│   ├── _common.py              # shared branch runner
│   ├── branch_A_class_weighting/{run.py,branch_config.yaml}
│   ├── branch_B_smote_enn/     {run.py,branch_config.yaml}
│   ├── branch_C_both/          {run.py,branch_config.yaml}
│   ├── branch_D_stratified_split/{run.py,branch_config.yaml}
│   └── branch_E_zero_day/      {run.py,branch_config.yaml}
├── tests/                      # pytest: leakage, preprocessing, FS, imbalance, models/fusion
│   ├── conftest.py
│   ├── synthetic_data.py       # CIC-IDS2017-shaped fixture generator
│   ├── test_preprocessing.py
│   ├── test_leakage.py
│   ├── test_feature_selection.py
│   ├── test_imbalance.py
│   └── test_models_fusion.py
├── data/interim/               # parquet cache, ranker-score cache, saved models
└── results/                    # JSON / CSV / PNG outputs
```

---

## Stage-to-file mapping

| Methodology | Implemented in | Key entry points |
|---|---|---|
| **Stage 1** — IDS dataset, label grouping (§1.3), session key (eq. 22) | `data_preprocessing.py` | `load_raw_dataset`, `group_labels`, `build_session_ids` |
| **Stage 2** — cleaning, imputation, Min-Max, one-hot (§2.2-2.5) | `data_preprocessing.py` | `ColumnCleaner`, `TrainFittedPreprocessor` |
| **Stage 3** — time-blocked / stratified / LOAO splits (§3.2) | `data_preprocessing.py` | `time_blocked_split`, `stratified_split`, `stratified_flow_split`, `leave_one_attack_out_split` |
| **Stage 4** — redundancy filter, 8 rankers, vote + rank, N\* (§4.2-4.8) | `feature_selection.py` | `RedundancyFilter`, `RANKERS`, `fuse_rankings`, `select_n_star`, `run_feature_selection` |
| **Stage 5** — class weights, SMOTE-ENN (§5.2) | `imbalance_handling.py` | `compute_class_weights`, `apply_smote_enn`, `apply_imbalance_handling` |
| **Stage 6** — RF, XGBoost, LSTM, 1-D CNN, GNN (§6.3-6.7) | `models.py` | `RandomForestModel`, `XGBoostModel`, `LSTMModel`, `CNN1DModel`, `GNNModel`, `build_session_graph` |
| **Stage 7** — calibration, reliability-aware fusion (§7.1-7.6) | `fusion.py` | `fit_calibrators`, `EvidenceFusion`, `run_fusion_ablations`, `worked_example` |
| **Stage 8** — metrics, plots, significance (§8.1-8.4) | `evaluation.py` | `evaluate_predictions`, `full_evaluation`, `compare_models`, `zero_day_metrics` |
| **Stage 8.2 / 8.3** — FS ablations and sensitivity | `feature_selection.py` | `run_feature_ablations`, `run_sensitivity` |
| **Stage 9** — FP-Growth, PrefixSpan, attack-state graph | `temporal_pattern_mining.py` | `mine_fp_growth`, `mine_prefixspan`, `AttackStateGraph`, `compute_sp`, `compute_tc` |
| Leakage controls (§8.4) | `utils.py`, `tests/test_leakage.py` | `assert_no_leakage` and friends |

### The eight rankers (FeS Set-1 … Set-8)

| Set | Ranker | Perspective | Score |
|---|---|---|---|
| 1 | MI | filter | mutual information, k-NN estimator (k=3), ≤50k stratified rows |
| 2 | RFI | embedded, bagging | Gini importance, 200 trees, `class_weight=balanced` |
| 3 | PI-SVM | wrapper | drop in macro-F1 of an RBF-SVM when a feature is permuted, measured on a **held-out slice of TRAIN** |
| 4 | SHAP-LGBM | explainability | mean \|SHAP\| over rows **and** classes, TreeExplainer on ~5k stratified rows |
| 5 | DMM | statistical | \|mean − median\| of the scaled train feature |
| 6 | SD | statistical | standard deviation of the scaled train feature |
| 7 | XGB | embedded, boosting | gain importance, balanced sample weights |
| 8 | LSTM | embedded, sequential | mean \|x · ∂L/∂x\| over windows, timesteps and classes |

Fusion: ψ(f) = number of top-K sets containing f; pool `P = {f : ψ(f) ≥ η}`;
`NR_m(f) = 1 − (rank−1)/(K−1)`; `Score(f) = mean_m NR_m(f)`; `F* = Top_{N*}(P)`
with N\* the smallest N within δ of the best **validation** macro-F1.

---

## Branches

| Branch | Imbalance | Split | Question it answers |
|---|---|---|---|
| **A** | class weighting only | time-blocked | What does cost-sensitive learning alone buy? |
| **B** | SMOTE-ENN only | time-blocked | What does resampling alone buy? |
| **C** | both | time-blocked | **The main experiment** (the root pipeline defaults to it) |
| **D** | both | stratified ×2 + time-blocked | How much of a reported score is the evaluation protocol? |
| **E** | both | leave-one-attack-out | Does it generalise to an attack family it has never seen? |

Branch D runs all three protocols on identical data and writes
`leakage_gap.csv`, whose `gap_vs_time_blocked_*` columns are the optimism of each
protocol relative to the honest one. Branch E loops over every attack family,
**refitting feature selection, models, calibrators and fusion weights per
family**, and reports unseen-family detection recall and the benign false-alarm
rate.

---

## Expected outputs

Everything lands in `results/` (and `branches/*/results/` per branch).

**Tables (CSV/JSON)**

| File | Contents |
|---|---|
| `class_distribution.csv` | raw label → grouped class → count (Stage 1.3) |
| `preprocessing_report.json` | duplicates removed, inf→NaN, sentinels preserved, columns excluded |
| `split_distribution.csv`, `split_report.json` | per-split class counts, block boundaries, gap size, leakage-check results |
| `feature_votes.csv` | per feature: ψ, fused Score, per-ranker membership and normalized rank |
| `ranker_scores.csv` | raw score of each of the eight rankers per feature |
| `redundancy_dropped.csv` | each dropped feature and why |
| `n_star_curve.csv` | validation macro-F1 / weighted-F1 / minority recall / PR-AUC / latency per N |
| `feature_stability.csv` | Stability(f) over repeated stratified subsamples (eq. 28) |
| `feature_selection.json` | F\*, N\*, pool size, the eight FeS sets, per-ranker timings |
| `imbalance_report.csv` | per-class counts before/after, k used, fallback policy taken |
| `model_efficiency.csv` | train time, inference latency per flow, parameter count, memory |
| `fusion_weights.csv/json` | per-session weight matrix, Q, tuned λ, τ_R |
| `metrics.csv`, `metrics_per_class.csv` | every Stage 8.1 metric, per model and for the fusion |
| `fusion_ablations.csv` | the Stage 7.6 baselines |
| `significance.csv` | pairwise McNemar / bootstrap tests |
| `fp_growth_itemsets.csv`, `prefixspan_patterns.csv` | mined patterns |
| `attack_state_graph.json/.graphml` | G_T with transition probabilities |
| `summary.json` | one-screen summary of the run |
| `branch_comparison.csv` | all branches side by side |

**Figures (PNG, dpi ≥ 200)** — class distribution; feature-vote heatmap; rank-score
bars; N\* curve; stability; SHAP summary; confusion matrices (raw + normalised)
per model; ROC and PR curves; reliability diagrams before/after calibration;
per-session fusion-weight heatmap; attack-state graph; top mined patterns;
branch comparison.

`run_all.py` finishes by printing a **stage → file → status** checklist.

---

## Leakage-controls checklist

Each item is enforced in code and covered by a test in `tests/test_leakage.py`.

| # | Control | Enforced by | Test |
|---|---|---|---|
| 1 | The split precedes imputation, scaling, encoding, ranking and resampling | `pipeline.run_pipeline` ordering | `test_preprocessor_statistics_come_only_from_train` |
| 2 | Train/val/test index sets are pairwise disjoint | `assert_disjoint_splits` | `test_splits_are_disjoint` |
| 3 | Every row is assigned to exactly one partition or the gap | construction | `test_splits_partition_every_row` |
| 4 | A session never crosses a split | `assert_sessions_not_split` | `test_sessions_stay_whole` |
| 5 | No val/test flow predates the preceding block | `assert_no_time_leakage` | `test_time_ordering_train_before_val_before_test` |
| 6 | The imputer/scaler/encoder are fitted on train rows only | `TrainFittedPreprocessor.fit` + `assert_fitted_on_train_only` | `test_preprocessor_fit_rejects_non_train_rows` |
| 7 | `transform` never updates fitted statistics | `TrainFittedPreprocessor.transform` | `test_transform_does_not_refit` |
| 8 | Unseen categories map to all zeros, never to a new column | one-hot encoder | `test_unseen_category_maps_to_all_zeros` |
| 9 | Rankers and the correlation filter see train only | `run_feature_selection` signature | `test_every_ranker_returns_one_score_per_feature` |
| 10 | PI-SVM's permutation score uses a held-out slice **of train** | `rank_pi_svm` | — |
| 11 | N\* is chosen on validation | `select_n_star` | `test_n_star_is_smallest_within_delta` |
| 12 | F\* is frozen and applied unchanged, in order | `assert_feature_mask_frozen` | `test_f_star_is_frozen_and_applied_unchanged` |
| 13 | Validation and test are never resampled | `assert_not_resampled` | `test_no_resampling_guard`, `test_eval_sets_are_never_resampled` |
| 14 | SMOTE-ENN runs on train only, **after** feature selection | `pipeline` ordering | `test_smote_enn_loses_no_class` |
| 15 | Calibrators, λ, Q and τ_R are fitted on validation only | `EvidenceFusion.fit` | `test_threshold_is_fitted_on_validation` |
| 16 | XGBoost early-stops on validation, never on test | `XGBoostModel.fit` | — |
| 17 | Identifier/time columns are excluded from X | `ColumnCleaner.identifier_columns` | `test_identifiers_excluded_from_features` |
| 18 | Destination Port is excluded by default and reported both ways | `data.use_dst_port` | `test_identifiers_excluded_from_features` |
| 19 | Graph edges never cross a split | graph built per split inside `predict_proba` | `test_session_graph_is_time_causal` |
| 20 | Stage 9 patterns and G_T are mined on train only | `run_temporal_mining(train_rows)` | — |
| 21 | Probability rows sum to 1; fusion weights sum to 1 per session | `assert_probabilities_valid`, `assert_weights_sum_to_one` | `test_probability_invariants`, `test_fusion_weights_sum_to_one_and_probabilities_valid` |

---

## Deviations / Assumptions

Where this implementation departs from a literal reading of the document, or
fills a gap the document leaves open. Each is configurable.

### Conflicts within the document

**D1 — Number of CSV files.** §1.2 says 8 files; §2.2 step 1 says "concatenation
of 7 daily CSV files". The code globs **every** CSV under `--data_dir` and logs
how many it found, so either release works.

**D2 — Imputation statistic.** The §2.2 table says "Median / Maximum
Imputation"; §2.3 states median throughout and gives the median formula. Median
is implemented (`preprocessing.imputer_strategy`).

**D3 — CNN final block.** The §6.6 prose diagram says "… → Conv1D → Global
pooling → Dense → Softmax", while the §6.6 architecture **table** specifies
`MaxPooling1D(2) → Flatten → Dense(128) → Dropout → Softmax` and gives a worked
shape example (k=46 → flattened 1,408). The **table** is implemented, as the
more specific statement.

### Data-dependent choices the document leaves open

**D4 — Releases without identifier columns.** `MachineLearningCVE/` ships no
Flow ID, IP or Timestamp column, yet Stage 3 needs an ordering and eq. 22 needs
endpoints. When they are absent the code logs a warning and falls back to: a
**surrogate time axis** from weekday-of-file plus row position (CICFlowMeter
writes in approximately flow-start order) and a **surrogate session key** of
(source file, Destination Port, Protocol) run-length grouped. Use the
`TrafficLabelling` release to avoid both surrogates.

**D5 — The session key makes singleton sessions.** Equation (22) is
`{IP_min, IP_max, Port_min, Port_max, Protocol}`. In CIC-IDS2017 the client port
is *ephemeral and effectively unique per flow*, so including it makes the key
almost a row identifier: on the synthetic fixture it produced 39,994 sessions
for 39,995 flows, which leaves the LSTM windows, the GNN graph and the Stage 9
sequences degenerate. Default is
`preprocessing.session_key_mode: service_port` — `{IP_min, IP_max, Port_min,
Protocol}`, i.e. eq. 22 **with the ephemeral high port dropped and the service
port kept**. Set it to `five_tuple` for eq. 22 verbatim (the code warns when
that produces singletons) or `host_pair` for endpoints only.

**D6 — Session idle timeout (not in the document).** Without one, a host pair
talking to port 80 all week is a *single* session spanning the whole capture, so
every session straddles every block boundary and the time-blocked split is
impossible. `preprocessing.session_timeout_s` (default 1800 s) starts a new
session after an idle gap, exactly as NetFlow/CICFlowMeter terminate flows;
`session_max_flows` caps one noisy pair. Set the timeout to 0 to disable.

**D7 — "Every class in all three partitions" vs "sessions kept whole".** §3.2
requires both, but they are *mutually unsatisfiable* for a class with fewer
separable sessions than partitions — Heartbleed (11 records) and Infiltration
(36) on the real data. Policy (`split.tiny_class_policy`, default `flow_level`):
attempt session-level blocking per class; if that leaves the class absent from a
partition, fall back to **flow-level** time blocking *for that class alone*,
record its sessions in `split.meta['exempt_sessions']`, log a warning, and
report the exemption in `split_report.json`. The session-integrity guard honours
the exemption list rather than silently passing. Set `tiny_class_policy:
session_whole` to keep sessions whole and accept that a tiny class may be
missing from a partition.

**D8 — The gap discards straddling sessions.** Block boundaries are flow-count
*time quantiles* (T1 at 70 %, T2 at 85 %). A session joins the training block
only if it **ends** by T1 and a later block only if it **starts** after the
previous boundary plus the gap; a session crossing a boundary is discarded into
the gap. This is what makes the ordering guarantee strict: a flow's timestamp is
never earlier than its own session's start, so "validation starts after training
ends" implies every validation flow is later than every training flow. The
discarded count and percentage are logged and saved.

### Choices forced by observed failures

Each of the following was adopted after the behaviour was *measured* during
development; the numbers quoted come from the smoke fixture.

**D9 — Oversampling is capped.** `imblearn`'s default raises every class to the
majority count. On CIC-IDS2017 that manufactures ~2.27 M synthetic Heartbleed
rows from 11 real ones. Observed on the fixture: 14 real Heartbleed flows became
787 synthetic ones, the CNN fitted those clusters and scored **0.75 train
accuracy against 0.14 validation accuracy**. Default is
`imbalance.smote_enn.target_strategy: capped` with `max_oversample_ratio: 20`:
no class is inflated beyond 20× its real size, and the residual imbalance is
left to class weighting — which is precisely why branch C combines the two. Set
`target_strategy: full` for the literal reading.

**D10 — The LSTM and GNN do not train on resampled data.** A SMOTE sample is an
interpolated point with no session membership, no timestamp and no host, so it
cannot occupy a position in a flow window or carry a graph edge. With
`imbalance.apply_to_sequence_models: false` (default) the sequence and graph
models train on the original ordered training flows **with class weighting**,
while RF, XGBoost and the CNN use the resampled set. Setting it `true` attaches
each synthetic row to a singleton session, which degrades the LSTM window to one
padded flow and isolates the GNN node.

**D11 — GNN node choice and label.** §6.7 permits "hosts or endpoints" as nodes.
**Sessions** are used: the label and F\* are defined per flow and aggregate
naturally to a session, whereas a host carries many unrelated labels across a
capture. Node attributes are the mean of F\* over the session's flows; the node
label is the session's **majority** class (the *rarest* class is used elsewhere,
to decide which side of a split a session falls on, where the goal is the
opposite). A node's prediction is broadcast to each of its flows so the output
aligns with the flow-level models.

**D12 — GNN residual connection and edge time-locality.** Measured on the
fixture, host-sharing edges joined same-class sessions **39.6 %** of the time
against a **38.5 %** random baseline — i.e. the graph carried essentially no
class signal, while an RF on the *same node features* reached 0.78. A GCN layer
averages over neighbours, so it was destroying the node's own evidence (0.04
accuracy). Two changes: `models.gnn.residual: true` adds a linear skip from the
raw node features to the output logits, so the GNN degrades gracefully to a
per-session classifier when the graph is uninformative and improves on it when
it is not; and `edge_max_time_gap_s` (default 900 s) keeps an edge only between
sessions close in time, since multi-stage attack activity is bursty. Equation
(23) remains the propagation rule. Result: 0.04 → 0.53 accuracy.

**D13 — CNN BatchNorm momentum.** Keras defaults to 0.99, which needs thousands
of updates for the inference-time moving averages to converge. On short
schedules the network trained on batch statistics and inferred on
near-initialisation ones: **0.82 train accuracy, 0.19 validation**.
`models.cnn.batchnorm_momentum: 0.9` fixes it (0.15 → 0.60 test accuracy). The
layer itself is exactly as the §6.6 table specifies.

**D14 — Smoke-mode budgets are larger, not smaller.** A smoke epoch has ~100×
fewer gradient steps than a full-run epoch, so the neural models need *more*
epochs and a higher learning rate there to reach the same place. The `smoke:`
block sets them accordingly; this affects the smoke test only.

### Stage 9, which the document does not specify

§7.2 and §7.3 consume `SP_t`, `TC_t` and `K_i(t)`, and Stage 9 is named as an
extension "not drawn in the figure", but no definition is given. The
implemented reading — every part configurable under `temporal:` — is:

**D15 — Event alphabet.** `event_definition` is one of `flow_state` (default: a
discretised duration/bytes/packets state such as `D0|B2|P1`, with quantile edges
fitted on TRAIN, keeping the sequence model independent of the labels),
`true_label`, or `predicted_class`.

**D16 — SP_t.** The maximum support among the mined TRAIN patterns that appear
as an ordered subsequence of the session, divided by the largest TRAIN support
(`sp_normalisation: max`). A session matching the most common pattern scores ≈1;
one matching nothing scores 0.

**D17 — TC_t.** The mean transition probability along the session's own path
through G_T, scaled by the number of states and clipped to [0,1], so a path of
merely chance-likely transitions lands near the bottom of the range and a
strongly expected path near the top. A session with fewer than
`fusion.min_events_for_temporal` (2) events has no transition, so its
availability `a(t)` is 0 and SP/TC receive **zero weight** for that session —
which is the document's own rule in §7.3.

**D18 — K_i(t) and smoothing.** `K_i(t) = P(state predicted by model i |
previous state)` read off G_T. Transition probabilities are Laplace-smoothed
(`laplace_alpha: 0.1`) so an unseen transition *lowers* a model's reliability
without zeroing its weight entirely.

**D19 — Risk score composition.** Equation (24) adds `w_i · P̂_i` (a class
distribution) to `w_S · SP_t` and `w_T · TC_t` (scalars). A model's scalar
attack evidence is taken as `1 − P̂_i(Benign)`, which is the document's own
definition in §6.6 for the CNN. The fused *class* probabilities (eq. 27) are
formed from the model terms alone and renormalised, since SP and TC carry no
class distribution.

### Other notes

**D20 — PI-SVM is a ranker only.** Per §4.4 it is "an auxiliary ranker and not
one of the final classifiers"; it appears in Stage 4 and nowhere in Stage 6.

**D21 — §7.4's worked example is illustrative.** The document says so
explicitly. `fusion.worked_example()` reproduces the *table layout* from real
fitted values; the numbers will not match the document's.

**D22 — Destination Port.** Excluded from X by default (`use_dst_port: false`)
because §8.4 flags it as a shortcut. Set it `true` and re-run to report both
ways; the cleaned-frame cache is keyed on the flag so the two do not collide.

**D23 — Optional packages.** `prefixspan` and `mlxtend` are used when
importable; equivalent implementations are bundled and used otherwise, and the
log records which ran. `torch_geometric` is used for `GCNConv`; a dense
normalised-adjacency GCN implementing eq. 23 is the documented fallback
(`models.gnn.allow_torch_fallback`), and the backend used is recorded in
`model_efficiency.csv`.

---

## Known limitations

* **Metrics in this repository come from a synthetic fixture.** No numbers here
  should be read as CIC-IDS2017 results; the fixture exists to exercise the code
  paths and the leakage guards. Run on the real data to obtain results.
* **The fixture cannot show a leakage gap, and does not.** Branch D on the
  synthetic data reports flow-level and time-blocked macro-F1 within ~0.001 of
  each other. That is the expected outcome, not a refutation of the premise: the
  generator draws each flow independently, so a session carries no
  near-duplicate flows for a torn split to leak across. The gap is a property of
  real captures, where consecutive flows of one session are highly redundant.
  Branch D is wired and reported; only real data can populate it meaningfully.
* **Session-level time blocking costs data.** Sessions overlapping a block
  boundary are discarded. On dense captures this is a small percentage, but the
  fraction is reported per run and should be checked.
* **Rare classes stay hard.** Heartbleed (11 records) and Infiltration (36)
  cannot support a meaningful 70/15/15 split; per-class recall for them is
  reported but is estimated from a handful of flows and will be noisy across
  seeds. Multi-seed runs (`seed.enable_multi_seed`) are the honest way to report
  them.
* **The GNN is only as good as the graph.** Where communication structure does
  not correlate with the labels, the residual connection means the GNN
  approximates a per-session classifier rather than adding relational evidence.
  The measured edge homophily is worth checking on real data before reading much
  into the GNN's contribution.
* **The fusion's λ are tuned on validation by grid search.** With a small
  validation partition the tuned λ may not transfer; the Stage 7.6 ablations are
  printed alongside precisely so that a λ choice which fails to beat
  `static_weights` is visible rather than hidden.
* **No streaming/online path.** Everything is batch. The per-flow inference
  latency in `model_efficiency.csv` is measured on batched prediction and is a
  lower bound for a streaming deployment.
* **Hyperparameter search is off by default** (`models.tuning.enabled: false`)
  to keep runtimes predictable; the published hyperparameters are the
  document's initial values, not tuned ones.
