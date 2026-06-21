# Refactor + PatchTST Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`). This is a **behavior-preserving refactor + empirical evaluation** — the existing `pytest` suite is the safety net (green before/after each refactor task), and several tasks are discovery-driven (the exact files to delete follow from the Task 2 trace).

**Goal:** Empirically evaluate the sklearn / HuggingFace / Lightning PatchTST implementations against a naive baseline, then clean up the repo (delete duplicate model code, unify `fit()`, split the `CurrencyPredictor` god-class, fix the README) without changing public behavior.

**Architecture:** Work on submodule branch `feat/refactor-and-eval` (off `dev`). Establish a green `pytest` baseline, map live-vs-dead code + the data pipeline, run a metrics evaluation, then refactor in small test-gated steps keeping the public API of `CurrencyPredictor` unchanged.

**Tech Stack:** Python ≥3.11, uv, pandas/numpy/scikit-learn, torch 2.8 + transformers + pytorch-lightning, pytest. yfinance data.

## Global Constraints

- All work in this repo (the submodule) on branch `feat/refactor-and-eval`; behavior-preserving; the existing `tests/` suite is the gate (green "baseline set" before and after each refactor task).
- Full deps via `uv sync` (incl. torch). No new runtime deps beyond `pyproject.toml`.
- Do NOT do the roadmap's DI container / Scaler·Feature strategy hierarchies (over-engineering). Do NOT build a DB storage adapter now (keep the storage boundary clean only). Do NOT touch `visualization/` (re-evaluate charting later). Do NOT fix unrelated pre-existing red tests (record them, exclude from the baseline set).
- Keep `CurrencyPredictor`'s public methods' signatures unchanged: `collect_and_store_data / prepare_training_data / train_model / predict / save_model / load_model / get_model_info`.
- Conventional Commits. Commit each task in the submodule. Do NOT push to the FX GitHub remote. Update the parent repo gitlink at the end.
- Plan/spec/eval docs live in `docs/development/` of this repo.
- Run commands from the submodule root: `external/currency_predict_attempt` (paths below are relative to it).

---

### Task 1: Environment + green baseline

**Files:** none changed (records a baseline note).

- [ ] **Step 1: Install full deps**

Run (from the submodule root): `uv sync` — installs everything in `pyproject.toml` incl. torch 2.8 / transformers / pytorch-lightning. (If `uv` is unavailable: `python -m venv .venv && . .venv/Scripts/activate && pip install -e .[dev,lightning]`.)
Expected: deps resolve and install (torch download is large; allow time).

- [ ] **Step 2: Run the full suite to capture the baseline**

Run: `uv run pytest -q 2>&1 | tail -40`
Record the pass/fail summary. Expected: a mix is possible (the repo is rough).

- [ ] **Step 3: Triage and record the baseline**

Create `docs/development/baseline-2026-06-21.md` listing: total passed/failed/errored, and for each failing test whether it is **(a) quickly fixable** or **(b) pre-existing / unrelated** to this refactor. Define the **"baseline-green set"** = the tests currently passing; these must stay green through every later task. Pre-existing failures are excluded and noted (NOT fixed here).

- [ ] **Step 4: Commit**

```bash
git add docs/development/baseline-2026-06-21.md
git commit -m "docs: pytest baseline before refactor (defines the must-stay-green set)"
```

---

### Task 2: Map live/dead model code + document the data-input flow

**Files:** creates `docs/development/code-map-and-data-flow.md`.

- [ ] **Step 1: Trace which model modules are live**

Run these and read the results:
```bash
grep -rn "import" src/currency_predictor/models/factory.py
grep -rn "patchtst" src/currency_predictor/models/factory.py
grep -rln "models.patchtst\|models\.models\|patchtst_transformer\|from .patchtst\|import PatchTST" src tests examples
```
Identify the **canonical** PatchTST package the factory + tests actually import (hypothesis: `src/currency_predictor/models/patchtst/{sklearn,huggingface,lightning}/`) vs **superseded** files (hypothesis: `src/currency_predictor/models.py`, `src/currency_predictor/models/patchtst.py`, `src/currency_predictor/models/patchtst_transformer.py`). Record each file as live or dead with the evidence (who imports it).

- [ ] **Step 2: Read and document the data-input pipeline**

Read `src/currency_predictor/data/collectors.py` (`YahooFinanceCollector`), `src/currency_predictor/data/storage.py` (`DataStorage`), and `src/currency_predictor/data_processor.py`. In `code-map-and-data-flow.md` document: the flow (yfinance → file storage → processor), the on-disk format (raw = CSV files under `data/raw/`, results = JSON/CSV under `results/`), the naming/timestamp scheme, and weaknesses — especially the `create_technical_indicators` + `create_lagged_features` → `dropna()` shrinkage that can reduce ~260 rows to near-zero (cross-ref `basic_usage_problems_analysis.md`).

- [ ] **Step 3: Compare to the parent project's DB approach**

In the same doc, add a short "Why a DB beats the file mechanism" section referencing the parent's PostgreSQL design (`yfinance_1d` table + Alembic migrations + upsert + queryable history) vs scattered timestamped CSVs. Note that Task 6's `DataManager` will keep the storage behind an interface so a DB adapter is a low-friction follow-on (not built now).

- [ ] **Step 4: Commit**

```bash
git add docs/development/code-map-and-data-flow.md
git commit -m "docs: map live/dead model code + data-input flow (vs parent DB)"
```

---

### Task 3: Evaluate the PatchTST implementations vs a naive baseline

**Files:** creates `scripts/evaluate_patchtst.py`, `docs/development/patchtst_evaluation.md`.

**Interfaces:**
- Produces: a runnable evaluation script and a metrics report. Uses the existing `CurrencyPredictor` facade (uniform `train_model`/`predict`) so each model is invoked through one API.

- [ ] **Step 1: Write the evaluation script** — `scripts/evaluate_patchtst.py`:

```python
"""Evaluate PatchTST implementations vs a naive random-walk baseline on FX data.

Runs each available model through the CurrencyPredictor facade (uniform API),
plus a naive baseline (predict[t] = actual[t-1]) on the same test window, and
reports RMSE / MAE / R^2 / directional accuracy. Honest answer to "how good?".
"""
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime

from currency_predictor.prediction.predictor import CurrencyPredictor
from currency_predictor.data.storage import DataStorage
from currency_predictor.data_processor import DataProcessor

logging.basicConfig(level=logging.INFO)
SYMBOL = "USDTWD=X"
PERIOD = "1y"
MODELS = ["patchtst_sklearn", "patchtst_transformer"]  # lightning added if runnable


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[-n:], y_pred[-n:]
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) or float("nan")
    r2 = 1 - ss_res / ss_tot if ss_tot == ss_tot else float("nan")
    # directional accuracy: did we get the sign of the day-over-day change right?
    if n >= 2:
        da = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))))
    else:
        da = float("nan")
    return {"n": int(n), "rmse": rmse, "mae": mae, "r2": r2, "directional_acc": da}


def naive_baseline(close_test):
    # predict[t] = actual[t-1]; align to predict the same series minus the first point
    y_true = np.asarray(close_test, float)[1:]
    y_pred = np.asarray(close_test, float)[:-1]
    return y_true, y_pred


def run():
    results = {"symbol": SYMBOL, "period": PERIOD, "models": {}}

    # Ensure data exists (collect once if missing).
    storage = DataStorage(base_dir="data")
    if storage.load_raw_data(SYMBOL.replace("=X", ""), PERIOD) is None:
        CurrencyPredictor().collect_and_store_data([SYMBOL], period=PERIOD)

    # Naive baseline on the processed test split (same processing as the models use).
    base = CurrencyPredictor()
    X_tr, y_tr, X_te, y_te = base.prepare_training_data(SYMBOL, PERIOD)
    results["test_rows"] = int(len(y_te))
    if len(y_te) >= 2:
        yt, yp = naive_baseline(y_te.values)
        results["models"]["naive_random_walk"] = metrics(yt, yp)

    # Each model via the facade.
    for name in MODELS:
        rec = {}
        try:
            p = CurrencyPredictor(model_name=name)
            train_res = p.train_model(SYMBOL, PERIOD)
            rec["trained"] = bool(train_res.get("training_completed"))
            rec["train_metrics"] = train_res.get("train_metrics")
            rec["test_metrics"] = train_res.get("test_metrics")
            rec["error"] = train_res.get("error")
        except Exception as e:  # noqa: BLE001 - eval must not abort on one model
            rec["trained"] = False
            rec["error"] = repr(e)
        results["models"][name] = rec

    print(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run the evaluation**

Run: `uv run python scripts/evaluate_patchtst.py 2>&1 | tail -60`
Capture: `test_rows` (the dropna shrinkage check — if near 0, training has no data), each model's `trained` flag + `test_metrics`, and the `naive_random_walk` metrics.

- [ ] **Step 3: Write the report** — `docs/development/patchtst_evaluation.md`:
Include a metrics table (rows: naive, patchtst_sklearn, patchtst_transformer, [lightning if run]; cols: trained?, RMSE, MAE, R², directional acc), the `test_rows` finding (does the dropna bug starve training?), and a **Conclusion**: which implementation (if any) trains end-to-end and beats naive — this is the input to Task 4's "keep which" decision. State plainly if none beat naive (expected for near-random-walk FX).

- [ ] **Step 4: Commit**

```bash
git add scripts/evaluate_patchtst.py docs/development/patchtst_evaluation.md
git commit -m "eval: PatchTST sklearn/HF vs naive baseline on USDTWD + report"
```

---

### Task 4: Consolidate duplicate model implementations

**Files:** deletes superseded model modules; repoints any stragglers; tests stay green.

- [ ] **Step 1: Confirm dead set** — from Task 2's `code-map-and-data-flow.md`, list the superseded files (hypothesis: `src/currency_predictor/models.py`, `src/currency_predictor/models/patchtst.py`, `src/currency_predictor/models/patchtst_transformer.py`). For each, run `grep -rln "<module-import-path>" src tests examples` to prove nothing live imports it. Only files with **zero live importers** are deletable; if a test imports a "dead" file, that test characterises the old code — note it and either repoint it to the canonical module or exclude it (don't expand scope).

- [ ] **Step 2: Delete the proven-dead files**

```bash
git rm src/currency_predictor/models.py src/currency_predictor/models/patchtst.py src/currency_predictor/models/patchtst_transformer.py
```
(Adjust to exactly the files proven dead in Step 1.)

- [ ] **Step 3: Repoint stragglers** — if Step 1 found a live importer of a deleted module, edit it to import the canonical package (`from currency_predictor.models.patchtst.sklearn.model import ...` / `...huggingface.model import ...`, per the factory's actual imports).

- [ ] **Step 4: Run the suite**

Run: `uv run pytest -q 2>&1 | tail -20`
Expected: the baseline-green set still passes (pre-existing red unchanged).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(models): remove superseded duplicate PatchTST implementations"
```

---

### Task 5: Unify the `fit()` interface

**Files:** `src/currency_predictor/models/base.py`, the two model `fit` methods, `src/currency_predictor/prediction/predictor.py`.

- [ ] **Step 1: Make BaseModel.fit accept **kwargs** — in `models/base.py`, change the abstract signature to `def fit(self, X, y, validation_data=None, **kwargs) -> 'BaseModel':`. Make the sklearn and HuggingFace model `fit` methods accept `**kwargs` (transformer keeps reading `num_epochs/batch_size/...` from kwargs; sklearn ignores extras). Do NOT change training math.

- [ ] **Step 2: Simplify the predictor special-casing** — in `prediction/predictor.py`, the `train_model` block (currently ~lines 215-243) special-cases `validation_split`. With a uniform `fit(**kwargs)`, keep the validation-split-to-validation_data conversion but drop any model-type branching that only existed for signature differences. Behavior (what gets trained) must be identical.

- [ ] **Step 3: Run the suite**

Run: `uv run pytest -q 2>&1 | tail -20`
Expected: baseline-green set passes. If a test asserted the old fit signature, update it to the unified one.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(models): unify fit() to (X, y, validation_data=None, **kwargs)"
```

---

### Task 6: Split the `CurrencyPredictor` god-class (Facade)

**Files:** creates `src/currency_predictor/prediction/data_manager.py`, `.../model_trainer.py`, `.../prediction_engine.py`; rewrites `prediction/predictor.py` as a thin Facade.

**Interfaces:**
- Produces: `DataManager(collect_and_store, load_processed)`, `ModelTrainer(train, evaluate)`, `PredictionEngine(predict)`. `CurrencyPredictor` keeps its 7 public methods, delegating to these.

- [ ] **Step 1: Extract `DataManager`** — move `collect_and_store_data` (collection) and the data-loading/processing half of `prepare_training_data` into `DataManager` in `prediction/data_manager.py`. Keep the **collection + processing logic identical**. `DataManager` holds `YahooFinanceCollector` + `DataStorage` + `DataProcessor`. The storage is accessed through the `DataStorage` object only (the clean boundary that a future DB adapter would replace — do not add the adapter now).

- [ ] **Step 2: Extract `ModelTrainer`** — move `train_model`'s training/validation-split + `_evaluate_model` into `ModelTrainer` in `prediction/model_trainer.py`, operating on a model instance + prepared `(X_train, y_train, X_test, y_test)`.

- [ ] **Step 3: Extract `PredictionEngine`** — move `predict`'s processing + model.predict + date generation into `PredictionEngine` in `prediction/prediction_engine.py`.

- [ ] **Step 4: Rewrite `CurrencyPredictor` as a Facade** — `prediction/predictor.py` keeps `__init__` (creating the model via factory + the three collaborators) and the 7 public methods, each delegating to a collaborator. Public signatures and return shapes unchanged.

- [ ] **Step 5: Run the suite**

Run: `uv run pytest -q 2>&1 | tail -20`
Expected: baseline-green set passes (especially `tests/test_currency_predictor.py`, `tests/test_prediction_pipeline.py`). Examples in `examples/` and `main.py` must still import and call the same way.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(predictor): split CurrencyPredictor into DataManager/ModelTrainer/PredictionEngine facade"
```

---

### Task 7: Fix the README + data-flow note

**Files:** `README.md`.

- [ ] **Step 1: Rewrite the README** — replace the stale structure section (which lists `data_collector.py/models.py/utils.py`) with the real package layout (`config/`, `data/`, `models/patchtst/{sklearn,huggingface,lightning}/`, `prediction/{predictor,data_manager,model_trainer,prediction_engine}.py`, `reporting/`, `visualization/`, `tests/`, `scripts/evaluate_patchtst.py`). Keep the existing Setup/Usage (`uv sync`, `uv run main.py`) sections; fix anything that no longer matches. Add a one-line pointer to `docs/development/patchtst_evaluation.md` and `code-map-and-data-flow.md`.

- [ ] **Step 2: Add a data-flow / DB note** — a short README subsection: data flows yfinance → CSV files (`data/raw/`) → processor; a DB-backed store (cf. the parent project's PostgreSQL) is a recommended follow-on, and `DataManager` keeps the storage boundary ready for it.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): align structure with real layout + data-flow/DB note"
```

---

### Task 8: Update the parent gitlink (finish)

**Files:** parent repo `external/currency_predict_attempt` gitlink.

- [ ] **Step 1: Confirm the submodule is clean and on the branch**

From submodule: `git status -sb` (clean) and `git log --oneline -8` (the refactor commits on `feat/refactor-and-eval`).

- [ ] **Step 2: Update the parent gitlink** — from the parent repo root:

```bash
git add external/currency_predict_attempt
git commit -m "chore(submodule): bump currency_predict_attempt to refactor+eval state"
```

(Per the user's standing preference, do NOT push the FX repo; the parent merge-to-master follows the project's auto-local-merge convention. The submodule branch stays local until the user chooses to push it.)

---

## Self-Review

**Spec coverage:**
- Phase 0 env + green baseline → Task 1. ✅
- Phase 1 map live/dead + data-input flow + DB comparison → Task 2. ✅
- Phase 1.5 evaluate 3 impls vs naive + report → Task 3 (lightning added only if runnable; noted). ✅
- Phase 2 consolidate duplicates (keep canonical) → Task 4. ✅
- Phase 3 unify fit() → Task 5. ✅
- Phase 4 split god-class, public API unchanged, storage boundary clean → Task 6. ✅
- Phase 5 README + data-flow note → Task 7. ✅
- Submodule-local commits + parent gitlink, no push → Task 8. ✅
- Non-goals respected: no DI/strategy zoo, no DB adapter, visualization untouched, no fixing unrelated red. ✅

**Placeholder scan:** the discovery-driven steps (Task 2 trace, Task 4 "delete the proven-dead files") carry the exact commands + decision rule + concrete hypothesis (named files), not deferred decisions — appropriate for a safe refactor. The eval script is concrete; the Lightning model is included only if Task 2 shows it runs.

**Consistency:** `CurrencyPredictor` public methods named identically across Tasks 5/6/8. The eval (Task 3) uses the facade's `train_model`/`prepare_training_data`, which Task 6 preserves. `DataManager`/`ModelTrainer`/`PredictionEngine` named consistently in Task 6. The "baseline-green set" defined in Task 1 is the gate referenced in Tasks 4/5/6.
