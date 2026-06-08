# Handoff — Cascade feature + full-pipeline review (next-session continuation)

- **Date**: 2026-06-08 09:58
- **Branch**: `feat/panel-training` (pushed). Stack (all pushed, unmerged):
  `fix/restore-models-and-uv` → `feat/stock-return-target` → `feat/panel-training`
  (the cascade feature was merged into `feat/panel-training`).
- **Gate suite**: **700 passed, 0 failed, 49 deselected** (`docker compose run --rm test`,
  default `-m 'not slow and not e2e'`). Green.
- **Run convention (REQUIRED)**:
  `APP_UID=$(id -u) APP_GID=$(id -g) docker compose run --rm test uv run pytest <path> -q`
  (docker compose + uv only; `.env` APP_UID/GID = host uid 1003).

## What was completed this session

1. **Cascade factor forecasting** (predict realized-vol + direction factors → inject → predict price).
   - `prediction/factors.py` (targets + `FactorModel` cross-fit), `prediction/cascade.py`
     (`CascadePredictor`, `evaluate_factor_lift`), `CascadeConfig`. 3-backend tested.
   - Spec: `docs/superpowers/specs/2026-06-07-cascade-factor-forecasting-design.md`;
     Plan: `docs/plans/2026-06-07-1131-cascade-factor-forecasting.md`.
2. **Honest investigation** (`docs/issues/2026-06-07-cascade-investigation.md`): only **volatility**
   is genuinely predictable; direction "0.70" was just the uptrend base rate (+1.3% over majority);
   factor-as-channel is useless for HF/Lightning (channel-independent).
3. **Eval bugs fixed** (`cec78c5`): direction majority-baseline+skill; return-space naive (zero-return);
   HF/Lightning multi-channel fail-loud target-channel guard; **`use_multi_channel` kwargs were being
   silently dropped** (fixed); **`models/naive.py` was untracked** (gitignore) → now force-added.
4. **Test quality**: multi-channel regression tests (`320ac43`); tightened stale/no-op/silent-swallow-
   masking tests (`a113b0c`).

## TODO — pipeline review findings to FIX (next session)

Source: 3-agent full-pipeline code review (收集→整理→3 模型×預測模式→視覺+流程). Recommended order: C1–C4, then I1–I3.

### Critical (real bugs; trigger in normal use, esp. recommended `target_transform=log_return`)
- [ ] **C1 — look-ahead leakage via `bfill()`**. `data_processor.py:214-215` (create_technical_indicators)
  and `:267-270` (create_lagged_features) backfill rolling/lag NaN with FUTURE values; backfilling a
  lag column is semantically inverted. Fix: drop warm-up rows or `ffill`-only; never `bfill` features.
  (Related: CAPM block `:461` same `bfill` + `effective_window` can be as small as 3.)
- [ ] **C2 — comparison chart space mismatch under log_return**. `comparer.py:237` sets
  `symbol_result['actual']=y_test` (return space ~0.001) but predictions are reconstructed to price
  (~150); `visualizer.py:758` (`plot_model_comparison`) and `interactive.py:228` plot both on one axis
  → meaningless. Fix: chart a price-space actual (read raw Close over test dates, or reconstruct), or
  plot returns-vs-returns. Default `price` path is fine.
- [ ] **C3 — `predict_with_uncertainty` ignores multi-channel** (HF `huggingface/model.py:807-873`;
  Lightning `wrapper.py:674-770`). Hardcoded single-channel: `scaler.transform` on 1 col after a
  multi-channel fit → shape crash / wrong scale. Fix: mirror the multi-channel branch from `predict()`
  (target-channel select + dummy full-feature inverse_transform).
- [ ] **C4 — multi-channel save/load loses `_target_channel_idx`**. HF metadata (~`:914-930`) and
  Lightning (~`:788-800`) don't persist `_target_channel_idx`/`_feature_columns`; after `load_model`
  they reset to 0/None → loaded multi-channel model predicts channel 0 (wrong scale). Fix: persist &
  restore both in save/load.

### Important
- [ ] **I1 — predictions CSV misalignment (#2)**. `comparer.py:585-596` pairs FUTURE forecast dates
  with PAST test actuals (and under log_return mixes spaces). Fix: drop `Actual` from the forecast CSV
  or write a separate aligned eval CSV (pipeline's `_save_predictions_csv` is already correct).
- [ ] **I2 — interactive chart uses calendar days**. `interactive.py:235-237` fabricates `freq="D"`
  dates instead of the model's `prediction_dates` → weekends + disagrees with matplotlib/JSON. Fix:
  pass and use real `prediction_dates` (like `plot_forecast`).
- [ ] **I3 — `predict()` swallows all errors** to `{success:False}` (`predictor.py:534-536`) → predictions
  can be silently all-empty while the run reports success (fail-loud gap; `_evaluate_model` was fixed,
  `predict` was not). Fix: only catch known-recoverable errors; surface prediction failure in exit code.
- [ ] **I4 — `collect_and_store_data` no all-failed guard** (`predictor.py:243-246`): returns all-False
  dict instead of raising when every symbol fails (panel path already raises). Fix: raise if `not any(results.values())`.
- [ ] **I5 — `_validate_cached_data` can claim "驗證通過" with 0 actual comparisons** (`predictor.py:144-183`):
  count real comparisons; don't report success if none happened.
- [ ] **I6 — `evaluate_single_shot` actual slice mis-aligned** (`base.py:262-263`): takes last `pred_len`
  of y_true (context end) vs future predictions → misleading metrics on short test sets (rolling is
  preferred but this fallback runs when `len(X) < seq_len+pred_len`).
- [ ] **I7 — `storage.load_raw_data(period=None)`** returns newest file regardless of period (foot-gun).
- [ ] **I8 — `naive.predict(horizon=1)` ignores horizon** → returns `pred_len` (cross-backend contract
  inconsistency; `naive.py:78`).

### Minor (nice-to-have)
- HF dead `target_scaler` (`huggingface/model.py:239`); sklearn `predict_with_uncertainty` is a
  placeholder band; forecast-JSON filename inconsistency pipeline (`symbol.replace('=','')...`) vs
  comparer (`_clean_symbol`) (#6); naive Dir-Acc shows 0 in chart (reads as "worst"); bare `except`
  (`visualizer.py:124`); deprecated `resample('M')`; `to_log_returns` no positivity guard (−inf not
  filtered by `valid=y.notna()`); `is_currency_pair`/`classify_symbol` duplicate the `=X` rule.

## Pre-existing / out-of-scope (known, not blocking the gate)
- **Deselected use_case integration tests** in `tests/test_use_case_currency_predictor.py` (marked
  slow/e2e) have a `test_days=30 < seq_len+pred_len` setup issue → 4 fail when run with `-m ''`.
  Not in the default gate. Fix their `test_days` if you want them green under override.
- **Baseline lint debt**: ~68 files would be black-reformatted; `flake8 src/` has 270+ pre-existing
  errors (W293 etc.). Handle in a dedicated `chore/lint-sweep`, not feature branches. New/changed files
  are kept clean. (See memory `baseline-lint-debt`.)
- **HF return-space**: HF/Lightning multi-channel forecast the Close PRICE channel; full return-space
  support for them is a larger refactor (cascade return-space is intentionally sklearn-only, fail-loud).

## Decision still pending (what the user wanted to decide next)
- Which review findings to fix: recommended **C1–C4 + I1/I2/I3**. None started yet.
- Optional follow-ups offered earlier: open PR(s) for the stack; `chore/lint-sweep`; cascade injection
  via channel_attention/exogenous head (to make factors actually affect HF/Lightning).

## Key reference docs
- `docs/issues/2026-06-07-cascade-investigation.md` (root causes + fixes + corrected conclusions)
- `docs/issues/2026-06-07-cascade-factor-lift-results.md`
- `docs/superpowers/specs/2026-06-07-cascade-factor-forecasting-design.md`
- `docs/plans/2026-06-07-1131-cascade-factor-forecasting.md`
- `docs/reference/external_websites/` (PatchTST serving + factor-forecasting research)
- `.claude/rules/fail-loud.md` (never swallow eval/metrics errors into empty results)
