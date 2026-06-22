# Fix Plan B — per-instance (window) normalization / RevIN — Plan (high-level)

> **Status: PLAN ONLY.** Fixes `2026-06-22-level-anchoring-issue.md`. Alternative: Plan A (`2026-06-22-fix-plan-a-predict-returns.md`).

**Goal:** Remove the training-era price anchor by normalizing **each input window with its own statistics** (not a global, train-fit `StandardScaler`), predicting in that local space, then de-normalizing with the **same** window's statistics. Keeps the price-level target but makes the model robust to distribution shift — this is exactly what the PatchTST paper uses (RevIN: Reversible Instance Normalization).

**Idea:** The current global `StandardScaler` is fit on the training window, so the model's output lives in "training-price space" (~1840 for 2330). Instead, for each input window subtract its **last value** (or mean) and divide by its std; the model predicts the normalized future; then **add the same last value back**. The anchor is always the window's own recent level, so a test window at price 2400 is centered at 2400 — no pull to the training mean.

## Approach
- For an input window `x` (the last `seq_len` closes): `mu = x[-1]` (or `mean(x)`), `sigma = std(x)`; `x_norm = (x - mu)/sigma`.
- Model predicts `y_norm` (the normalized future); de-normalize: `y = y_norm * sigma + mu`.
- **Reversible**: the exact `mu/sigma` used to normalize the input are reused to de-normalize the output of *that* instance. No global scaler.

## Where it changes
- The model wrappers' `_prepare_data` / `predict` (`models/patchtst/{lightning,huggingface,sklearn}/`): replace the global `StandardScaler` (fit on training) with **per-window normalization**. For HuggingFace `PatchTSTForPrediction`, this is partly its built-in `scaling='std'`/`norm` — the bug is the wrapper layering an *extra* global scaler on top; the fix may be as small as **removing the wrapper's global scaler and trusting the model's instance scaling** + ensuring predict de-normalizes per-instance.
- A shared `revin_normalize(window) -> (x_norm, mu, sigma)` / `revin_denormalize(y_norm, mu, sigma)` helper so all backends behave the same.

## Tasks (detail in a future spec)
- **B1** — shared RevIN helpers (`revin_normalize`/`revin_denormalize`), unit-tested (round-trip; a window at level 2400 normalizes to ~0-centered and de-normalizes back to ~2400).
- **B2** — replace the wrappers' global `StandardScaler` with per-instance normalization in fit (per training window) + predict (per input window); for HF, audit whether to drop the wrapper scaler and use the model's native `scaling`.
- **B3** — re-run the walk-forward eval for 2330 + USDTWD; record RMSE/corr/bias. **Success bar: prediction mean tracks the live level (bias ≈ 0), corr ≫ 0.06.**

## Pros / cons
- **Pros:** the **PatchTST-native** fix (RevIN is standard for distribution shift in modern TSF); keeps the intuitive price-level target; the chart shows prices directly.
- **Cons:** if `sigma` of a window is tiny (flat window) the normalization is unstable — needs an epsilon; doesn't make the *target* stationary the way returns do, so a model that still under-fits could mean-revert *within* the window's local space (smaller, but possible).
- **Relation to Plan A:** orthogonal; could stack (instance-normalize a returns target). B is the smaller, more "correct-by-design" change to the existing architecture; A is the more aggressive reformulation of the target.

## Open questions for the spec
- Anchor on the window's **last value** (mu = x[-1]) vs its **mean**; std vs a fixed scale; the epsilon for flat windows.
- For HF: remove the wrapper's global scaler vs implement RevIN explicitly; confirm the model's built-in `scaling` is per-instance.
- Apply to all three backends or the best one first; combine with Plan A or not.
