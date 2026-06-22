# Fix Plan A — predict returns/diffs instead of price levels — Plan (high-level)

> **Status: PLAN ONLY.** Fixes `2026-06-22-level-anchoring-issue.md`. Alternative: Plan B (`2026-06-22-fix-plan-b-instance-norm.md`).

**Goal:** Make the PatchTST forecasters **level-agnostic** by predicting the *change* (log-return or first difference) rather than the absolute price, so a trending test series no longer pulls predictions back to the training-window price level.

**Idea:** A trending price series is non-stationary; its **returns** are roughly stationary (mean ≈ 0, stable variance). Train the model to predict the return sequence, then reconstruct the price by cumulating returns from the last known close. The model never sees absolute price, so it cannot be anchored to the training-era level.

## Approach
- **Target transform:** `r_t = log(close_t) - log(close_{t-1})` (log-return; simple diff is the cheaper alternative). The model's target sequence is returns over the prediction horizon.
- **Input:** still a window of (log-)returns (and/or other stationary features), scaled — but now the scaled space is returns (~N(0, small)), which is stationary across train/test.
- **Reconstruction:** `close_{T+h} = close_T * exp(sum_{i=1..h} r_pred_i)` (or `close_T + sum diff_i`), anchored to the **last known** close `close_T` (not the training mean). This is the key: the level comes from the live anchor, the model only supplies the shape.

## Where it changes
- The model wrappers' data prep (`models/patchtst/{lightning,huggingface,sklearn}/`): add a `target='returns'` mode in `_prepare_data` / `fit` (compute returns, scale, build sequences) and in `predict` (predict returns → cumulate from the last close → return prices). Best behind one shared helper so all three backends get it consistently.
- Keep the existing `target='level'` path for comparison (config flag, default flips to `returns` after validation).

## Tasks (detail in a future spec)
- **A1** — shared returns transform helpers: `to_returns(close, kind='log')`, `from_returns(last_close, r, kind='log')`. Unit-tested (round-trip: `from_returns(c0, to_returns(series))` reconstructs the series).
- **A2** — thread a `target` option through the wrappers' fit/predict (returns vs level); default-validate on USDTWD + 2330.
- **A3** — re-run the walk-forward eval (`evaluate_patchtst` / `generate_eval`) for 2330 + USDTWD; record RMSE/corr/bias vs the level baseline and vs naive. **Success bar: directional corr ≫ 0.06 and no systematic level bias; ideally RMSE ≤ naive on at least some symbols.**

## Pros / cons
- **Pros:** stationary target (the textbook finance fix); level set by the live anchor (no training-mean pull); simple, model-agnostic.
- **Cons:** multi-step error **compounds** (cumulating noisy returns drifts); a return-RMSE that looks fine can still give a wide price band at horizon 7. Needs honest multi-step reporting.
- **Relation to Plan B:** orthogonal — could combine (predict returns *and* instance-normalize). If both are cheap, A is the more fundamental fix for the *bias*; B is the more "PatchTST-native" fix for *distribution shift*.

## Open questions for the spec
- log-return vs simple diff; how to handle splits/gaps; whether to clip extreme returns.
- 1-step (the eval) vs multi-step (the chart) reconstruction + how to show the compounding uncertainty.
- Do we apply it to all three backends or pick the best one first.
