# RevIN — Reversible Instance Normalization

- **URL:** https://seharanul17.github.io/RevIN/
- **Accessed:** 2026-06-22
- **Why read:** validate Fix Plan B (per-instance normalization) — is RevIN the standard PatchTST way to handle the non-stationarity/level-anchoring our model suffers?

## Key takeaways
- RevIN = a generally applicable **normalize → (network) → denormalize** method with a learnable affine transform.
- **Input layer** removes non-stationary stats (mean & variance); the network runs on normalized data; **output layer restores** the original statistics. Symmetric & reversible **per instance**.
- Solves **distribution shift** — "statistical properties such as mean and variance change over time", named a main blocker for accurate forecasting. This is exactly our level-anchoring issue (training-era price level ≠ test price level).
- Reported significant gains on ETT/ECL/**Nasdaq** (financial) data.
- **Implication for us:** RevIN is the PatchTST-native fix → our wrappers layering a **global, train-fit StandardScaler** on top is precisely what defeats instance normalization and anchors predictions to the training price level. Plan B = drop the global scaler / normalize per window.
