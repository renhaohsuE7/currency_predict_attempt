"""Generate an FX eval-metrics JSON for the parent app's importer.

Runs the same leakage-free walk-forward evaluation as scripts/evaluate_patchtst.py
(naive + the three PatchTST impls) and writes fx_eval_<pair>.json. Runs in this
submodule env (has torch); the parent web app never imports this.

Usage:
  uv run python scripts/generate_eval.py --out fx_eval_USDTWD.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone
from typing import Dict, Optional

# Make the repo root importable so `from scripts.evaluate_patchtst` resolves when
# run directly (`uv run python scripts/generate_eval.py` puts scripts/ on sys.path,
# not the repo root). pytest already adds the root via pyproject pythonpath=["."].
_repo_root = pathlib.Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.evaluate_patchtst import (  # noqa: E402  (after sys.path bootstrap)
    Metrics, load_close, build_features, naive_walkforward, model_walkforward,
    MODELS, TEST_FRAC,
)


def build_eval_payload(
    pair: str,
    generated_at: str,
    metrics_by_model: Dict[str, Optional[Metrics]],
) -> dict:
    """Assemble the importer JSON. None metrics (a model that failed) are skipped."""
    models = []
    for name, m in metrics_by_model.items():
        if m is None:
            continue
        models.append({
            "model": name,
            "n": m.n,
            "rmse": m.rmse,
            "mae": m.mae,
            "r2": m.r2,
            "directional_acc": m.directional_acc,
        })
    return {"pair": pair, "generated_at": generated_at, "models": models}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="USDTWD=X")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    close = load_close()
    feats, predict_full = build_features(close)
    n = len(predict_full)
    cut = int(n * (1 - TEST_FRAC))
    closes = predict_full["Close"].values.astype(float)

    metrics_by_model: Dict[str, Optional[Metrics]] = {
        "naive": naive_walkforward(closes, cut)}
    for name in MODELS:
        metrics_by_model[name] = model_walkforward(name, feats, predict_full, cut)

    payload = build_eval_payload(
        args.pair, datetime.now(timezone.utc).isoformat(), metrics_by_model)
    out = args.out or "fx_eval_{}.json".format(args.pair.replace("=X", ""))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("wrote", out, "with", len(payload["models"]), "models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
