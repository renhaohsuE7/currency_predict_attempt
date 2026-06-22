"""Generate an FX forecast JSON for the parent app's importer.

Runs the model (default patchtst_lightning) plus a naive random-walk baseline
and writes forecast_<pair>.json. Runs in this submodule's env (has torch);
the parent web app never imports this.

Usage:
  uv run python scripts/generate_forecast.py --pair USDTWD=X --horizon 7 --out forecast_USDTWD.json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from currency_predictor.prediction import CurrencyPredictor


def build_payload(
    pair: str,
    generated_at: str,
    prediction_dates: List,
    model_name: str,
    model_values: List[float],
    naive_value: float,
    lowers: Optional[List[float]] = None,
    uppers: Optional[List[float]] = None,
) -> dict:
    """Assemble the importer JSON. naive = flat random walk (last close repeated).

    target_date is taken from prediction_dates (ISO strings). The model and naive
    series share the same target dates and step indices.
    """
    horizon = len(model_values)
    model_steps = []
    for i in range(horizon):
        step = {
            "target_date": _iso(prediction_dates[i]),
            "value": float(model_values[i]),
            "step": i + 1,
        }
        if lowers is not None and uppers is not None:
            step["lower"] = float(lowers[i])
            step["upper"] = float(uppers[i])
        model_steps.append(step)
    naive_steps = [
        {"target_date": _iso(prediction_dates[i]),
         "value": float(naive_value), "step": i + 1}
        for i in range(horizon)
    ]
    return {
        "pair": pair,
        "generated_at": generated_at,
        "horizon": horizon,
        "series": [
            {"model": model_name, "steps": model_steps},
            {"model": "naive", "steps": naive_steps},
        ],
    }


def _iso(d) -> str:
    return d.isoformat() if hasattr(d, "isoformat") else str(d)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="USDTWD=X")
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--model", default="patchtst_lightning")
    parser.add_argument("--period", default="1y")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    predictor = CurrencyPredictor(model_name=args.model)
    predictor.collect_and_store_data([args.pair], period=args.period)
    predictor.train_model(args.pair, period=args.period)
    res = predictor.predict(args.pair, horizon=args.horizon,
                            period=args.period, return_uncertainty=True)
    if res.get("error"):
        print("predict failed:", res["error"]); return 1

    values = [float(v) for v in res["predictions"]]
    dates = res["prediction_dates"]
    naive_value = float(res["last_known_value"])
    lowers = [float(v) for v in res["lower_bound"]] if "lower_bound" in res else None
    uppers = [float(v) for v in res["upper_bound"]] if "upper_bound" in res else None
    generated_at = _iso(res["last_known_date"])  # present on success; main returns early on error

    payload = build_payload(args.pair, generated_at, dates, args.model,
                            values, naive_value, lowers, uppers)
    out = args.out or "forecast_{}.json".format(args.pair.replace("=X", ""))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("wrote", out, "with", len(values), "steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
