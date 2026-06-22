"""Generate an FX/stock forecast JSON for the parent app's importer.

Runs each requested model (default: lightning + sklearn + huggingface) plus a
naive random-walk baseline and writes forecast_<pair>.json with one series per
model. Runs in this submodule's env (has torch); the parent web app never
imports this. Only the band model (default lightning) carries a CI band.

Usage:
  uv run python scripts/generate_forecast.py --pair USDTWD=X --horizon 7 \
      --models patchtst_lightning patchtst_sklearn patchtst_huggingface \
      --out forecast_USDTWD.json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Tuple

from currency_predictor.prediction import CurrencyPredictor

# (model_name, values, lowers|None, uppers|None) for one model's forecast steps
ModelSeries = Tuple[str, List[float], Optional[List[float]], Optional[List[float]]]


def build_multimodel_payload(
    pair: str,
    generated_at: str,
    prediction_dates: List,
    model_series: List[ModelSeries],
    naive_value: float,
) -> dict:
    """Assemble the importer JSON from N model series + a naive baseline.

    Each model series shares the same target dates / step indices. A series with
    lowers/uppers carries a per-step CI band; naive = flat last-close repeated.
    """
    horizon = len(prediction_dates)
    series = []
    for model_name, values, lowers, uppers in model_series:
        steps = []
        for i in range(len(values)):
            step = {
                "target_date": _iso(prediction_dates[i]),
                "value": float(values[i]),
                "step": i + 1,
            }
            if lowers is not None and uppers is not None:
                step["lower"] = float(lowers[i])
                step["upper"] = float(uppers[i])
            steps.append(step)
        series.append({"model": model_name, "steps": steps})
    series.append({
        "model": "naive",
        "steps": [
            {"target_date": _iso(prediction_dates[i]),
             "value": float(naive_value), "step": i + 1}
            for i in range(horizon)
        ],
    })
    return {"pair": pair, "generated_at": generated_at,
            "horizon": horizon, "series": series}


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
    """Single-model convenience wrapper over build_multimodel_payload."""
    return build_multimodel_payload(
        pair, generated_at, prediction_dates,
        [(model_name, model_values, lowers, uppers)], naive_value)


def _iso(d) -> str:
    return d.isoformat() if hasattr(d, "isoformat") else str(d)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="USDTWD=X")
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument(
        "--models", nargs="+",
        default=["patchtst_lightning", "patchtst_sklearn", "patchtst_huggingface"])
    parser.add_argument(
        "--band-model", default="patchtst_lightning",
        help="which model carries the CI band (others are point-only)")
    parser.add_argument("--period", default="1y")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # Collect data once (shared data/raw store); each model trains on it.
    bootstrap = CurrencyPredictor(model_name=args.models[0])
    bootstrap.collect_and_store_data([args.pair], period=args.period)

    model_series: List[ModelSeries] = []
    dates = naive_value = generated_at = None
    for m in args.models:
        predictor = CurrencyPredictor(model_name=m)
        predictor.train_model(args.pair, period=args.period)
        want_band = (m == args.band_model)
        res = predictor.predict(args.pair, horizon=args.horizon,
                                period=args.period, return_uncertainty=want_band)
        if res.get("error"):
            print("predict failed for", m, ":", res["error"])
            continue
        values = [float(v) for v in res["predictions"]]
        lowers = ([float(v) for v in res["lower_bound"]]
                  if want_band and "lower_bound" in res else None)
        uppers = ([float(v) for v in res["upper_bound"]]
                  if want_band and "upper_bound" in res else None)
        model_series.append((m, values, lowers, uppers))
        dates = res["prediction_dates"]
        naive_value = float(res["last_known_value"])
        generated_at = _iso(res["last_known_date"])
        print("forecast", m, "->", len(values), "steps")

    if not model_series:
        print("no model produced a forecast")
        return 1

    payload = build_multimodel_payload(
        args.pair, generated_at, dates, model_series, naive_value)
    out = args.out or "forecast_{}.json".format(args.pair.replace("=X", ""))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("wrote", out, "with", len(model_series), "models +", "naive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
