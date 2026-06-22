"""Unit test for the forecast JSON assembly (no torch / no network)."""
from datetime import datetime, timedelta

from scripts.generate_forecast import build_payload


def test_build_payload_shapes_model_and_naive():
    base = datetime(2026, 6, 22)
    dates = [base + timedelta(days=i + 1) for i in range(3)]
    payload = build_payload(
        pair="USDTWD=X", generated_at=base.isoformat(),
        prediction_dates=dates, model_name="patchtst_lightning",
        model_values=[31.1, 31.2, 31.3], naive_value=31.0,
        lowers=[30.9, 31.0, 31.1], uppers=[31.3, 31.4, 31.5])

    assert payload["pair"] == "USDTWD=X"
    assert payload["horizon"] == 3
    names = {s["model"] for s in payload["series"]}
    assert names == {"patchtst_lightning", "naive"}
    model = [s for s in payload["series"] if s["model"] == "patchtst_lightning"][0]
    assert model["steps"][0]["value"] == 31.1
    assert model["steps"][0]["lower"] == 30.9
    assert model["steps"][0]["step"] == 1
    naive = [s for s in payload["series"] if s["model"] == "naive"][0]
    assert all(s["value"] == 31.0 for s in naive["steps"])   # flat random walk
    assert "lower" not in naive["steps"][0]
