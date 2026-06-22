"""Unit test for the eval JSON assembly (no torch / no network)."""
from scripts.evaluate_patchtst import Metrics
from scripts.generate_eval import build_eval_payload


def test_build_eval_payload_skips_none_and_maps_fields():
    mbm = {
        "naive": Metrics(n=50, rmse=0.08, mae=0.06, r2=0.36, directional_acc=0.0),
        "patchtst_lightning": Metrics(n=50, rmse=0.20, mae=0.16, r2=-3.6, directional_acc=0.62),
        "patchtst_sklearn": None,   # a failed model is skipped
    }
    payload = build_eval_payload("USDTWD=X", "2026-06-22T00:00:00+00:00", mbm)

    assert payload["pair"] == "USDTWD=X"
    names = [m["model"] for m in payload["models"]]
    assert names == ["naive", "patchtst_lightning"]   # sklearn skipped
    naive = [m for m in payload["models"] if m["model"] == "naive"][0]
    assert naive["rmse"] == 0.08
    assert naive["directional_acc"] == 0.0
    lt = [m for m in payload["models"] if m["model"] == "patchtst_lightning"][0]
    assert lt["n"] == 50 and lt["r2"] == -3.6
