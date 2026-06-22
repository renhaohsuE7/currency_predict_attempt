"""Unit test for the eval JSON assembly (no torch / no network)."""
from scripts.evaluate_patchtst import Metrics
from scripts.generate_eval import build_eval_payload, storage_symbol


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


def test_storage_symbol_strips_fx_suffix_but_keeps_stock_tickers():
    # FX pair: '=X' suffix stripped to match DataStorage filenames (USDTWD_*).
    assert storage_symbol("USDTWD=X") == "USDTWD"
    # Stock tickers carry no '=X' -> returned unchanged (incl. TPEx '.TWO').
    assert storage_symbol("2330.TW") == "2330.TW"
    assert storage_symbol("5483.TWO") == "5483.TWO"


def test_build_eval_payload_preserves_stock_pair_label():
    mbm = {"naive": Metrics(n=47, rmse=40.0, mae=30.0, r2=0.9, directional_acc=0.5)}
    payload = build_eval_payload("2330.TW", "2026-06-22T00:00:00+00:00", mbm)
    assert payload["pair"] == "2330.TW"
    assert payload["models"][0]["model"] == "naive"
