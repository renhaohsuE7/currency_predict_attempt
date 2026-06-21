"""PatchTST 效果評估 — sklearn 版 vs HuggingFace 版 vs naive 隨機漫步基準。

回答兩件事(對應使用者「驗證他 torch patch tst, huggingface patch tst 效果如何」):
  1. 端到端可跑性:每個模型能否在真實 FX 資料上 fit + predict 不爆。
  2. 預測效果:用 *無洩漏* 的固定模型 walk-forward 單步預測,對照 naive 隨機漫步,
     量 RMSE / MAE / R² / 方向準確率。匯率近隨機漫步,**模型須贏過 naive 才算有效**。

刻意 *不* 走 god-class 的 `prepare_training_data`,因為它用當日 Close 當目標(無 shift,
近似洩漏,見 docs/development/code-map-and-data-flow.md),會讓數字虛高、對照失真。
這支腳本自己用乾淨的次日預測設定,scaler 只在訓練段擬合。

用法:
    uv run python scripts/evaluate_patchtst.py
"""

from __future__ import annotations

import io
import logging
import sys
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from currency_predictor.data.storage import DataStorage
from currency_predictor.models.factory import ModelFactory

# Windows 主控台預設 cp950,無法輸出中文/符號;強制 UTF-8(對齊專案踩過的同一坑)。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

warnings.filterwarnings("ignore")  # 評估輸出保持乾淨;模型內部 UserWarning 不在本次重點
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("evaluate_patchtst")

# --- 共用設定(對兩個模型一致,確保公平對照)---------------------------------
SYMBOL = "USDTWD"          # DataStorage 以去掉 '=X' 的符號命名
PERIOD = "1y"
SEQ_LEN = 60               # context_length:60 個交易日歷史
PRED_LEN = 5               # 模型內建多步長度;walk-forward 只取第 1 步
PATCH_LEN = 8
STRIDE = 4
HF_EPOCHS = 30             # CPU 上可接受;給 transformer 公平訓練量
TEST_FRAC = 0.2
RANDOM_STATE = 42

MODELS = ["patchtst_sklearn", "patchtst_huggingface", "patchtst_lightning"]


@dataclass
class Metrics:
    n: int
    rmse: float
    mae: float
    r2: float
    directional_acc: float


def metrics(y_true: np.ndarray, y_pred: np.ndarray, prev: np.ndarray) -> Metrics:
    """RMSE/MAE/R²/方向準確率。

    prev = 每個預測點的前一已知值(close_{t-1}),用來判定漲跌方向。
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    prev = np.asarray(prev, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    true_dir = np.sign(y_true - prev)
    pred_dir = np.sign(y_pred - prev)
    directional_acc = float(np.mean(true_dir == pred_dir))
    return Metrics(len(y_true), rmse, mae, r2, directional_acc)


def load_close() -> pd.Series:
    """以專案自己的 DataStorage 載入 USDTWD Close(忠於其資料輸入流程)。"""
    storage = DataStorage(base_dir="data")
    df = storage.load_raw_data(SYMBOL, PERIOD)
    if df is None or df.empty:
        raise SystemExit(f"找不到 {SYMBOL} {PERIOD} 資料於 data/raw/")
    close = df["Close"].astype(float).dropna()
    close.index = pd.to_datetime(close.index)
    return close.sort_index()


def build_features(close: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """從 Close 造**因果**特徵(只用過去,rolling/pct_change 預設不看未來,無洩漏)。

    回傳 (feats, predict_full):
      - feats: 不含 Close 的特徵欄(當 fit 的 X)
      - predict_full: feats + Close(Close 置末),欄數 = feats+1。

    這是為了滿足模型的隱性契約:`predict` 的輸入欄數必須等於 `fit` 的
    `X` 欄數 + `y` 欄數(sklearn fit 內部 `concat([X, y])`,predict 卻只吃 X;
    god-class 剛好因為 predict 餵入「含 Close 的全欄」而對上)。單欄 Close 會
    觸發 scaler 維度不一致,正是本次評估發現的真實 bug。
    """
    df = pd.DataFrame({"Close": close.astype(float)})
    df["ret1"] = df["Close"].pct_change()
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma10"] = df["Close"].rolling(10).mean()
    df["vol10"] = df["Close"].pct_change().rolling(10).std()
    df = df.dropna()
    feat_cols = ["ret1", "ma5", "ma10", "vol10"]
    return df[feat_cols], df[feat_cols + ["Close"]]


def naive_walkforward(close: np.ndarray, cut: int) -> Metrics:
    """naive 隨機漫步:預測 close_t = close_{t-1}。"""
    y_true = close[cut:]
    prev = close[cut - 1:-1]
    y_pred = prev.copy()  # 預測值 = 前一日
    return metrics(y_true, y_pred, prev)


def model_walkforward(
    model_name: str, feats: pd.DataFrame, predict_full: pd.DataFrame, cut: int
) -> Optional[Metrics]:
    """固定模型 walk-forward 單步預測。

    在訓練段 fit 一次(scaler 只看訓練段),再對每個測試點餵入到該點為止的歷史,
    取模型多步輸出的第 1 步當次日預測。無重訓、無洩漏。
    fit 的 X=feats、y=Close;predict 餵 feats+Close 全欄以滿足欄數契約。
    """
    values = predict_full["Close"].values.astype(float)
    X_train = feats.iloc[:cut]
    y_train = predict_full["Close"].iloc[:cut]

    model = ModelFactory.create_model(
        model_name,
        seq_len=SEQ_LEN, pred_len=PRED_LEN, patch_len=PATCH_LEN, stride=STRIDE,
        random_state=RANDOM_STATE,
    )
    try:
        model.fit(X_train, y_train, num_epochs=HF_EPOCHS)  # sklearn 由 **kwargs 吞掉 num_epochs
    except Exception as e:  # noqa: BLE001 — 誠實記錄「無法訓練」
        log.warning("%s fit 失敗: %s", model_name, e)
        return None

    # 各家 predict 對輸入欄數的隱性要求不一致(本身就是發現之一):
    #   - sklearn:需 feats+Close 全欄(scaler 維度=fit 的 concat 欄數)
    #   - huggingface / lightning:單變量,只吃 Close
    multivar = model_name == "patchtst_sklearn"
    feed = predict_full if multivar else predict_full[["Close"]]

    preds, trues, prevs = [], [], []
    for t in range(cut, len(values)):
        hist = feed.iloc[:t]
        if len(hist) < SEQ_LEN:
            continue
        try:
            step = float(np.asarray(model.predict(hist)).ravel()[0])
        except Exception as e:  # noqa: BLE001
            log.warning("%s predict@%d 失敗: %s", model_name, t, e)
            return None
        preds.append(step)
        trues.append(values[t])
        prevs.append(values[t - 1])

    if len(trues) < 2:
        log.warning("%s:有效測試點不足(%d)", model_name, len(trues))
        return None
    return metrics(np.array(trues), np.array(preds), np.array(prevs))


def facade_runs_end_to_end(model_name: str) -> tuple[bool, str]:
    """端到端可跑性:走專案真正的 facade 路徑(含其洩漏式 prep),只看會不會爆。"""
    from currency_predictor.prediction.predictor import CurrencyPredictor

    try:
        predictor = CurrencyPredictor(
            model_name=model_name,
            model_params=dict(seq_len=SEQ_LEN, pred_len=PRED_LEN,
                              patch_len=PATCH_LEN, stride=STRIDE),
            data_storage_path="data",
        )
        res = predictor.train_model(f"{SYMBOL}=X", period=PERIOD, num_epochs=HF_EPOCHS)
        ok = bool(res.get("training_completed"))
        return ok, ("training_completed=True" if ok else f"error={res.get('error')}")
    except Exception as e:  # noqa: BLE001
        return False, f"exception={e}"


def main() -> int:
    np.random.seed(RANDOM_STATE)
    close = load_close()
    feats, predict_full = build_features(close)
    n = len(predict_full)
    cut = int(n * (1 - TEST_FRAC))
    print(f"\n資料:{SYMBOL} {PERIOD} · 造特徵後 {n} 點 · 訓練 {cut} / 測試 {n - cut}")
    print(f"設定:seq_len={SEQ_LEN} pred_len={PRED_LEN} patch_len={PATCH_LEN} "
          f"stride={STRIDE} HF_epochs={HF_EPOCHS} · 特徵 {list(feats.columns)}\n")

    closes = predict_full["Close"].values.astype(float)
    rows: dict[str, Optional[Metrics]] = {}
    # naive 與模型都評估測試區間 [cut, n)。前提:cut > SEQ_LEN(1y 資料 cut≈192 ≫ 60),
    # 故模型不會因 len(hist)<SEQ_LEN 跳點,兩者點集一致、RMSE 可直接對照。
    rows["naive_random_walk"] = naive_walkforward(closes, cut)

    runnable: dict[str, tuple[bool, str]] = {}
    for name in MODELS:
        print(f"--- 評估 {name} ---")
        rows[name] = model_walkforward(name, feats, predict_full, cut)
        runnable[name] = facade_runs_end_to_end(name)
        print(f"    facade 端到端:{runnable[name][0]} ({runnable[name][1]})\n")

    # 報表 ------------------------------------------------------------------
    header = f"{'model':22} {'n':>4} {'RMSE':>10} {'MAE':>10} {'R2':>8} {'dir_acc':>8}"
    print(header)
    print("-" * len(header))
    for name, m in rows.items():
        if m is None:
            print(f"{name:22} {'—  無法評估(見上方 warning)':>40}")
            continue
        print(f"{name:22} {m.n:>4} {m.rmse:>10.5f} {m.mae:>10.5f} "
              f"{m.r2:>8.3f} {m.directional_acc:>8.3f}")

    # 結論判定:模型 RMSE 是否低於 naive(用 ASCII 標記,避開 Windows cp950 編碼坑)
    naive = rows["naive_random_walk"]
    print("\n結論(模型須贏過 naive RMSE 才算有效):")
    for name in MODELS:
        m = rows[name]
        if m is None:
            print(f"  - {name}: [N/A] 無法端到端有效評估")
        elif naive and m.rmse < naive.rmse:
            print(f"  - {name}: [WIN]  RMSE {m.rmse:.5f} < naive {naive.rmse:.5f} — 贏過隨機漫步")
        else:
            print(f"  - {name}: [LOSS] RMSE {m.rmse:.5f} >= naive {naive.rmse:.5f} — 未贏過隨機漫步")

    # 回傳結構化結果供（必要時）程式化使用
    return 0


if __name__ == "__main__":
    sys.exit(main())
