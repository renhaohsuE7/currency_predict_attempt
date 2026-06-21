# 程式碼活/死碼地圖 + 資料輸入流程檢查 — 2026-06-21

> Phase 1 of `2026-06-21-refactor-and-evaluation-plan.md`。
> 回答兩件事:(1) 哪些模型/資料模組是 canonical、哪些被取代(供 Task 4 安全刪除);(2) 資料輸入流程的機制與弱點,對照父專案 PostgreSQL。

---

## Part 1 — 活碼 / 死碼地圖

### Canonical(保留)

| 模組 | 角色 |
|---|---|
| `models/base.py` | `BaseModel` / `ModelType` 等抽象基底 |
| `models/factory.py` | `ModelFactory` + `create_patchtst_model`(唯一對外建模入口) |
| `models/patchtst/` (package) | **唯一 canonical 模型實作**:`config.py`、`sklearn/`、`huggingface/`、`lightning/` |
| `models/__init__.py` | 由 `.patchtst` 套件 re-export;`from .patchtst import ...` 解析到**套件**(見下) |
| `data/collectors.py` (`YahooFinanceCollector`) | 現行收集器(predictor / examples / tests 都用它) |
| `data/storage.py` (`DataStorage`) | 現行檔案儲存 |
| `data_processor.py` (`DataProcessor`) | 現行特徵工程 |
| `prediction/predictor.py` (`CurrencyPredictor`) | 現行 god-class(Task 6 拆 Facade) |
| `prediction/pipeline.py` (`PredictionPipeline`) | 多貨幣編排 |

### 死碼 / 被取代(Task 4 刪除標的,共 5 項)

| 檔案 | 為何是死碼 | 刪除注意 |
|---|---|---|
| `src/currency_predictor/models.py` | 與套件目錄 `models/` **同名**。Python import 解析中**套件(目錄)優先於同名 `.py`**,故 `import currency_predictor.models` 永遠拿到 `models/__init__.py`,此檔**完全不可達**。內容是舊版 sklearn 集成 `CurrencyPredictor`(RandomForest/GradientBoosting),早被 `prediction/predictor.py` 取代。 | 不可達,直接刪,無 import 受影響 |
| `src/currency_predictor/models/patchtst.py` | 與套件 `models/patchtst/` **同名**,被套件 shadow,不可達。所有 `from ...models.patchtst import` 都解析到套件 `__init__.py`(測試 import 成功即證)。 | 不可達,直接刪 |
| `src/currency_predictor/models/patchtst_transformer.py` | 唯一引用者是 `models/transformer/__init__.py`,而後者本身已壞(見下)且無人 import。 | 與 `transformer/` 一起刪 |
| `src/currency_predictor/models/transformer/` (僅 `__init__.py`) | `__init__.py` 寫 `from .patchtst_transformer import PatchTSTTransformer`,但 `patchtst_transformer.py` 在 `models/` 而非 `models/transformer/`,**此 import 已壞**;且全 repo 無人 import `models.transformer`。 | 整個資料夾刪 |
| `src/currency_predictor/data_collector.py` (`CurrencyDataCollector`) | 舊版收集器,已被 `data/collectors.py` 的 `YahooFinanceCollector` 取代。唯一引用是 `currency_predictor/__init__.py` 的 re-export。 | **刪前先改** `currency_predictor/__init__.py`:移除 `from .data_collector import CurrencyDataCollector` 與 `__all__` 內 `"CurrencyDataCollector"`,否則 `import currency_predictor` 會壞、波及綠測試 |

### Import 解析證據

- `tests/test_models_patchtst.py` 的 `from src.currency_predictor.models.patchtst import PatchTST` 能成功 import(失敗在 assertion,不是 ImportError)→ 證明取到**套件**而非 `patchtst.py`。
- grep 全 repo:無任何檔案 import `currency_predictor.models`(top-level `.py`)、`models.transformer`、或 `data_collector` 之 `CurrencyDataCollector`(除 `__init__.py` re-export)。

---

## Part 2 — 資料輸入流程(collector → storage → processor → train/predict)

### 流程

```
YahooFinanceCollector.get_currency_data(symbol, period, interval)
  └ yf.Ticker(symbol).history(...)  → _clean_data: dropna / 排序 / 去重
DataStorage.save_raw_data(df, clean_symbol, period)
  └ 寫 data/raw/{symbol}_{period}_{YYYYmmdd_HHMMSS}.csv  (+ 可選同名 .json metadata)
DataStorage.load_raw_data(symbol, period)
  └ glob 符合樣式的 csv → 取 **mtime 最新** 的那個
DataProcessor:
  clean_data            → 去重 / dropna / 去 tz / 排序
  create_technical_indicators → MA/EMA/MACD/RSI/BB/Volatility/Price_Change；尾段 bfill().ffill()
  create_lagged_features(lags=[1,2,3]) → Close/Volume/Price_Change 的 lag；ffill().bfill()，殘餘才 dropna
  →（god-class 內聯）80/20 時間切分
```

### 關鍵弱點(影響「效果」與可靠性)

1. **【嚴重·影響效果】訓練目標未做未來位移(look-ahead / 近似洩漏)。**
   `CurrencyPredictor.prepare_training_data`(predictor.py:170-171)設
   `y = processed_data['Close']`(**當日** Close),特徵卻含**當日** Open/High/Low 與 `Close_lag_*`。
   全流程**沒有** `shift(-horizon)`(對比 `DataProcessor.prepare_features_target` 其實有做位移,但 god-class 沒用它)。
   後果:模型是用「當日其他價格」回歸「當日收盤」,並非預測未來;任何回報的 R² 都會虛高、對真實預測無意義。**評估(Task 3)須明列此點,並以正確的次日預測設定對照 naive。**

2. **【中·洩漏】指標的 `bfill()` 用未來值回填起始 NaN。**
   `create_technical_indicators` 末段 `df.bfill().ffill()`:rolling 視窗造成的開頭 NaN 被「之後」算出的值回填 → 未來資訊向前洩漏。原作以「填補取代 dropna」解決資料砍量,但填補本身是正確性問題,非單純數量問題。

3. **【中·可靠性】時間戳 CSV 檔不斷累積、以 mtime 取最新。**
   每次收集都寫一個新檔 `{symbol}_{period}_{timestamp}.csv`;載入靠 glob + mtime。後果:檔案無上限增生、跨檔不去重、無單一真實來源、`load_raw_data` 隨檔數 O(n);若手動搬動/複製檔案,mtime 可能選錯版本。

4. **【小】metadata 存於旁置 `.json`**,與資料分離可能不同步;CSV round-trip 失去 dtype/tz(靠 `parse_dates` 還原)。

5. **【小】無 schema、無並發保護、無索引/查詢**;欄位靠約定。

### 對照父專案 PostgreSQL(為何 DB 較佳)

父專案 `yfinance-project-demo-412` 以 PostgreSQL 存報價(`yfinance_1d` 等表 + migration + upsert):

- **冪等 upsert by (symbol, date)** → 無重複、無檔案增生、單一真實來源(解弱點 3)。
- **schema + 型別 + 索引** → dtype/tz 一致、可查詢、O(log n) 取值(解弱點 4、5)。
- **交易/並發安全**。

### 重構介面化方向(本次不建 DB adapter,只留乾淨介面)

Task 6 拆 `DataManager` 時,把「儲存」收斂為一個窄介面(如
`load_raw(symbol, period) -> DataFrame` / `save_raw(df, symbol, period) -> bool`),
現以 `DataStorage`(檔案)實作;日後可加一個 `PostgresDataStore` 實作同介面、接父專案 DB,而不動上層邏輯(YAGNI:本次不寫 DB adapter)。
