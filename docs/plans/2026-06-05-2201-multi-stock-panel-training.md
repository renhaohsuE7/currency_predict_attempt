# Plan: B — 多股 Panel 訓練(全域模型)

- **Date**: 2026-06-05 22:01
- **Status**: completed
- **Decisions**: 兩市場各一份範例(panel_config.json 台股 + panel_config_us.json 美股);sklearn-first;評估報酬空間
- **驗證**: 全測 652 passed, 0 failed;mypy clean(42 檔);panel_config.json 容器內載入正確(enabled/30 symbols/log_return/CAPM)
- **Module**: models（panel fit hook）、prediction（PanelTrainer）、data、config、main CLI
- **Goal 對應**: 解決 Transformer 版 PatchTST 樣本量不足 —— 用數十檔股票訓練單一全域模型
- **Depends on**: D（log-return 目標,已完成,讓跨股目標可比)、A（10y 歷史,已完成)

## 背景與核心概念

單股 10y ≈ 2500 交易日,扣掉 seq_len+pred_len 窗口僅 ~2400 筆序列,對 Transformer 仍偏少。
**Panel(面板)訓練**:對 N 檔股票各自抽取序列,**concat 成一個大訓練集,訓練單一全域模型**,
再用該模型對每檔預測。N=30 檔 → ~7 萬筆序列,量級提升 30×。

**關鍵前提(已就緒)**:目標用 **log-return**(D),報酬跨股可比;價格水準不可比,故 panel 必須在報酬空間。

**關鍵限制(必須處理)**:
1. **不可跨 symbol 建序列** —— 每檔獨立抽 window,再 concat 樣本(樣本間獨立,sklearn/視窗模型皆可)。
2. **特徵 schema 必須跨股一致** —— adaptive 技術指標會因資料長度產生不同欄位;panel 需強制統一欄位集(取交集或固定清單)後再抽序列,否則 feature 維度對不上。
3. **特徵標準化** —— 在 concat 後的全域特徵上 fit 單一 scaler(現有 `fit_transform` 即如此,只要先 concat)。

## 範圍(第一版)

- **模型**:先做 **sklearn**(always available);在 `BaseModel` 設計 panel hook,HF/Lightning 留後續。
- **流程**:新增 `PanelTrainer` orchestration + `--panel` CLI 模式。
- **不在範圍**:HF/Lightning panel、跨市場混訓、fundamentals/news。

## 設計

### 1. 模型層 hook（最小改動,可重用既有序列抽取）
`PatchTSTSklearn` 已將「frame→序列」(`_extract_features_from_data`)與「序列→fit」分離。新增:

```python
# BaseModel (抽象/預設)
def fit_panel(self, datasets: list[tuple[pd.DataFrame, pd.Series]],
              training_config=None) -> "BaseModel":
    """對多個 (X, y)（每檔一組）各自抽序列後 concat,單一 fit。"""
```

`PatchTSTSklearn.fit_panel` 實作:
- 對每個 (X_i, y_i) 呼叫 `self._extract_features_from_data` → (feat_i, tgt_i)
- `features = np.vstack([feat_i...])`、`targets = np.vstack([tgt_i...])`
- `scaler.fit_transform(features)`、`target_scaler.fit_transform(targets)`、`ensemble_model.fit(...)`
- `self._feature_columns` = 共用欄位集;`self.is_fitted = True`
- predict() 不變(用全域 scaler/model)。

> 注意:`_extract_features_from_data` 需要每檔 `X_i` 欄位數一致(見限制 2)。

### 2. PanelTrainer（orchestration,新檔 `prediction/panel_trainer.py`）
職責:
1. 讀 universe(`panel.symbols`),逐檔:collect → clean → technical indicators → (CAPM,股票) → lagged。
2. **統一特徵欄位**:計算所有檔的欄位交集(或 config 固定清單),reindex 對齊。
3. **log-return 目標**:`y_i = to_log_returns(Close_i)`,丟首列 NaN。
4. **per-symbol 時間切分**:各檔保留尾段 test_days 當 test;train 段組成 panel。
5. `model.fit_panel([(X_train_i, y_train_i) ...])`。
6. **評估**:對每檔用全域模型 `evaluate_rolling`(報酬空間)→ per-symbol metrics + 聚合(平均)。
7. **儲存**:單一全域模型 + panel manifest(含 universe、欄位集)。

### 3. Config
```json
"panel": {
  "enabled": true,
  "symbols": ["2330.TW", "2317.TW", "..."],
  "feature_columns": null,        // null = 自動取交集
  "min_history_days": 500
}
```
新增 `PanelConfig`(settings.py)。

### 4. CLI / main
- 新增 `--panel`(或 `panel.enabled`)→ `_run_panel_mode()`:訓練全域模型並對 universe 各檔輸出預測/報表。
- 沿用 `RunManager`(op_type="panel")。

## 影響範圍
| 檔案 | 變更 |
| --- | --- |
| `models/base.py` | 新增 `fit_panel()`（抽象或預設）|
| `models/patchtst/sklearn/model.py` | 實作 `fit_panel()`（concat 序列）|
| `prediction/panel_trainer.py` | **新檔** `PanelTrainer` |
| `prediction/__init__.py` | export `PanelTrainer` |
| `data_processor.py` | （視需要）欄位對齊 helper `align_feature_columns(frames)` |
| `config/settings.py` | 新增 `PanelConfig` + 掛到 AppSettings |
| `main.py` | `--panel` flag + `_run_panel_mode()` |
| `panel_config.json` | **新檔** 範例 universe（台股/美股各一） |
| `tests/test_panel_trainer.py` | **新檔** 單元 + 小型整合（mock 資料,2–3 檔）|

## 實作步驟
1. `settings.PanelConfig` + config 範例。
2. `data_processor.align_feature_columns()`（取交集、reindex）。
3. `PatchTSTSklearn.fit_panel()` + `BaseModel.fit_panel()` 介面;單元測試（2 檔合成資料,驗證序列數 = Σ 各檔序列數）。
4. `PanelTrainer`:collect→process→align→split→fit_panel→evaluate;per-symbol + aggregate metrics。
5. `main.py --panel` + `_run_panel_mode()`;`RunManager` op_type="panel"。
6. 測試（mock 離線資料,避免網路）+ 端對端 `panel_config.json` 小 universe 實跑。
7. 全測 + black/flake8/mypy（僅新/改檔)。

## 風險評估
- **欄位不一致** → 強制交集 + reindex;log 出被丟棄欄位(no silent cap)。
- **記憶體**:N×序列可能大;sklearn MultiOutputRegressor 可承受數萬列;必要時 config 限制 universe 大小。
- **資料抓取時間/網路**:逐檔 yfinance;沿用既有 cache/驗證;測試用 mock 離線資料。
- **跨股尺度**:報酬空間 + 全域 scaler 已處理;但極端波動股可能影響;可選 winsorize（後續）。
- **HF/Lightning 未支援 panel**:第一版只 sklearn;`fit_panel` 預設對未支援模型丟 `NotImplementedError` 並清楚訊息。

## 完成標準
- [ ] `fit_panel` 對 N 檔 concat 序列、單一全域模型訓練成功(序列數 = Σ 各檔)
- [ ] `PanelTrainer` 端對端:train 全域模型 → 對 universe 各檔預測 + per-symbol/aggregate metrics
- [ ] 特徵欄位跨股強制一致(交集),被丟欄位有 log
- [ ] `--panel` CLI 可運行(小 universe)
- [ ] 新增測試通過;全測維持綠(641+）
- [ ] 新/改檔 black/flake8/mypy clean

## 後續(不在本計畫)
- HF/Lightning 的 panel fit
- 跨市場、fundamentals、winsorize/robust scaling
- C 多變量脈絡(同儕股為輸入 channel)
