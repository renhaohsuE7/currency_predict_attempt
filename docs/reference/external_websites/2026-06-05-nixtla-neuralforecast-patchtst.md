# Nixtla NeuralForecast — PatchTST

- **URL**: https://nixtlaverse.nixtla.io/neuralforecast/models.patchtst.html
- **Fetched**: 2026-06-05
- **Topic**: NeuralForecast(基於 PyTorch Lightning)的 PatchTST 實作與 train/predict/serving

## 核心參數
- `h`(forecast horizon)、`input_size`(看回窗)— 兩個必填
- `patch_len`(預設 16,自動限制為 `min(patch_len, input_size+stride)`)、`stride`(預設 8)
- `encoder_layers=3`、`n_heads=16`、`hidden_size=128`、`linear_hidden_size=256`
- `revin=True`(reversible instance normalization,穩定時間尺度 — 等同我們關心的非平穩問題對策)
- 訓練:`max_steps=5000`、`learning_rate=1e-4`、`batch_size=32`、`windows_batch_size=1024`

## 訓練/推論(走 Lightning Trainer)
```python
fit(dataset, val_size=0, test_size=0, ...)        # 內部用 PL Trainer,可傳 trainer_kwargs
predict(dataset, step_size=1, quantiles=[0.1,0.5,0.9], h=None, ...)  # 內部走 predict_step
```
- `predict()` 內部呼叫 PL `predict_step`;`h` 可覆蓋 fitted horizon;`quantiles` → **機率/區間預測**。
- 預設 **不存 checkpoint**(省磁碟),需 `enable_checkpointing=True`。

## 生產考量
- `inference_windows_batch_size`(預設 1024)控制推論記憶體;`valid_batch_size` 可獨立調。
- `training_data_availability_threshold`:訓練窗最低有效資料比例。
- **Channel-independence**:每 channel 當獨立 univariate series 處理,跨大量序列可擴展 —— 正是 panel 訓練的理論基礎。

## 對本專案的啟示
- NeuralForecast 是「panel(多序列)+ Lightning PatchTST」的成熟參考實作。我們自製的 panel 訓練概念一致(多序列共享權重)。
- **`revin`(reversible instance norm)** 值得借鏡:它在模型內做可逆正規化,等同處理跨股尺度差異 —— 是我們 log-return 目標之外的另一條路。
- 若不想自己維護 panel/serving,**NeuralForecast 可作為替代或對照**(它已封裝 train/predict/quantiles/批次推論)。
