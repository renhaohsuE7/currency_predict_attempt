# HuggingFace Transformers — PatchTST

- **URL**: https://huggingface.co/docs/transformers/en/model_doc/patchtst
- **Fetched**: 2026-06-05
- **Topic**: HF 版 PatchTST 的模型類別與推論 API

## 模型類別
- `PatchTSTModel` — 純 backbone(輸出 hidden states)
- **`PatchTSTForPrediction`** — 預測(我們要的)
- `PatchTSTForClassification` / `PatchTSTForRegression` / `PatchTSTForPretraining`(自監督遮罩預訓練)
- 全部是標準 `nn.Module` / `PreTrainedModel`,支援 `from_pretrained()` / `save_pretrained()`。

## 輸入格式
- `past_values`: `(batch, sequence_length, num_input_channels)` —— **channel 維在最後**。
- `past_observed_mask`(選填):1=觀測到、0=缺失(NaN 補 0)。
- 訓練時additionally給 `future_values`;推論時**只給 past_values**。

## 推論用法(關鍵)
```python
from transformers import PatchTSTForPrediction
import torch

model = PatchTSTForPrediction.from_pretrained("namctin/patchtst_etth1_forecast")
model.eval()
with torch.no_grad():
    outputs = model(past_values=batch["past_values"])  # 只給過去
preds = outputs.prediction_outputs   # (batch, prediction_length, num_channels)
```

## 重要 config 參數(`PatchTSTConfig`)
- `context_length`(看回窗)、`prediction_length`(預測長度)、`num_input_channels`
- `patch_length` / `patch_stride`(patch 化)
- `d_model` / `num_hidden_layers` / `num_attention_heads` / `ffn_dim`
- `scaling`("std"/"mean"/None) — **內建 instance scaling**(免自己標準化)
- `distribution_output`("student_t"/"normal"/"negative_binomial")+ `loss="nll"` → **機率預測**;`num_parallel_samples`(預設 100)平行取樣
- `do_mask_input` / `mask_type`("random"/"forecast")— 自監督預訓練
- 預設權重對齊 `ibm-granite/granite-timeseries-patchtst`(可拿來 fine-tune)

## 對本專案的啟示
- 我們現有 `patchtst_huggingface` 模型即基於此。若做 panel/預測服務:
  - 輸入要組成 `(batch, context_length, n_channels)`,**channel-independence** 天然適合 panel(多檔股票各為一 channel/series 共享權重)。
  - `scaling="std"` 內建,和我們的 log-return 目標可搭配(或擇一)。
  - `distribution_output` 可給**預測區間**(對應我們想要的不確定性)。
  - 部署:`from_pretrained` + `eval()+no_grad()`,純 nn.Module 易 serve。
