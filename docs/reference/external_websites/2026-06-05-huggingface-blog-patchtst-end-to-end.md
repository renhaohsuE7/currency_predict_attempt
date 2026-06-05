# HuggingFace Blog — PatchTST End-to-End

- **URL**: https://huggingface.co/blog/patchtst
- **Fetched**: 2026-06-05
- **Topic**: PatchTST 從資料→訓練→推論→部署的完整實作流程

## 資料準備
```python
context_length = 512      # 看回窗
forecast_horizon = 96     # 預測長度
patch_length = 16
```
- 用 `TimeSeriesPreprocessor`(`scaling=True` → 以 train split 標準化,套用到所有資料)。
- `id_columns`(多序列識別,univariate 給空 list)、`forecast_columns`。
- **手動算 index 確保 context 窗不跨 train/val/test 邊界**(避免 leakage — 與我們時間切分一致)。

## 訓練(用 HF Trainer)
```python
training_args = TrainingArguments(
    num_train_epochs=100, per_device_train_batch_size=64,
    evaluation_strategy="epoch", load_best_model_at_end=True,
    metric_for_best_model="eval_loss")
trainer = Trainer(model=model, args=training_args,
    train_dataset=train_ds, eval_dataset=valid_ds,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)])
trainer.train()
```

## 推論 / 服務
```python
model = PatchTSTForPrediction.from_pretrained("path/to/checkpoint")
results = trainer.evaluate(test_dataset)   # 或手動 model(past_values=...)
```
**Serving tips(部落格明列):**
- **把 preprocessor 跟 model 一起存**:`time_series_preprocessor.save_pretrained(save_dir)`,推論時用**同一個** preprocessor 套 scaling(scaling 必須一致)。
- batch 推論:調 `per_device_eval_batch_size`;部落格報 ETTh1 約 2600–2900 samples/sec。

## Transfer Learning(三段)
1. **Zero-shot**(直接套預訓練,MSE 0.370)
2. **Linear probing**(凍結 backbone,只訓 head,MSE 0.357)
3. **Full fine-tune**(全更新,MSE 0.354)
→ 每段載入前一段 checkpoint 再 `trainer.train()`。

## 對本專案的啟示
- **「preprocessor 與 model 綁定存檔」**是我們缺的一塊:目前 sklearn 模型存了 scaler,但 config/feature schema(panel 的 41 欄交集)也應一起持久化,serving 時保證一致。
- Transfer learning 路線(zero-shot → linear probe → fine-tune)對「樣本不足」很實用 —— 可拿 `ibm-granite/granite-timeseries-patchtst` 預訓練權重 fine-tune,而非從零訓 panel。
- scaling 一致性 + 不跨邊界切窗,與我們既有做法吻合。
