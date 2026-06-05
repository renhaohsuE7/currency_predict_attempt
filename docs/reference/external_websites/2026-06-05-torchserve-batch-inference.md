# TorchServe — Batch Inference

- **URL**: https://docs.pytorch.org/serve/batch_inference_with_ts.html
- **Fetched**: 2026-06-05
- **Topic**: 用 TorchServe 把 torch 模型做成批次推論服務的最佳實作

## 核心參數
- `batch_size`:模型一次最多處理的 batch 大小。
- `max_batch_delay`(ms):等湊滿 batch_size 的最長等待;超時就用已收到的 requests 推論。
- → 用 **latency vs throughput** 取捨來調這兩個值。

## 設定方式
- Management API:`curl -X POST "localhost:8081/models?url=model.mar&batch_size=8&max_batch_delay=50"`
- 或 `config.properties` 的 `batchSize` / `maxBatchDelay`。

## 自訂 handler 結構(三方法)
- `preprocess`:整理輸入(time series 需組成 `(batch, context_length, channels)`)。
- `inference`:跑模型(`eval()`+`no_grad()`)。
- `postprocess`:格式化輸出(此處做 log-return → 價格還原、反 scaling)。
- handler **必須自行實作 batch 邏輯**;HF transformer handler 可當範本。

## 為何要 batch
- 大多數 ML/DL 框架對 batch 請求最佳化 → 提升 host 資源利用率、降低 serving 成本。

## 調校建議
- 從保守的 batch_size / max_batch_delay 起,監測 GPU/記憶體,再依吞吐/延遲需求調整。
- 預設 handler 多數支援自動 batching。
- ⚠️ **注意**:TorchServe 目前處於 limited maintenance mode(無新功能規劃)—— 新專案可考慮 FastAPI 自管 或其他 serving 方案。

## 對本專案的啟示
- 若把 PatchTST(lightning/HF)做成預測服務:用 custom handler 的 preprocess/inference/postprocess 三段對應我們的「特徵組裝 → 模型 → log-return 還原」。
- batch + max_batch_delay 適合多檔股票同時請求(panel 場景)。
- 因 TorchServe 維護趨緩,**FastAPI + 抽出的 nn.Module(eval/no_grad)** 可能是更輕、更可控的替代(見 lightning-production-inference 摘要)。
