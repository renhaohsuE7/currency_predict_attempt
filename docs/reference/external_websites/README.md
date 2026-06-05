# External Research — PatchTST 預測服務(torch + lightning)最佳實作

每查一個網站留一份摘要(本資料夾)。以下為綜整結論。

## 來源清單
| 摘要 | 來源 | 重點 |
| --- | --- | --- |
| [lightning-production-inference](2026-06-05-lightning-production-inference.md) | PyTorch Lightning docs | predict_step、to_onnx/to_torchscript、**生產環境抽出 nn.Module 脫離 Lightning** |
| [huggingface-patchtst](2026-06-05-huggingface-patchtst.md) | HF Transformers docs | `PatchTSTForPrediction`、輸入 `(B, ctx_len, channels)`、內建 scaling、機率預測 |
| [huggingface-blog-patchtst-end-to-end](2026-06-05-huggingface-blog-patchtst-end-to-end.md) | HF Blog | 完整 train→infer、**preprocessor 與 model 綁定存檔**、transfer learning 三段 |
| [nixtla-neuralforecast-patchtst](2026-06-05-nixtla-neuralforecast-patchtst.md) | Nixtla NeuralForecast | panel(多序列)+ Lightning、`revin` 可逆正規化、quantiles、批次推論參數 |
| [torchserve-batch-inference](2026-06-05-torchserve-batch-inference.md) | TorchServe docs | batch_size/max_batch_delay、handler 三段、TorchServe 維護趨緩 |

## 綜整最佳實作(預測服務)

1. **訓練與服務分離**:用 Lightning/HF Trainer 訓練;**服務端抽出核心 `nn.Module`**(或匯出 TorchScript/ONNX),以 `model.eval()` + `torch.no_grad()` 跑,serving 容器不裝整套 Lightning。
2. **輸入契約固定**:PatchTST 吃 `(batch, context_length, num_channels)`;channel-independence 天然適合多檔股票 panel(共享權重)。
3. **scaling/preprocessor 一致且綁定持久化**:訓練用的 scaler / feature schema 必須與 model 一起存、推論時套同一份(HF blog 明列;對應我們 panel 的 41 欄交集 + scaler)。
4. **不確定性/區間**:用 `distribution_output`(HF)或 `quantiles`(NeuralForecast)做機率預測 —— 對金融預測比點估計更有價值。
5. **批次服務**:多檔同時請求用 batch + `max_batch_delay` 湊批(TorchServe);但 TorchServe 維護趨緩,**FastAPI + nn.Module(eval/no_grad)** 是更輕的替代。
6. **後處理放在 predict_step / handler.postprocess**:log-return → 價格還原、反 scaling。
7. **樣本不足對策**:`revin`(可逆 instance norm)或拿 `ibm-granite/granite-timeseries-patchtst` 預訓練權重做 **transfer learning(zero-shot → linear-probe → fine-tune)**,而非從零訓 Transformer。

## 對本專案的下一步建議
- 服務化:`PatchTSTLightningWrapper` / HF 模型 → 匯出 TorchScript 或抽 nn.Module + FastAPI 端點;postprocess 做 log-return 還原。
- 持久化:把 scaler + feature_columns(panel 交集)+ target_transform 設定與模型一起存,確保 serving 一致。
- 預測力:評估 `revin` 與 HF transfer learning,作為提升(目前 panel MASE≈1.2 打不過 naive)的方向。
