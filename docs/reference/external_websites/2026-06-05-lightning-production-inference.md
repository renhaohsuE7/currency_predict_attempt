# PyTorch Lightning — Inference in Production

- **URL**: https://lightning.ai/docs/pytorch/1.6.4/common/production_inference.html
- **Fetched**: 2026-06-05
- **Topic**: 把 Lightning 模型部署成預測服務的最佳實作

## 重點摘要

### 1. 推論 API
- `Trainer.predict()` + 在 `LightningModule` override `predict_step()`(預設呼叫 `forward()`)。
- `predict_step` 是放**前處理/後處理**的地方(例如 Monte Carlo Dropout 不確定性估計)。

### 2. 匯出格式(framework-agnostic 部署)
- **ONNX**:`model.to_onnx(filepath, input_sample, export_params=True)`;可定義 `example_input_array` property 省略 input_sample。之後用 ONNX Runtime 跑,**不需 PyTorch**。
- **TorchScript**:`model.to_torchscript()` → 用 `torch.jit.save()` 存;可在 C++/非 Python 環境載入。要匯出 `forward` 以外的方法,用 `@torch.jit.export` 裝飾。

### 3. 生產環境建議「脫離 Lightning」
- 官方明說:Lightning 有額外依賴,**生產環境用 raw PyTorch 可能更有利**。
- 做法:`load_from_checkpoint()` 取得權重後,抽出核心 `nn.Module`:
  ```python
  model.eval()
  with torch.no_grad():
      output = model(input_tensor)
  ```
- 把核心 `nn.Module` 與 `LightningModule` 分離,部署時只帶最小依賴。

### 4. 其他
- 推論務必 `model.eval()` + `torch.no_grad()`(關掉 dropout/batchnorm 訓練行為)。
- checkpoint 用 `weights_only=True` 較安全/可攜。

## 對本專案的啟示
- 我們的 `PatchTSTLightningWrapper` 若要做預測服務:訓練用 Lightning,**服務端抽出 nn.Module(或匯出 TorchScript/ONNX)**,以 `eval()+no_grad()` 跑,避免在 serving 容器裝整套 Lightning。
- `predict_step` 適合放 log-return → 價格還原等後處理(對應我們 `target_transform` 的還原邏輯)。
