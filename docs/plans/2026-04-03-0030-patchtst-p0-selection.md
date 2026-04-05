# P0 項目選擇：HuggingFace vs Lightning PatchTST

- **Date**: 2026-04-03 00:30
- **Status**: completed
- **Module**: models/patchtst/

## 目標

從兩個 P0 候選項目中選擇一個優先實作，並建立該項目的獨立實作計畫。

## 候選比較

| 項目 | HuggingFace PatchTST | PyTorch Lightning PatchTST |
| --- | --- | --- |
| 現有骨架 | `models/patchtst/huggingface/` 76% 覆蓋 | `models/patchtst/lightning/` 14-23% 覆蓋 |
| 既有 plan | `patchtst_transformer_implementation_plan.md` Phase 1-3 | `patchtst_implementation_roadmap.md` 預估 3-5 天 |
| 外部依賴 | `transformers`, `datasets` | `pytorch-lightning` |
| 預訓練支援 | 有（HuggingFace Hub） | 無，需自行訓練 |
| 成熟度 | 骨架較完整，76% code 已覆蓋 | 骨架存在但覆蓋率極低 |
| 適用場景 | Production：預訓練 + fine-tune | Research：自定義訓練迴圈 |

## 建議

**優先實作 HuggingFace 版**，理由：

1. 骨架完成度高（76% 覆蓋率表示大部分 code path 已可運行）
2. 有預訓練模型可直接使用，較快看到成果
3. 既有 plan 文件（`patchtst_transformer_implementation_plan.md`）已有詳細 Phase 分拆
4. Lightning 版可在 HuggingFace 版穩定後再實作

## 實作步驟

1. 確認選擇後，建立獨立 plan：`docs/plans/YYYY-MM-DD-HHmm-patchtst-huggingface-integration.md`
2. 參考 `docs/patchtst_transformer_implementation_plan.md` 的 Phase 1-3 拆解
3. Phase 1: TimeSeriesDataset 整合
4. Phase 2: Model 改進與 fine-tune pipeline
5. Phase 3: Integration test + CurrencyPredictor 對接

## 風險評估

- HuggingFace PatchTST 需要 `transformers>=4.30` 且可能需 GPU 做 fine-tune
- 如無 GPU 環境，可先以 CPU inference（預訓練 → predict）驗證

## 執行結果（2026-04-03）

- 選定 **HuggingFace PatchTST**
- 實作 plan: [`2026-04-03-0040-patchtst-huggingface-integration.md`](2026-04-03-0040-patchtst-huggingface-integration.md)

## 完成標準

- [x] 選定一個 P0 項目 → HuggingFace
- [x] 建立該項目的獨立實作 plan → #0040
- [x] 在 roadmap 更新 status
