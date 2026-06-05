# Currency Predictor - Jupyter Notebooks

本目錄包含用於探索性資料分析、原型開發和視覺化的 Jupyter Notebooks。

---

## 📓 Notebook 清單

### 1. currency_prediction_pipeline.ipynb

**大小：** 1.0 MB
**狀態：** 完整的端到端流程
**最後更新：** 2024-10-07

**說明：**
完整的貨幣預測流程，使用 HuggingFace Transformer (PatchTST) 模型。這是最完整的 notebook，包含從資料收集到預測的所有步驟。

**主要內容：**
1. 🎨 **資料收集與視覺化**
   - 使用 Yahoo Finance API 收集貨幣資料
   - 資料品質檢查
   - 初步視覺化分析

2. 🔧 **資料處理與特徵工程**
   - 資料清理
   - 添加技術指標
   - 特徵選擇策略

3. 🤖 **模型訓練**
   - PatchTST Transformer 模型
   - 訓練循環
   - 測試集評估

4. 📊 **模型性能對比**
   - PatchTST vs SimpleBaseline
   - 多種評估指標（RMSE, MAE, R², MAPE）
   - 視覺化對比

5. 🔮 **未來預測**
   - 7 天預測
   - 信心區間計算
   - 投資建議生成

**適合：**
- 了解完整的預測流程
- 研究模型性能
- 視覺化分析

**執行要求：**
- 需要安裝所有依賴（transformers, torch 等）
- 需要中文字型支援（用於視覺化）
- 執行時間較長（包含模型訓練）

---

### 2. data_collection_demo.ipynb

**大小：** 608 KB
**狀態：** 資料收集示範
**最後更新：** 2024-09-28

**說明：**
展示如何使用專案的資料收集模組收集和儲存貨幣資料。

**主要內容：**
- 資料收集器使用示範
- Yahoo Finance API 使用
- 資料儲存格式
- 基本視覺化

**適合：**
- 了解資料收集流程
- 測試 API 連線
- 檢查資料品質

**執行要求：**
- 網路連線（Yahoo Finance API）
- 基本依賴（pandas, yfinance）

---

### 3. currency_analysis.ipynb

**大小：** 19 KB
**狀態：** 基本分析
**最後更新：** 2024-09-28

**說明：**
基本的貨幣資料分析 notebook。

**主要內容：**
- 歷史資料分析
- 基本統計
- 簡單視覺化

**適合：**
- 快速查看貨幣資料
- 基本統計分析
- 初學者了解資料結構

---

### 4. zhtw_font_test.ipynb

**大小：** 22 KB
**狀態：** 測試工具
**最後更新：** 2024-10-07

**說明：**
測試中文字型在 matplotlib 圖表中的顯示。

**主要內容：**
- 中文字型設置
- 字型路徑配置
- 圖表中文顯示測試

**適合：**
- 解決中文顯示問題
- 配置視覺化環境
- 測試字型設置

---

## 🚀 快速開始

### 啟動 Jupyter Notebook

```bash
# 使用 uv
uv run jupyter notebook notebooks/

# 或使用 pip
jupyter notebook notebooks/
```

### 執行順序建議

**初次使用：**
1. `zhtw_font_test.ipynb` - 確保中文顯示正常
2. `data_collection_demo.ipynb` - 了解資料收集
3. `currency_analysis.ipynb` - 基本分析
4. `currency_prediction_pipeline.ipynb` - 完整流程

**快速預測：**
- 直接執行 `currency_prediction_pipeline.ipynb`

---

## 📝 使用注意事項

### 資料收集

- Yahoo Finance API 有速率限制
- 建議使用較短的資料期間進行測試（如 "3mo"）
- 資料收集失敗時可以稍後重試

### 模型訓練

- PatchTST 訓練需要較長時間
- 建議使用 GPU 加速（如果可用）
- 可以調整 batch size 和 epochs 來加快訓練

### 視覺化

- 需要安裝中文字型
- Windows: 使用 Microsoft JhengHei
- Linux/Mac: 需要安裝對應的中文字型

### 記憶體使用

- `currency_prediction_pipeline.ipynb` 需要較多記憶體
- 如果遇到 OOM 錯誤，可以：
  - 減少資料期間
  - 減少 batch size
  - 重啟 kernel 清理記憶體

---

## 🔧 環境設置

本專案一律使用 **docker compose + uv**，所有指令在容器內執行（禁止 `pip` / `uv pip install`）。

### 安裝所有依賴（含可選）

```bash
# 在容器內安裝全部依賴（runtime + optional + dev）
uv sync --all-extras --all-groups
```

### 新增依賴

```bash
uv add <package>                       # runtime
uv add --optional <group> <package>    # optional（如 interactive plotly）
uv add --dev <package>                 # 開發工具
```

---

## 📊 資料目錄

Notebooks 可能會創建以下資料：

```
notebooks/
├── data/              # Notebook 產生的臨時資料
├── figures/           # 儲存的圖表（如果有）
└── checkpoints/       # 模型 checkpoints（如果有）
```

**注意：** 這些資料不會提交到 git（已在 .gitignore 中）

---

## 🐛 常見問題

### Q: 執行 notebook 時出現 ModuleNotFoundError

**A:** 確保已安裝專案：
```bash
uv sync --all-extras --all-groups
```

### Q: 中文顯示為方框

**A:** 執行 `zhtw_font_test.ipynb` 檢查字型設置，或在 notebook 開頭添加：
```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # Windows
# 或
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']    # Mac
```

### Q: Yahoo Finance API 失敗

**A:** 可能原因：
- 網路連線問題
- API 速率限制
- 貨幣符號錯誤

解決方案：
- 檢查網路連線
- 稍後重試
- 確認貨幣符號格式（如 "USDTWD=X"）

### Q: Kernel 崩潰或記憶體不足

**A:**
- 重啟 kernel
- 減少資料量
- 分批處理資料

---

## 📖 與專案其他部分的關係

### Notebooks vs 範例 (examples/)

| 用途 | Notebooks | Examples |
|------|-----------|----------|
| 目的 | 探索、分析、原型 | 可重用的使用範例 |
| 格式 | .ipynb | .py |
| 執行方式 | Jupyter | Python script |
| 視覺化 | 豐富的圖表 | 基本輸出 |
| 代碼組織 | 探索性 | 結構化 |

### Notebooks vs 模組 (src/)

- Notebooks 用於**探索和實驗**
- 經過驗證的代碼應該**提取到模組**中
- Notebooks 可以導入並使用模組中的類別和函數

### 開發流程

1. 在 Notebook 中**探索和實驗**
2. 將可重用的代碼**提取到模組**
3. 創建**測試**確保功能正確
4. 編寫**範例**展示使用方式
5. 保留 Notebook 作為**分析記錄**

---

## 💡 最佳實踐

### 代碼組織

- 在 cell 開頭添加註解說明目的
- 使用 markdown cell 分隔不同部分
- 定義函數而不是重複代碼
- 保持 cell 簡短易讀

### 視覺化

- 使用有意義的標題和標籤
- 設置適當的圖表大小
- 保存重要的圖表到文件
- 使用顏色時考慮色盲友好

### 資料管理

- 不要在 notebook 中硬編碼路徑
- 使用相對路徑或配置
- 清理臨時變數釋放記憶體
- 定期重啟 kernel 測試完整執行

### 版本控制

- 提交前清除輸出（節省空間）
- 添加有意義的 commit 訊息
- 大型 notebook 考慮拆分
- 使用 .gitignore 排除資料文件

---

## 📚 相關文檔

- [專案 README](../README.md)
- [使用範例](../examples/README.md)
- [配置管理說明](../docs/usage_descriptions/config_management.md)
- [開發規範](../.claude/project_guidelines.md)

---

## 🤝 貢獻指南

如果要添加新的 notebook：

1. 確保 notebook 有清晰的標題和說明
2. 使用 markdown cell 組織結構
3. 添加必要的註解
4. 更新此 README
5. 清除輸出後提交

---

## 📄 授權

與專案主體相同的授權條款。

---

**Happy Exploring! 🔬📊**
