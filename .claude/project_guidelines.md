# Currency Predictor 專案開發規範

> 本文件定義專案的開發規範和最佳實踐，每次開發時都應遵循這些規則。

## 📁 目錄結構規範

### 1. 測試文件組織規則

**所有測試相關的腳本必須放在 `/tests/` 目錄下**

- ✅ Shell 測試腳本：`/tests/test_*.sh`
- ✅ Batch 測試腳本：`/tests/test_*.bat`
- ✅ Pytest 測試文件：`/tests/test_*.py`

**範例：**
```
tests/
├── test_data_collector.py          # pytest 單元測試
├── test_models.py                  # pytest 模型測試
├── test_integration.sh             # Shell 整合測試
├── test_pipeline.bat               # Batch 流程測試
└── conftest.py                     # pytest 配置
```

**❌ 錯誤做法：**
- 不要在專案根目錄放置測試腳本（如 `test_*.py`）
- 不要在 `src/` 目錄內放置測試文件

---

### 2. 文檔組織規則

**除了 `README.md` 以外，所有文檔必須分門別類放在 `/docs/` 目錄下**

#### 文檔分類結構

```
docs/
├── testing_reports/              # 測試報告
│   ├── unit_test_report.md
│   ├── integration_test_report.md
│   └── performance_test_report.md
│
├── usage_descriptions/           # 使用說明文檔
│   ├── quick_start.md
│   ├── api_reference.md
│   └── configuration_guide.md
│
├── architectures/                # 架構文檔
│   ├── system_architecture.md
│   ├── data_flow.md
│   └── model_design.md
│
├── development/                  # 開發文檔
│   ├── contributing.md
│   ├── coding_standards.md
│   └── git_workflow.md
│
└── design_decisions/             # 設計決策記錄
    ├── adr_001_model_selection.md
    ├── adr_002_data_storage.md
    └── adr_003_api_design.md
```

#### 文檔類型說明

| 類型 | 目錄 | 用途 |
|------|------|------|
| 測試報告 | `/docs/testing_reports/` | 測試執行結果、覆蓋率報告、性能測試 |
| 使用說明 | `/docs/usage_descriptions/` | 使用教學、API 文檔、配置指南 |
| 架構文檔 | `/docs/architectures/` | 系統架構、模組設計、資料流程 |
| 開發文檔 | `/docs/development/` | 貢獻指南、編碼標準、開發流程 |
| 設計決策 | `/docs/design_decisions/` | ADR (Architecture Decision Records) |

**❌ 錯誤做法：**
- 不要在專案根目錄放置 `Architecture.md`（應移至 `/docs/architectures/`）
- 不要將所有文檔混在一起不分類

---

### 3. 函數/方法開發規範

**每個函數或方法必須同時具備：測試 + 文檔**

#### ✅ 完整的函數開發流程

1. **撰寫函數實現**
   ```python
   def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
       """
       計算相對強弱指標 (RSI)

       Args:
           prices: 價格序列
           period: RSI 計算週期，默認為 14

       Returns:
           RSI 值序列 (0-100)

       Raises:
           ValueError: 當 period < 1 時

       Example:
           >>> prices = pd.Series([100, 102, 101, 103, 105])
           >>> rsi = calculate_rsi(prices, period=14)
       """
       # 實現...
   ```

2. **撰寫單元測試** (`/tests/test_indicators.py`)
   ```python
   def test_calculate_rsi():
       """測試 RSI 計算功能"""
       prices = pd.Series([100, 102, 101, 103, 105])
       rsi = calculate_rsi(prices, period=3)

       assert len(rsi) == len(prices)
       assert 0 <= rsi.min() <= 100
       assert 0 <= rsi.max() <= 100
   ```

3. **撰寫使用文檔** (`/docs/usage_descriptions/technical_indicators.md`)
   ```markdown
   ## RSI 指標使用說明

   ### 函數簽名
   `calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series`

   ### 參數說明
   - `prices`: 價格序列數據
   - `period`: 計算週期，建議使用 14

   ### 使用範例
   [完整的使用範例和說明...]
   ```

#### 📋 檢查清單

在提交代碼前，確認：

- [ ] 函數有完整的 docstring（包含 Args, Returns, Raises, Example）
- [ ] 在 `/tests/` 目錄下有對應的單元測試
- [ ] 測試覆蓋主要功能路徑和邊界情況
- [ ] 在 `/docs/usage_descriptions/` 下有使用文檔
- [ ] 所有測試通過 (`pytest tests/`)

---

## 🔧 開發工作流程

### 新增功能時

1. **規劃階段**
   - 在 `/docs/design_decisions/` 記錄設計決策
   - 更新 `/docs/architectures/` 相關架構文檔

2. **實現階段**
   - 撰寫函數/方法實現
   - 同步撰寫 docstring

3. **測試階段**
   - 在 `/tests/` 創建測試文件
   - 執行測試確保通過

4. **文檔階段**
   - 在 `/docs/usage_descriptions/` 撰寫使用說明
   - 更新 API 文檔

5. **驗證階段**
   - 運行完整測試套件
   - 檢查文檔完整性
   - 更新 `README.md` (如需要)

### 修復 Bug 時

1. 在 `/tests/` 添加重現 bug 的測試（先寫測試，確保會失敗）
2. 修復 bug
3. 確認測試通過
4. 在 `/docs/testing_reports/` 記錄 bug 修復報告

---

## 📝 文件命名規範

### 測試文件
- Python 測試：`test_<module_name>.py`
- Shell 測試：`test_<feature_name>.sh`
- Batch 測試：`test_<feature_name>.bat`

### 文檔文件
- 使用小寫字母和底線：`api_reference.md`
- 架構決策記錄：`adr_<number>_<title>.md`
- 測試報告：`<test_type>_test_report_<date>.md`

---

## ✅ 範例：完整的功能開發

假設要添加新功能「布林通道指標計算」：

1. **實現** (`src/currency_predictor/indicators.py`)
   ```python
   def calculate_bollinger_bands(prices, period=20, std_dev=2):
       """布林通道計算"""
       # ... 帶完整 docstring
   ```

2. **測試** (`/tests/test_indicators.py`)
   ```python
   def test_calculate_bollinger_bands():
       """測試布林通道計算"""
       # ... 完整測試
   ```

3. **文檔** (`/docs/usage_descriptions/technical_indicators.md`)
   ```markdown
   ## 布林通道 (Bollinger Bands)
   ### 使用方法
   ### 參數說明
   ### 範例代碼
   ```

4. **測試報告** (`/docs/testing_reports/indicators_test_report.md`)
   ```markdown
   # 技術指標測試報告
   - 布林通道測試：✅ PASS
   - 測試覆蓋率：95%
   ```

---

## 🚫 禁止事項

1. **不要**在專案根目錄放置測試腳本
2. **不要**在根目錄放置除 `README.md` 以外的文檔
3. **不要**提交沒有測試的新功能
4. **不要**提交沒有 docstring 的函數
5. **不要**提交沒有使用文檔的公開 API

---

## 📊 當前需要整理的文件

以下文件需要按規範重新組織：

- [ ] `Architecture.md` → 移至 `/docs/architectures/system_architecture.md`
- [ ] `test_*.py` (根目錄) → 移至 `/tests/`
- [ ] 其他散落的文檔 → 按分類移至 `/docs/` 對應目錄

---

**最後更新：** 2026-01-24
**版本：** 1.0
