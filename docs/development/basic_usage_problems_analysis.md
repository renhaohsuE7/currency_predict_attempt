# Basic Usage Problems Analysis Report

**Date**: 2026-01-25
**Task**: Analyze and fix issues in `examples/basic_usage.py` execution

---

## Executive Summary

Execution of `examples/basic_usage.py` revealed 4 critical issues preventing successful model training and prediction:

1. **Model Parameter Mismatch**: PatchTST doesn't accept `validation_split` parameter
2. **Zero Training Data**: Feature engineering creates excessive NaN values, resulting in 0 training samples
3. **Unicode Encoding Error**: Emoji characters incompatible with Windows console encoding
4. **Missing Result Key**: Pipeline results missing `end_time` key during save

---

## Problem 1: Model Training Parameter Mismatch

### Error Message
```
ERROR: 模型訓練失敗: PatchTST.fit() got an unexpected keyword argument 'validation_split'
```

### Root Cause Analysis

**Location**: [src/currency_predictor/prediction/predictor.py:214](src/currency_predictor/prediction/predictor.py#L214)

```python
def train_model(self, symbol: str, period: str = "1y",
                target_column: str = 'Close',
                feature_columns: Optional[List[str]] = None,
                **train_kwargs) -> Dict[str, Any]:
    # ... preparation code ...

    # Line 214: Pass all kwargs to model.fit()
    self.model.fit(X_train, y_train, **train_kwargs)  # ❌ ERROR HERE
```

**PatchTST.fit() Signature** ([src/currency_predictor/models/patchtst.py:151-156](src/currency_predictor/models/patchtst.py#L151-L156)):

```python
def fit(
    self,
    X: pd.DataFrame,
    y: pd.Series,
    validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None  # ⚠️ Not validation_split!
) -> 'PatchTST':
```

**The Issue**:
- Caller passes `validation_split=0.2` in `train_kwargs`
- PatchTST.fit() expects `validation_data` (tuple), not `validation_split` (float)
- Unexpected keyword argument causes TypeError

**Data Flow**:
```
basic_usage.py (line 35)
  → calls pipeline.run_full_pipeline()
    → calls predictor.train_model(**train_kwargs)
      → calls model.fit(X_train, y_train, validation_split=0.2)  # ❌ FAILS
```

---

## Problem 2: Zero Training Data from Feature Engineering

### Error Message
```
2026-01-25 00:46:44,004 - Lagged features created: 0 records after removing NaN (had 485 NaN values)
2026-01-25 00:46:44,013 - 訓練資料: 0 筆，測試資料: 0 筆
```

### Root Cause Analysis

**Initial Data**: 260 records collected from Yahoo Finance

**Feature Engineering Pipeline**:

#### Step 1: Technical Indicators ([data_processor.py:71-127](src/currency_predictor/data_processor.py#L71-L127))

Creates rolling window features:
- `MA_5` (5-period moving average) → 4 NaN at start
- `MA_10` (10-period moving average) → 9 NaN at start
- `MA_20` (20-period moving average) → 19 NaN at start
- `MA_50` (50-period moving average) → 49 NaN at start ⚠️
- `EMA_12`, `EMA_26` → exponential smoothing creates NaN
- `RSI` (10-period) → 9 NaN at start
- `Bollinger Bands` (15-period) → 14 NaN at start
- Price changes with `pct_change(periods=10)` → 10 NaN at start

**NaN Created**: ~50+ rows have NaN in at least one column

#### Step 2: Lagged Features ([data_processor.py:129-155](src/currency_predictor/data_processor.py#L129-L155))

Creates shifted features with lags=[1, 2, 3, 5, 10]:
```python
for lag in lags:
    df[f'Close_lag_{lag}'] = df['Close'].shift(lag)  # Creates NaN
    df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
    df[f'Price_Change_lag_{lag}'] = df['Price_Change'].shift(lag)
```

Each lag creates additional NaN values. With max lag=10, first 10 rows have NaN in lagged columns.

**Total NaN Count**: 485 NaN values across all cells

#### Step 3: dropna() Removes All Rows

```python
df = df.dropna()  # Line 152
logger.info(f"Lagged features created: {len(df)} records after removing NaN")
```

**Result**: 0 records remaining

### Why All Rows Have NaN

With 260 records:
1. First 50 rows: NaN from MA_50
2. Rows with lagged features: NaN from shift operations
3. Price change features: NaN from percentage calculations
4. Every row has AT LEAST one NaN value in some column
5. `dropna()` removes ALL rows → 0 training data

### Data Loss Calculation

```
Initial: 260 records
After technical indicators: ~210 complete rows (lost ~50 to MA_50)
After lagged features: 0 rows (lost ALL to dropna())
```

---

## Problem 3: Unicode Encoding Error

### Error Message
```
UnicodeEncodeError: 'cp950' codec can't encode character '\u2705' in position XX:
illegal multibyte sequence
```

### Root Cause

Output contains emoji characters (✅/❌) incompatible with Windows console encoding (cp950/cp1252).

**Location**: Result formatter output

**Cause**: Windows console uses legacy code page (cp950 for Traditional Chinese) that doesn't support Unicode emoji.

---

## Problem 4: Missing end_time Key

### Error Message
```
發生錯誤無法儲存: 'end_time'
```

### Root Cause

Pipeline results dictionary missing `end_time` key when attempting to save results.

**Location**: Result saving logic in pipeline

---

## Model Version Analysis

### Current Model: sklearn-based PatchTST

From [src/currency_predictor/models/factory.py:136-155](src/currency_predictor/models/factory.py#L136-L155):

```python
def create_patchtst_model(use_transformer: bool = None, **kwargs) -> BaseModel:
    if use_transformer is None:
        # Automatically select best version
        model_name = ModelFactory.get_recommended_model(prefer_accuracy=True)
```

**Default Model**: `patchtst_sklearn` (not the transformer version)

**Available Models**:
1. `patchtst_sklearn`: Uses sklearn RandomForestRegressor and GradientBoostingRegressor
2. `patchtst_transformer`: Requires transformers library (not currently available)

**Current Implementation** ([src/currency_predictor/models/patchtst.py](src/currency_predictor/models/patchtst.py)):
- Simulates PatchTST using traditional ML
- Uses patching (sliding windows) + ensemble models
- Does NOT support `validation_split` parameter
- Only supports `validation_data` as tuple of (X_val, y_val)

---

## Data Flow Diagram

```
Yahoo Finance API
    ↓
[260 records collected]
    ↓
DataProcessor.create_technical_indicators()
    ↓ (creates MA_50, RSI, Bollinger Bands, etc.)
[~210 records with complete data, ~50 with NaN]
    ↓
DataProcessor.create_lagged_features(lags=[1,2,3,5,10])
    ↓ (creates shifted features)
[485 total NaN values distributed across all rows]
    ↓
dropna()
    ↓
[0 records remaining] ❌
    ↓
train_test_split()
    ↓
[X_train: 0 rows, X_test: 0 rows]
    ↓
model.fit(X_train, y_train, validation_split=0.2)
    ↓
[TypeError: unexpected keyword argument 'validation_split'] ❌
```

---

## Impact Assessment

| Problem | Severity | Impact | Blocks Training | Blocks Prediction |
|---------|----------|--------|-----------------|-------------------|
| Parameter Mismatch | Critical | Training fails immediately | ✅ Yes | ✅ Yes |
| Zero Training Data | Critical | No data to train on | ✅ Yes | ✅ Yes |
| Unicode Encoding | Medium | Output formatting fails | ❌ No | ❌ No |
| Missing end_time | Low | Results can't be saved | ❌ No | ❌ No |

**Current Status**: Pipeline is completely non-functional due to Problems 1 & 2.

---

## Recommended Solutions

### Solution 1: Fix Parameter Passing

**Option A**: Filter train_kwargs before passing to model.fit()

```python
# In predictor.py train_model()
allowed_params = {'validation_data'}  # Only params that PatchTST.fit() accepts
filtered_kwargs = {k: v for k, v in train_kwargs.items() if k in allowed_params}
self.model.fit(X_train, y_train, **filtered_kwargs)
```

**Option B**: Convert validation_split to validation_data

```python
if 'validation_split' in train_kwargs:
    val_split = train_kwargs.pop('validation_split')
    split_idx = int(len(X_train) * (1 - val_split))
    X_val = X_train.iloc[split_idx:]
    y_val = y_train.iloc[split_idx:]
    X_train = X_train.iloc[:split_idx]
    y_train = y_train.iloc[:split_idx]
    train_kwargs['validation_data'] = (X_val, y_val)

self.model.fit(X_train, y_train, **train_kwargs)
```

### Solution 2: Fix Data Processing

**Option A**: Use fillna() instead of dropna()

```python
# In data_processor.py
df = df.fillna(method='ffill')  # Forward fill
df = df.fillna(method='bfill')  # Backward fill
df = df.fillna(0)  # Or fill with 0
```

**Option B**: Only drop rows with NaN in critical columns

```python
# Only drop NaN in essential columns
essential_cols = ['Close', 'Open', 'High', 'Low']
df = df.dropna(subset=essential_cols)
```

**Option C**: Reduce feature engineering aggressiveness

```python
# Use smaller windows that preserve more data
df['MA_10'] = df['Close'].rolling(window=10).mean()  # Instead of MA_50
df['MA_20'] = df['Close'].rolling(window=20).mean()
# Remove MA_50 entirely

# Use smaller lags
lags = [1, 2, 3, 5]  # Instead of [1, 2, 3, 5, 10]
```

**Option D**: Fill NaN after each feature creation step

```python
def create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_20'] = df['MA_20'].fillna(method='bfill')  # Fill immediately

    # ... other indicators with immediate filling

    return df
```

### Solution 3: Fix Unicode Encoding

Replace emoji characters with ASCII-safe alternatives:

```python
# In formatter
success_marker = "[OK]"  # Instead of ✅
failure_marker = "[FAIL]"  # Instead of ❌
```

### Solution 4: Fix Missing end_time

Ensure pipeline sets end_time before saving:

```python
results['end_time'] = datetime.now().isoformat()
```

---

## Testing Strategy

### Test 1: Parameter Handling
```python
def test_patchtst_fit_parameters():
    """Test that PatchTST.fit() handles parameters correctly"""
    model = PatchTST()
    X_train = pd.DataFrame(...)
    y_train = pd.Series(...)

    # Should not raise error
    model.fit(X_train, y_train)

    # Should raise TypeError
    with pytest.raises(TypeError):
        model.fit(X_train, y_train, validation_split=0.2)
```

### Test 2: Data Processing NaN Handling
```python
def test_data_processor_preserves_records():
    """Test that feature engineering doesn't remove all data"""
    processor = DataProcessor()

    # Create test data with 100 records
    df = create_test_currency_data(n_records=100)

    # Apply feature engineering
    df = processor.create_technical_indicators(df)
    df = processor.create_lagged_features(df)

    # Should still have data remaining
    assert len(df) > 0, "All records were dropped!"
    assert len(df) > 50, f"Too many records lost: {100 - len(df)}"
```

### Test 3: Full Pipeline Integration
```python
def test_basic_usage_pipeline():
    """Test the complete pipeline from basic_usage.py"""
    config_manager = ConfigManager()
    config = config_manager.get_config()

    pipeline = PredictionPipeline(config, output_dir="test_results")

    results = pipeline.run_full_pipeline(
        symbols=["USDTWD=X"],
        prediction_horizon=7,
        save_results=False
    )

    # Verify success
    assert results.get('success', False), "Pipeline failed"
    assert len(results.get('predictions', {})) > 0, "No predictions generated"
```

---

## Implementation Priority

1. **HIGH PRIORITY**: Fix Solution 1 (parameter passing) - Blocks all training
2. **HIGH PRIORITY**: Fix Solution 2 (data processing) - Blocks all training
3. **MEDIUM PRIORITY**: Fix Solution 3 (unicode encoding) - Improves UX
4. **LOW PRIORITY**: Fix Solution 4 (missing end_time) - Minor issue

---

## Next Steps

1. ✅ Complete problem analysis (DONE)
2. ⏳ Create pytest tests for each issue
3. ⏳ Implement fixes for Problems 1 & 2
4. ⏳ Run tests to verify fixes
5. ⏳ Implement fixes for Problems 3 & 4
6. ⏳ Execute basic_usage.py to verify full pipeline
7. ⏳ Update documentation with lessons learned

---

## Related Files

- [src/currency_predictor/prediction/predictor.py](src/currency_predictor/prediction/predictor.py) - Parameter passing issue
- [src/currency_predictor/models/patchtst.py](src/currency_predictor/models/patchtst.py) - Model signature
- [src/currency_predictor/data_processor.py](src/currency_predictor/data_processor.py) - Feature engineering
- [examples/basic_usage.py](examples/basic_usage.py) - Entry point
- [tests/test_visualizer.py](tests/test_visualizer.py) - Example test structure

---

## Appendix: Full Error Log

```
2026-01-25 00:46:39,929 - Starting Currency Prediction Pipeline (Basic Usage)
2026-01-25 00:46:39,936 - ====================================================================================================
2026-01-25 00:46:39,936 - 步驟 1/5: 載入配置
2026-01-25 00:46:39,936 - ====================================================================================================
2026-01-25 00:46:40,037 - Configuration loaded successfully from: d:\learn_python\currency_predict_attempt\.currency_predictor\config.json
2026-01-25 00:46:40,038 - ====================================================================================================
2026-01-25 00:46:40,038 - 步驟 2/5: 收集貨幣資料
2026-01-25 00:46:40,038 - ====================================================================================================
2026-01-25 00:46:40,039 - 開始收集 USDTWD=X 的資料...
2026-01-25 00:46:41,050 - 成功收集 USDTWD=X 資料: 260 筆
2026-01-25 00:46:41,051 - 資料儲存至: data\USDTWD=X.csv
2026-01-25 00:46:41,054 - 收集了 260 筆資料，時間範圍: 2024-02-22 to 2026-01-24
2026-01-25 00:46:41,055 - ====================================================================================================
2026-01-25 00:46:41,055 - 步驟 3/5: 準備訓練資料
2026-01-25 00:46:41,055 - ====================================================================================================
2026-01-25 00:46:41,059 - 開始準備訓練資料...
2026-01-25 00:46:41,059 - Data cleaned: 260 records remaining
2026-01-25 00:46:41,076 - Technical indicators created: 30 total features
2026-01-25 00:46:44,004 - Lagged features created: 0 records after removing NaN (had 485 NaN values)
2026-01-25 00:46:44,004 - Features prepared: 0 features, 0 samples
2026-01-25 00:46:44,013 - Data split: 0 train, 0 test samples
2026-01-25 00:46:44,013 - 訓練資料: 0 筆，測試資料: 0 筆
2026-01-25 00:46:44,013 - ====================================================================================================
2026-01-25 00:46:44,013 - 步驟 4/5: 訓練 PatchTST 模型
2026-01-25 00:46:44,013 - ====================================================================================================
2026-01-25 00:46:44,013 - 開始訓練 PatchTST 模型
2026-01-25 00:46:44,013 - ERROR: 模型訓練失敗: PatchTST.fit() got an unexpected keyword argument 'validation_split'
```
