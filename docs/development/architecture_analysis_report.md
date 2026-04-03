# 專案架構分析報告

## 1. 現有類別架構分析

### 1.1 繼承層次結構

```
BaseModel (ABC)                          # 抽象基類 - 定義所有模型的介面
├── TimeSeriesModel                      # 時間序列模型基類
│   ├── SklearnBasedModel               # sklearn 模型基類
│   │   └── PatchTST                    # sklearn 版 PatchTST 實作
│   └── TransformerBasedModel           # Transformer 模型基類
│       └── PatchTSTTransformer         # HuggingFace PatchTST 實作
```

### 1.2 ABC (Abstract Base Class) 使用情況

#### BaseModel (base.py:24-148)
```python
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y, validation_data=None) -> 'BaseModel': pass

    @abstractmethod
    def predict(self, X, horizon=1) -> np.ndarray: pass

    @abstractmethod
    def predict_with_uncertainty(self, X, horizon=1, confidence_level=0.95) -> Dict: pass
```

**評估**: ✅ 正確使用 ABC，強制子類實作核心方法

#### TransformerBasedModel (base.py:233-266)
```python
class TransformerBasedModel(TimeSeriesModel):
    @abstractmethod
    def prepare_data_for_transformer(self, data) -> Dict[str, Any]: pass

    @abstractmethod
    def setup_model(self, **model_kwargs): pass
```

**評估**: ✅ 正確使用 ABC，為 Transformer 模型定義額外的抽象方法

---

## 2. 問題識別

### 2.1 介面不一致問題

#### 問題 A: fit() 方法參數不一致

| 模型 | fit() 參數 |
|------|-----------|
| `PatchTST` | `X, y, validation_data=None` |
| `PatchTSTTransformer` | `X, y, validation_data=None, num_epochs=50, batch_size=32, learning_rate=1e-4, ...` |

**問題**: 兩個模型的 `fit()` 介面不一致，`PatchTSTTransformer` 有大量額外參數。

**建議**:
```python
# 選項 1: 使用 TrainingConfig 物件
class TrainingConfig:
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    ...

def fit(self, X, y, validation_data=None, config: TrainingConfig = None)

# 選項 2: 使用 **kwargs 並在內部處理
def fit(self, X, y, validation_data=None, **train_kwargs)
```

### 2.2 職責不清問題

#### 問題 B: CurrencyPredictor 職責過多

`CurrencyPredictor` 目前負責:
1. 資料收集 (`collect_and_store_data`)
2. 資料處理 (`prepare_training_data`)
3. 模型訓練 (`train_model`)
4. 模型評估 (`_evaluate_model`)
5. 預測執行 (`predict`)
6. 模型持久化 (`save_model`, `load_model`)

**違反單一職責原則 (SRP)**

**建議架構**:
```
CurrencyPredictor (Facade)
├── DataManager           # 資料收集與儲存
├── FeatureEngineer       # 特徵工程
├── ModelTrainer          # 模型訓練
├── ModelEvaluator        # 模型評估
└── PredictionEngine      # 預測執行
```

### 2.3 缺少 Trainer 抽象

#### 問題 C: 訓練邏輯分散

訓練邏輯分散在:
1. `PatchTST.fit()` - sklearn 訓練邏輯
2. `PatchTSTTransformer.fit()` - HuggingFace Trainer 邏輯
3. `CurrencyPredictor.train_model()` - 資料準備和驗證分割

**建議**: 引入 `Trainer` 抽象類別

```python
class BaseTrainer(ABC):
    @abstractmethod
    def train(self, model, train_data, val_data=None, config=None): pass

    @abstractmethod
    def evaluate(self, model, data): pass

class SklearnTrainer(BaseTrainer): ...
class TransformerTrainer(BaseTrainer): ...
```

### 2.4 策略模式未充分利用

#### 問題 D: Scaler 策略寫死

```python
# PatchTST
self.scaler = StandardScaler()

# PatchTSTTransformer
self.scaler = StandardScaler()
```

**建議**: 使用策略模式

```python
class ScalerStrategy(ABC):
    @abstractmethod
    def fit_transform(self, data): pass
    @abstractmethod
    def inverse_transform(self, data): pass

class StandardScalerStrategy(ScalerStrategy): ...
class MinMaxScalerStrategy(ScalerStrategy): ...
class RobustScalerStrategy(ScalerStrategy): ...
```

---

## 3. 良好設計確認

### 3.1 正確使用的設計模式

| 模式 | 位置 | 評估 |
|------|------|------|
| **Factory Pattern** | `ModelFactory` | ✅ 正確實作 |
| **Template Method** | `BaseModel` 的 `get_model_info()`, `save_model()`, `load_model()` | ✅ 提供預設實作 |
| **Strategy Pattern** | `ModelType` enum | ✅ 區分模型類型 |
| **Pipeline Pattern** | `PredictionPipeline` | ✅ 流程編排 |

### 3.2 ABC 使用正確的地方

```python
# base.py - 強制子類實作核心方法
class BaseModel(ABC):
    @abstractmethod
    def fit(...): pass

    @abstractmethod
    def predict(...): pass

    @abstractmethod
    def predict_with_uncertainty(...): pass

# TransformerBasedModel - 為 Transformer 定義額外合約
class TransformerBasedModel(TimeSeriesModel):
    @abstractmethod
    def prepare_data_for_transformer(...): pass

    @abstractmethod
    def setup_model(...): pass
```

---

## 4. 重構建議

### 4.1 短期改進 (Low Effort, High Impact)

#### 4.1.1 統一 fit() 介面

```python
# 修改 BaseModel
class BaseModel(ABC):
    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple] = None,
        **kwargs  # 允許額外參數
    ) -> 'BaseModel':
        pass
```

#### 4.1.2 引入 TrainingConfig

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class TrainingConfig:
    """訓練配置"""
    # 通用參數
    validation_split: float = 0.2
    random_state: int = 42

    # Transformer 專用參數
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    early_stopping_patience: int = 10

    # sklearn 專用參數
    n_estimators: int = 100
    max_depth: int = 10

    # 額外參數
    extra_params: Dict[str, Any] = field(default_factory=dict)
```

### 4.2 中期改進 (Medium Effort)

#### 4.2.1 拆分 CurrencyPredictor

```python
# 新的模組結構
src/currency_predictor/
├── data/
│   ├── manager.py          # DataManager - 資料收集與儲存
│   └── feature_engineer.py # FeatureEngineer - 特徵工程
├── training/
│   ├── base.py             # BaseTrainer
│   ├── sklearn_trainer.py  # SklearnTrainer
│   └── transformer_trainer.py  # TransformerTrainer
├── evaluation/
│   └── evaluator.py        # ModelEvaluator
└── prediction/
    ├── engine.py           # PredictionEngine (核心預測)
    └── predictor.py        # CurrencyPredictor (Facade)
```

#### 4.2.2 引入 Trainer 抽象

```python
class BaseTrainer(ABC):
    """訓練器基類"""

    @abstractmethod
    def train(
        self,
        model: BaseModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        config: Optional[TrainingConfig] = None
    ) -> Dict[str, Any]:
        """執行訓練"""
        pass

    @abstractmethod
    def evaluate(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """評估模型"""
        pass


class SklearnTrainer(BaseTrainer):
    """sklearn 模型訓練器"""

    def train(self, model, X_train, y_train, X_val=None, y_val=None, config=None):
        config = config or TrainingConfig()

        validation_data = (X_val, y_val) if X_val is not None else None
        model.fit(X_train, y_train, validation_data=validation_data)

        return {'status': 'completed'}

    def evaluate(self, model, X, y):
        return model.evaluate(X, y)


class TransformerTrainer(BaseTrainer):
    """Transformer 模型訓練器"""

    def train(self, model, X_train, y_train, X_val=None, y_val=None, config=None):
        config = config or TrainingConfig()

        validation_data = (X_val, y_val) if X_val is not None else None

        model.fit(
            X_train, y_train,
            validation_data=validation_data,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            early_stopping_patience=config.early_stopping_patience
        )

        return {
            'status': 'completed',
            'history': model.training_history
        }

    def evaluate(self, model, X, y):
        return model.evaluate(X, y)
```

### 4.3 長期改進 (High Effort)

#### 4.3.1 完整策略模式

```python
# 資料標準化策略
class ScalerStrategy(ABC):
    @abstractmethod
    def fit(self, data: np.ndarray) -> 'ScalerStrategy': pass

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray: pass

    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray: pass

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)


# 特徵工程策略
class FeatureStrategy(ABC):
    @abstractmethod
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame: pass


class TechnicalIndicatorStrategy(FeatureStrategy):
    """技術指標特徵"""
    def create_features(self, data):
        # 創建 SMA, EMA, MACD, RSI 等
        pass


class LaggedFeatureStrategy(FeatureStrategy):
    """滯後特徵"""
    def __init__(self, lags: List[int]):
        self.lags = lags

    def create_features(self, data):
        # 創建滯後特徵
        pass


# 組合策略
class FeatureEngineer:
    def __init__(self, strategies: List[FeatureStrategy]):
        self.strategies = strategies

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        for strategy in self.strategies:
            result = strategy.create_features(result)
        return result
```

#### 4.3.2 依賴注入容器

```python
from typing import TypeVar, Type

T = TypeVar('T')

class Container:
    """簡單的依賴注入容器"""
    _registry: Dict[Type, Any] = {}

    @classmethod
    def register(cls, interface: Type[T], implementation: T):
        cls._registry[interface] = implementation

    @classmethod
    def resolve(cls, interface: Type[T]) -> T:
        if interface not in cls._registry:
            raise KeyError(f"No implementation registered for {interface}")
        return cls._registry[interface]


# 使用
Container.register(BaseTrainer, TransformerTrainer())
Container.register(ScalerStrategy, StandardScalerStrategy())

trainer = Container.resolve(BaseTrainer)
scaler = Container.resolve(ScalerStrategy)
```

---

## 5. 重構優先順序

| 優先級 | 任務 | 複雜度 | 影響 |
|--------|------|--------|------|
| 1 | 統一 fit() 介面使用 **kwargs | 低 | 高 |
| 2 | 引入 TrainingConfig dataclass | 低 | 中 |
| 3 | 引入 BaseTrainer 抽象 | 中 | 高 |
| 4 | 拆分 CurrencyPredictor 職責 | 中 | 高 |
| 5 | 實作 ScalerStrategy | 低 | 中 |
| 6 | 實作 FeatureStrategy | 中 | 中 |
| 7 | 依賴注入容器 | 高 | 中 |

---

## 6. 類別圖 (重構後)

```
                          ┌─────────────────┐
                          │   BaseModel     │
                          │     (ABC)       │
                          ├─────────────────┤
                          │ +fit()          │
                          │ +predict()      │
                          │ +predict_with_  │
                          │  uncertainty()  │
                          └────────┬────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
   ┌──────────▼─────────┐ ┌───────▼────────┐ ┌────────▼────────┐
   │ SklearnBasedModel  │ │TransformerBased│ │  FutureModel    │
   │                    │ │     Model      │ │    (擴展)        │
   └──────────┬─────────┘ └───────┬────────┘ └─────────────────┘
              │                   │
   ┌──────────▼─────────┐ ┌───────▼────────┐
   │     PatchTST       │ │ PatchTST       │
   │    (sklearn)       │ │ Transformer    │
   └────────────────────┘ └────────────────┘


                          ┌─────────────────┐
                          │  BaseTrainer    │
                          │     (ABC)       │
                          ├─────────────────┤
                          │ +train()        │
                          │ +evaluate()     │
                          └────────┬────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
   ┌──────────▼─────────┐ ┌───────▼────────┐ ┌────────▼────────┐
   │  SklearnTrainer    │ │Transformer     │ │ CustomTrainer   │
   │                    │ │   Trainer      │ │    (擴展)        │
   └────────────────────┘ └────────────────┘ └─────────────────┘


   ┌─────────────────────────────────────────────────────────────┐
   │                    CurrencyPredictor                        │
   │                       (Facade)                              │
   ├─────────────────────────────────────────────────────────────┤
   │ - data_manager: DataManager                                 │
   │ - feature_engineer: FeatureEngineer                         │
   │ - trainer: BaseTrainer                                      │
   │ - evaluator: ModelEvaluator                                 │
   │ - model: BaseModel                                          │
   ├─────────────────────────────────────────────────────────────┤
   │ + collect_data()                                            │
   │ + train()                                                   │
   │ + predict()                                                 │
   │ + evaluate()                                                │
   └─────────────────────────────────────────────────────────────┘
```

---

## 7. 結論

### 目前架構優點
1. ✅ 正確使用 ABC 定義模型介面
2. ✅ Factory Pattern 用於模型創建
3. ✅ 繼承層次結構清晰
4. ✅ Pipeline 模式用於流程編排

### 需要改進的地方
1. ⚠️ fit() 介面參數不一致
2. ⚠️ CurrencyPredictor 職責過多
3. ⚠️ 缺少 Trainer 抽象
4. ⚠️ 策略模式未充分利用

### 建議下一步
1. 先實作 `TrainingConfig` dataclass
2. 引入 `BaseTrainer` 抽象類別
3. 逐步拆分 `CurrencyPredictor` 的職責
