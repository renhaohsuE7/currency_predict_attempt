"""統一 fit() 介面的契約測試(Phase 3)。

三個 PatchTST 實作與抽象基底都應採同一 fit 簽名:
    fit(self, X, y, validation_data=None, **kwargs)
使上層 CurrencyPredictor.train_model 能用同一條呼叫路徑驅動任一實作。
此處用 inspect 檢查簽名,毋須真的訓練(快速、不需資料/GPU)。
"""

import inspect

import pytest

from currency_predictor.models.base import BaseModel
from currency_predictor.models.patchtst import (
    PatchTSTSklearn,
    PatchTSTHuggingFace,
    PatchTSTLightningWrapper,
)

CLASSES = [
    pytest.param(BaseModel, id="base"),
    pytest.param(PatchTSTSklearn, id="sklearn"),
    pytest.param(
        PatchTSTHuggingFace, id="huggingface",
        marks=pytest.mark.skipif(PatchTSTHuggingFace is None, reason="transformers 未安裝"),
    ),
    pytest.param(
        PatchTSTLightningWrapper, id="lightning",
        marks=pytest.mark.skipif(PatchTSTLightningWrapper is None, reason="lightning 未安裝"),
    ),
]


@pytest.mark.parametrize("cls", CLASSES)
def test_fit_has_validation_data_param(cls):
    params = inspect.signature(cls.fit).parameters
    assert "validation_data" in params, f"{cls.__name__}.fit 缺少 validation_data 參數"
    assert params["validation_data"].default is None


@pytest.mark.parametrize("cls", CLASSES)
def test_fit_accepts_var_keyword(cls):
    params = inspect.signature(cls.fit).parameters
    kinds = {p.kind for p in params.values()}
    assert inspect.Parameter.VAR_KEYWORD in kinds, f"{cls.__name__}.fit 應接受 **kwargs"


@pytest.mark.parametrize("cls", CLASSES)
def test_fit_leading_params_are_x_y(cls):
    names = list(inspect.signature(cls.fit).parameters)
    # self, X, y, ...
    assert names[:3] == ["self", "X", "y"], f"{cls.__name__}.fit 前三個參數應為 self, X, y"
