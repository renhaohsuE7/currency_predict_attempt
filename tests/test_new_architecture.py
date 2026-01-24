#!/usr/bin/env python3
"""
測試新的模型架構

驗證 sklearn 和 transformer 版本的 PatchTST
"""

import sys
import logging
from pathlib import Path

# 添加 src 目錄到 Python 路徑
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from currency_predictor.models.factory import ModelFactory, create_patchtst_model

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_model_factory():
    """測試模型工廠"""
    print("=" * 60)
    print("測試模型工廠")
    print("=" * 60)
    
    # 列印可用模型
    ModelFactory.print_model_info()
    
    print("\\n" + "=" * 60)
    print("測試模型創建")
    print("=" * 60)
    
    # 測試 sklearn 版本
    try:
        print("\\n1. 測試 Sklearn 版本 PatchTST:")
        sklearn_model = ModelFactory.create_model("patchtst_sklearn")
        print(f"   ✅ 成功創建: {sklearn_model.get_model_info()}")
    except Exception as e:
        print(f"   ❌ 創建失敗: {str(e)}")
    
    # 測試 transformer 版本
    try:
        print("\\n2. 測試 Transformer 版本 PatchTST:")
        transformer_model = ModelFactory.create_model("patchtst_transformer")
        print(f"   ✅ 成功創建: {transformer_model.get_model_info()}")
    except Exception as e:
        print(f"   ❌ 創建失敗: {str(e)}")
    
    # 測試便利函數
    try:
        print("\\n3. 測試便利函數:")
        auto_model = create_patchtst_model()
        print(f"   ✅ 自動選擇模型: {auto_model.get_model_info()}")
        
        sklearn_model_via_func = create_patchtst_model(use_transformer=False)
        print(f"   ✅ 強制選擇 sklearn: {sklearn_model_via_func.get_model_info()}")
        
        transformer_model_via_func = create_patchtst_model(use_transformer=True)
        print(f"   ✅ 強制選擇 transformer: {transformer_model_via_func.get_model_info()}")
        
    except Exception as e:
        print(f"   ❌ 便利函數測試失敗: {str(e)}")


def test_currency_predictor():
    """測試貨幣預測器"""
    print("\\n" + "=" * 60)
    print("測試貨幣預測器整合")
    print("=" * 60)
    
    from currency_predictor.prediction.predictor import CurrencyPredictor
    
    # 測試不同模型的預測器
    models_to_test = ["patchtst_sklearn"]
    
    # 如果有 transformers，也測試 transformer 版本
    available_models = ModelFactory.get_available_models()
    if available_models['patchtst_transformer']['available']:
        models_to_test.append("patchtst_transformer")
    
    for model_name in models_to_test:
        try:
            print(f"\\n測試預測器 ({model_name}):")
            predictor = CurrencyPredictor(
                model_name=model_name,
                model_params={'context_length': 32}  # 減少上下文長度以便快速測試
            )
            print(f"   ✅ 成功創建預測器: {predictor.model.get_model_info()}")
            
        except Exception as e:
            print(f"   ❌ 預測器創建失敗: {str(e)}")


if __name__ == "__main__":
    print("🚀 開始測試新的模型架構...")
    
    try:
        test_model_factory()
        test_currency_predictor()
        
        print("\\n" + "=" * 60)
        print("✅ 所有測試完成!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"測試過程中發生錯誤: {str(e)}")
        print("\\n" + "=" * 60)
        print("❌ 測試失敗!")
        print("=" * 60)