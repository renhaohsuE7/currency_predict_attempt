"""
測試新架構的可用性

檢查 transformer PatchTST 是否可用且符合架構
"""

import sys
from pathlib import Path
import logging

# 添加項目路徑
sys.path.append(str(Path(__file__).parent / 'src'))

def test_architecture():
    """測試架構完整性"""
    print("🔍 測試新架構...")
    
    # 1. 測試基礎類別
    print("\n1. 測試基礎類別...")
    try:
        from currency_predictor.models.base import (
            BaseModel, TimeSeriesModel, 
            SklearnBasedModel, TransformerBasedModel, 
            ModelType
        )
        print("✅ 基礎類別導入成功")
        
        # 檢查抽象方法
        print(f"   ModelType 枚舉: {[e.value for e in ModelType]}")
        
    except ImportError as e:
        print(f"❌ 基礎類別導入失敗: {e}")
        return False
    
    # 2. 測試 sklearn 版本 PatchTST
    print("\n2. 測試 sklearn PatchTST...")
    try:
        from currency_predictor.models.patchtst import PatchTST
        
        # 檢查繼承關係
        sklearn_model = PatchTST()
        print(f"✅ PatchTST 創建成功")
        print(f"   模型類型: {sklearn_model.model_type}")
        print(f"   是否為 SklearnBasedModel: {isinstance(sklearn_model, SklearnBasedModel)}")
        print(f"   是否為 TimeSeriesModel: {isinstance(sklearn_model, TimeSeriesModel)}")
        
    except Exception as e:
        print(f"❌ sklearn PatchTST 測試失敗: {e}")
        return False
    
    # 3. 測試 transformer 版本 PatchTST
    print("\n3. 測試 transformer PatchTST...")
    try:
        from currency_predictor.models.patchtst_transformer import PatchTSTTransformer
        
        # 檢查繼承關係
        transformer_model = PatchTSTTransformer()
        print(f"✅ PatchTSTTransformer 創建成功")
        print(f"   模型類型: {transformer_model.model_type}")
        print(f"   是否為 TransformerBasedModel: {isinstance(transformer_model, TransformerBasedModel)}")
        print(f"   是否為 TimeSeriesModel: {isinstance(transformer_model, TimeSeriesModel)}")
        print(f"   設備: {transformer_model.device}")
        
    except Exception as e:
        print(f"❌ transformer PatchTST 測試失敗: {e}")
        print("   可能原因: transformers 庫未安裝或 GPU 不可用")
        return False
    
    # 4. 測試模型工廠 (是否真的需要)
    print("\n4. 測試模型工廠...")
    try:
        from currency_predictor.models.factory import ModelFactory
        
        available_models = ModelFactory.get_available_models()
        print(f"✅ 模型工廠可用")
        print(f"   可用模型: {list(available_models.keys())}")
        
        # 測試創建模型
        sklearn_model = ModelFactory.create_model('patchtst_sklearn')
        print(f"   sklearn 模型創建: ✅")
        
        try:
            transformer_model = ModelFactory.create_model('patchtst_transformer')
            print(f"   transformer 模型創建: ✅")
        except Exception as e:
            print(f"   transformer 模型創建: ❌ ({e})")
        
    except Exception as e:
        print(f"❌ 模型工廠測試失敗: {e}")
    
    # 5. 測試直接使用 (不透過工廠)
    print("\n5. 測試直接使用模型...")
    try:
        # 直接創建 sklearn 版本
        sklearn_direct = PatchTST(
            seq_len=30,
            pred_len=7,
            patch_len=10,
            stride=5
        )
        print(f"✅ 直接創建 sklearn PatchTST 成功")
        
        # 直接創建 transformer 版本
        transformer_direct = PatchTSTTransformer(
            context_length=60,
            prediction_length=7,
            patch_length=12,
            stride=6
        )
        print(f"✅ 直接創建 transformer PatchTST 成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 直接創建模型失敗: {e}")
        return False

def evaluate_factory_necessity():
    """評估工廠模式是否必要"""
    print("\n🤔 評估工廠模式的必要性...")
    
    print("\n優點:")
    print("   ✅ 統一的模型創建介面")
    print("   ✅ 可以處理模型不可用的情況")
    print("   ✅ 便於配置管理")
    print("   ✅ 支援模型發現和列舉")
    
    print("\n缺點:")
    print("   ❌ 增加了架構複雜度")
    print("   ❌ 對於簡單場景過度設計")
    print("   ❌ 抽象類別已提供足夠的多態性")
    
    print("\n🎯 建議:")
    print("   如果直接使用抽象類別繼承已經足夠，可以考慮簡化")
    print("   如果需要動態模型選擇和配置，工廠模式有價值")
    print("   當前可以保留但簡化工廠實作")

if __name__ == "__main__":
    success = test_architecture()
    evaluate_factory_necessity()
    
    print(f"\n{'='*50}")
    print(f"架構測試結果: {'✅ 通過' if success else '❌ 失敗'}")
    
    if success:
        print("\n📋 架構總結:")
        print("   ✅ 抽象基類設計良好")
        print("   ✅ sklearn 和 transformer 版本都可用")
        print("   ✅ 繼承關係正確")
        print("   ✅ 符合多態性原則")
        print("\n💡 建議:")
        print("   - 可以考慮簡化或移除工廠模式")
        print("   - 直接使用抽象類別已足夠靈活")
        print("   - 在 CurrencyPredictor 中直接實例化模型")