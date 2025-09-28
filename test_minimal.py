"""
簡化版測試腳本

使用最小的技術指標進行測試
"""

import sys
from pathlib import Path
import logging
import pandas as pd

# 添加項目路徑
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.currency_predictor.prediction import CurrencyPredictor
from src.currency_predictor.utils import setup_logging

def test_minimal_functionality():
    """測試最小功能"""
    print("=== 測試最小功能 ===")
    
    # 設置日誌
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)
    
    try:
        # 1. 創建預測器，使用最小參數
        print("1. 創建預測器...")
        predictor = CurrencyPredictor(
            model_name="PatchTST",
            model_params={
                'seq_len': 30,   # 最小序列長度
                'pred_len': 3,   # 預測3天
                'patch_len': 5,  # 小patch
                'stride': 2,     # 小步長
                'n_estimators': 10,
                'max_depth': 3,
                'random_state': 42
            }
        )
        print("✅ 預測器創建成功")
        
        # 2. 測試資料收集
        print("2. 測試資料收集...")
        symbol = "USDTWD=X"
        data_result = predictor.collect_and_store_data(
            [symbol], 
            period="1y",  # 使用一年的資料
            force_update=True
        )
        print(f"資料收集結果: {data_result}")
        
        if data_result.get(symbol, False):
            print("✅ 資料收集成功")
        else:
            print("❌ 資料收集失敗")
            return False
        
        # 3. 直接載入資料並檢查
        print("3. 檢查載入的資料...")
        raw_data = predictor.data_storage.load_raw_data("USDTWD", "1y")
        print(f"原始資料形狀: {raw_data.shape}")
        
        # 簡單清理資料
        cleaned_data = predictor.data_processor.clean_data(raw_data)
        print(f"清理後資料形狀: {cleaned_data.shape}")
        
        # 只創建最基本的特徵
        print("4. 創建最基本特徵...")
        
        # 手動創建簡單特徵避免過多NaN
        feature_data = cleaned_data.copy()
        feature_data['Price_Change'] = feature_data['Close'].pct_change()
        feature_data['SMA_5'] = feature_data['Close'].rolling(window=5).mean()
        feature_data['Volume_MA'] = feature_data.get('Volume', pd.Series(index=feature_data.index, data=1)).rolling(window=5).mean()
        
        # 刪除NaN值
        feature_data = feature_data.dropna()
        print(f"特徵資料形狀: {feature_data.shape}")
        
        if len(feature_data) < 50:
            print(f"❌ 資料太少 ({len(feature_data)} 筆)，無法訓練")
            return False
        
        # 5. 手動準備訓練資料
        print("5. 準備訓練資料...")
        
        # 使用簡單的特徵
        feature_columns = ['Open', 'High', 'Low', 'Price_Change', 'SMA_5']
        available_features = [col for col in feature_columns if col in feature_data.columns]
        
        X = feature_data[available_features]
        y = feature_data['Close']
        
        # 簡單分割
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        print(f"訓練集: {len(X_train)} 筆，測試集: {len(X_test)} 筆")
        
        # 6. 直接訓練模型
        print("6. 訓練模型...")
        predictor.model.fit(X_train, y_train)
        print("✅ 模型訓練成功")
        
        # 7. 測試預測
        print("7. 測試預測...")
        predictions = predictor.model.predict(X_test[-30:], horizon=3)  # 使用最後30筆資料預測3天
        print(f"預測結果: {predictions}")
        print("✅ 預測成功")
        
        # 8. 測試模型儲存
        print("8. 測試模型儲存...")
        Path("test_models").mkdir(exist_ok=True)
        model_path = "test_models/test_minimal_model.joblib"
        save_success = predictor.save_model(model_path)
        
        if save_success:
            print("✅ 模型儲存成功")
        else:
            print("❌ 模型儲存失敗")
            return False
        
        print("\n🎉 最小功能測試通過！")
        return True
        
    except Exception as e:
        print(f"❌ 測試過程發生錯誤: {str(e)}")
        logger.error(f"測試錯誤詳細信息: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    """主測試程序"""
    print("貨幣預測系統最小功能測試")
    print("=" * 50)
    
    # 運行測試
    test_result = test_minimal_functionality()
    
    # 結果摘要
    print("\n" + "=" * 50)
    print("測試結果摘要")
    print("=" * 50)
    print(f"最小功能測試: {'✅ PASS' if test_result else '❌ FAIL'}")
    
    if test_result:
        print("\n🎉 最小功能測試通過！系統基本架構運作正常。")
        print("\n接下來可以:")
        print("1. 優化技術指標計算，減少NaN值")
        print("2. 改進資料處理流程")
        print("3. 完善完整的訓練和預測流程")
    else:
        print("\n❌ 測試失敗，需要進一步檢查和修正。")