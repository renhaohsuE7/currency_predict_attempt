"""
簡單測試腳本

測試新的預測系統是否正常工作
"""

import sys
from pathlib import Path
import logging

# 添加項目路徑
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.currency_predictor.prediction import CurrencyPredictor
from src.currency_predictor.utils import setup_logging

def test_basic_functionality():
    """測試基本功能"""
    print("=== 測試基本功能 ===")
    
    # 設置日誌
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)
    
    try:
        # 1. 創建預測器
        print("1. 創建預測器...")
        predictor = CurrencyPredictor(
            model_name="PatchTST",
            model_params={
                'seq_len': 48,  # 使用較短的序列長度進行測試
                'pred_len': 7,   # 預測7天
                'patch_len': 6,  # 使用較小的patch_len進行測試
                'stride': 3,     # 步長
                'n_estimators': 10,  # 減少估計器數量加速測試
                'max_depth': 5,
                'random_state': 42
            }
        )
        print("✅ 預測器創建成功")
        
        # 2. 測試資料收集
        print("2. 測試資料收集...")
        symbol = "USDTWD=X"
        data_result = predictor.collect_and_store_data(
            [symbol], 
            period="6mo",  # 使用6個月期間獲得更多資料
            force_update=True
        )
        print(f"資料收集結果: {data_result}")
        
        if data_result.get(symbol, False):
            print("✅ 資料收集成功")
        else:
            print("❌ 資料收集失敗")
            return False
        
        # 3. 測試模型訓練
        print("3. 測試模型訓練...")
        training_result = predictor.train_model(
            symbol, 
            period="6mo"
        )
        
        if training_result.get('training_completed', False):
            print("✅ 模型訓練成功")
            print(f"訓練集大小: {training_result.get('train_size', 0)}")
            print(f"測試集大小: {training_result.get('test_size', 0)}")
            
            if training_result.get('test_metrics'):
                metrics = training_result['test_metrics']
                print(f"測試 RMSE: {metrics.get('rmse', 0):.6f}")
                print(f"測試 MAE: {metrics.get('mae', 0):.6f}")
        else:
            print("❌ 模型訓練失敗")
            error = training_result.get('error', '未知錯誤')
            print(f"錯誤: {error}")
            return False
        
        # 4. 測試預測
        print("4. 測試預測...")
        prediction_result = predictor.predict(
            symbol, 
            horizon=3,  # 預測3天減少測試時間
            return_uncertainty=True
        )
        
        if not prediction_result.get('error'):
            print("✅ 預測成功")
            print(f"最後已知值: {prediction_result.get('last_known_value', 0):.4f}")
            predictions = prediction_result.get('predictions', [])
            print(f"預測值: {predictions}")
            
            if 'uncertainty' in prediction_result:
                print(f"不確定性: {prediction_result['uncertainty']}")
        else:
            print("❌ 預測失敗")
            print(f"錯誤: {prediction_result.get('error', '未知錯誤')}")
            return False
        
        # 5. 測試模型儲存和載入
        print("5. 測試模型儲存和載入...")
        
        # 創建模型目錄
        Path("test_models").mkdir(exist_ok=True)
        
        # 儲存模型
        model_path = f"test_models/test_{symbol.replace('=X', '')}_PatchTST.joblib"
        save_success = predictor.save_model(model_path)
        
        if save_success:
            print("✅ 模型儲存成功")
            
            # 創建新的預測器並載入模型
            new_predictor = CurrencyPredictor()
            load_success = new_predictor.load_model(model_path)
            
            if load_success:
                print("✅ 模型載入成功")
                
                # 測試載入的模型是否能預測
                test_prediction = new_predictor.predict(symbol, horizon=1)
                if not test_prediction.get('error'):
                    print("✅ 載入的模型預測成功")
                else:
                    print("❌ 載入的模型預測失敗")
                    return False
            else:
                print("❌ 模型載入失敗")
                return False
        else:
            print("❌ 模型儲存失敗")
            return False
        
        print("\n🎉 所有基本功能測試通過！")
        return True
        
    except Exception as e:
        print(f"❌ 測試過程發生錯誤: {str(e)}")
        logger.error(f"測試錯誤詳細信息: {str(e)}", exc_info=True)
        return False


def test_model_info():
    """測試模型資訊功能"""
    print("\n=== 測試模型資訊功能 ===")
    
    try:
        predictor = CurrencyPredictor()
        info = predictor.get_model_info()
        
        print("模型資訊:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("✅ 模型資訊獲取成功")
        return True
        
    except Exception as e:
        print(f"❌ 模型資訊測試失敗: {str(e)}")
        return False


if __name__ == "__main__":
    """主測試程序"""
    print("貨幣預測系統基本功能測試")
    print("=" * 50)
    
    # 運行測試
    test1 = test_basic_functionality()
    test2 = test_model_info()
    
    # 結果摘要
    print("\n" + "=" * 50)
    print("測試結果摘要")
    print("=" * 50)
    print(f"基本功能測試: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"模型資訊測試: {'✅ PASS' if test2 else '❌ FAIL'}")
    
    overall_success = test1 and test2
    print(f"整體測試結果: {'✅ PASS' if overall_success else '❌ FAIL'}")
    
    if overall_success:
        print("\n🎉 所有測試通過！新的預測系統運作正常。")
        print("\n下一步可以:")
        print("1. 運行 main.py 進行完整預測")
        print("2. 運行 example_usage.py 查看更多使用範例") 
        print("3. 檢查 'test_models/' 目錄中儲存的測試模型")
    else:
        print("\n❌ 部分測試失敗，請檢查錯誤信息並修正。")
    
    # 清理測試檔案（可選）
    import shutil
    try:
        if Path("test_models").exists():
            # shutil.rmtree("test_models")  # 取消註解以清理測試檔案
            pass
    except:
        pass