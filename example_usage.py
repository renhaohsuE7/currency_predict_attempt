"""
貨幣預測系統使用範例

展示如何使用預測系統進行端到端的貨幣預測
"""

import sys
from pathlib import Path
import json
import logging

# 添加項目路徑到 sys.path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.currency_predictor.prediction import CurrencyPredictor, PredictionPipeline

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def example_single_prediction():
    """單一貨幣預測範例"""
    print("=== 單一貨幣預測範例 ===")
    
    # 創建預測器
    predictor = CurrencyPredictor(
        model_name="PatchTST",
        model_params={
            'seq_len': 168,
            'pred_len': 24,
            'patch_len': 16,
            'stride': 8,
            'n_estimators': 50,
            'max_depth': 8,
            'random_state': 42
        }
    )
    
    # 要預測的貨幣對
    symbol = "USDTWD=X"
    
    try:
        # 1. 收集資料
        print(f"1. 收集 {symbol} 資料...")
        data_result = predictor.collect_and_store_data([symbol], period="6mo")
        print(f"資料收集結果: {data_result}")
        
        # 2. 訓練模型
        print(f"2. 訓練 {symbol} 模型...")
        training_result = predictor.train_model(symbol, period="6mo")
        print(f"訓練完成: {training_result.get('training_completed', False)}")
        
        if training_result.get('test_metrics'):
            print(f"測試集 RMSE: {training_result['test_metrics'].get('rmse', 0):.6f}")
        
        # 3. 進行預測
        print(f"3. 預測 {symbol} 未來 7 天...")
        prediction_result = predictor.predict(symbol, horizon=7, return_uncertainty=True)
        
        if not prediction_result.get('error'):
            print(f"預測成功！")
            print(f"最後已知值: {prediction_result['last_known_value']:.4f}")
            print(f"預測值: {prediction_result['predictions']}")
            
            if 'uncertainty' in prediction_result:
                print(f"不確定性: {prediction_result['uncertainty']}")
        else:
            print(f"預測失敗: {prediction_result['error']}")
        
        # 4. 儲存模型
        model_path = f"models/{symbol.replace('=X', '')}_PatchTST.joblib"
        predictor.save_model(model_path)
        print(f"模型已儲存至: {model_path}")
        
    except Exception as e:
        print(f"範例執行失敗: {str(e)}")


def example_pipeline_prediction():
    """完整流程預測範例"""
    print("\n=== 完整流程預測範例 ===")
    
    # 配置
    config = {
        "model_name": "PatchTST",
        "model_params": {
            "seq_len": 168,
            "pred_len": 24,
            "patch_len": 16,
            "stride": 8,
            "n_estimators": 50,
            "max_depth": 8,
            "random_state": 42
        },
        "data_storage_path": "data",
        "log_level": "INFO",
        "data_collection": {
            "period": "6mo",
            "interval": "1d",
            "force_update": False
        },
        "model_training": {
            "period": "6mo",
            "target_column": "Close",
            "train_params": {}
        },
        "prediction": {
            "period": "6mo",
            "return_uncertainty": True
        }
    }
    
    # 要預測的貨幣對
    symbols = ["USDTWD=X", "EURUSD=X"]
    
    try:
        # 創建預測管道
        pipeline = PredictionPipeline(config, output_dir="results")
        
        # 執行完整流程
        print("執行完整預測流程...")
        results = pipeline.run_full_pipeline(
            symbols=symbols,
            prediction_horizon=7,
            save_results=True,
            force_retrain=False
        )
        
        # 輸出結果摘要
        print(f"\n流程執行完成，成功: {results.get('success', False)}")
        print(f"處理貨幣對: {len(symbols)} 個")
        
        # 顯示各階段狀態
        status = results.get('pipeline_status', {})
        print(f"資料收集: {'✅' if status.get('data_collection') else '❌'}")
        print(f"模型訓練: {'✅' if status.get('model_training') else '❌'}")
        print(f"預測執行: {'✅' if status.get('prediction') else '❌'}")
        print(f"結果儲存: {'✅' if status.get('results_saved') else '❌'}")
        
        # 顯示預測結果
        predictions = results.get('predictions', [])
        for prediction in predictions:
            symbol = prediction['symbol']
            if not prediction.get('error'):
                last_value = prediction.get('last_known_value', 0)
                first_pred = prediction.get('predictions', [0])[0] if prediction.get('predictions') else 0
                change = ((first_pred - last_value) / last_value * 100) if last_value != 0 else 0
                print(f"{symbol}: 預測變化 {change:+.2f}%")
            else:
                print(f"{symbol}: 預測失敗 - {prediction.get('error', '')}")
        
    except Exception as e:
        print(f"流程執行失敗: {str(e)}")


def example_batch_prediction():
    """批次預測範例（使用已訓練模型）"""
    print("\n=== 批次預測範例 ===")
    
    config = {
        "model_name": "PatchTST",
        "data_storage_path": "data"
    }
    
    symbols = ["USDTWD=X", "EURUSD=X"]
    model_paths = {
        "USDTWD=X": "models/USDTWD_PatchTST.joblib",
        "EURUSD=X": "models/EURUSD_PatchTST.joblib"
    }
    
    try:
        pipeline = PredictionPipeline(config)
        
        results = pipeline.run_batch_prediction(
            symbols=symbols,
            model_paths=model_paths,
            prediction_horizon=7
        )
        
        print("批次預測完成:")
        for result in results:
            symbol = result['symbol']
            if not result.get('error'):
                print(f"{symbol}: ✅ 預測成功")
                if result.get('predictions'):
                    print(f"  預測值: {result['predictions'][:3]}...")  # 顯示前3個
            else:
                print(f"{symbol}: ❌ {result.get('error', '')}")
        
    except Exception as e:
        print(f"批次預測失敗: {str(e)}")


def example_from_config_file():
    """從配置檔案執行範例"""
    print("\n=== 從配置檔案執行範例 ===")
    
    config_file = "config_example.json"
    
    try:
        # 從配置檔案創建流程
        pipeline = PredictionPipeline.from_config_file(config_file)
        
        # 載入配置中的設定
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        symbols = config.get('symbols', ['USDTWD=X'])
        horizon = config.get('prediction_horizon', 7)
        
        # 執行流程
        results = pipeline.run_full_pipeline(
            symbols=symbols,
            prediction_horizon=horizon
        )
        
        print(f"從配置檔案執行完成: {results.get('success', False)}")
        
    except Exception as e:
        print(f"配置檔案執行失敗: {str(e)}")


if __name__ == "__main__":
    """主程序"""
    print("貨幣預測系統使用範例")
    print("=" * 50)
    
    # 執行各種範例
    example_single_prediction()
    example_pipeline_prediction()
    
    # 以下範例需要先有已訓練的模型
    # example_batch_prediction()
    # example_from_config_file()
    
    print("\n所有範例執行完成！")
    print("\n提示:")
    print("1. 檢查 'results/' 目錄查看輸出結果")
    print("2. 檢查 'models/' 目錄查看已儲存的模型")
    print("3. 檢查 'data/' 目錄查看收集的資料")