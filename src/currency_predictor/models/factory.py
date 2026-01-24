"""
模型工廠

提供便利的模型創建介面
"""

import logging
from typing import Dict, Any, Optional, Union
from .base import ModelType, BaseModel
from .patchtst import PatchTST

logger = logging.getLogger(__name__)

# 嘗試導入 transformer 模型
try:
    from .patchtst_transformer import PatchTSTTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    PatchTSTTransformer = None
    HAS_TRANSFORMERS = False
    logger.warning("Transformers 庫不可用，無法使用 PatchTSTTransformer")


class ModelFactory:
    """模型工廠類別"""
    
    @staticmethod
    def get_available_models() -> Dict[str, Dict[str, Any]]:
        """
        取得可用的模型列表
        
        Returns:
            可用模型的字典
        """
        models = {
            'patchtst_sklearn': {
                'class': PatchTST,
                'type': ModelType.SKLEARN_BASED,
                'description': '基於 sklearn 的簡化版 PatchTST (快速但功能有限)',
                'available': True
            }
        }
        
        if HAS_TRANSFORMERS:
            models['patchtst_transformer'] = {
                'class': PatchTSTTransformer,
                'type': ModelType.TRANSFORMER_BASED,
                'description': '真正的 HuggingFace PatchTST Transformer (功能完整)',
                'available': True
            }
        else:
            models['patchtst_transformer'] = {
                'class': None,
                'type': ModelType.TRANSFORMER_BASED,
                'description': '真正的 HuggingFace PatchTST Transformer (需要安裝 transformers)',
                'available': False
            }
        
        return models
    
    @staticmethod
    def create_model(model_name: str, **kwargs) -> BaseModel:
        """
        創建模型實例
        
        Args:
            model_name: 模型名稱 ('patchtst_sklearn' 或 'patchtst_transformer')
            **kwargs: 模型配置參數
            
        Returns:
            模型實例
            
        Raises:
            ValueError: 如果模型名稱無效或模型不可用
        """
        available_models = ModelFactory.get_available_models()
        
        if model_name not in available_models:
            raise ValueError(f"未知的模型名稱: {model_name}. 可用模型: {list(available_models.keys())}")
        
        model_info = available_models[model_name]
        
        if not model_info['available']:
            raise ValueError(f"模型 {model_name} 不可用: {model_info['description']}")
        
        model_class = model_info['class']
        
        try:
            logger.info(f"創建模型: {model_name} ({model_info['description']})")
            return model_class(**kwargs)
        except Exception as e:
            logger.error(f"創建模型 {model_name} 失敗: {str(e)}")
            raise
    
    @staticmethod
    def get_recommended_model(prefer_accuracy: bool = True) -> str:
        """
        取得推薦的模型
        
        Args:
            prefer_accuracy: 是否偏好準確性 (True) 還是速度 (False)
            
        Returns:
            推薦的模型名稱
        """
        available_models = ModelFactory.get_available_models()
        
        if prefer_accuracy and available_models['patchtst_transformer']['available']:
            return 'patchtst_transformer'
        else:
            return 'patchtst_sklearn'
    
    @staticmethod
    def print_model_info():
        """列印所有可用模型的資訊"""
        models = ModelFactory.get_available_models()
        
        print("可用的模型:")
        print("=" * 80)
        
        for name, info in models.items():
            status = "✅ 可用" if info['available'] else "❌ 不可用"
            print(f"模型名稱: {name}")
            print(f"類型: {info['type'].value}")
            print(f"狀態: {status}")
            print(f"描述: {info['description']}")
            print("-" * 40)
        
        recommended = ModelFactory.get_recommended_model(prefer_accuracy=True)
        print(f"\n推薦模型 (準確性優先): {recommended}")
        
        recommended_speed = ModelFactory.get_recommended_model(prefer_accuracy=False)
        print(f"推薦模型 (速度優先): {recommended_speed}")


def create_patchtst_model(use_transformer: bool = None, **kwargs) -> BaseModel:
    """
    便利函數：創建 PatchTST 模型
    
    Args:
        use_transformer: 是否使用 transformer 版本。None 表示自動選擇
        **kwargs: 模型參數
        
    Returns:
        PatchTST 模型實例
    """
    if use_transformer is None:
        # 自動選擇最好的版本
        model_name = ModelFactory.get_recommended_model(prefer_accuracy=True)
    elif use_transformer:
        model_name = 'patchtst_transformer'
    else:
        model_name = 'patchtst_sklearn'
    
    return ModelFactory.create_model(model_name, **kwargs)


if __name__ == "__main__":
    # 測試模型工廠
    ModelFactory.print_model_info()