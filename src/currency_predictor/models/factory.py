"""
模型工廠

提供便利的模型創建介面
"""

import logging
from typing import Dict, Any, Optional, Union
from .base import ModelType, BaseModel

# 導入新的 patchtst 模組
from .patchtst import PatchTSTSklearn, PatchTSTConfig

logger = logging.getLogger(__name__)

# 嘗試導入 HuggingFace 版本
try:
    from .patchtst import PatchTSTHuggingFace
    HAS_TRANSFORMERS = True
except (ImportError, TypeError):
    PatchTSTHuggingFace = None  # type: ignore[assignment,misc]
    HAS_TRANSFORMERS = False
    logger.warning("Transformers 庫不可用，無法使用 PatchTSTHuggingFace")

# 嘗試導入 Lightning 版本
try:
    from .patchtst import PatchTSTLightningWrapper
    # 檢查是否實際可用 (可能導入成功但為 None)
    HAS_LIGHTNING = PatchTSTLightningWrapper is not None
    if not HAS_LIGHTNING:
        logger.info("PyTorch Lightning 未安裝，PatchTSTLightning 不可用")
except (ImportError, TypeError):
    PatchTSTLightningWrapper = None  # type: ignore[assignment,misc]
    HAS_LIGHTNING = False
    logger.info("PyTorch Lightning 未安裝，PatchTSTLightning 不可用")

# 向後兼容的別名
PatchTST = PatchTSTSklearn
PatchTSTTransformer = PatchTSTHuggingFace


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
            # sklearn 版本 (總是可用)
            'patchtst_sklearn': {
                'class': PatchTSTSklearn,
                'type': ModelType.SKLEARN_BASED,
                'description': '基於 sklearn 的簡化版 PatchTST (快速但功能有限)',
                'available': True,
                'implementation': 'sklearn'
            },
        }

        # HuggingFace 版本
        if HAS_TRANSFORMERS:
            models['patchtst_huggingface'] = {
                'class': PatchTSTHuggingFace,
                'type': ModelType.TRANSFORMER_BASED,
                'description': '基於 HuggingFace 的 PatchTST Transformer (功能完整)',
                'available': True,
                'implementation': 'huggingface'
            }
            # 向後兼容別名
            models['patchtst_transformer'] = {
                'class': PatchTSTHuggingFace,
                'type': ModelType.TRANSFORMER_BASED,
                'description': '基於 HuggingFace 的 PatchTST Transformer (別名)',
                'available': True,
                'implementation': 'huggingface',
                'alias_for': 'patchtst_huggingface'
            }
        else:
            models['patchtst_huggingface'] = {
                'class': None,
                'type': ModelType.TRANSFORMER_BASED,
                'description': 'HuggingFace PatchTST (需要安裝 transformers)',
                'available': False,
                'implementation': 'huggingface'
            }
            models['patchtst_transformer'] = {
                'class': None,
                'type': ModelType.TRANSFORMER_BASED,
                'description': 'HuggingFace PatchTST (需要安裝 transformers)',
                'available': False,
                'implementation': 'huggingface',
                'alias_for': 'patchtst_huggingface'
            }

        # Lightning 版本
        if HAS_LIGHTNING:
            models['patchtst_lightning'] = {
                'class': PatchTSTLightningWrapper,
                'type': ModelType.TRANSFORMER_BASED,
                'description': '基於 PyTorch Lightning 的 PatchTST (靈活，適合研究)',
                'available': True,
                'implementation': 'lightning'
            }
        else:
            models['patchtst_lightning'] = {
                'class': None,
                'type': ModelType.TRANSFORMER_BASED,
                'description': 'PyTorch Lightning PatchTST (需要安裝 pytorch-lightning)',
                'available': False,
                'implementation': 'lightning'
            }

        return models

    @staticmethod
    def create_model(model_name: str, **kwargs) -> BaseModel:
        """
        創建模型實例

        Args:
            model_name: 模型名稱
                - 'patchtst_sklearn': sklearn 版本
                - 'patchtst_huggingface': HuggingFace 版本
                - 'patchtst_transformer': HuggingFace 版本 (別名)
                - 'patchtst_lightning': PyTorch Lightning 版本
            **kwargs: 模型配置參數

        Returns:
            模型實例

        Raises:
            ValueError: 如果模型名稱無效或模型不可用
        """
        available_models = ModelFactory.get_available_models()

        if model_name not in available_models:
            raise ValueError(
                f"未知的模型名稱: {model_name}. "
                f"可用模型: {list(available_models.keys())}"
            )

        model_info = available_models[model_name]

        if not model_info['available']:
            raise ValueError(
                f"模型 {model_name} 不可用: {model_info['description']}"
            )

        model_class = model_info['class']

        try:
            logger.info(
                f"創建模型: {model_name} "
                f"(實作: {model_info['implementation']})"
            )
            model: BaseModel = model_class(**kwargs)
            return model
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

        if prefer_accuracy:
            # 優先順序: huggingface > lightning > sklearn
            if available_models['patchtst_huggingface']['available']:
                return 'patchtst_huggingface'
            elif available_models['patchtst_lightning']['available']:
                return 'patchtst_lightning'
            else:
                return 'patchtst_sklearn'
        else:
            # 速度優先
            return 'patchtst_sklearn'

    @staticmethod
    def print_model_info():
        """列印所有可用模型的資訊"""
        models = ModelFactory.get_available_models()

        print("可用的模型:")
        print("=" * 80)

        for name, info in models.items():
            if info.get('alias_for'):
                continue  # 跳過別名

            status = "[OK] 可用" if info['available'] else "[X] 不可用"
            print(f"模型名稱: {name}")
            print(f"類型: {info['type'].value}")
            print(f"實作: {info['implementation']}")
            print(f"狀態: {status}")
            print(f"描述: {info['description']}")
            print("-" * 40)

        recommended = ModelFactory.get_recommended_model(prefer_accuracy=True)
        print(f"\n推薦模型 (準確性優先): {recommended}")

        recommended_speed = ModelFactory.get_recommended_model(prefer_accuracy=False)
        print(f"推薦模型 (速度優先): {recommended_speed}")

    @staticmethod
    def get_model_by_implementation(implementation: str) -> Optional[str]:
        """
        根據實作類型取得模型名稱

        Args:
            implementation: 實作類型 ('sklearn', 'huggingface', 'lightning')

        Returns:
            模型名稱，如果不可用則返回 None
        """
        impl_map = {
            'sklearn': 'patchtst_sklearn',
            'huggingface': 'patchtst_huggingface',
            'transformer': 'patchtst_huggingface',  # 別名
            'lightning': 'patchtst_lightning',
        }

        model_name = impl_map.get(implementation.lower())
        if model_name is None:
            return None

        available_models = ModelFactory.get_available_models()
        if available_models.get(model_name, {}).get('available', False):
            return model_name

        return None


def create_patchtst_model(
    use_transformer: Optional[bool] = None,
    implementation: Optional[str] = None,
    **kwargs
) -> BaseModel:
    """
    便利函數：創建 PatchTST 模型

    Args:
        use_transformer: 是否使用 transformer 版本 (向後兼容)
            - None: 自動選擇最好的版本
            - True: 使用 HuggingFace 版本
            - False: 使用 sklearn 版本
        implementation: 指定實作類型
            - 'sklearn': sklearn 版本
            - 'huggingface': HuggingFace 版本
            - 'lightning': PyTorch Lightning 版本
        **kwargs: 模型參數

    Returns:
        PatchTST 模型實例
    """
    # 如果指定了 implementation，優先使用
    if implementation is not None:
        model_name = ModelFactory.get_model_by_implementation(implementation)
        if model_name is None:
            logger.warning(
                f"實作 '{implementation}' 不可用，回退到 sklearn 版本"
            )
            model_name = 'patchtst_sklearn'
    elif use_transformer is None:
        # 自動選擇最好的版本
        model_name = ModelFactory.get_recommended_model(prefer_accuracy=True)
    elif use_transformer:
        model_name = 'patchtst_huggingface'
    else:
        model_name = 'patchtst_sklearn'

    return ModelFactory.create_model(model_name, **kwargs)


if __name__ == "__main__":
    # 測試模型工廠
    ModelFactory.print_model_info()
