"""
PatchTST PyTorch Lightning 模組

實作 PyTorch Lightning 的訓練邏輯
"""

import logging
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ConstantLR, SequentialLR

from ..config import PatchTSTConfig, TrainingConfig
from .modules import PatchTSTModel

logger = logging.getLogger(__name__)

# 嘗試導入 PyTorch Lightning
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        LearningRateMonitor
    )
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    pl = None
    logger.warning("pytorch_lightning 未安裝，PatchTSTLightning 將無法使用")


if HAS_LIGHTNING:

    class PatchTSTLightning(pl.LightningModule):
        """
        PatchTST PyTorch Lightning 模組

        封裝 PatchTSTModel 並提供訓練、驗證、測試邏輯

        Args:
            config: PatchTST 配置
            num_features: 輸入特徵數量
            learning_rate: 學習率
            weight_decay: 權重衰減
            warmup_ratio: 預熱比例
            lr_scheduler: 學習率調度器類型
        """

        def __init__(
            self,
            config: PatchTSTConfig,
            num_features: int = 1,
            learning_rate: float = 1e-4,
            weight_decay: float = 0.01,
            warmup_ratio: float = 0.1,
            lr_scheduler: str = 'cosine',
            **kwargs
        ):
            super().__init__()

            # 保存超參數
            self.save_hyperparameters(ignore=['config'])

            self.config = config
            self.num_features = num_features
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.warmup_ratio = warmup_ratio
            self.lr_scheduler_type = lr_scheduler

            # 創建模型
            self.model = PatchTSTModel(config, num_features)

            # 損失函數
            self.loss_fn = nn.MSELoss()

            # 訓練歷史
            self.training_step_outputs: List[Dict] = []
            self.validation_step_outputs: List[Dict] = []

            logger.info(
                f"PatchTSTLightning 初始化完成: "
                f"parameters={self.model.get_num_parameters():,}"
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            前向傳播

            Args:
                x: (batch, context_length, num_features)

            Returns:
                predictions: (batch, prediction_length, num_features)
            """
            return self.model(x)

        def _align_output(
            self,
            y_hat: torch.Tensor,
            y: torch.Tensor
        ) -> torch.Tensor:
            """
            對齊模型輸出與目標的形狀

            當 y_hat 是 3D (batch, pred_len, n_features) 且
            y 是 2D (batch, pred_len) 時，對特徵維度取平均
            """
            if y_hat.dim() == 3 and y.dim() == 2:
                y_hat = y_hat.mean(dim=-1)  # (batch, pred_len)
            elif y_hat.dim() == 3 and y.dim() == 3:
                pass  # 形狀一致，不需調整
            return y_hat

        def training_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int
        ) -> torch.Tensor:
            """
            訓練步驟

            Args:
                batch: 包含 'past_values' 和 'future_values' 的字典
                batch_idx: 批次索引

            Returns:
                損失值
            """
            x = batch['past_values']
            y = batch['future_values']

            # 前向傳播
            y_hat = self._align_output(self(x), y)

            # 計算損失
            loss = self.loss_fn(y_hat, y)

            # 記錄
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

            self.training_step_outputs.append({'loss': loss.detach()})

            return loss

        def on_train_epoch_end(self):
            """訓練 epoch 結束時的處理"""
            avg_loss = torch.stack(
                [x['loss'] for x in self.training_step_outputs]
            ).mean()
            self.log('train_loss_epoch', avg_loss, prog_bar=True)
            self.training_step_outputs.clear()

        def _eval_step(
            self,
            batch: Dict[str, torch.Tensor],
            prefix: str = 'val'
        ) -> Dict[str, torch.Tensor]:
            """
            共用的評估步驟

            Args:
                batch: 包含 'past_values' 和 'future_values' 的字典
                prefix: 指標前綴 ('val' 或 'test')

            Returns:
                包含 loss, mae, rmse 的字典
            """
            x = batch['past_values']
            y = batch['future_values']

            # 前向傳播
            y_hat = self._align_output(self(x), y)

            # 計算損失
            loss = self.loss_fn(y_hat, y)

            # 計算額外指標
            mae = torch.mean(torch.abs(y_hat - y))
            rmse = torch.sqrt(torch.mean((y_hat - y) ** 2))

            # 記錄
            self.log(f'{prefix}_loss', loss, on_epoch=True, prog_bar=True)
            self.log(f'{prefix}_mae', mae, on_epoch=True)
            self.log(f'{prefix}_rmse', rmse, on_epoch=True)

            return {
                'loss': loss.detach(),
                'mae': mae.detach(),
                'rmse': rmse.detach()
            }

        def validation_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int
        ) -> torch.Tensor:
            """驗證步驟"""
            result = self._eval_step(batch, prefix='val')
            self.validation_step_outputs.append(result)
            return result['loss']

        def on_validation_epoch_end(self):
            """驗證 epoch 結束時的處理"""
            if self.validation_step_outputs:
                avg_loss = torch.stack(
                    [x['loss'] for x in self.validation_step_outputs]
                ).mean()
                avg_mae = torch.stack(
                    [x['mae'] for x in self.validation_step_outputs]
                ).mean()
                avg_rmse = torch.stack(
                    [x['rmse'] for x in self.validation_step_outputs]
                ).mean()

                self.log('val_loss_epoch', avg_loss)
                self.log('val_mae_epoch', avg_mae)
                self.log('val_rmse_epoch', avg_rmse)

            self.validation_step_outputs.clear()

        def test_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int
        ) -> torch.Tensor:
            """測試步驟"""
            result = self._eval_step(batch, prefix='test')
            return result['loss']

        def predict_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int
        ) -> torch.Tensor:
            """
            預測步驟

            Args:
                batch: 包含 'past_values' 的字典
                batch_idx: 批次索引

            Returns:
                預測值
            """
            x = batch['past_values']
            return self(x)

        def configure_optimizers(self):
            """
            配置優化器和學習率調度器
            """
            # 優化器
            optimizer = AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

            # 學習率調度器
            if self.trainer is not None:
                total_steps = self.trainer.estimated_stepping_batches
            else:
                total_steps = 1000  # 默認值

            warmup_steps = int(total_steps * self.warmup_ratio)

            if self.lr_scheduler_type == 'cosine':
                main_scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max(total_steps - warmup_steps, 1),
                    eta_min=self.learning_rate * 0.01
                )
            elif self.lr_scheduler_type == 'linear':
                main_scheduler = LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=0.01,
                    total_iters=max(total_steps - warmup_steps, 1)
                )
            else:
                main_scheduler = ConstantLR(optimizer, factor=1.0)

            # Compose warmup + main scheduler
            if warmup_steps > 0:
                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=warmup_steps
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[warmup_steps]
                )
            else:
                scheduler = main_scheduler

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }

        def get_model_info(self) -> Dict[str, Any]:
            """獲取模型資訊"""
            return {
                'model_name': 'PatchTST_Lightning',
                'num_parameters': self.model.get_num_parameters(),
                'num_trainable_parameters': self.model.get_num_trainable_parameters(),
                'config': self.config.to_dict(),
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
            }


    def get_lightning_callbacks(
        training_config: TrainingConfig,
        checkpoint_dir: str = './checkpoints'
    ) -> List:
        """
        獲取 Lightning 回調

        Args:
            training_config: 訓練配置
            checkpoint_dir: 檢查點目錄

        Returns:
            回調列表
        """
        callbacks = []

        # 早停
        if training_config.early_stopping_patience > 0:
            callbacks.append(
                EarlyStopping(
                    monitor='val_loss',
                    patience=training_config.early_stopping_patience,
                    min_delta=training_config.early_stopping_threshold,
                    mode='min',
                    verbose=True
                )
            )

        # 模型檢查點
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename='patchtst-{epoch:02d}-{val_loss:.4f}',
                monitor='val_loss',
                mode='min',
                save_top_k=3,
                save_last=True
            )
        )

        # 學習率監控
        callbacks.append(LearningRateMonitor(logging_interval='step'))

        return callbacks


else:
    # 如果沒有 Lightning，提供空的佔位符
    class PatchTSTLightning:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "pytorch_lightning 未安裝，請執行: pip install pytorch-lightning"
            )

    def get_lightning_callbacks(*args, **kwargs):
        raise ImportError(
            "pytorch_lightning 未安裝，請執行: pip install pytorch-lightning"
        )
