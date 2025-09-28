"""
資料儲存模組

提供資料儲存和載入功能
"""

import pandas as pd
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataStorage:
    """資料儲存管理器"""
    
    def __init__(self, base_dir: str = "data"):
        """
        初始化儲存管理器
        
        Args:
            base_dir: 資料儲存的基礎目錄
        """
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        
        # 建立必要的目錄
        self._ensure_directories()
        
        logger.info(f"資料儲存管理器已初始化，基礎目錄: {self.base_dir}")
    
    def _ensure_directories(self):
        """確保必要的目錄存在"""
        directories = [self.base_dir, self.raw_dir, self.processed_dir]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"確保目錄存在: {directory}")
    
    def save_raw_data(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        period: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        儲存原始資料
        
        Args:
            data: 要儲存的資料
            symbol: 貨幣對符號
            period: 資料期間
            metadata: 額外的元資料
            
        Returns:
            True 如果儲存成功，False 如果失敗
        """
        try:
            # 建立檔案名稱
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{period}_{timestamp}.csv"
            filepath = self.raw_dir / filename
            
            # 處理時區問題後儲存 CSV 檔案
            data_to_save = data.copy()
            if hasattr(data_to_save.index, 'tz') and data_to_save.index.tz is not None:
                data_to_save.index = data_to_save.index.tz_localize(None)
            
            data_to_save.to_csv(filepath, index=True)
            
            # 儲存元資料
            if metadata:
                metadata_file = filepath.with_suffix('.json')
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"原始資料已儲存至: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"儲存原始資料時發生錯誤: {str(e)}")
            return False
    
    def load_raw_data(self, symbol: str, period: str = None) -> Optional[pd.DataFrame]:
        """
        載入原始資料
        
        Args:
            symbol: 貨幣對符號
            period: 資料期間 (如果未指定，載入最新的)
            
        Returns:
            載入的資料 DataFrame，如果失敗則返回 None
        """
        try:
            # 尋找符合條件的檔案
            pattern = f"{symbol}_*.csv" if period is None else f"{symbol}_{period}_*.csv"
            files = list(self.raw_dir.glob(pattern))
            
            if not files:
                logger.warning(f"未找到符合 {pattern} 的檔案")
                return None
            
            # 選擇最新的檔案
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            
            # 載入資料
            data = pd.read_csv(latest_file, index_col=0, parse_dates=True)
            
            # 確保索引名稱一致
            if data.index.name is None:
                data.index.name = 'Date'
            
            # 處理時區問題
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            logger.info(f"已載入資料: {latest_file}")
            return data
            
        except Exception as e:
            logger.error(f"載入原始資料時發生錯誤: {str(e)}")
            return None
    
    def list_available_data(self) -> Dict[str, list]:
        """
        列出可用的資料檔案
        
        Returns:
            包含可用資料檔案資訊的字典
        """
        try:
            raw_files = list(self.raw_dir.glob("*.csv"))
            processed_files = list(self.processed_dir.glob("*.csv"))
            
            result = {
                'raw_files': [f.name for f in raw_files],
                'processed_files': [f.name for f in processed_files]
            }
            
            logger.info(f"找到 {len(raw_files)} 個原始資料檔案，{len(processed_files)} 個處理後資料檔案")
            return result
            
        except Exception as e:
            logger.error(f"列出可用資料時發生錯誤: {str(e)}")
            return {'raw_files': [], 'processed_files': []}
    
    def get_data_info(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        取得資料檔案的資訊
        
        Args:
            filepath: 資料檔案路徑
            
        Returns:
            包含資料檔案資訊的字典，如果失敗則返回 None
        """
        try:
            full_path = Path(filepath) if os.path.isabs(filepath) else self.base_dir / filepath
            
            if not full_path.exists():
                logger.warning(f"檔案不存在: {full_path}")
                return None
            
            # 讀取基本檔案資訊
            stat = full_path.stat()
            
            # 嘗試讀取資料來取得更多資訊
            data = pd.read_csv(full_path, index_col=0, parse_dates=True)
            
            info = {
                'filepath': str(full_path),
                'filename': full_path.name,
                'size_bytes': stat.st_size,
                'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'rows': len(data),
                'columns': len(data.columns),
                'column_names': list(data.columns),
                'date_range': {
                    'start': data.index.min().isoformat() if not data.empty else None,
                    'end': data.index.max().isoformat() if not data.empty else None
                }
            }
            
            return info
            
        except Exception as e:
            logger.error(f"取得檔案資訊時發生錯誤: {str(e)}")
            return None