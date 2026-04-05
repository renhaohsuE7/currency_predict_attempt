"""
金融資料收集器

提供從 Yahoo Finance 收集貨幣匯率及股票資料的功能
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class YahooFinanceCollector:
    """Yahoo Finance 金融資料收集器（支援貨幣對、股票、加密貨幣等）"""

    def __init__(self):
        """初始化收集器"""
        self.default_symbols = [
            'EURUSD=X', 'GBPUSD=X', 'USDTWD=X', 'TWD=X',
            'JPYUSD=X', 'AUDUSD=X', 'CADUSD=X'
        ]
        # 向後相容
        self.supported_pairs = self.default_symbols
        logger.info("Yahoo Finance 收集器已初始化")

    @staticmethod
    def is_currency_pair(symbol: str) -> bool:
        """判斷符號是否為貨幣對（結尾為 =X）"""
        return symbol.endswith('=X')
    
    def get_currency_data(
        self, 
        symbol: str, 
        period: str = "1mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        取得指定貨幣對的匯率資料
        
        Args:
            symbol: 金融符號 (例: 'USDTWD=X', 'AAPL', 'BTC-USD')
            period: 時間範圍 ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: 資料間隔 ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            包含 OHLCV 資料的 DataFrame，如果失敗則返回 None
        """
        try:
            logger.info(f"正在取得 {symbol} 的資料，期間: {period}")
            
            # 建立 yfinance Ticker 物件
            ticker = yf.Ticker(symbol)
            
            # 取得歷史資料
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"未找到 {symbol} 的資料")
                return None
            
            # 清理資料
            data = self._clean_data(data)
            
            logger.info(f"成功取得 {symbol} 的 {len(data)} 筆資料")
            return data
            
        except Exception as e:
            logger.error(f"取得 {symbol} 資料時發生錯誤: {str(e)}")
            return None
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清理原始資料
        
        Args:
            data: 原始 OHLCV 資料
            
        Returns:
            清理後的資料
        """
        # 移除空值
        data = data.dropna()
        
        # 確保索引是日期時間格式
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # 按時間排序
        data = data.sort_index()
        
        # 移除重複資料
        data = data[~data.index.duplicated(keep='first')]
        
        return data
    
    def check_symbol_availability(self, symbol: str) -> bool:
        """
        檢查貨幣對符號是否可用
        
        Args:
            symbol: 貨幣對符號
            
        Returns:
            True 如果可用，False 如果不可用
        """
        try:
            ticker = yf.Ticker(symbol)
            # 嘗試取得 1 天的資料來測試
            test_data = ticker.history(period="1d")
            return not test_data.empty
        except Exception:
            return False
    
    def get_available_pairs(self) -> List[str]:
        """
        取得可用的貨幣對清單
        
        Returns:
            可用的貨幣對符號清單
        """
        available_pairs = []
        
        for pair in self.supported_pairs:
            logger.info(f"檢查 {pair} 的可用性...")
            if self.check_symbol_availability(pair):
                available_pairs.append(pair)
                logger.info(f"✓ {pair} 可用")
            else:
                logger.warning(f"✗ {pair} 不可用")
        
        return available_pairs
    
    def get_spot_check_data(
        self,
        symbol: str,
        target_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """下載指定日期附近的小範圍資料，用於抽樣驗證快取資料。

        Args:
            symbol: 金融符號
            target_date: 要驗證的目標日期

        Returns:
            包含目標日期附近資料的 DataFrame，失敗回傳 None
        """
        try:
            start = (target_date - timedelta(days=3)).strftime("%Y-%m-%d")
            end = (target_date + timedelta(days=3)).strftime("%Y-%m-%d")
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start, end=end, interval="1d")
            if data.empty:
                return None
            data = self._clean_data(data)
            return data
        except Exception as e:
            logger.debug(f"Spot check failed for {symbol} @ {target_date}: {e}")
            return None

    def get_currency_info(self, symbol: str) -> Optional[Dict]:
        """
        取得貨幣對的基本資訊
        
        Args:
            symbol: 貨幣對符號
            
        Returns:
            包含貨幣對資訊的字典，如果失敗則返回 None
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', ''),
                'currency': info.get('currency', ''),
                'exchange': info.get('exchange', ''),
                'timezone': info.get('exchangeTimezoneName', '')
            }
        except Exception as e:
            logger.error(f"取得 {symbol} 資訊時發生錯誤: {str(e)}")
            return None