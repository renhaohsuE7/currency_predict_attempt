"""
貨幣視覺化模組

提供貨幣資料的各種視覺化功能，包含中文字型支援
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties, fontManager
import seaborn as sns

logger = logging.getLogger(__name__)


def setup_chinese_font(font_path: Optional[str] = None) -> bool:
    """
    設定中文字型

    Args:
        font_path: 自定義字型路徑。如果為 None，則嘗試使用系統字型

    Returns:
        是否成功設定字型
    """
    try:
        if font_path and os.path.exists(font_path):
            # 使用自定義字型
            logger.info(f"Loading custom font: {font_path}")

            chinese_font = FontProperties(fname=font_path)
            font_name = chinese_font.get_name()

            # 註冊字型
            fontManager.addfont(font_path)

            # 設定字型
            mpl.rcParams['font.family'] = ['sans-serif']
            mpl.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Arial Unicode MS']
            plt.rcParams['font.family'] = font_name

            logger.info(f"Custom font set: {font_name}")
        else:
            # 使用系統字型
            logger.info("Using system fonts")

            # Windows
            if os.name == 'nt':
                mpl.rcParams['font.sans-serif'] = [
                    'Microsoft JhengHei',
                    'Microsoft YaHei',
                    'SimHei',
                    'DejaVu Sans'
                ]
            # Mac
            elif os.uname().sysname == 'Darwin':
                mpl.rcParams['font.sans-serif'] = [
                    'Arial Unicode MS',
                    'Heiti TC',
                    'DejaVu Sans'
                ]
            # Linux
            else:
                mpl.rcParams['font.sans-serif'] = [
                    'Noto Sans CJK TC',
                    'WenQuanYi Micro Hei',
                    'DejaVu Sans'
                ]

        # 解決負號顯示問題
        mpl.rcParams['axes.unicode_minus'] = False

        logger.info("Chinese font setup completed")
        return True

    except Exception as e:
        logger.warning(f"Failed to setup Chinese font: {e}")
        logger.warning("Falling back to default font")
        return False


class CurrencyVisualizer:
    """
    貨幣視覺化類別

    提供各種圖表繪製功能，用於分析和展示貨幣資料
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        output_dir: str = "results/figures",
        font_path: Optional[str] = None,
        style: str = "seaborn-v0_8-darkgrid",
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        初始化視覺化器

        Args:
            data_path: 資料檔案路徑或目錄
            output_dir: 圖表輸出目錄
            font_path: 自定義字型路徑
            style: matplotlib 樣式
            figsize: 預設圖表大小
        """
        self.data_path = Path(data_path) if data_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.figsize = figsize

        # 設定中文字型
        setup_chinese_font(font_path)

        # 設定樣式
        try:
            plt.style.use(style)
        except:
            logger.warning(f"Style '{style}' not available, using default")
            plt.style.use('default')

        # 設定顏色調色板
        sns.set_palette("husl")

        # 設定預設參數
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10

        logger.info(f"CurrencyVisualizer initialized. Output dir: {self.output_dir}")

    def load_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        data_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        載入貨幣資料

        Args:
            symbol: 貨幣符號（如 "USDTWD=X"）
            start_date: 開始日期（格式：YYYY-MM-DD）
            end_date: 結束日期（格式：YYYY-MM-DD）
            data_path: 資料路徑（覆蓋預設路徑）

        Returns:
            DataFrame 包含貨幣資料
        """
        path = Path(data_path) if data_path else self.data_path

        if path is None:
            raise ValueError("Data path not specified")

        # 如果 path 是目錄，嘗試找到對應的資料檔案
        if path.is_dir():
            # 嘗試找到符號對應的檔案
            possible_files = [
                path / f"{symbol}.csv",
                path / f"{symbol.replace('=X', '')}.csv",
                path / f"currency_data_{symbol}.csv",
            ]

            for file_path in possible_files:
                if file_path.exists():
                    path = file_path
                    break
            else:
                raise FileNotFoundError(f"Data file for {symbol} not found in {path}")

        # 讀取資料
        logger.info(f"Loading data from: {path}")
        df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')

        # 篩選日期範圍
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        logger.info(f"Loaded {len(df)} rows for {symbol}")
        return df

    def plot_price_history(
        self,
        df: pd.DataFrame,
        symbol: str,
        columns: List[str] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        繪製價格歷史圖

        Args:
            df: 資料 DataFrame
            symbol: 貨幣符號
            columns: 要繪製的欄位（預設：['Close']）
            title: 圖表標題
            save_path: 儲存路徑

        Returns:
            matplotlib Figure 物件
        """
        if columns is None:
            columns = ['Close']

        fig, ax = plt.subplots(figsize=self.figsize)

        for col in columns:
            if col in df.columns:
                ax.plot(df.index, df[col], label=col, linewidth=2)

        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('價格', fontsize=12)
        ax.set_title(title or f'{symbol} 價格歷史', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 旋轉 x 軸標籤
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_candlestick(
        self,
        df: pd.DataFrame,
        symbol: str,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        繪製 K 線圖

        Args:
            df: 資料 DataFrame（需要包含 Open, High, Low, Close）
            symbol: 貨幣符號
            title: 圖表標題
            save_path: 儲存路徑

        Returns:
            matplotlib Figure 物件
        """
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        fig, ax = plt.subplots(figsize=self.figsize)

        # 繪製 K 線
        for idx, row in df.iterrows():
            # 決定顏色（漲紅跌綠）
            color = 'red' if row['Close'] >= row['Open'] else 'green'

            # 繪製高低線
            ax.plot([idx, idx], [row['Low'], row['High']],
                   color=color, linewidth=1)

            # 繪製實體
            height = abs(row['Close'] - row['Open'])
            bottom = min(row['Close'], row['Open'])
            ax.bar(idx, height, bottom=bottom, color=color,
                  width=0.8, alpha=0.8)

        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('價格', fontsize=12)
        ax.set_title(title or f'{symbol} K 線圖', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_volume(
        self,
        df: pd.DataFrame,
        symbol: str,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        繪製成交量圖

        Args:
            df: 資料 DataFrame（需要包含 Volume）
            symbol: 貨幣符號
            title: 圖表標題
            save_path: 儲存路徑

        Returns:
            matplotlib Figure 物件
        """
        if 'Volume' not in df.columns:
            raise ValueError("DataFrame must contain 'Volume' column")

        fig, ax = plt.subplots(figsize=self.figsize)

        # 決定顏色（根據價格漲跌）
        colors = ['red' if close >= open_price else 'green'
                 for close, open_price in zip(df['Close'], df['Open'])]

        ax.bar(df.index, df['Volume'], color=colors, alpha=0.6)

        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('成交量', fontsize=12)
        ax.set_title(title or f'{symbol} 成交量', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_price_and_volume(
        self,
        df: pd.DataFrame,
        symbol: str,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        繪製價格和成交量組合圖

        Args:
            df: 資料 DataFrame
            symbol: 貨幣符號
            title: 圖表標題
            save_path: 儲存路徑

        Returns:
            matplotlib Figure 物件
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5),
                                       gridspec_kw={'height_ratios': [3, 1]})

        # 價格圖
        ax1.plot(df.index, df['Close'], label='收盤價', linewidth=2, color='blue')
        if 'Open' in df.columns:
            ax1.plot(df.index, df['Open'], label='開盤價', linewidth=1,
                    color='orange', alpha=0.7)

        ax1.set_ylabel('價格', fontsize=12)
        ax1.set_title(title or f'{symbol} 價格與成交量', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 成交量圖
        if 'Volume' in df.columns:
            colors = ['red' if close >= open_price else 'green'
                     for close, open_price in zip(df['Close'], df['Open'])]
            ax2.bar(df.index, df['Volume'], color=colors, alpha=0.6)
            ax2.set_ylabel('成交量', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='y')

        ax2.set_xlabel('日期', fontsize=12)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_returns(
        self,
        df: pd.DataFrame,
        symbol: str,
        period: str = 'daily',
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        繪製報酬率圖

        Args:
            df: 資料 DataFrame
            symbol: 貨幣符號
            period: 計算週期（'daily', 'weekly', 'monthly'）
            title: 圖表標題
            save_path: 儲存路徑

        Returns:
            matplotlib Figure 物件
        """
        # 計算報酬率
        returns = df['Close'].pct_change()

        if period == 'weekly':
            returns = df['Close'].resample('W').last().pct_change()
        elif period == 'monthly':
            returns = df['Close'].resample('M').last().pct_change()

        fig, ax = plt.subplots(figsize=self.figsize)

        # 繪製報酬率
        colors = ['red' if r > 0 else 'green' for r in returns]
        ax.bar(returns.index, returns * 100, color=colors, alpha=0.6)

        # 添加零線
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('報酬率 (%)', fontsize=12)
        ax.set_title(title or f'{symbol} {period.capitalize()} 報酬率',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_moving_averages(
        self,
        df: pd.DataFrame,
        symbol: str,
        windows: List[int] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        繪製移動平均線

        Args:
            df: 資料 DataFrame
            symbol: 貨幣符號
            windows: 移動平均窗口列表（預設：[5, 20, 60]）
            title: 圖表標題
            save_path: 儲存路徑

        Returns:
            matplotlib Figure 物件
        """
        if windows is None:
            windows = [5, 20, 60]

        fig, ax = plt.subplots(figsize=self.figsize)

        # 繪製收盤價
        ax.plot(df.index, df['Close'], label='收盤價', linewidth=2, alpha=0.7)

        # 繪製移動平均線
        colors = ['orange', 'green', 'red']
        for i, window in enumerate(windows):
            ma = df['Close'].rolling(window=window).mean()
            color = colors[i % len(colors)]
            ax.plot(df.index, ma, label=f'MA{window}',
                   linewidth=2, linestyle='--', color=color)

        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('價格', fontsize=12)
        ax.set_title(title or f'{symbol} 移動平均線', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_comparison(
        self,
        data_dict: Dict[str, pd.DataFrame],
        column: str = 'Close',
        normalize: bool = True,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        繪製多個貨幣對比圖

        Args:
            data_dict: 字典，key 為符號，value 為 DataFrame
            column: 要比較的欄位
            normalize: 是否標準化（以第一個值為基準）
            title: 圖表標題
            save_path: 儲存路徑

        Returns:
            matplotlib Figure 物件
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for symbol, df in data_dict.items():
            if column not in df.columns:
                logger.warning(f"Column '{column}' not found in data for {symbol}")
                continue

            data = df[column]

            if normalize:
                # 標準化為以第一個值為 100
                data = (data / data.iloc[0]) * 100

            ax.plot(data.index, data, label=symbol, linewidth=2)

        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('標準化價格 (基準=100)' if normalize else column, fontsize=12)
        ax.set_title(title or '貨幣對比', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if normalize:
            ax.axhline(y=100, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_prediction_results(
        self,
        actual: pd.Series,
        predicted: pd.Series,
        symbol: str,
        confidence_interval: Optional[Tuple[pd.Series, pd.Series]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        繪製預測結果對比圖

        Args:
            actual: 實際值
            predicted: 預測值
            symbol: 貨幣符號
            confidence_interval: 信心區間 (lower, upper)
            title: 圖表標題
            save_path: 儲存路徑

        Returns:
            matplotlib Figure 物件
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # 繪製實際值
        ax.plot(actual.index, actual.values, label='實際值',
               linewidth=2, color='blue', marker='o')

        # 繪製預測值
        ax.plot(predicted.index, predicted.values, label='預測值',
               linewidth=2, color='red', linestyle='--', marker='s')

        # 繪製信心區間
        if confidence_interval is not None:
            lower, upper = confidence_interval
            ax.fill_between(predicted.index, lower, upper,
                           alpha=0.2, color='red', label='95% 信心區間')

        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('價格', fontsize=12)
        ax.set_title(title or f'{symbol} 預測結果', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def _save_figure(self, fig: plt.Figure, save_path: str):
        """
        儲存圖表

        Args:
            fig: matplotlib Figure 物件
            save_path: 儲存路徑
        """
        # 如果只提供檔名，使用預設輸出目錄
        path = Path(save_path)
        if not path.is_absolute():
            path = self.output_dir / path

        # 確保目錄存在
        path.parent.mkdir(parents=True, exist_ok=True)

        # 儲存
        fig.savefig(path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to: {path}")

    def create_dashboard(
        self,
        df: pd.DataFrame,
        symbol: str,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        創建完整的分析儀表板

        Args:
            df: 資料 DataFrame
            symbol: 貨幣符號
            save_path: 儲存路徑

        Returns:
            matplotlib Figure 物件
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. 價格歷史
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df.index, df['Close'], linewidth=2, color='blue')
        ax1.set_title(f'{symbol} 價格歷史', fontsize=14, fontweight='bold')
        ax1.set_ylabel('價格')
        ax1.grid(True, alpha=0.3)

        # 2. 移動平均線
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(df.index, df['Close'], label='收盤價', linewidth=2, alpha=0.7)
        for window, color in zip([5, 20, 60], ['orange', 'green', 'red']):
            ma = df['Close'].rolling(window=window).mean()
            ax2.plot(df.index, ma, label=f'MA{window}', linewidth=2,
                    linestyle='--', color=color)
        ax2.set_title('移動平均線', fontsize=12, fontweight='bold')
        ax2.set_ylabel('價格')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. 成交量
        ax3 = fig.add_subplot(gs[1, 1])
        if 'Volume' in df.columns:
            colors = ['red' if close >= open_price else 'green'
                     for close, open_price in zip(df['Close'], df['Open'])]
            ax3.bar(df.index, df['Volume'], color=colors, alpha=0.6)
        ax3.set_title('成交量', fontsize=12, fontweight='bold')
        ax3.set_ylabel('成交量')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. 報酬率分布
        ax4 = fig.add_subplot(gs[2, 0])
        returns = df['Close'].pct_change().dropna() * 100
        ax4.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=1)
        ax4.set_title('報酬率分布', fontsize=12, fontweight='bold')
        ax4.set_xlabel('報酬率 (%)')
        ax4.set_ylabel('頻率')
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. 統計資訊
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')

        stats_text = f"""
統計資訊

資料期間: {df.index[0].date()} ~ {df.index[-1].date()}
資料筆數: {len(df)}

價格統計:
  最高: {df['Close'].max():.4f}
  最低: {df['Close'].min():.4f}
  平均: {df['Close'].mean():.4f}
  標準差: {df['Close'].std():.4f}

報酬率統計:
  平均: {returns.mean():.4f}%
  標準差: {returns.std():.4f}%
  最大: {returns.max():.4f}%
  最小: {returns.min():.4f}%
        """

        ax5.text(0.1, 0.5, stats_text, fontsize=10,
                verticalalignment='center', fontfamily='monospace')

        plt.suptitle(f'{symbol} 分析儀表板', fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            self._save_figure(fig, save_path)

        return fig
