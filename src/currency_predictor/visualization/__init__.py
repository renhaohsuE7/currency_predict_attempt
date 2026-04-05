"""
視覺化模組

提供貨幣資料的視覺化功能
"""

from .visualizer import CurrencyVisualizer, setup_chinese_font
from .themes import ThemeManager

try:
    from .interactive import InteractiveVisualizer
except ImportError:
    InteractiveVisualizer = None  # type: ignore[assignment,misc]

__all__ = ['CurrencyVisualizer', 'setup_chinese_font', 'ThemeManager', 'InteractiveVisualizer']
