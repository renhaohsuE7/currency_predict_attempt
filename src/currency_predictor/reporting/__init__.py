"""
報告和結果展示模組

提供結果格式化、報告生成等功能
"""

from .formatter import ResultFormatter, StatusFormatter

try:
    from .pdf_generator import ReportGenerator
except ImportError:
    ReportGenerator = None  # type: ignore[assignment,misc]

__all__ = ['ResultFormatter', 'StatusFormatter', 'ReportGenerator']
