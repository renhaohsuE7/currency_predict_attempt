"""Tests for asset type classification utility."""

import pytest

from currency_predictor.utils.asset_type import AssetType, classify_symbol


class TestAssetType:
    """Test AssetType enum values."""

    def test_enum_values(self):
        assert AssetType.FOREX == "forex"
        assert AssetType.STOCK == "stock"
        assert AssetType.CRYPTO == "crypto"
        assert AssetType.INDEX == "index"


class TestClassifySymbol:
    """Test classify_symbol() for various Yahoo Finance symbol formats."""

    @pytest.mark.parametrize(
        "symbol",
        ["USDTWD=X", "EURUSD=X", "GBPUSD=X", "JPYUSD=X"],
    )
    def test_forex_symbols(self, symbol):
        assert classify_symbol(symbol) == AssetType.FOREX

    @pytest.mark.parametrize(
        "symbol",
        ["AAPL", "TSLA", "MSFT", "GOOGL", "TSM"],
    )
    def test_stock_symbols(self, symbol):
        assert classify_symbol(symbol) == AssetType.STOCK

    @pytest.mark.parametrize(
        "symbol",
        ["BTC-USD", "ETH-USD", "SOL-USD"],
    )
    def test_crypto_symbols(self, symbol):
        assert classify_symbol(symbol) == AssetType.CRYPTO

    @pytest.mark.parametrize(
        "symbol",
        ["^GSPC", "^DJI", "^IXIC", "^IRX"],
    )
    def test_index_symbols(self, symbol):
        assert classify_symbol(symbol) == AssetType.INDEX

    def test_returns_strenum(self):
        result = classify_symbol("AAPL")
        assert isinstance(result, str)
        assert result == "stock"
