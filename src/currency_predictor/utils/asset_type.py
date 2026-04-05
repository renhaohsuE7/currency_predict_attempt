"""Asset type classification for Yahoo Finance symbols."""

from enum import StrEnum


class AssetType(StrEnum):
    """Supported financial asset types."""

    FOREX = "forex"
    STOCK = "stock"
    CRYPTO = "crypto"
    INDEX = "index"


def classify_symbol(symbol: str) -> AssetType:
    """Classify a Yahoo Finance symbol by its format.

    Rules:
    - Ends with ``=X`` → forex (e.g. ``USDTWD=X``, ``EURUSD=X``)
    - Contains ``-`` → crypto (e.g. ``BTC-USD``, ``ETH-USD``)
    - Starts with ``^`` → index (e.g. ``^GSPC``, ``^DJI``)
    - Otherwise → stock (e.g. ``AAPL``, ``TSLA``)
    """
    if symbol.endswith("=X"):
        return AssetType.FOREX
    if "-" in symbol:
        return AssetType.CRYPTO
    if symbol.startswith("^"):
        return AssetType.INDEX
    return AssetType.STOCK
