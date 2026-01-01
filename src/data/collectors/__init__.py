"""
Data Collectors Package

Collects real-time data from private market sources (no government data).
"""

from .bitcoin import BitcoinCollector, get_bitcoin_price, get_bitcoin_market_data
from .commodities import (
    CommoditiesCollector,
    get_commodity_prices,
    get_gold_price,
    get_commodity_inflation,
)
from .market_sentiment import (
    MarketSentimentCollector,
    get_market_fear_level,
    get_market_indices,
    get_crypto_fear_greed,
)
from .forex import (
    ForexCollector,
    get_forex_rate,
    get_country_forex_rate,
    get_all_forex_rates,
)
from .country_collector import (
    CountryDataCollector,
    CountryConditions,
    MultiCountryCollector,
    get_country_data,
    get_country_conditions,
    get_multi_country_comparison,
)

__all__ = [
    # Bitcoin
    "BitcoinCollector",
    "get_bitcoin_price",
    "get_bitcoin_market_data",
    # Commodities
    "CommoditiesCollector",
    "get_commodity_prices",
    "get_gold_price",
    "get_commodity_inflation",
    # Market Sentiment
    "MarketSentimentCollector",
    "get_market_fear_level",
    "get_market_indices",
    "get_crypto_fear_greed",
    # Forex
    "ForexCollector",
    "get_forex_rate",
    "get_country_forex_rate",
    "get_all_forex_rates",
    # Country
    "CountryDataCollector",
    "CountryConditions",
    "MultiCountryCollector",
    "get_country_data",
    "get_country_conditions",
    "get_multi_country_comparison",
]
