"""
Data Collectors Package

Collects real-time data from private market sources (no government data).
"""

from .bitcoin import BitcoinCollector, get_bitcoin_market_data, get_bitcoin_price
from .commodities import (
    CommoditiesCollector,
    get_commodity_inflation,
    get_commodity_prices,
    get_gold_price,
)
from .country_collector import (
    CountryConditions,
    CountryDataCollector,
    MultiCountryCollector,
    get_country_conditions,
    get_country_data,
    get_multi_country_comparison,
)
from .forex import (
    ForexCollector,
    get_all_forex_rates,
    get_country_forex_rate,
    get_forex_rate,
)
from .market_sentiment import (
    MarketSentimentCollector,
    get_crypto_fear_greed,
    get_market_fear_level,
    get_market_indices,
)

__all__ = [
    # Bitcoin
    "BitcoinCollector",
    # Commodities
    "CommoditiesCollector",
    "CountryConditions",
    # Country
    "CountryDataCollector",
    # Forex
    "ForexCollector",
    # Market Sentiment
    "MarketSentimentCollector",
    "MultiCountryCollector",
    "get_all_forex_rates",
    "get_bitcoin_market_data",
    "get_bitcoin_price",
    "get_commodity_inflation",
    "get_commodity_prices",
    "get_country_conditions",
    "get_country_data",
    "get_country_forex_rate",
    "get_crypto_fear_greed",
    "get_forex_rate",
    "get_gold_price",
    "get_market_fear_level",
    "get_market_indices",
    "get_multi_country_comparison",
]
