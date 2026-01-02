"""
Country Data Collector

Collects economic data for a specific country from Yahoo Finance.

Austrian Theory Relevance:
- Stock indices reflect economic health
- Bond yields reveal inflation expectations
- Currency strength shows CB policy effects
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import yfinance as yf

from .forex import ForexCollector


@dataclass
class CountryConditions:
    """Current economic conditions for a country"""

    country_code: str
    timestamp: datetime

    # Stock market
    stock_index_value: Decimal
    stock_index_change_pct: float
    stock_index_52w_high: Decimal
    stock_index_52w_low: Decimal

    # Currency
    currency_code: str
    fx_rate_vs_usd: Decimal
    fx_change_30d_pct: float

    # Interest rates (proxy)
    implied_rate: float

    # Volatility
    market_volatility: float

    # Sentiment
    is_bullish: bool
    fear_level: float  # 0-1


class CountryDataCollector:
    """
    Collects economic data for a specific country.

    Uses Yahoo Finance for:
    - Stock indices
    - Currency rates
    - Bond yields (when available)
    """

    # Stock index tickers for all 20 countries
    STOCK_INDEX_TICKERS = {
        "USA": "^GSPC",  # S&P 500
        "CAN": "^GSPTSE",  # S&P/TSX Composite
        "MEX": "^MXX",  # IPC Mexico
        "BRA": "^BVSP",  # Bovespa
        "ARG": "^MERV",  # Merval
        "GBR": "^FTSE",  # FTSE 100
        "DEU": "^GDAXI",  # DAX
        "FRA": "^FCHI",  # CAC 40
        "SWE": "^OMX",  # OMX Stockholm 30
        "NOR": "^OSEAX",  # Oslo All Share
        "CHE": "^SSMI",  # SMI
        "LIE": "^SSMI",  # Uses Swiss SMI
        "CHN": "000001.SS",  # Shanghai Composite
        "JPN": "^N225",  # Nikkei 225
        "IND": "^BSESN",  # BSE Sensex
        "IDN": "^JKSE",  # Jakarta Composite
        "ARE": "^DFMGI",  # Dubai Financial Market
        "SAU": "^TASI",  # Tadawul All Share
        "RUS": "IMOEX.ME",  # MOEX Russia
        "TUR": "XU100.IS",  # BIST 100
    }

    # Bond yield tickers (10-year when available)
    BOND_YIELD_TICKERS = {
        "USA": "^TNX",  # 10-Year Treasury
        "DEU": None,  # Use ECB rate proxy
        "GBR": None,  # Use BoE rate proxy
        "JPN": None,  # Use BoJ rate proxy
    }

    # Country currencies
    COUNTRY_CURRENCIES = {
        "USA": "USD",
        "CAN": "CAD",
        "MEX": "MXN",
        "BRA": "BRL",
        "ARG": "ARS",
        "GBR": "GBP",
        "DEU": "EUR",
        "FRA": "EUR",
        "SWE": "SEK",
        "NOR": "NOK",
        "CHE": "CHF",
        "LIE": "CHF",
        "CHN": "CNY",
        "JPN": "JPY",
        "IND": "INR",
        "IDN": "IDR",
        "ARE": "AED",
        "SAU": "SAR",
        "RUS": "RUB",
        "TUR": "TRY",
    }

    def __init__(self, country_code: str):
        """
        Initialize collector for a specific country.

        Args:
            country_code: ISO 3166-1 alpha-3 (e.g., 'USA', 'BRA')
        """
        self.country_code = country_code
        self.forex_collector = ForexCollector()
        self._cache: dict[str, Any] = {}
        self._cache_time: datetime | None = None
        self._cache_ttl = timedelta(minutes=15)

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self._cache_time:
            return False
        return datetime.now() - self._cache_time < self._cache_ttl

    @property
    def currency_code(self) -> str:
        """Get currency code for this country"""
        return self.COUNTRY_CURRENCIES.get(self.country_code, "USD")

    @property
    def stock_ticker(self) -> str | None:
        """Get stock index ticker for this country"""
        return self.STOCK_INDEX_TICKERS.get(self.country_code)

    def get_stock_index(self) -> dict[str, Any]:
        """
        Get stock index data for this country.

        Returns:
            Dict with value, change, high, low
        """
        ticker = self.stock_ticker
        if not ticker:
            return self._get_mock_stock_data()

        try:
            data = yf.Ticker(ticker)
            hist = data.history(period="1mo")

            if not hist.empty:
                current = Decimal(str(round(hist["Close"].iloc[-1], 2)))
                prev = Decimal(str(round(hist["Close"].iloc[0], 2)))
                change_pct = float((current - prev) / prev * 100) if prev > 0 else 0

                # Get 52-week data
                hist_1y = data.history(period="1y")
                high_52w = (
                    Decimal(str(round(hist_1y["High"].max(), 2)))
                    if not hist_1y.empty
                    else current
                )
                low_52w = (
                    Decimal(str(round(hist_1y["Low"].min(), 2)))
                    if not hist_1y.empty
                    else current
                )

                return {
                    "ticker": ticker,
                    "value": current,
                    "change_30d_pct": change_pct,
                    "high_52w": high_52w,
                    "low_52w": low_52w,
                    "volume": int(hist["Volume"].iloc[-1]) if "Volume" in hist else 0,
                    "volatility": float(
                        hist["Close"].std() / hist["Close"].mean() * 100
                    ),
                }

        except Exception as e:
            print(f"Error fetching {self.country_code} stock index: {e}")

        return self._get_mock_stock_data()

    def _get_mock_stock_data(self) -> dict[str, Any]:
        """Return mock stock data for testing"""
        mock_values = {
            "USA": 5000,
            "CAN": 22000,
            "MEX": 55000,
            "BRA": 130000,
            "ARG": 1500000,  # High due to peso devaluation
            "GBR": 7500,
            "DEU": 18000,
            "FRA": 7500,
            "SWE": 2500,
            "NOR": 1400,
            "CHE": 12000,
            "LIE": 12000,
            "CHN": 3200,
            "JPN": 38000,
            "IND": 75000,
            "IDN": 7000,
            "ARE": 4000,
            "SAU": 12000,
            "RUS": 3000,
            "TUR": 9000,
        }
        value = Decimal(str(mock_values.get(self.country_code, 10000)))
        return {
            "ticker": self.stock_ticker,
            "value": value,
            "change_30d_pct": 0.0,
            "high_52w": value * Decimal("1.15"),
            "low_52w": value * Decimal("0.85"),
            "volume": 0,
            "volatility": 15.0,
        }

    def get_currency_rate(self) -> dict[str, Any]:
        """
        Get currency exchange rate vs USD.

        Returns:
            Dict with rate, change, volatility
        """
        return self.forex_collector.get_rate_with_history(self.currency_code)

    def get_interest_rate_proxy(self) -> float:
        """
        Get implied interest rate for this country.

        Uses bond yields when available, otherwise estimates
        based on currency volatility and regime type.

        Austrian Note:
        - Natural rate should equal time preference
        - CB-manipulated rates cause distortions
        """
        # Try to get bond yield
        bond_ticker = self.BOND_YIELD_TICKERS.get(self.country_code)
        if bond_ticker:
            try:
                data = yf.Ticker(bond_ticker)
                info = data.fast_info
                yield_rate = info.get("lastPrice", 0)
                if yield_rate and yield_rate > 0:
                    return yield_rate / 100  # Convert to decimal

            except Exception:
                pass

        # Estimate based on country
        # Higher rates in EM, lower in developed markets
        estimated_rates = {
            "USA": 0.05,
            "CAN": 0.045,
            "MEX": 0.11,
            "BRA": 0.1075,
            "ARG": 0.80,  # Very high (inflation)
            "GBR": 0.05,
            "DEU": 0.035,
            "FRA": 0.035,
            "SWE": 0.04,
            "NOR": 0.045,
            "CHE": 0.015,  # Low (safe haven)
            "LIE": 0.015,
            "CHN": 0.035,
            "JPN": 0.005,  # Near zero (deflation)
            "IND": 0.065,
            "IDN": 0.06,
            "ARE": 0.05,
            "SAU": 0.05,
            "RUS": 0.16,  # High (sanctions, inflation)
            "TUR": 0.45,  # Very high (Erdogan policy)
        }
        return estimated_rates.get(self.country_code, 0.05)

    def get_market_sentiment(self) -> dict[str, Any]:
        """
        Get market sentiment indicators for this country.

        Returns:
            Dict with bullish/bearish, fear level, momentum
        """
        stock_data = self.get_stock_index()
        fx_data = self.get_currency_rate()

        # Calculate sentiment from price momentum
        stock_momentum = stock_data.get("change_30d_pct", 0)
        fx_momentum = fx_data.get("change_pct", 0)

        # Bullish if both stock and currency are up
        is_bullish = stock_momentum > 0 and fx_momentum >= 0

        # Fear level based on volatility and negative momentum
        volatility = stock_data.get("volatility", 15)
        fear_level = min(
            1.0, max(0.0, (volatility / 30) + (0.3 if stock_momentum < -5 else 0))
        )

        return {
            "is_bullish": is_bullish,
            "fear_level": fear_level,
            "stock_momentum": stock_momentum,
            "fx_momentum": fx_momentum,
            "volatility": volatility,
        }

    def get_country_conditions(self) -> CountryConditions:
        """
        Get comprehensive economic conditions for this country.

        Returns:
            CountryConditions dataclass with all data
        """
        stock = self.get_stock_index()
        fx = self.get_currency_rate()
        sentiment = self.get_market_sentiment()
        rate = self.get_interest_rate_proxy()

        return CountryConditions(
            country_code=self.country_code,
            timestamp=datetime.now(),
            stock_index_value=stock["value"],
            stock_index_change_pct=stock["change_30d_pct"],
            stock_index_52w_high=stock["high_52w"],
            stock_index_52w_low=stock["low_52w"],
            currency_code=self.currency_code,
            fx_rate_vs_usd=fx["rate"],
            fx_change_30d_pct=fx["change_pct"],
            implied_rate=rate,
            market_volatility=sentiment["volatility"],
            is_bullish=sentiment["is_bullish"],
            fear_level=sentiment["fear_level"],
        )

    def get_all_data(self) -> dict[str, Any]:
        """
        Get all available data for this country.

        Returns:
            Comprehensive dict with all economic data
        """
        if self._is_cache_valid() and self._cache:
            return self._cache.copy()

        conditions = self.get_country_conditions()
        data = {
            "country_code": self.country_code,
            "currency_code": self.currency_code,
            "timestamp": conditions.timestamp.isoformat(),
            "stock_market": {
                "index_value": str(conditions.stock_index_value),
                "change_30d_pct": conditions.stock_index_change_pct,
                "high_52w": str(conditions.stock_index_52w_high),
                "low_52w": str(conditions.stock_index_52w_low),
            },
            "currency": {
                "rate_vs_usd": str(conditions.fx_rate_vs_usd),
                "change_30d_pct": conditions.fx_change_30d_pct,
            },
            "interest_rate": conditions.implied_rate,
            "sentiment": {
                "is_bullish": conditions.is_bullish,
                "fear_level": conditions.fear_level,
                "volatility": conditions.market_volatility,
            },
        }

        self._cache = data.copy()
        self._cache_time = datetime.now()
        return data


class MultiCountryCollector:
    """
    Collects data for multiple countries simultaneously.

    Useful for multi-country simulations and comparisons.
    """

    ALL_COUNTRIES = [
        "USA",
        "CAN",
        "MEX",
        "BRA",
        "ARG",
        "GBR",
        "DEU",
        "FRA",
        "SWE",
        "NOR",
        "CHE",
        "LIE",
        "CHN",
        "JPN",
        "IND",
        "IDN",
        "ARE",
        "SAU",
        "RUS",
        "TUR",
    ]

    def __init__(self, countries: list[str] | None = None):
        """
        Initialize multi-country collector.

        Args:
            countries: List of country codes (default: all 20)
        """
        self.countries = countries or self.ALL_COUNTRIES
        self._collectors: dict[str, CountryDataCollector] = {}

        for country in self.countries:
            self._collectors[country] = CountryDataCollector(country)

    def get_all_conditions(self) -> dict[str, CountryConditions]:
        """
        Get conditions for all countries.

        Returns:
            Dict mapping country codes to CountryConditions
        """
        conditions = {}
        for country, collector in self._collectors.items():
            try:
                conditions[country] = collector.get_country_conditions()
            except Exception as e:
                print(f"Error getting conditions for {country}: {e}")
        return conditions

    def get_comparison_data(self) -> dict[str, dict[str, Any]]:
        """
        Get comparison data across all countries.

        Returns:
            Dict with rankings and comparisons
        """
        conditions = self.get_all_conditions()

        # Sort by various metrics
        by_stock_performance = sorted(
            conditions.items(), key=lambda x: x[1].stock_index_change_pct, reverse=True
        )
        by_currency_strength = sorted(
            conditions.items(), key=lambda x: x[1].fx_change_30d_pct, reverse=True
        )
        by_interest_rate = sorted(
            conditions.items(), key=lambda x: x[1].implied_rate, reverse=True
        )
        by_fear_level = sorted(
            conditions.items(), key=lambda x: x[1].fear_level, reverse=True
        )

        return {
            "stock_performance_ranking": [
                (c, f"{cond.stock_index_change_pct:+.1f}%")
                for c, cond in by_stock_performance
            ],
            "currency_strength_ranking": [
                (c, f"{cond.fx_change_30d_pct:+.1f}%")
                for c, cond in by_currency_strength
            ],
            "interest_rate_ranking": [
                (c, f"{cond.implied_rate:.1%}") for c, cond in by_interest_rate
            ],
            "fear_ranking": [
                (c, f"{cond.fear_level:.2f}") for c, cond in by_fear_level
            ],
            "bullish_countries": [
                c for c, cond in conditions.items() if cond.is_bullish
            ],
            "bearish_countries": [
                c for c, cond in conditions.items() if not cond.is_bullish
            ],
        }


# Simple interface
def get_country_data(country_code: str) -> dict[str, Any]:
    """Get all data for a country"""
    collector = CountryDataCollector(country_code)
    return collector.get_all_data()


def get_country_conditions(country_code: str) -> CountryConditions:
    """Get conditions for a country"""
    collector = CountryDataCollector(country_code)
    return collector.get_country_conditions()


def get_multi_country_comparison(
    countries: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Get comparison across multiple countries"""
    collector = MultiCountryCollector(countries)
    return collector.get_comparison_data()
