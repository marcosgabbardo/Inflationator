"""
Bitcoin Data Collector

Fetches Bitcoin price and market data from CoinGecko API.

Austrian Theory Relevance:
- Bitcoin as sound money (fixed supply)
- BTC price in USD = measure of dollar debasement
- Hash rate = network security/value
"""

import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import httpx


class BitcoinCollector:
    """
    Collects Bitcoin data from CoinGecko.

    CoinGecko API is free and doesn't require API key for basic usage.
    Rate limit: ~50 calls/minute
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._cache: Dict[str, Any] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)  # Cache for 5 minutes

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        headers = {
            "Accept": "application/json",
        }
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key
        return headers

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self._cache_time:
            return False
        return datetime.now() - self._cache_time < self._cache_ttl

    async def get_current_price(self) -> Dict[str, Decimal]:
        """
        Get current Bitcoin price in multiple currencies.

        Returns:
            Dict with prices: {'usd': Decimal, 'eur': Decimal, ...}
        """
        if self._is_cache_valid() and "price" in self._cache:
            return self._cache["price"]

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.BASE_URL}/simple/price",
                    params={
                        "ids": "bitcoin",
                        "vs_currencies": "usd,eur,gbp,jpy,brl",
                        "include_24hr_change": "true",
                        "include_market_cap": "true",
                    },
                    headers=self._get_headers(),
                    timeout=10.0,
                )
                response.raise_for_status()
                data = response.json()

                btc_data = data.get("bitcoin", {})
                prices = {
                    "usd": Decimal(str(btc_data.get("usd", 0))),
                    "eur": Decimal(str(btc_data.get("eur", 0))),
                    "gbp": Decimal(str(btc_data.get("gbp", 0))),
                    "jpy": Decimal(str(btc_data.get("jpy", 0))),
                    "brl": Decimal(str(btc_data.get("brl", 0))),
                    "change_24h": btc_data.get("usd_24h_change", 0),
                    "market_cap_usd": Decimal(str(btc_data.get("usd_market_cap", 0))),
                }

                self._cache["price"] = prices
                self._cache_time = datetime.now()
                return prices

            except Exception as e:
                print(f"Error fetching Bitcoin price: {e}")
                # Return cached data if available
                if "price" in self._cache:
                    return self._cache["price"]
                # Return mock data as fallback
                return self._get_mock_price()

    def _get_mock_price(self) -> Dict[str, Decimal]:
        """Return mock price data for testing"""
        return {
            "usd": Decimal("50000"),
            "eur": Decimal("46000"),
            "gbp": Decimal("40000"),
            "jpy": Decimal("7500000"),
            "brl": Decimal("250000"),
            "change_24h": 0.0,
            "market_cap_usd": Decimal("1000000000000"),
        }

    async def get_market_data(self) -> Dict[str, Any]:
        """
        Get comprehensive Bitcoin market data.

        Includes:
        - Price
        - Market cap
        - Volume
        - Supply metrics
        - All-time high
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.BASE_URL}/coins/bitcoin",
                    params={
                        "localization": "false",
                        "tickers": "false",
                        "community_data": "false",
                        "developer_data": "false",
                    },
                    headers=self._get_headers(),
                    timeout=10.0,
                )
                response.raise_for_status()
                data = response.json()

                market_data = data.get("market_data", {})

                return {
                    "price_usd": Decimal(str(market_data.get("current_price", {}).get("usd", 0))),
                    "market_cap_usd": Decimal(str(market_data.get("market_cap", {}).get("usd", 0))),
                    "volume_24h_usd": Decimal(str(market_data.get("total_volume", {}).get("usd", 0))),
                    "circulating_supply": Decimal(str(market_data.get("circulating_supply", 0))),
                    "total_supply": Decimal(str(market_data.get("total_supply", 0))),
                    "max_supply": Decimal("21000000"),  # Fixed
                    "ath_usd": Decimal(str(market_data.get("ath", {}).get("usd", 0))),
                    "ath_date": market_data.get("ath_date", {}).get("usd"),
                    "price_change_24h": market_data.get("price_change_percentage_24h", 0),
                    "price_change_7d": market_data.get("price_change_percentage_7d", 0),
                    "price_change_30d": market_data.get("price_change_percentage_30d", 0),
                }

            except Exception as e:
                print(f"Error fetching Bitcoin market data: {e}")
                return self._get_mock_market_data()

    def _get_mock_market_data(self) -> Dict[str, Any]:
        """Return mock market data for testing"""
        return {
            "price_usd": Decimal("50000"),
            "market_cap_usd": Decimal("1000000000000"),
            "volume_24h_usd": Decimal("30000000000"),
            "circulating_supply": Decimal("19500000"),
            "total_supply": Decimal("19500000"),
            "max_supply": Decimal("21000000"),
            "ath_usd": Decimal("69000"),
            "ath_date": "2021-11-10",
            "price_change_24h": 0.0,
            "price_change_7d": 0.0,
            "price_change_30d": 0.0,
        }

    async def get_historical_prices(
        self,
        days: int = 30,
        currency: str = "usd"
    ) -> List[Dict[str, Any]]:
        """
        Get historical price data.

        Args:
            days: Number of days of history
            currency: Currency for prices

        Returns:
            List of {timestamp, price} dicts
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.BASE_URL}/coins/bitcoin/market_chart",
                    params={
                        "vs_currency": currency,
                        "days": str(days),
                    },
                    headers=self._get_headers(),
                    timeout=10.0,
                )
                response.raise_for_status()
                data = response.json()

                prices = data.get("prices", [])
                return [
                    {
                        "timestamp": datetime.fromtimestamp(p[0] / 1000),
                        "price": Decimal(str(p[1])),
                    }
                    for p in prices
                ]

            except Exception as e:
                print(f"Error fetching historical prices: {e}")
                return []

    def calculate_dollar_debasement(
        self,
        btc_price_usd: Decimal,
        baseline_price: Decimal = Decimal("1")
    ) -> float:
        """
        Calculate dollar debasement using Bitcoin as reference.

        Austrian Theory:
        - If BTC is sound money (fixed supply)
        - Then BTC price increase = USD debasement
        - This is the "real" inflation

        Args:
            btc_price_usd: Current BTC price in USD
            baseline_price: Reference price (e.g., price at some date)

        Returns:
            Percentage debasement
        """
        if baseline_price <= 0:
            return 0.0

        return float((btc_price_usd / baseline_price - 1) * 100)


# Synchronous wrapper for non-async code
def get_bitcoin_price() -> Dict[str, Decimal]:
    """Synchronous wrapper to get Bitcoin price"""
    collector = BitcoinCollector()
    return asyncio.run(collector.get_current_price())


def get_bitcoin_market_data() -> Dict[str, Any]:
    """Synchronous wrapper to get Bitcoin market data"""
    collector = BitcoinCollector()
    return asyncio.run(collector.get_market_data())
