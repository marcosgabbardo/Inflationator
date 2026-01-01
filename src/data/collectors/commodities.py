"""
Commodities Data Collector

Fetches commodity prices (Gold, Silver, Oil) from Yahoo Finance.

Austrian Theory Relevance:
- Gold/Silver as historical sound money
- Commodity prices reveal real inflation
- Oil as economic activity indicator
"""

from decimal import Decimal
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import yfinance as yf


class CommoditiesCollector:
    """
    Collects commodity data from Yahoo Finance.

    Tickers:
    - GC=F: Gold futures
    - SI=F: Silver futures
    - CL=F: Crude Oil futures
    - NG=F: Natural Gas futures
    - ZC=F: Corn futures
    - ZW=F: Wheat futures
    """

    TICKERS = {
        "gold": "GC=F",
        "silver": "SI=F",
        "oil": "CL=F",
        "natural_gas": "NG=F",
        "corn": "ZC=F",
        "wheat": "ZW=F",
    }

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=15)

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self._cache_time:
            return False
        return datetime.now() - self._cache_time < self._cache_ttl

    def get_current_prices(self) -> Dict[str, Decimal]:
        """
        Get current prices for all commodities.

        Returns:
            Dict with commodity names and prices in USD
        """
        if self._is_cache_valid() and self._cache:
            return self._cache.copy()

        prices = {}

        for name, ticker in self.TICKERS.items():
            try:
                data = yf.Ticker(ticker)
                info = data.fast_info

                # Get current price
                price = info.get("lastPrice", 0)
                if price and price > 0:
                    prices[name] = Decimal(str(round(price, 2)))
                else:
                    prices[name] = self._get_mock_price(name)

            except Exception as e:
                print(f"Error fetching {name}: {e}")
                prices[name] = self._get_mock_price(name)

        self._cache = prices.copy()
        self._cache_time = datetime.now()
        return prices

    def _get_mock_price(self, commodity: str) -> Decimal:
        """Return mock price for testing"""
        mock_prices = {
            "gold": Decimal("2000"),
            "silver": Decimal("25"),
            "oil": Decimal("80"),
            "natural_gas": Decimal("3"),
            "corn": Decimal("450"),
            "wheat": Decimal("600"),
        }
        return mock_prices.get(commodity, Decimal("100"))

    def get_gold_price(self) -> Dict[str, Any]:
        """
        Get detailed gold price data.

        Austrian Theory:
        - Gold is the historical standard of sound money
        - Gold price in USD = measure of USD debasement
        """
        try:
            gold = yf.Ticker(self.TICKERS["gold"])
            hist = gold.history(period="1mo")

            if not hist.empty:
                current_price = Decimal(str(round(hist["Close"].iloc[-1], 2)))
                prev_price = Decimal(str(round(hist["Close"].iloc[0], 2)))
                change_pct = float((current_price - prev_price) / prev_price * 100)

                return {
                    "price_usd": current_price,
                    "change_30d_pct": change_pct,
                    "high_30d": Decimal(str(round(hist["High"].max(), 2))),
                    "low_30d": Decimal(str(round(hist["Low"].min(), 2))),
                    "volume_avg": Decimal(str(round(hist["Volume"].mean(), 0))),
                }

        except Exception as e:
            print(f"Error fetching gold data: {e}")

        return {
            "price_usd": Decimal("2000"),
            "change_30d_pct": 0.0,
            "high_30d": Decimal("2050"),
            "low_30d": Decimal("1950"),
            "volume_avg": Decimal("100000"),
        }

    def get_silver_price(self) -> Dict[str, Any]:
        """Get detailed silver price data"""
        try:
            silver = yf.Ticker(self.TICKERS["silver"])
            hist = silver.history(period="1mo")

            if not hist.empty:
                current_price = Decimal(str(round(hist["Close"].iloc[-1], 2)))
                prev_price = Decimal(str(round(hist["Close"].iloc[0], 2)))
                change_pct = float((current_price - prev_price) / prev_price * 100)

                # Gold/Silver ratio (important metric)
                gold_price = self.get_gold_price()["price_usd"]
                gs_ratio = float(gold_price / current_price) if current_price > 0 else 80

                return {
                    "price_usd": current_price,
                    "change_30d_pct": change_pct,
                    "gold_silver_ratio": gs_ratio,
                }

        except Exception as e:
            print(f"Error fetching silver data: {e}")

        return {
            "price_usd": Decimal("25"),
            "change_30d_pct": 0.0,
            "gold_silver_ratio": 80.0,
        }

    def get_oil_price(self) -> Dict[str, Any]:
        """
        Get oil price data.

        Economic Indicator:
        - Oil is lifeblood of modern economy
        - Rising oil = economic activity OR supply constraint
        - Falling oil = recession signal OR supply glut
        """
        try:
            oil = yf.Ticker(self.TICKERS["oil"])
            hist = oil.history(period="1mo")

            if not hist.empty:
                current_price = Decimal(str(round(hist["Close"].iloc[-1], 2)))
                prev_price = Decimal(str(round(hist["Close"].iloc[0], 2)))
                change_pct = float((current_price - prev_price) / prev_price * 100)

                return {
                    "price_usd": current_price,
                    "change_30d_pct": change_pct,
                    "high_30d": Decimal(str(round(hist["High"].max(), 2))),
                    "low_30d": Decimal(str(round(hist["Low"].min(), 2))),
                }

        except Exception as e:
            print(f"Error fetching oil data: {e}")

        return {
            "price_usd": Decimal("80"),
            "change_30d_pct": 0.0,
            "high_30d": Decimal("85"),
            "low_30d": Decimal("75"),
        }

    def get_historical_prices(
        self,
        commodity: str,
        period: str = "1y"
    ) -> List[Dict[str, Any]]:
        """
        Get historical price data for a commodity.

        Args:
            commodity: One of 'gold', 'silver', 'oil', etc.
            period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'

        Returns:
            List of {date, open, high, low, close, volume}
        """
        ticker = self.TICKERS.get(commodity)
        if not ticker:
            return []

        try:
            data = yf.Ticker(ticker)
            hist = data.history(period=period)

            return [
                {
                    "date": index.to_pydatetime(),
                    "open": Decimal(str(round(row["Open"], 2))),
                    "high": Decimal(str(round(row["High"], 2))),
                    "low": Decimal(str(round(row["Low"], 2))),
                    "close": Decimal(str(round(row["Close"], 2))),
                    "volume": int(row["Volume"]),
                }
                for index, row in hist.iterrows()
            ]

        except Exception as e:
            print(f"Error fetching {commodity} history: {e}")
            return []

    def calculate_commodity_inflation(
        self,
        current_prices: Dict[str, Decimal],
        baseline_prices: Optional[Dict[str, Decimal]] = None
    ) -> Dict[str, float]:
        """
        Calculate inflation using commodity basket.

        Austrian Theory:
        - Commodities reveal real inflation
        - Government CPI understates true inflation
        - Use hard assets as inflation measure

        Args:
            current_prices: Current commodity prices
            baseline_prices: Reference prices (default: mock historical)

        Returns:
            Dict with inflation rates per commodity and weighted average
        """
        if baseline_prices is None:
            # Mock baseline (approximately 5 years ago)
            baseline_prices = {
                "gold": Decimal("1300"),
                "silver": Decimal("15"),
                "oil": Decimal("55"),
                "natural_gas": Decimal("2.5"),
                "corn": Decimal("380"),
                "wheat": Decimal("500"),
            }

        # Weights for basket
        weights = {
            "gold": 0.30,
            "silver": 0.10,
            "oil": 0.30,
            "natural_gas": 0.10,
            "corn": 0.10,
            "wheat": 0.10,
        }

        inflation_rates = {}
        weighted_sum = 0.0

        for commodity, current in current_prices.items():
            baseline = baseline_prices.get(commodity, current)
            if baseline > 0:
                rate = float((current - baseline) / baseline * 100)
                inflation_rates[commodity] = rate
                weighted_sum += rate * weights.get(commodity, 0)

        inflation_rates["weighted_average"] = weighted_sum

        # Annualized (assuming 5 year baseline)
        inflation_rates["annualized"] = weighted_sum / 5

        return inflation_rates

    def get_commodity_basket_index(self) -> Dict[str, Any]:
        """
        Create our own commodity price index.

        This is the "real" inflation indicator,
        not the government-manipulated CPI.
        """
        prices = self.get_current_prices()
        inflation = self.calculate_commodity_inflation(prices)

        return {
            "prices": {k: str(v) for k, v in prices.items()},
            "inflation_from_baseline": inflation,
            "timestamp": datetime.now().isoformat(),
            "note": "Commodity-based inflation - more accurate than CPI",
        }


# Simple interface
def get_commodity_prices() -> Dict[str, Decimal]:
    """Get current commodity prices"""
    collector = CommoditiesCollector()
    return collector.get_current_prices()


def get_gold_price() -> Decimal:
    """Get current gold price"""
    collector = CommoditiesCollector()
    return collector.get_gold_price()["price_usd"]


def get_commodity_inflation() -> float:
    """Get commodity-based inflation rate"""
    collector = CommoditiesCollector()
    prices = collector.get_current_prices()
    inflation = collector.calculate_commodity_inflation(prices)
    return inflation.get("annualized", 0.0)
