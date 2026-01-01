"""
Forex (Foreign Exchange) Data Collector

Fetches currency exchange rates from Yahoo Finance.

Austrian Theory Relevance:
- Exchange rates reflect relative monetary policies
- Currency devaluation = central bank damage
- Strong currencies = less intervention
"""

from decimal import Decimal
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import yfinance as yf


class ForexCollector:
    """
    Collects forex data from Yahoo Finance.

    All rates are quoted against USD (base currency).
    Ticker format: XXXUSD=X (e.g., EURUSD=X, BRLUSD=X)
    """

    # Currency tickers for all 20 countries (vs USD)
    CURRENCY_TICKERS = {
        # Americas
        "USD": None,           # Base currency
        "CAD": "CADUSD=X",     # Canadian Dollar
        "MXN": "MXNUSD=X",     # Mexican Peso
        "BRL": "BRLUSD=X",     # Brazilian Real
        "ARS": "ARSUSD=X",     # Argentine Peso

        # Europe
        "GBP": "GBPUSD=X",     # British Pound
        "EUR": "EURUSD=X",     # Euro (DEU, FRA)
        "SEK": "SEKUSD=X",     # Swedish Krona
        "NOK": "NOKUSD=X",     # Norwegian Krone
        "CHF": "CHFUSD=X",     # Swiss Franc (CHE, LIE)

        # Asia
        "CNY": "CNYUSD=X",     # Chinese Yuan
        "JPY": "JPYUSD=X",     # Japanese Yen
        "INR": "INRUSD=X",     # Indian Rupee
        "IDR": "IDRUSD=X",     # Indonesian Rupiah

        # Middle East
        "AED": "AEDUSD=X",     # UAE Dirham
        "SAR": "SARUSD=X",     # Saudi Riyal

        # Eurasia
        "RUB": "RUBUSD=X",     # Russian Ruble
        "TRY": "TRYUSD=X",     # Turkish Lira
    }

    # Country to currency mapping
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

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=15)

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self._cache_time:
            return False
        return datetime.now() - self._cache_time < self._cache_ttl

    def get_rate(self, currency: str) -> Decimal:
        """
        Get current exchange rate for a currency vs USD.

        Args:
            currency: ISO 4217 currency code (e.g., 'BRL', 'EUR')

        Returns:
            Exchange rate (1 unit of currency in USD)
        """
        if currency == "USD":
            return Decimal("1.0")

        ticker = self.CURRENCY_TICKERS.get(currency)
        if not ticker:
            return self._get_mock_rate(currency)

        try:
            data = yf.Ticker(ticker)
            info = data.fast_info
            rate = info.get("lastPrice", 0)

            if rate and rate > 0:
                return Decimal(str(round(rate, 6)))

        except Exception as e:
            print(f"Error fetching {currency} rate: {e}")

        return self._get_mock_rate(currency)

    def get_country_rate(self, country_code: str) -> Decimal:
        """
        Get exchange rate for a country's currency vs USD.

        Args:
            country_code: ISO 3166-1 alpha-3 (e.g., 'BRA', 'JPN')

        Returns:
            Exchange rate
        """
        currency = self.COUNTRY_CURRENCIES.get(country_code, "USD")
        return self.get_rate(currency)

    def get_all_rates(self) -> Dict[str, Decimal]:
        """
        Get current exchange rates for all currencies.

        Returns:
            Dict with currency codes and rates vs USD
        """
        if self._is_cache_valid() and self._cache:
            return self._cache.copy()

        rates = {"USD": Decimal("1.0")}

        for currency, ticker in self.CURRENCY_TICKERS.items():
            if ticker:
                rates[currency] = self.get_rate(currency)

        self._cache = rates.copy()
        self._cache_time = datetime.now()
        return rates

    def _get_mock_rate(self, currency: str) -> Decimal:
        """Return mock rate for testing"""
        mock_rates = {
            "CAD": Decimal("0.74"),
            "MXN": Decimal("0.058"),
            "BRL": Decimal("0.20"),
            "ARS": Decimal("0.0011"),   # Very weak (inflation)
            "GBP": Decimal("1.27"),
            "EUR": Decimal("1.08"),
            "SEK": Decimal("0.095"),
            "NOK": Decimal("0.092"),
            "CHF": Decimal("1.12"),
            "CNY": Decimal("0.14"),
            "JPY": Decimal("0.0067"),
            "INR": Decimal("0.012"),
            "IDR": Decimal("0.000063"),
            "AED": Decimal("0.27"),
            "SAR": Decimal("0.27"),
            "RUB": Decimal("0.011"),    # Weakened by sanctions
            "TRY": Decimal("0.031"),    # Very weak (Erdogan policy)
        }
        return mock_rates.get(currency, Decimal("1.0"))

    def get_rate_with_history(
        self,
        currency: str,
        period: str = "1mo"
    ) -> Dict[str, Any]:
        """
        Get exchange rate with historical data.

        Args:
            currency: Currency code
            period: '1d', '5d', '1mo', '3mo', '6mo', '1y'

        Returns:
            Dict with current rate, change, high, low
        """
        if currency == "USD":
            return {
                "currency": "USD",
                "rate": Decimal("1.0"),
                "change_pct": 0.0,
                "high": Decimal("1.0"),
                "low": Decimal("1.0"),
            }

        ticker = self.CURRENCY_TICKERS.get(currency)
        if not ticker:
            rate = self._get_mock_rate(currency)
            return {
                "currency": currency,
                "rate": rate,
                "change_pct": 0.0,
                "high": rate * Decimal("1.05"),
                "low": rate * Decimal("0.95"),
            }

        try:
            data = yf.Ticker(ticker)
            hist = data.history(period=period)

            if not hist.empty:
                current = Decimal(str(round(hist["Close"].iloc[-1], 6)))
                prev = Decimal(str(round(hist["Close"].iloc[0], 6)))
                change_pct = float((current - prev) / prev * 100) if prev > 0 else 0

                return {
                    "currency": currency,
                    "rate": current,
                    "change_pct": change_pct,
                    "high": Decimal(str(round(hist["High"].max(), 6))),
                    "low": Decimal(str(round(hist["Low"].min(), 6))),
                    "volatility": float(hist["Close"].std()),
                }

        except Exception as e:
            print(f"Error fetching {currency} history: {e}")

        rate = self._get_mock_rate(currency)
        return {
            "currency": currency,
            "rate": rate,
            "change_pct": 0.0,
            "high": rate,
            "low": rate,
        }

    def calculate_currency_strength(self) -> Dict[str, float]:
        """
        Calculate relative currency strength.

        Austrian Interpretation:
        - Strong currency = less central bank intervention
        - Weak currency = more money printing
        - Switzerland (CHF) should be strongest
        - Argentina (ARS), Turkey (TRY) should be weakest
        """
        rates = self.get_all_rates()

        # Get 30-day changes for each currency
        strength = {}
        for currency in rates.keys():
            if currency == "USD":
                strength[currency] = 0.0  # Baseline
                continue

            data = self.get_rate_with_history(currency, "1mo")
            # Positive change = currency strengthened vs USD
            strength[currency] = data.get("change_pct", 0.0)

        return strength

    def get_major_pairs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get major currency pairs data.

        Major pairs for simulation:
        - EUR/USD (most traded)
        - USD/JPY (carry trade)
        - GBP/USD
        - USD/CHF (safe haven)
        - USD/CNY (trade war indicator)
        - USD/BRL (emerging markets)
        """
        major_pairs = {
            "EUR/USD": self.get_rate_with_history("EUR"),
            "GBP/USD": self.get_rate_with_history("GBP"),
            "USD/JPY": self._invert_rate(self.get_rate_with_history("JPY")),
            "USD/CHF": self._invert_rate(self.get_rate_with_history("CHF")),
            "USD/CNY": self._invert_rate(self.get_rate_with_history("CNY")),
            "USD/BRL": self._invert_rate(self.get_rate_with_history("BRL")),
        }
        return major_pairs

    def _invert_rate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Invert rate data (for USD/XXX quotes)"""
        if data["rate"] > 0:
            data["rate"] = Decimal("1") / data["rate"]
            data["change_pct"] = -data["change_pct"]
            old_high = data.get("high", data["rate"])
            old_low = data.get("low", data["rate"])
            data["high"] = Decimal("1") / old_low if old_low > 0 else data["rate"]
            data["low"] = Decimal("1") / old_high if old_high > 0 else data["rate"]
        return data

    def calculate_dollar_debasement(
        self,
        years: int = 10
    ) -> Dict[str, float]:
        """
        Calculate USD debasement against various currencies.

        Austrian Theory:
        - USD loses value due to Fed money printing
        - Strong currencies (CHF) gain vs USD
        - Gold is ultimate measure of debasement
        """
        # Historical rates (approximate, N years ago)
        historical_rates = {
            "CHF": Decimal("0.95"),   # Was ~0.95, now ~1.12
            "EUR": Decimal("1.30"),   # Was ~1.30, now ~1.08
            "GBP": Decimal("1.55"),   # Was ~1.55, now ~1.27
            "JPY": Decimal("0.0095"), # Was ~0.0095, now ~0.0067
            "CNY": Decimal("0.16"),   # Was ~0.16, now ~0.14
            "BRL": Decimal("0.40"),   # Was ~0.40, now ~0.20
            "ARS": Decimal("0.25"),   # Was ~0.25, now ~0.001
            "TRY": Decimal("0.50"),   # Was ~0.50, now ~0.03
        }

        current_rates = self.get_all_rates()
        debasement = {}

        for currency, hist_rate in historical_rates.items():
            curr_rate = current_rates.get(currency, hist_rate)
            if hist_rate > 0:
                change = float((curr_rate - hist_rate) / hist_rate * 100)
                debasement[currency] = change

        return debasement


# Simple interface
def get_forex_rate(currency: str) -> Decimal:
    """Get current exchange rate vs USD"""
    collector = ForexCollector()
    return collector.get_rate(currency)


def get_country_forex_rate(country_code: str) -> Decimal:
    """Get exchange rate for a country's currency"""
    collector = ForexCollector()
    return collector.get_country_rate(country_code)


def get_all_forex_rates() -> Dict[str, Decimal]:
    """Get all exchange rates"""
    collector = ForexCollector()
    return collector.get_all_rates()
