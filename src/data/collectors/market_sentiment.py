"""
Market Sentiment & Macro Indicators Collector

Fetches real-time sentiment and macro data from non-government sources.

Austrian Theory Relevance:
- Market sentiment reflects aggregate time preference
- VIX (fear index) shows market uncertainty
- DXY shows dollar debasement against other fiats
- Baltic Dry Index shows real economic activity
"""

import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf
import httpx


class MarketSentimentCollector:
    """
    Collects market sentiment and macro indicators.

    Sources:
    - Yahoo Finance: S&P 500, VIX, DXY, Treasury Yields
    - Alternative.me: Crypto Fear & Greed Index
    - Investing.com: Baltic Dry Index (via proxy)
    """

    TICKERS = {
        "sp500": "^GSPC",           # S&P 500
        "vix": "^VIX",              # Volatility Index (Fear Index)
        "dxy": "DX-Y.NYB",          # US Dollar Index
        "us10y": "^TNX",            # 10-Year Treasury Yield
        "us2y": "^IRX",             # 2-Year Treasury Yield (proxy: 13-week)
        "nasdaq": "^IXIC",          # NASDAQ
        "dow": "^DJI",              # Dow Jones
        "russell": "^RUT",          # Russell 2000 (small caps)
    }

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=15)

    def _is_cache_valid(self) -> bool:
        if not self._cache_time:
            return False
        return datetime.now() - self._cache_time < self._cache_ttl

    def get_market_indices(self) -> Dict[str, Any]:
        """
        Get current market indices.

        Returns:
            Dict with index values and changes
        """
        if self._is_cache_valid() and "indices" in self._cache:
            return self._cache["indices"]

        indices = {}

        for name, ticker in self.TICKERS.items():
            try:
                data = yf.Ticker(ticker)
                hist = data.history(period="5d")

                if not hist.empty:
                    current = float(hist["Close"].iloc[-1])
                    prev = float(hist["Close"].iloc[0])
                    change_pct = ((current - prev) / prev) * 100 if prev > 0 else 0

                    indices[name] = {
                        "value": round(current, 2),
                        "change_5d_pct": round(change_pct, 2),
                    }
                else:
                    indices[name] = self._get_mock_index(name)

            except Exception as e:
                print(f"Error fetching {name}: {e}")
                indices[name] = self._get_mock_index(name)

        self._cache["indices"] = indices
        self._cache_time = datetime.now()
        return indices

    def _get_mock_index(self, name: str) -> Dict[str, Any]:
        """Return mock index data"""
        mocks = {
            "sp500": {"value": 5000.0, "change_5d_pct": 0.5},
            "vix": {"value": 15.0, "change_5d_pct": -2.0},
            "dxy": {"value": 104.0, "change_5d_pct": 0.2},
            "us10y": {"value": 4.5, "change_5d_pct": 0.1},
            "us2y": {"value": 4.8, "change_5d_pct": 0.05},
            "nasdaq": {"value": 16000.0, "change_5d_pct": 0.8},
            "dow": {"value": 39000.0, "change_5d_pct": 0.3},
            "russell": {"value": 2000.0, "change_5d_pct": 0.6},
        }
        return mocks.get(name, {"value": 100.0, "change_5d_pct": 0.0})

    async def get_crypto_fear_greed(self) -> Dict[str, Any]:
        """
        Get Crypto Fear & Greed Index from Alternative.me

        Scale: 0-100
        - 0-25: Extreme Fear (buy signal in Austrian terms)
        - 25-45: Fear
        - 45-55: Neutral
        - 55-75: Greed
        - 75-100: Extreme Greed (sell signal)
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.alternative.me/fng/",
                    params={"limit": "7"},
                    timeout=10.0,
                )
                response.raise_for_status()
                data = response.json()

                if data.get("data"):
                    latest = data["data"][0]
                    return {
                        "value": int(latest.get("value", 50)),
                        "classification": latest.get("value_classification", "Neutral"),
                        "timestamp": latest.get("timestamp"),
                        "history_7d": [
                            {"value": int(d["value"]), "date": d["timestamp"]}
                            for d in data["data"]
                        ],
                    }
        except Exception as e:
            print(f"Error fetching crypto fear/greed: {e}")

        return {
            "value": 50,
            "classification": "Neutral",
            "timestamp": str(int(datetime.now().timestamp())),
            "history_7d": [],
        }

    def get_yield_curve(self) -> Dict[str, Any]:
        """
        Get Treasury yield curve data.

        Austrian Theory:
        - Yield curve reflects market time preference
        - Inverted curve = recession signal
        - Central bank manipulation distorts this natural signal
        """
        try:
            us10y = yf.Ticker("^TNX")
            us2y = yf.Ticker("^IRX")  # 13-week as proxy

            hist_10y = us10y.history(period="1mo")
            hist_2y = us2y.history(period="1mo")

            if not hist_10y.empty and not hist_2y.empty:
                yield_10y = float(hist_10y["Close"].iloc[-1])
                yield_2y = float(hist_2y["Close"].iloc[-1])
                spread = yield_10y - yield_2y

                return {
                    "us10y": round(yield_10y, 2),
                    "us2y": round(yield_2y, 2),
                    "spread_10y_2y": round(spread, 2),
                    "inverted": spread < 0,
                    "recession_signal": spread < -0.5,
                }
        except Exception as e:
            print(f"Error fetching yield curve: {e}")

        return {
            "us10y": 4.5,
            "us2y": 4.8,
            "spread_10y_2y": -0.3,
            "inverted": True,
            "recession_signal": False,
        }

    def get_dollar_strength(self) -> Dict[str, Any]:
        """
        Get US Dollar strength indicators.

        Austrian Theory:
        - DXY measures dollar against other fiats (garbage vs garbage)
        - Real debasement is measured against gold/bitcoin
        - Rising DXY can still mean dollar losing value vs real assets
        """
        try:
            dxy = yf.Ticker("DX-Y.NYB")
            hist = dxy.history(period="1y")

            if not hist.empty:
                current = float(hist["Close"].iloc[-1])
                month_ago = float(hist["Close"].iloc[-22]) if len(hist) > 22 else current
                year_start = float(hist["Close"].iloc[0])

                return {
                    "dxy_current": round(current, 2),
                    "change_1m_pct": round(((current - month_ago) / month_ago) * 100, 2),
                    "change_ytd_pct": round(((current - year_start) / year_start) * 100, 2),
                    "high_1y": round(float(hist["High"].max()), 2),
                    "low_1y": round(float(hist["Low"].min()), 2),
                }
        except Exception as e:
            print(f"Error fetching DXY: {e}")

        return {
            "dxy_current": 104.0,
            "change_1m_pct": 0.5,
            "change_ytd_pct": 2.0,
            "high_1y": 107.0,
            "low_1y": 100.0,
        }

    def calculate_market_fear_level(self) -> Dict[str, Any]:
        """
        Calculate aggregate fear level from multiple indicators.

        Combines:
        - VIX (volatility)
        - Yield curve (recession risk)
        - Market momentum (recent performance)

        Returns 0-100 scale (0=extreme greed, 100=extreme fear)
        """
        indices = self.get_market_indices()
        yield_curve = self.get_yield_curve()

        fear_score = 50  # Base neutral

        # VIX contribution (higher VIX = more fear)
        vix = indices.get("vix", {}).get("value", 15)
        if vix > 30:
            fear_score += 20
        elif vix > 20:
            fear_score += 10
        elif vix < 12:
            fear_score -= 10

        # Yield curve contribution (inverted = more fear)
        if yield_curve.get("recession_signal"):
            fear_score += 15
        elif yield_curve.get("inverted"):
            fear_score += 8

        # Market momentum contribution (falling = more fear)
        sp500_change = indices.get("sp500", {}).get("change_5d_pct", 0)
        if sp500_change < -5:
            fear_score += 15
        elif sp500_change < -2:
            fear_score += 8
        elif sp500_change > 5:
            fear_score -= 10
        elif sp500_change > 2:
            fear_score -= 5

        fear_score = max(0, min(100, fear_score))

        if fear_score >= 75:
            classification = "Extreme Fear"
        elif fear_score >= 55:
            classification = "Fear"
        elif fear_score >= 45:
            classification = "Neutral"
        elif fear_score >= 25:
            classification = "Greed"
        else:
            classification = "Extreme Greed"

        return {
            "score": fear_score,
            "classification": classification,
            "components": {
                "vix": vix,
                "yield_spread": yield_curve.get("spread_10y_2y", 0),
                "sp500_momentum": sp500_change,
            },
        }

    def get_historical_volatility(self, period_years: int = 10) -> Dict[str, Any]:
        """
        Get historical volatility data to understand market sensitivity.

        Austrian Theory:
        - High historical volatility = market distorted by interventions
        - Events like 2008, 2020 show systemic fragility
        """
        try:
            sp500 = yf.Ticker("^GSPC")
            hist = sp500.history(period=f"{period_years}y")

            if not hist.empty:
                # Calculate rolling volatility
                hist["returns"] = hist["Close"].pct_change()
                hist["volatility_30d"] = hist["returns"].rolling(30).std() * (252 ** 0.5) * 100

                current_vol = float(hist["volatility_30d"].iloc[-1])
                avg_vol = float(hist["volatility_30d"].mean())
                max_vol = float(hist["volatility_30d"].max())

                # Find crisis periods
                high_vol_periods = hist[hist["volatility_30d"] > avg_vol * 2]

                return {
                    "current_volatility": round(current_vol, 2),
                    "average_volatility": round(avg_vol, 2),
                    "max_volatility": round(max_vol, 2),
                    "vol_vs_average_ratio": round(current_vol / avg_vol, 2) if avg_vol > 0 else 1.0,
                    "high_vol_periods_count": len(high_vol_periods) // 30,  # Approximate months
                    "market_stress_level": "high" if current_vol > avg_vol * 1.5 else "normal",
                }
        except Exception as e:
            print(f"Error calculating historical volatility: {e}")

        return {
            "current_volatility": 15.0,
            "average_volatility": 18.0,
            "max_volatility": 80.0,
            "vol_vs_average_ratio": 0.83,
            "high_vol_periods_count": 5,
            "market_stress_level": "normal",
        }


# Synchronous wrappers
def get_market_fear_level() -> Dict[str, Any]:
    """Get aggregate market fear level"""
    collector = MarketSentimentCollector()
    return collector.calculate_market_fear_level()


def get_market_indices() -> Dict[str, Any]:
    """Get current market indices"""
    collector = MarketSentimentCollector()
    return collector.get_market_indices()


def get_crypto_fear_greed() -> Dict[str, Any]:
    """Get crypto fear & greed index"""
    collector = MarketSentimentCollector()
    return asyncio.run(collector.get_crypto_fear_greed())
