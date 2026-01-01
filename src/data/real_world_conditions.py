"""
Real World Conditions Aggregator

Fetches and aggregates real-time economic data from multiple non-government sources
to create realistic initial conditions for the simulation.

Austrian Theory:
- Initial conditions must reflect TODAY's actual economy
- Historical context (10-20 years) shapes agent behavior and expectations
- Market prices contain distributed information (Hayek)
- Government statistics lie; use market data instead
"""

import asyncio
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Any, Optional, List
from datetime import datetime
import random

from .collectors.bitcoin import BitcoinCollector, get_bitcoin_price
from .collectors.commodities import CommoditiesCollector, get_commodity_prices
from .collectors.market_sentiment import (
    MarketSentimentCollector,
    get_market_indices,
    get_market_fear_level,
)


@dataclass
class EconomicConditions:
    """
    Aggregated real-world economic conditions.

    Used to initialize the simulation with TODAY's economy.
    """

    # Asset prices
    btc_price_usd: Decimal = Decimal("50000")
    gold_price_usd: Decimal = Decimal("2000")
    silver_price_usd: Decimal = Decimal("25")
    oil_price_usd: Decimal = Decimal("80")

    # Market indices
    sp500_value: float = 5000.0
    vix_value: float = 15.0
    dxy_value: float = 104.0

    # Interest rates (market-derived)
    treasury_10y: float = 4.5
    treasury_2y: float = 4.8
    yield_curve_inverted: bool = False

    # Sentiment
    market_fear_level: int = 50  # 0-100 (higher = more fear)
    crypto_fear_greed: int = 50  # 0-100 (higher = more greed)
    market_sentiment: str = "neutral"  # fear, neutral, greed

    # Historical context
    inflation_estimate: float = 5.0  # Commodity-based, not CPI
    dollar_debasement_10y: float = 50.0  # % vs gold/btc over 10 years
    volatility_ratio: float = 1.0  # Current vs average volatility

    # Geopolitical
    geopolitical_risk_level: str = "moderate"  # low, moderate, high, extreme
    trade_tensions: float = 0.3  # 0-1

    # Derived metrics
    recession_probability: float = 0.2  # 0-1
    monetary_expansion_signal: str = "moderate"  # tight, moderate, loose

    # Timestamp
    fetched_at: datetime = field(default_factory=datetime.now)


class RealWorldConditionsCollector:
    """
    Aggregates data from multiple sources to create realistic initial conditions.
    """

    def __init__(self, country: str = "USA"):
        self.country = country
        self.btc_collector = BitcoinCollector()
        self.commodities_collector = CommoditiesCollector()
        self.sentiment_collector = MarketSentimentCollector()

    async def fetch_all_conditions(self) -> EconomicConditions:
        """
        Fetch all real-world conditions in parallel.

        Returns:
            EconomicConditions with current market state
        """
        conditions = EconomicConditions()

        # Fetch in parallel where possible
        try:
            # Async fetches
            btc_task = self.btc_collector.get_current_price()
            btc_market_task = self.btc_collector.get_market_data()
            crypto_fg_task = self.sentiment_collector.get_crypto_fear_greed()

            btc_price, btc_market, crypto_fg = await asyncio.gather(
                btc_task, btc_market_task, crypto_fg_task,
                return_exceptions=True
            )

            # Process BTC data
            if isinstance(btc_price, dict):
                conditions.btc_price_usd = btc_price.get("usd", Decimal("50000"))
            if isinstance(crypto_fg, dict):
                conditions.crypto_fear_greed = crypto_fg.get("value", 50)

        except Exception as e:
            print(f"Error in async fetches: {e}")

        # Sync fetches
        try:
            commodities = self.commodities_collector.get_current_prices()
            conditions.gold_price_usd = commodities.get("gold", Decimal("2000"))
            conditions.silver_price_usd = commodities.get("silver", Decimal("25"))
            conditions.oil_price_usd = commodities.get("oil", Decimal("80"))
        except Exception as e:
            print(f"Error fetching commodities: {e}")

        try:
            indices = self.sentiment_collector.get_market_indices()
            conditions.sp500_value = indices.get("sp500", {}).get("value", 5000.0)
            conditions.vix_value = indices.get("vix", {}).get("value", 15.0)
            conditions.dxy_value = indices.get("dxy", {}).get("value", 104.0)
        except Exception as e:
            print(f"Error fetching indices: {e}")

        try:
            yield_curve = self.sentiment_collector.get_yield_curve()
            conditions.treasury_10y = yield_curve.get("us10y", 4.5)
            conditions.treasury_2y = yield_curve.get("us2y", 4.8)
            conditions.yield_curve_inverted = yield_curve.get("inverted", False)
        except Exception as e:
            print(f"Error fetching yield curve: {e}")

        try:
            fear_level = self.sentiment_collector.calculate_market_fear_level()
            conditions.market_fear_level = fear_level.get("score", 50)
            conditions.market_sentiment = self._classify_sentiment(
                conditions.market_fear_level
            )
        except Exception as e:
            print(f"Error calculating fear level: {e}")

        try:
            volatility = self.sentiment_collector.get_historical_volatility(10)
            conditions.volatility_ratio = volatility.get("vol_vs_average_ratio", 1.0)
        except Exception as e:
            print(f"Error fetching volatility: {e}")

        # Calculate derived metrics
        conditions.inflation_estimate = self._estimate_real_inflation(conditions)
        conditions.dollar_debasement_10y = self._calculate_debasement(conditions)
        conditions.recession_probability = self._estimate_recession_prob(conditions)
        conditions.monetary_expansion_signal = self._assess_monetary_policy(conditions)
        conditions.geopolitical_risk_level = self._assess_geopolitical_risk(conditions)
        conditions.trade_tensions = self._assess_trade_tensions(conditions)

        conditions.fetched_at = datetime.now()
        return conditions

    def _classify_sentiment(self, fear_level: int) -> str:
        if fear_level >= 60:
            return "fear"
        elif fear_level <= 40:
            return "greed"
        return "neutral"

    def _estimate_real_inflation(self, conditions: EconomicConditions) -> float:
        """
        Estimate real inflation using commodity prices.

        Austrian Theory:
        - CPI is manipulated; use hard assets instead
        - Gold and BTC are sound money benchmarks
        """
        # Compare current prices to historical baselines
        # These baselines are from approximately 5 years ago
        gold_baseline = Decimal("1300")  # Gold ~5 years ago
        btc_baseline = Decimal("10000")  # BTC ~5 years ago

        gold_inflation = float((conditions.gold_price_usd - gold_baseline) / gold_baseline) * 100 / 5
        btc_inflation = float((conditions.btc_price_usd - btc_baseline) / btc_baseline) * 100 / 5

        # Weighted average (gold more stable, btc more volatile)
        real_inflation = gold_inflation * 0.7 + btc_inflation * 0.1

        # Cap at reasonable range
        return max(0, min(30, real_inflation))

    def _calculate_debasement(self, conditions: EconomicConditions) -> float:
        """
        Calculate dollar debasement over 10 years.

        Uses gold and BTC as benchmarks.
        """
        # 10-year baselines
        gold_10y_ago = Decimal("1200")
        btc_10y_ago = Decimal("500")

        gold_debasement = float((conditions.gold_price_usd / gold_10y_ago - 1) * 100)
        btc_debasement = float((conditions.btc_price_usd / btc_10y_ago - 1) * 100)

        # Use more conservative gold measure
        return gold_debasement

    def _estimate_recession_prob(self, conditions: EconomicConditions) -> float:
        """
        Estimate recession probability from market signals.

        Austrian Theory:
        - Yield curve inversion is a market signal
        - High VIX shows uncertainty
        - These are natural market signals, not models
        """
        prob = 0.15  # Base probability

        # Yield curve inversion is strong signal
        if conditions.yield_curve_inverted:
            prob += 0.25

        # High VIX indicates stress
        if conditions.vix_value > 30:
            prob += 0.15
        elif conditions.vix_value > 20:
            prob += 0.08

        # High fear level
        if conditions.market_fear_level > 70:
            prob += 0.10

        # High volatility ratio
        if conditions.volatility_ratio > 1.5:
            prob += 0.10

        return min(0.85, prob)

    def _assess_monetary_policy(self, conditions: EconomicConditions) -> str:
        """
        Assess current monetary policy stance from market signals.

        Austrian Theory:
        - Interest rates should reflect natural time preference
        - Current rates are artificially manipulated
        """
        # Use yield as proxy (higher = tighter)
        if conditions.treasury_10y > 5.0:
            return "tight"
        elif conditions.treasury_10y < 3.0:
            return "loose"
        return "moderate"

    def _assess_geopolitical_risk(self, conditions: EconomicConditions) -> str:
        """
        Assess geopolitical risk from market signals.

        Higher VIX, falling markets, high fear = more risk
        """
        risk_score = 0

        if conditions.vix_value > 30:
            risk_score += 3
        elif conditions.vix_value > 20:
            risk_score += 1

        if conditions.market_fear_level > 70:
            risk_score += 2
        elif conditions.market_fear_level > 55:
            risk_score += 1

        if conditions.oil_price_usd > Decimal("100"):
            risk_score += 1

        if risk_score >= 5:
            return "extreme"
        elif risk_score >= 3:
            return "high"
        elif risk_score >= 1:
            return "moderate"
        return "low"

    def _assess_trade_tensions(self, conditions: EconomicConditions) -> float:
        """
        Assess trade tensions (0-1 scale).

        Currently simplified; could be enhanced with shipping data.
        """
        # Use DXY as proxy (high dollar can indicate trade stress)
        if conditions.dxy_value > 110:
            return 0.7
        elif conditions.dxy_value > 105:
            return 0.4
        elif conditions.dxy_value < 95:
            return 0.2
        return 0.3

    def to_simulation_config(self, conditions: EconomicConditions) -> Dict[str, Any]:
        """
        Convert economic conditions to simulation initial state.

        Returns:
            Dict with parameters for SimulationEngine
        """
        return {
            # Market prices
            "initial_prices": {
                "crypto": float(conditions.btc_price_usd),
                "precious_metals": float(conditions.gold_price_usd),
                "energy": float(conditions.oil_price_usd),
            },

            # Central bank parameters
            "central_bank": {
                "policy_rate": conditions.treasury_2y / 100,  # Convert to decimal
                "inflation_target": 0.02,
                "qe_active": conditions.monetary_expansion_signal == "loose",
            },

            # Agent behavior modifiers
            "agent_modifiers": {
                "fear_level": conditions.market_fear_level / 100,
                "time_preference_shift": self._calculate_time_pref_shift(conditions),
                "risk_tolerance_shift": self._calculate_risk_shift(conditions),
                "inflation_expectations": conditions.inflation_estimate / 100,
            },

            # Market conditions
            "market_conditions": {
                "volatility_multiplier": conditions.volatility_ratio,
                "recession_probability": conditions.recession_probability,
                "dollar_strength": conditions.dxy_value / 100,
            },

            # Geopolitical
            "geopolitics": {
                "risk_level": conditions.geopolitical_risk_level,
                "trade_tensions": conditions.trade_tensions,
            },

            # Metadata
            "data_timestamp": conditions.fetched_at.isoformat(),
            "country": self.country,
        }

    def _calculate_time_pref_shift(self, conditions: EconomicConditions) -> float:
        """
        Calculate shift in agent time preference based on conditions.

        Austrian Theory:
        - Fear increases time preference (want things NOW)
        - Low rates artificially lower time preference
        - Inflation increases time preference
        """
        shift = 0.0

        # Fear increases time preference
        if conditions.market_fear_level > 70:
            shift += 0.15
        elif conditions.market_fear_level > 55:
            shift += 0.08

        # High inflation increases time preference
        if conditions.inflation_estimate > 10:
            shift += 0.10
        elif conditions.inflation_estimate > 5:
            shift += 0.05

        # Low rates artificially decrease (distortion)
        if conditions.treasury_10y < 3.0:
            shift -= 0.10

        return max(-0.2, min(0.2, shift))

    def _calculate_risk_shift(self, conditions: EconomicConditions) -> float:
        """
        Calculate shift in agent risk tolerance based on conditions.

        Greed = more risk taking
        Fear = less risk taking
        """
        # Convert fear level to risk shift
        # 50 = neutral, lower = more risk, higher = less risk
        shift = (50 - conditions.market_fear_level) / 100

        # VIX adjustment
        if conditions.vix_value > 30:
            shift -= 0.15
        elif conditions.vix_value < 15:
            shift += 0.10

        return max(-0.3, min(0.3, shift))


class RealWorldInitializer:
    """
    Initializes simulation agents and markets with real-world conditions.
    """

    def __init__(self, conditions: EconomicConditions, country: str = "USA"):
        self.conditions = conditions
        self.country = country

    def get_initial_wealth_distribution(self, num_agents: int) -> List[Decimal]:
        """
        Generate realistic wealth distribution based on current conditions.

        Uses log-normal distribution adjusted for current inequality levels.
        """
        # Gini coefficient proxy (higher fear = more hoarding by wealthy)
        gini_factor = 1.0 + (self.conditions.market_fear_level - 50) / 200

        wealth_list = []
        for _ in range(num_agents):
            # Log-normal distribution
            base_wealth = random.lognormvariate(10, 1.5 * gini_factor)

            # Adjust for inflation (higher inflation = nominal wealth higher)
            inflation_factor = 1 + self.conditions.inflation_estimate / 100

            wealth = Decimal(str(max(100, base_wealth * inflation_factor)))
            wealth_list.append(wealth)

        return wealth_list

    def get_initial_time_preferences(self, num_agents: int) -> List[float]:
        """
        Generate realistic time preference distribution.

        Austrian Theory:
        - Natural distribution of time preferences
        - Current conditions shift the entire distribution
        """
        base_mean = 0.5
        base_std = 0.15

        # Adjust for current conditions
        shift = 0.0
        if self.conditions.inflation_estimate > 8:
            shift += 0.10  # High inflation = higher time preference
        if self.conditions.market_fear_level > 65:
            shift += 0.08  # Fear = higher time preference
        if self.conditions.recession_probability > 0.5:
            shift += 0.05

        preferences = []
        for _ in range(num_agents):
            pref = random.gauss(base_mean + shift, base_std)
            pref = max(0.1, min(0.9, pref))
            preferences.append(pref)

        return preferences

    def get_employment_rate(self) -> float:
        """
        Estimate initial employment rate based on market conditions.

        Uses market signals instead of government statistics.
        """
        # Base rate (not government's fake 4%)
        base_rate = 0.88  # ~12% real unemployment

        # Adjust for conditions
        if self.conditions.recession_probability > 0.6:
            base_rate -= 0.05
        elif self.conditions.recession_probability < 0.2:
            base_rate += 0.02

        if self.conditions.market_sentiment == "fear":
            base_rate -= 0.02

        return max(0.70, min(0.95, base_rate))

    def get_initial_consumer_prices(self) -> Dict[str, Decimal]:
        """
        Get realistic initial prices for consumer goods.

        Based on commodity prices and inflation estimates.
        """
        # Base prices adjusted for current conditions
        inflation_multiplier = 1 + self.conditions.inflation_estimate / 100

        # Approximate real prices (2024 baseline)
        base_prices = {
            "food": Decimal("15"),  # Average meal
            "housing": Decimal("2000"),  # Monthly rent
            "transportation": Decimal("500"),  # Monthly
            "healthcare": Decimal("400"),  # Monthly
            "education": Decimal("1000"),  # Monthly
            "entertainment": Decimal("200"),  # Monthly
        }

        # Energy costs tied to oil
        energy_ratio = float(self.conditions.oil_price_usd) / 80  # 80 as baseline

        adjusted_prices = {}
        for category, price in base_prices.items():
            adjusted = price * Decimal(str(inflation_multiplier))

            # Transportation and housing more affected by energy
            if category in ["transportation", "housing"]:
                adjusted *= Decimal(str(0.7 + 0.3 * energy_ratio))

            adjusted_prices[category] = adjusted.quantize(Decimal("0.01"))

        return adjusted_prices


# Main functions
def get_real_world_conditions(country: str = "USA") -> EconomicConditions:
    """
    Fetch current real-world economic conditions.

    Synchronous wrapper for async function.
    """
    collector = RealWorldConditionsCollector(country)
    return asyncio.run(collector.fetch_all_conditions())


def get_simulation_initial_state(country: str = "USA") -> Dict[str, Any]:
    """
    Get initial state for simulation based on real-world conditions.
    """
    collector = RealWorldConditionsCollector(country)
    conditions = asyncio.run(collector.fetch_all_conditions())
    return collector.to_simulation_config(conditions)


def print_conditions_summary(conditions: EconomicConditions):
    """Print a summary of current conditions."""
    print("\n" + "=" * 50)
    print("REAL WORLD CONDITIONS - TODAY'S ECONOMY")
    print("=" * 50)

    print(f"\n{'ASSET PRICES':^50}")
    print("-" * 50)
    print(f"  Bitcoin:  ${conditions.btc_price_usd:,.0f}")
    print(f"  Gold:     ${conditions.gold_price_usd:,.0f}")
    print(f"  Silver:   ${conditions.silver_price_usd:,.0f}")
    print(f"  Oil:      ${conditions.oil_price_usd:,.0f}")

    print(f"\n{'MARKET INDICES':^50}")
    print("-" * 50)
    print(f"  S&P 500:  {conditions.sp500_value:,.0f}")
    print(f"  VIX:      {conditions.vix_value:.1f}")
    print(f"  DXY:      {conditions.dxy_value:.1f}")

    print(f"\n{'INTEREST RATES':^50}")
    print("-" * 50)
    print(f"  10Y Treasury:  {conditions.treasury_10y:.2f}%")
    print(f"  2Y Treasury:   {conditions.treasury_2y:.2f}%")
    print(f"  Yield Curve:   {'INVERTED' if conditions.yield_curve_inverted else 'Normal'}")

    print(f"\n{'SENTIMENT':^50}")
    print("-" * 50)
    print(f"  Market Fear Level:    {conditions.market_fear_level}/100 ({conditions.market_sentiment})")
    print(f"  Crypto Fear & Greed:  {conditions.crypto_fear_greed}/100")

    print(f"\n{'DERIVED METRICS':^50}")
    print("-" * 50)
    print(f"  Real Inflation Est:   {conditions.inflation_estimate:.1f}% (commodity-based)")
    print(f"  Dollar Debasement:    {conditions.dollar_debasement_10y:.1f}% (10 years)")
    print(f"  Recession Prob:       {conditions.recession_probability:.0%}")
    print(f"  Monetary Policy:      {conditions.monetary_expansion_signal}")

    print(f"\n{'GEOPOLITICAL':^50}")
    print("-" * 50)
    print(f"  Risk Level:           {conditions.geopolitical_risk_level}")
    print(f"  Trade Tensions:       {conditions.trade_tensions:.0%}")

    print(f"\n  Data fetched at: {conditions.fetched_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
