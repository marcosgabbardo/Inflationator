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
from .collectors.country_collector import CountryDataCollector
from .collectors.forex import ForexCollector


@dataclass
class EconomicConditions:
    """
    Aggregated real-world economic conditions.

    Used to initialize the simulation with TODAY's economy.
    Now supports country-specific data.
    """

    # Country identifier
    country: str = "USA"
    currency_code: str = "USD"

    # Asset prices (global)
    btc_price_usd: Decimal = Decimal("50000")
    gold_price_usd: Decimal = Decimal("2000")
    silver_price_usd: Decimal = Decimal("25")
    oil_price_usd: Decimal = Decimal("80")

    # Country-specific stock index (not just S&P 500)
    stock_index_name: str = "S&P 500"
    stock_index_value: float = 5000.0
    stock_index_change_pct: float = 0.0

    # US Market indices (global reference)
    sp500_value: float = 5000.0
    vix_value: float = 15.0
    dxy_value: float = 104.0

    # Country-specific exchange rate vs USD
    fx_rate_vs_usd: float = 1.0
    fx_change_30d_pct: float = 0.0

    # Country-specific interest rate
    local_interest_rate: float = 0.05
    treasury_10y: float = 4.5  # US reference
    treasury_2y: float = 4.8   # US reference
    yield_curve_inverted: bool = False

    # Sentiment
    market_fear_level: int = 50  # 0-100 (higher = more fear)
    crypto_fear_greed: int = 50  # 0-100 (higher = more greed)
    market_sentiment: str = "neutral"  # fear, neutral, greed
    local_market_bullish: bool = True

    # Historical context
    inflation_estimate: float = 5.0  # Commodity-based, not CPI
    local_inflation_estimate: float = 5.0  # Country-specific estimate
    dollar_debasement_10y: float = 50.0  # % vs gold/btc over 10 years
    volatility_ratio: float = 1.0  # Current vs average volatility
    local_volatility: float = 15.0  # Country stock market volatility

    # Historical economic data (10-year averages)
    historical_gdp_growth: float = 2.0  # Average annual GDP growth
    historical_unemployment: float = 7.0  # Average unemployment rate

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

    Now supports country-specific data collection.
    """

    # Stock index names per country
    STOCK_INDEX_NAMES = {
        "USA": "S&P 500",
        "CAN": "S&P/TSX",
        "MEX": "IPC Mexico",
        "BRA": "Bovespa",
        "ARG": "Merval",
        "GBR": "FTSE 100",
        "DEU": "DAX",
        "FRA": "CAC 40",
        "SWE": "OMX Stockholm",
        "NOR": "Oslo All Share",
        "CHE": "SMI",
        "LIE": "SMI (CH)",
        "CHN": "Shanghai Composite",
        "JPN": "Nikkei 225",
        "IND": "BSE Sensex",
        "IDN": "Jakarta Composite",
        "ARE": "Dubai Financial",
        "SAU": "Tadawul",
        "RUS": "MOEX Russia",
        "TUR": "BIST 100",
    }

    # Estimated local inflation rates (Austrian estimate, not CPI)
    LOCAL_INFLATION_ESTIMATES = {
        "USA": 8.0,   # Real inflation higher than CPI claims
        "CAN": 6.0,
        "MEX": 12.0,
        "BRA": 15.0,
        "ARG": 150.0,  # Hyperinflation
        "GBR": 7.0,
        "DEU": 5.0,
        "FRA": 5.5,
        "SWE": 6.0,
        "NOR": 4.0,
        "CHE": 2.0,   # Low intervention = low inflation
        "LIE": 2.0,
        "CHN": 8.0,
        "JPN": 3.0,   # Deflation history
        "IND": 10.0,
        "IDN": 8.0,
        "ARE": 4.0,
        "SAU": 4.0,
        "RUS": 20.0,  # Sanctions effect
        "TUR": 80.0,  # Erdogan policy
    }

    # Historical average GDP growth rates (10-year average, real terms)
    HISTORICAL_GDP_GROWTH = {
        "USA": 2.3,    # Slow but steady
        "CAN": 1.8,
        "MEX": 1.5,    # Stagnant
        "BRA": 0.5,    # Lost decade
        "ARG": -1.0,   # Negative growth (intervention damage)
        "GBR": 1.5,
        "DEU": 1.2,
        "FRA": 1.0,
        "SWE": 2.0,
        "NOR": 1.5,
        "CHE": 1.8,    # Stable growth despite low intervention
        "LIE": 2.5,    # High growth (low intervention)
        "CHN": 5.5,    # Slowing from previous highs
        "JPN": 0.8,    # Stagnation
        "IND": 6.0,    # Strong growth
        "IDN": 5.0,
        "ARE": 3.5,    # Oil-driven
        "SAU": 2.0,    # Oil-dependent
        "RUS": 1.0,    # Sanctions impact
        "TUR": 3.0,    # Volatile but growing
    }

    # Historical unemployment rates (baseline, not government manipulated)
    # Austrian estimate: real unemployment is higher than official stats
    HISTORICAL_UNEMPLOYMENT = {
        "USA": 7.0,    # Real: higher than official 4%
        "CAN": 7.5,
        "MEX": 8.0,
        "BRA": 12.0,   # High structural unemployment
        "ARG": 15.0,   # Very high
        "GBR": 6.0,
        "DEU": 5.5,
        "FRA": 9.0,    # High due to regulations
        "SWE": 8.0,
        "NOR": 4.5,    # Low (oil wealth)
        "CHE": 3.5,    # Very low (minarchy works)
        "LIE": 2.0,    # Lowest (monarchy + low intervention)
        "CHN": 8.0,    # Hidden unemployment
        "JPN": 4.0,    # Low but underemployment
        "IND": 10.0,   # High informal economy
        "IDN": 7.0,
        "ARE": 3.0,    # Low (oil + migrant workers)
        "SAU": 12.0,   # High among nationals
        "RUS": 8.0,
        "TUR": 12.0,   # High and volatile
    }

    def __init__(self, country: str = "USA"):
        self.country = country
        self.btc_collector = BitcoinCollector()
        self.commodities_collector = CommoditiesCollector()
        self.sentiment_collector = MarketSentimentCollector()
        self.country_collector = CountryDataCollector(country)
        self.forex_collector = ForexCollector()

    async def fetch_all_conditions(self) -> EconomicConditions:
        """
        Fetch all real-world conditions in parallel.

        Now includes country-specific data for non-US countries.

        Returns:
            EconomicConditions with current market state
        """
        conditions = EconomicConditions()
        conditions.country = self.country
        conditions.currency_code = self.country_collector.currency_code

        # Fetch country-specific data first
        try:
            country_data = self.country_collector.get_country_conditions()
            conditions.stock_index_name = self.STOCK_INDEX_NAMES.get(self.country, "Stock Index")
            conditions.stock_index_value = float(country_data.stock_index_value)
            conditions.stock_index_change_pct = country_data.stock_index_change_pct
            conditions.fx_rate_vs_usd = float(country_data.fx_rate_vs_usd)
            conditions.fx_change_30d_pct = country_data.fx_change_30d_pct
            conditions.local_interest_rate = country_data.implied_rate
            conditions.local_volatility = country_data.market_volatility
            conditions.local_market_bullish = country_data.is_bullish
            conditions.local_inflation_estimate = self.LOCAL_INFLATION_ESTIMATES.get(
                self.country, 5.0
            )
            # Add historical data
            conditions.historical_gdp_growth = self.HISTORICAL_GDP_GROWTH.get(
                self.country, 2.0
            )
            conditions.historical_unemployment = self.HISTORICAL_UNEMPLOYMENT.get(
                self.country, 7.0
            )
        except Exception as e:
            print(f"Error fetching country-specific data for {self.country}: {e}")

        # Fetch global data (BTC, commodities)
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
    """Print a summary of current conditions (now country-aware)."""
    country = conditions.country
    is_usa = country == "USA"

    print("\n" + "=" * 50)
    if is_usa:
        print("REAL WORLD CONDITIONS - TODAY'S ECONOMY")
    else:
        print(f"REAL WORLD CONDITIONS - {country}")
    print("=" * 50)

    print(f"\n{'ASSET PRICES (Global)':^50}")
    print("-" * 50)
    print(f"  Bitcoin:  ${conditions.btc_price_usd:,.0f}")
    print(f"  Gold:     ${conditions.gold_price_usd:,.0f}")
    print(f"  Silver:   ${conditions.silver_price_usd:,.0f}")
    print(f"  Oil:      ${conditions.oil_price_usd:,.0f}")

    # Show country-specific stock index
    print(f"\n{f'{country} MARKET':^50}")
    print("-" * 50)
    print(f"  {conditions.stock_index_name}:  {conditions.stock_index_value:,.0f}")
    print(f"  30-Day Change:  {conditions.stock_index_change_pct:+.1f}%")
    print(f"  Market Trend:   {'BULLISH' if conditions.local_market_bullish else 'BEARISH'}")
    print(f"  Volatility:     {conditions.local_volatility:.1f}%")

    # Show currency info for non-USD countries
    if not is_usa:
        print(f"\n{f'{conditions.currency_code} CURRENCY':^50}")
        print("-" * 50)
        print(f"  Rate vs USD:    {conditions.fx_rate_vs_usd:.4f}")
        print(f"  30-Day Change:  {conditions.fx_change_30d_pct:+.1f}%")

    # Show global reference indices
    print(f"\n{'GLOBAL REFERENCE':^50}")
    print("-" * 50)
    print(f"  S&P 500:  {conditions.sp500_value:,.0f}")
    print(f"  VIX:      {conditions.vix_value:.1f}")
    print(f"  DXY:      {conditions.dxy_value:.1f}")

    print(f"\n{'INTEREST RATES':^50}")
    print("-" * 50)
    print(f"  {country} Rate:     {conditions.local_interest_rate:.2%}")
    print(f"  US 10Y Treasury:  {conditions.treasury_10y:.2f}%")
    print(f"  Yield Curve:      {'INVERTED' if conditions.yield_curve_inverted else 'Normal'}")

    print(f"\n{'SENTIMENT':^50}")
    print("-" * 50)
    print(f"  Market Fear Level:    {conditions.market_fear_level}/100 ({conditions.market_sentiment})")
    print(f"  Crypto Fear & Greed:  {conditions.crypto_fear_greed}/100")

    print(f"\n{'INFLATION ESTIMATES':^50}")
    print("-" * 50)
    print(f"  {country} Real Inflation: {conditions.local_inflation_estimate:.1f}% (Austrian estimate)")
    print(f"  Global (commodity):     {conditions.inflation_estimate:.1f}%")
    print(f"  Dollar Debasement:      {conditions.dollar_debasement_10y:.1f}% (10 years)")

    print(f"\n{'MACRO INDICATORS':^50}")
    print("-" * 50)
    print(f"  Recession Prob:       {conditions.recession_probability:.0%}")
    print(f"  Monetary Policy:      {conditions.monetary_expansion_signal}")
    print(f"  Geopolitical Risk:    {conditions.geopolitical_risk_level}")
    print(f"  Trade Tensions:       {conditions.trade_tensions:.0%}")

    print(f"\n  Data fetched at: {conditions.fetched_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
