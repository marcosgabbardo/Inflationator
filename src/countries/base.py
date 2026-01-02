"""
Country Configuration Base Classes

Defines the CountryConfig dataclass and related types for multi-country simulation.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any

from src.agents.government import RegimeType


class RelationType(str, Enum):
    """Relationship types between countries"""

    ALLY = "ally"  # Positive relationship, low conflict risk
    NEUTRAL = "neutral"  # No strong ties either way
    RIVAL = "rival"  # Competitive, moderate conflict risk
    ENEMY = "enemy"  # Hostile, high conflict risk


@dataclass
class CountryConfig:
    """
    Complete configuration for a country in the multi-country simulation.

    Contains economic parameters, regime type, central bank info,
    market tickers, and historical context that influences behavior.
    """

    # Identification
    code: str  # ISO 3166-1 alpha-3 (e.g., "USA", "BRA")
    name: str  # Full name
    currency: str  # ISO 4217 currency code (e.g., "USD", "BRL")

    # Political regime (Hoppe hierarchy)
    regime_type: RegimeType
    intervention_level: float  # 0-1 (0=ancap, 1=totalitarian)

    # Economic parameters (initial values)
    gdp_nominal_usd: Decimal  # GDP in USD
    population: int
    unemployment_rate: float  # 0-1
    inflation_rate: float  # Official (government-reported) - usually lies
    real_inflation_estimate: float  # Austrian estimate (commodity-based)

    # Central Bank
    central_bank_name: str
    base_money: Decimal  # In local currency
    policy_rate: float  # Current interest rate (0-1)
    inflation_target: float  # CB's stated target (usually 0.02)

    # Fiscal position
    debt_to_gdp: float  # Government debt / GDP ratio
    tax_burden: float  # Total tax as % of GDP

    # International trade
    trade_openness: float  # (Exports + Imports) / GDP
    main_exports: list[str] = field(default_factory=list)
    main_imports: list[str] = field(default_factory=list)

    # Market data (Yahoo Finance tickers)
    stock_index_ticker: str | None = None
    currency_ticker: str | None = None  # vs USD (e.g., "BRLUSD=X")
    bond_yield_ticker: str | None = None

    # Historical context (influences sensitivities and behavior)
    historical_context: dict[str, Any] = field(default_factory=dict)

    # Country-specific sensitivities (0-1 scale)
    usd_sensitivity: float = 0.5  # How much USD affects this country
    inflation_memory: float = 0.5  # Historical inflation trauma
    commodity_exposure: float = 0.5  # Dependency on commodity prices
    geopolitical_volatility: float = 0.5  # Political instability risk

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not 0 <= self.intervention_level <= 1:
            raise ValueError(
                f"intervention_level must be 0-1, got {self.intervention_level}"
            )
        if not 0 <= self.unemployment_rate <= 1:
            raise ValueError(
                f"unemployment_rate must be 0-1, got {self.unemployment_rate}"
            )
        if not 0 <= self.tax_burden <= 1:
            raise ValueError(f"tax_burden must be 0-1, got {self.tax_burden}")

    @property
    def freedom_index(self) -> float:
        """Austrian freedom index (inverse of intervention)"""
        return (1 - self.intervention_level) * 100

    @property
    def agent_scale(self) -> int:
        """
        Number of agents to create for this country.
        Proportional to GDP (USA = 100K baseline).
        """
        usa_gdp = Decimal("27000000000000")  # $27T
        ratio = float(self.gdp_nominal_usd / usa_gdp)
        # Scale: USA = 100K, minimum 1K
        return max(1000, int(100000 * ratio))

    def get_regime_parameters(self) -> dict[str, float]:
        """Get default parameters based on regime type"""
        from config.settings import REGIME_PARAMETERS

        return REGIME_PARAMETERS.get(self.regime_type, {})

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "code": self.code,
            "name": self.name,
            "currency": self.currency,
            "regime_type": self.regime_type.value,
            "intervention_level": self.intervention_level,
            "gdp_nominal_usd": str(self.gdp_nominal_usd),
            "population": self.population,
            "unemployment_rate": self.unemployment_rate,
            "inflation_rate": self.inflation_rate,
            "real_inflation_estimate": self.real_inflation_estimate,
            "central_bank_name": self.central_bank_name,
            "base_money": str(self.base_money),
            "policy_rate": self.policy_rate,
            "inflation_target": self.inflation_target,
            "debt_to_gdp": self.debt_to_gdp,
            "tax_burden": self.tax_burden,
            "trade_openness": self.trade_openness,
            "main_exports": self.main_exports,
            "main_imports": self.main_imports,
            "stock_index_ticker": self.stock_index_ticker,
            "currency_ticker": self.currency_ticker,
            "bond_yield_ticker": self.bond_yield_ticker,
            "historical_context": self.historical_context,
            "freedom_index": self.freedom_index,
            "agent_scale": self.agent_scale,
        }


@dataclass
class BilateralRelationship:
    """
    Represents the relationship between two countries.

    Includes trade data, political relationship, and war probability
    for simulation purposes.
    """

    country_a: str  # ISO code
    country_b: str  # ISO code

    # Relationship type and strength
    relationship_type: RelationType
    strength: float  # -1 (enemy) to +1 (strong ally)

    # Trade data
    trade_volume_usd: Decimal  # Annual bilateral trade in USD
    trade_balance: Decimal = Decimal("0")  # Positive = A exports more to B
    tariff_a_to_b: float = 0.0  # Tariff A imposes on B's goods
    tariff_b_to_a: float = 0.0  # Tariff B imposes on A's goods

    # Sanctions (lists of sanction types)
    sanctions_a_on_b: list[str] = field(default_factory=list)
    sanctions_b_on_a: list[str] = field(default_factory=list)

    # Historical and current tensions
    historical_conflicts: int = 0  # Number of past conflicts
    current_tensions: list[str] = field(default_factory=list)

    # Economic dependency (0-1)
    a_dependency_on_b: float = 0.0  # How much A depends on B
    b_dependency_on_a: float = 0.0  # How much B depends on A

    # War probability indicators
    war_probability: float = 0.0  # 0-1, calculated dynamically
    war_triggers: list[str] = field(default_factory=list)
    escalation_level: float = 0.0  # Current level of escalation (0-1)

    def __post_init__(self):
        """Validate relationship after initialization"""
        if not -1 <= self.strength <= 1:
            raise ValueError(f"strength must be -1 to 1, got {self.strength}")
        if not 0 <= self.war_probability <= 1:
            raise ValueError(f"war_probability must be 0-1, got {self.war_probability}")

    @property
    def is_hostile(self) -> bool:
        """Returns True if relationship is hostile (rival or enemy)"""
        return self.relationship_type in [RelationType.RIVAL, RelationType.ENEMY]

    @property
    def has_active_sanctions(self) -> bool:
        """Returns True if either country has sanctions on the other"""
        return bool(self.sanctions_a_on_b or self.sanctions_b_on_a)

    @property
    def trade_interdependence(self) -> float:
        """
        Measure of trade interdependence (0-1).
        High interdependence reduces war probability (peace through commerce).
        """
        return min(self.a_dependency_on_b, self.b_dependency_on_a)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "country_a": self.country_a,
            "country_b": self.country_b,
            "relationship_type": self.relationship_type.value,
            "strength": self.strength,
            "trade_volume_usd": str(self.trade_volume_usd),
            "trade_balance": str(self.trade_balance),
            "tariff_a_to_b": self.tariff_a_to_b,
            "tariff_b_to_a": self.tariff_b_to_a,
            "sanctions_a_on_b": self.sanctions_a_on_b,
            "sanctions_b_on_a": self.sanctions_b_on_a,
            "historical_conflicts": self.historical_conflicts,
            "current_tensions": self.current_tensions,
            "a_dependency_on_b": self.a_dependency_on_b,
            "b_dependency_on_a": self.b_dependency_on_a,
            "war_probability": self.war_probability,
            "war_triggers": self.war_triggers,
            "escalation_level": self.escalation_level,
            "is_hostile": self.is_hostile,
            "has_active_sanctions": self.has_active_sanctions,
            "trade_interdependence": self.trade_interdependence,
        }
