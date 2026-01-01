"""
Geopolitical Influence Calculator

Calculates how countries economically influence each other.

Austrian Theory Relevance:
- Cantillon effect extends internationally (USD first receivers benefit)
- Trade creates interdependence and price transmission
- Monetary policy spills over to trading partners
- Commodity producers have outsized influence on importers
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum

from src.countries.base import CountryConfig, BilateralRelationship, RelationType


class InfluenceType(str, Enum):
    """Types of economic influence"""
    MONETARY = "monetary"           # Central bank policy transmission
    TRADE = "trade"                 # Export/import price effects
    COMMODITY = "commodity"         # Commodity price shocks
    FINANCIAL = "financial"         # Capital flows, interest rates
    CURRENCY = "currency"           # Exchange rate effects
    SENTIMENT = "sentiment"         # Market contagion
    CRISIS = "crisis"               # Crisis propagation


@dataclass
class InfluenceResult:
    """Result of influence calculation between two countries"""
    source_country: str
    target_country: str
    influence_type: InfluenceType
    magnitude: float                # 0-1 (how much influence)
    direction: str                  # "positive", "negative", "neutral"
    transmission_channels: List[str]
    lag_months: int                 # How long for effect to manifest


class CountrySensitivities:
    """
    Country-specific sensitivities based on historical context.

    Some countries are more sensitive to:
    - USD movements (dollarized, high USD debt)
    - Commodity prices (exporters/importers)
    - Inflation (historical trauma)
    - China (trade dependency)
    - Geopolitical events
    """

    SENSITIVITIES = {
        "USA": {
            "usd_sensitivity": 0.0,      # Is the USD issuer
            "inflation_memory": 0.3,     # Low memory (1970s long ago)
            "commodity_exposure": 0.3,   # Net importer but diversified
            "china_dependency": 0.4,     # Significant trade
            "geopolitical_volatility": 0.3,
        },
        "ARG": {
            "usd_sensitivity": 0.95,     # Highly dollarized, USD debt
            "inflation_memory": 0.98,    # Hyperinflation trauma (multiple)
            "commodity_exposure": 0.7,   # Agro exporter
            "china_dependency": 0.3,
            "geopolitical_volatility": 0.6,
            "imf_trauma": 0.9,           # Historical IMF issues
        },
        "BRA": {
            "usd_sensitivity": 0.6,
            "inflation_memory": 0.75,    # Plano Real 1994
            "commodity_exposure": 0.8,   # Soy, iron ore, oil
            "china_dependency": 0.85,    # Largest trade partner
            "geopolitical_volatility": 0.5,
        },
        "TUR": {
            "usd_sensitivity": 0.85,     # USD debt, weak lira
            "inflation_memory": 0.8,
            "commodity_exposure": 0.6,
            "geopolitical_volatility": 0.9,
            "erdogan_policy_risk": 0.9,  # Unorthodox monetary policy
        },
        "RUS": {
            "usd_sensitivity": 0.5,      # De-dollarizing
            "inflation_memory": 0.6,
            "oil_dependency": 0.85,      # Major oil/gas exporter
            "sanctions_impact": 0.7,
            "china_pivot": 0.6,
        },
        "JPN": {
            "usd_sensitivity": 0.4,
            "deflation_history": 0.8,    # Lost decades
            "yen_carry_trade": 0.7,
            "aging_population_drag": 0.8,
            "china_dependency": 0.5,
        },
        "CHN": {
            "usd_sensitivity": 0.5,      # Manages CNY vs USD
            "commodity_exposure": 0.8,   # Major importer
            "export_dependency": 0.6,
            "property_bubble_risk": 0.7,
            "us_trade_sensitivity": 0.6,
        },
        "DEU": {
            "usd_sensitivity": 0.3,      # Euro zone
            "china_dependency": 0.6,     # Export market
            "russia_energy": 0.4,        # Reduced after 2022
            "inflation_memory": 0.5,     # Weimar, but long ago
        },
        "GBR": {
            "usd_sensitivity": 0.4,
            "inflation_memory": 0.4,
            "brexit_sensitivity": 0.5,
            "financial_sector_exposure": 0.7,
        },
        "CHE": {
            "usd_sensitivity": 0.2,      # Safe haven
            "safe_haven_flows": 0.9,     # Receives crisis flows
            "euro_dependency": 0.6,
            "inflation_memory": 0.2,     # Very low
        },
        "SAU": {
            "oil_price_sensitivity": 0.95,  # Oil economy
            "usd_peg": 0.9,                 # SAR pegged to USD
            "geopolitical_volatility": 0.6,
        },
        "IND": {
            "usd_sensitivity": 0.5,
            "oil_import_dependency": 0.8,   # Major importer
            "china_rivalry": 0.6,
            "inflation_memory": 0.5,
        },
        "MEX": {
            "usd_sensitivity": 0.7,
            "us_trade_dependency": 0.8,
            "remittance_dependency": 0.4,
            "oil_exposure": 0.5,
        },
        "CAN": {
            "usd_sensitivity": 0.5,
            "us_trade_dependency": 0.7,
            "commodity_exposure": 0.6,
            "oil_exposure": 0.5,
        },
        "SWE": {
            "euro_sensitivity": 0.6,
            "export_dependency": 0.5,
            "geopolitical_volatility": 0.4,
        },
        "NOR": {
            "oil_price_sensitivity": 0.7,
            "euro_sensitivity": 0.4,
            "sovereign_wealth_buffer": 0.8,  # Oil fund reduces impact
        },
    }

    @classmethod
    def get_sensitivity(cls, country: str, factor: str) -> float:
        """Get sensitivity for a country and factor"""
        country_data = cls.SENSITIVITIES.get(country, {})
        return country_data.get(factor, 0.5)  # Default 0.5

    @classmethod
    def get_all_sensitivities(cls, country: str) -> Dict[str, float]:
        """Get all sensitivities for a country"""
        return cls.SENSITIVITIES.get(country, {})


class InfluenceCalculator:
    """
    Calculates how country A's economic events affect country B.

    Key mechanisms:
    1. Trade influence: Prices flow through trade channels
    2. Monetary influence: CB policy affects partner currencies
    3. Commodity influence: Producer price changes affect importers
    4. Financial influence: Capital flows and interest rate differentials
    5. Crisis contagion: Problems spread through connected markets
    """

    # Major commodity producers
    COMMODITY_PRODUCERS = {
        "oil": ["SAU", "RUS", "USA", "CAN", "BRA", "NOR", "ARE", "MEX"],
        "natural_gas": ["RUS", "USA", "NOR", "SAU"],
        "iron_ore": ["BRA", "CHN", "IND", "RUS"],
        "soybeans": ["BRA", "USA", "ARG"],
        "gold": ["CHN", "RUS", "USA", "CAN", "MEX"],
        "copper": ["CAN", "CHN", "RUS"],
        "wheat": ["RUS", "USA", "CAN", "FRA", "ARG"],
    }

    # Major commodity importers
    COMMODITY_IMPORTERS = {
        "oil": ["CHN", "IND", "JPN", "DEU", "FRA", "GBR", "IDN", "TUR"],
        "natural_gas": ["DEU", "JPN", "CHN", "GBR", "FRA"],
        "iron_ore": ["CHN", "JPN", "DEU"],
        "soybeans": ["CHN"],
        "copper": ["CHN", "DEU", "JPN"],
        "wheat": ["IDN", "TUR", "BRA", "JPN"],
    }

    def __init__(self):
        self._influence_cache: Dict[Tuple[str, str], List[InfluenceResult]] = {}

    def calculate_trade_influence(
        self,
        source: str,
        target: str,
        trade_volume: Decimal
    ) -> InfluenceResult:
        """
        Calculate trade-based influence.

        Larger trade volume = more price transmission.
        """
        # Normalize trade volume (max ~$800B USA-CHN)
        magnitude = min(1.0, float(trade_volume) / 500e9)

        # Trade always transmits prices (neutral direction)
        return InfluenceResult(
            source_country=source,
            target_country=target,
            influence_type=InfluenceType.TRADE,
            magnitude=magnitude,
            direction="neutral",
            transmission_channels=["import_prices", "export_demand"],
            lag_months=1,
        )

    def calculate_monetary_influence(
        self,
        source: str,
        target: str
    ) -> InfluenceResult:
        """
        Calculate monetary policy influence.

        USD (Fed) has strongest influence globally.
        EUR affects Europe. CNY affects EM commodity exporters.
        """
        # USD influence is global
        if source == "USA":
            target_usd_sensitivity = CountrySensitivities.get_sensitivity(
                target, "usd_sensitivity"
            )
            return InfluenceResult(
                source_country=source,
                target_country=target,
                influence_type=InfluenceType.MONETARY,
                magnitude=target_usd_sensitivity * 0.8,
                direction="negative",  # Fed tightening hurts EM
                transmission_channels=["interest_rate_differential", "capital_flows", "currency"],
                lag_months=2,
            )

        # EUR influence on Europe
        eurozone = {"DEU", "FRA"}
        if source in eurozone and target in {"SWE", "NOR", "CHE", "GBR"}:
            return InfluenceResult(
                source_country=source,
                target_country=target,
                influence_type=InfluenceType.MONETARY,
                magnitude=0.4,
                direction="neutral",
                transmission_channels=["ecb_policy", "euro_fx"],
                lag_months=1,
            )

        # CNY influence on trading partners
        if source == "CHN":
            china_dep = CountrySensitivities.get_sensitivity(target, "china_dependency")
            if china_dep > 0.3:
                return InfluenceResult(
                    source_country=source,
                    target_country=target,
                    influence_type=InfluenceType.MONETARY,
                    magnitude=china_dep * 0.5,
                    direction="neutral",
                    transmission_channels=["pboc_policy", "cny_fx", "trade_flows"],
                    lag_months=2,
                )

        # Default low influence
        return InfluenceResult(
            source_country=source,
            target_country=target,
            influence_type=InfluenceType.MONETARY,
            magnitude=0.1,
            direction="neutral",
            transmission_channels=["fx_spillover"],
            lag_months=3,
        )

    def calculate_commodity_influence(
        self,
        source: str,
        target: str
    ) -> Optional[InfluenceResult]:
        """
        Calculate commodity price influence.

        Major producers affect importers through price changes.
        """
        channels = []
        magnitude = 0.0

        for commodity, producers in self.COMMODITY_PRODUCERS.items():
            if source in producers:
                importers = self.COMMODITY_IMPORTERS.get(commodity, [])
                if target in importers:
                    # Source produces, target imports
                    channels.append(f"{commodity}_price")
                    magnitude += 0.2

        if not channels:
            return None

        return InfluenceResult(
            source_country=source,
            target_country=target,
            influence_type=InfluenceType.COMMODITY,
            magnitude=min(1.0, magnitude),
            direction="negative",  # Price increases hurt importers
            transmission_channels=channels,
            lag_months=1,
        )

    def calculate_crisis_contagion(
        self,
        source: str,
        target: str,
        trade_volume: Decimal
    ) -> InfluenceResult:
        """
        Calculate crisis contagion risk.

        Crises spread through:
        - Trade linkages
        - Financial linkages
        - Sentiment/panic
        """
        # Trade linkage
        trade_factor = min(0.5, float(trade_volume) / 200e9)

        # Same region increases contagion
        region_factor = 0.0
        latam = {"BRA", "ARG", "MEX"}
        europe = {"DEU", "FRA", "GBR", "SWE", "NOR", "CHE", "LIE"}
        asia = {"CHN", "JPN", "IND", "IDN"}
        mideast = {"SAU", "ARE", "TUR"}

        for region in [latam, europe, asia, mideast]:
            if source in region and target in region:
                region_factor = 0.3
                break

        # EM to EM contagion is high
        em_countries = {"BRA", "ARG", "MEX", "TUR", "IND", "IDN", "RUS"}
        if source in em_countries and target in em_countries:
            region_factor = max(region_factor, 0.4)

        magnitude = trade_factor + region_factor

        return InfluenceResult(
            source_country=source,
            target_country=target,
            influence_type=InfluenceType.CRISIS,
            magnitude=min(1.0, magnitude),
            direction="negative",
            transmission_channels=["panic", "capital_flight", "risk_off"],
            lag_months=0,  # Immediate
        )

    def calculate_all_influences(
        self,
        source: str,
        target: str,
        relationship: Optional[BilateralRelationship] = None
    ) -> List[InfluenceResult]:
        """
        Calculate all types of influence from source to target.

        Returns:
            List of InfluenceResults for different influence types
        """
        results = []

        trade_volume = relationship.trade_volume_usd if relationship else Decimal("10000000000")

        # Trade influence
        results.append(self.calculate_trade_influence(source, target, trade_volume))

        # Monetary influence
        results.append(self.calculate_monetary_influence(source, target))

        # Commodity influence (if applicable)
        commodity = self.calculate_commodity_influence(source, target)
        if commodity:
            results.append(commodity)

        # Crisis contagion potential
        results.append(self.calculate_crisis_contagion(source, target, trade_volume))

        return results

    def calculate_influence_matrix(
        self,
        countries: List[str],
        relationships: Dict[Tuple[str, str], BilateralRelationship]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate influence matrix for all country pairs.

        Returns:
            Dict[source][target] = total_influence (0-1)
        """
        matrix = {c: {t: 0.0 for t in countries} for c in countries}

        for source in countries:
            for target in countries:
                if source == target:
                    continue

                rel = relationships.get((source, target))
                influences = self.calculate_all_influences(source, target, rel)

                # Sum all influence types (weighted)
                total = sum(inf.magnitude for inf in influences) / len(influences)
                matrix[source][target] = total

        return matrix

    def get_most_influential_countries(
        self,
        countries: List[str],
        relationships: Dict[Tuple[str, str], BilateralRelationship]
    ) -> List[Tuple[str, float]]:
        """
        Get countries ranked by total outbound influence.

        USA should be #1 (USD hegemony).
        """
        matrix = self.calculate_influence_matrix(countries, relationships)

        totals = []
        for source in countries:
            total_influence = sum(matrix[source].values())
            totals.append((source, total_influence))

        return sorted(totals, key=lambda x: x[1], reverse=True)

    def get_most_influenced_countries(
        self,
        countries: List[str],
        relationships: Dict[Tuple[str, str], BilateralRelationship]
    ) -> List[Tuple[str, float]]:
        """
        Get countries ranked by total inbound influence (vulnerability).

        Small open economies and EM should be most vulnerable.
        """
        matrix = self.calculate_influence_matrix(countries, relationships)

        totals = []
        for target in countries:
            total_vulnerability = sum(matrix[source][target] for source in countries)
            totals.append((target, total_vulnerability))

        return sorted(totals, key=lambda x: x[1], reverse=True)

    def simulate_shock_propagation(
        self,
        source: str,
        shock_magnitude: float,
        countries: List[str],
        relationships: Dict[Tuple[str, str], BilateralRelationship]
    ) -> Dict[str, float]:
        """
        Simulate how a shock in one country propagates.

        Args:
            source: Country where shock originates
            shock_magnitude: Size of shock (0-1)
            countries: All countries
            relationships: Relationship data

        Returns:
            Dict[country] = impact (0-1)
        """
        matrix = self.calculate_influence_matrix(countries, relationships)
        impacts = {c: 0.0 for c in countries}
        impacts[source] = shock_magnitude

        # Propagate through 3 rounds (direct + 2 indirect)
        for round in range(3):
            new_impacts = impacts.copy()
            for affected, current_impact in impacts.items():
                if current_impact > 0.01:  # Only propagate significant impacts
                    for target in countries:
                        if target != affected:
                            transmission = matrix[affected][target] * current_impact * 0.5
                            new_impacts[target] = min(1.0, new_impacts[target] + transmission)
            impacts = new_impacts

        return impacts
