"""
War Probability Calculator

Calculates the probability of armed conflict between countries.

Austrian Theory Relevance:
- Wars destroy capital and wealth
- Wars increase government power (Randolph Bourne: "War is the health of the state")
- Trade interdependence reduces war probability (peace through commerce)
- Democracies fighting each other is rare (but they still start wars)
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

from src.agents.government import RegimeType
from src.countries.base import BilateralRelationship, RelationType


class WarTrigger(str, Enum):
    """Types of events that can trigger war"""

    TERRITORIAL_DISPUTE = "territorial_dispute"
    RESOURCE_COMPETITION = "resource_competition"
    IDEOLOGICAL_CONFLICT = "ideological_conflict"
    ECONOMIC_SANCTIONS = "economic_sanctions"
    PROXY_CONFLICT = "proxy_conflict"
    ARMS_RACE = "arms_race"
    ALLIANCE_OBLIGATION = "alliance_obligation"
    REGIME_CHANGE = "regime_change"
    NATIONALIST_INCIDENT = "nationalist_incident"
    CYBER_ATTACK = "cyber_attack"


class WarType(str, Enum):
    """Types of conflicts"""

    CONVENTIONAL = "conventional"  # Traditional military
    PROXY = "proxy"  # Through third parties
    ECONOMIC = "economic"  # Trade war, sanctions
    CYBER = "cyber"  # Cyber warfare
    LIMITED = "limited"  # Limited strikes
    TOTAL = "total"  # Full-scale war
    NUCLEAR = "nuclear"  # Nuclear exchange


@dataclass
class WarRiskAssessment:
    """Assessment of war risk between two countries"""

    country_a: str
    country_b: str
    probability: float  # 0-1
    risk_level: str  # "low", "moderate", "high", "critical"
    primary_triggers: list[WarTrigger]
    most_likely_type: WarType
    escalation_factors: list[str]
    de_escalation_factors: list[str]
    economic_impact_estimate: Decimal  # Estimated damage in USD
    nuclear_risk: bool


class WarProbabilityCalculator:
    """
    Calculates probability of war between countries.

    Factors that INCREASE probability:
    - Historical conflicts
    - Territorial disputes
    - Ideological differences (regime type)
    - Active sanctions
    - Arms race / military buildup
    - Nationalist governments
    - Resource competition

    Factors that DECREASE probability:
    - Trade interdependence (peace through commerce)
    - Alliance with nuclear powers
    - Mutual Assured Destruction (MAD)
    - International organization membership
    - Democratic peace theory (democracies rarely fight each other)
    - Economic cost of war
    """

    # Known territorial disputes
    TERRITORIAL_DISPUTES = {
        ("CHN", "JPN"): ["senkaku_islands"],
        ("CHN", "IND"): ["aksai_chin", "arunachal_pradesh"],
        ("RUS", "JPN"): ["kuril_islands"],
        ("ARG", "GBR"): ["falklands_malvinas"],
        ("RUS", "NOR"): ["svalbard", "arctic"],
        ("CHN", "IDN"): ["south_china_sea"],
    }

    # Countries with nuclear weapons
    NUCLEAR_POWERS = {"USA", "RUS", "CHN", "GBR", "FRA", "IND"}

    # NATO members (simplified)
    NATO_MEMBERS = {"USA", "CAN", "GBR", "DEU", "FRA", "NOR", "TUR"}

    # Country regimes for ideological conflict calculation
    COUNTRY_REGIMES = {
        "USA": RegimeType.DEMOCRACY_LIBERAL,
        "CAN": RegimeType.DEMOCRACY_SOCIALIST,
        "MEX": RegimeType.DEMOCRACY_SOCIALIST,
        "BRA": RegimeType.DEMOCRACY_SOCIALIST,
        "ARG": RegimeType.DEMOCRACY_SOCIALIST,
        "GBR": RegimeType.DEMOCRACY_SOCIALIST,
        "DEU": RegimeType.DEMOCRACY_SOCIALIST,
        "FRA": RegimeType.DEMOCRACY_SOCIALIST,
        "SWE": RegimeType.DEMOCRACY_SOCIALIST,
        "NOR": RegimeType.DEMOCRACY_SOCIALIST,
        "CHE": RegimeType.MINARCHY,
        "LIE": RegimeType.MONARCHY,
        "CHN": RegimeType.TOTALITARIAN,
        "JPN": RegimeType.DEMOCRACY_LIBERAL,
        "IND": RegimeType.DEMOCRACY_SOCIALIST,
        "IDN": RegimeType.DEMOCRACY_SOCIALIST,
        "ARE": RegimeType.MONARCHY,
        "SAU": RegimeType.MONARCHY,
        "RUS": RegimeType.TOTALITARIAN,
        "TUR": RegimeType.TOTALITARIAN,
    }

    def __init__(self):
        self._assessments: dict[tuple[str, str], WarRiskAssessment] = {}

    def calculate_base_probability(self, relationship: BilateralRelationship) -> float:
        """
        Calculate base war probability from relationship type.

        Returns:
            Base probability (0-1)
        """
        base_probs = {
            RelationType.ALLY: 0.001,  # Very unlikely
            RelationType.NEUTRAL: 0.01,  # Unlikely
            RelationType.RIVAL: 0.05,  # Possible
            RelationType.ENEMY: 0.15,  # Significant risk
        }
        return base_probs.get(relationship.relationship_type, 0.01)

    def calculate_historical_factor(self, relationship: BilateralRelationship) -> float:
        """
        Factor based on historical conflicts.

        Past conflicts increase future war probability.
        """
        conflicts = relationship.historical_conflicts
        return min(0.1, conflicts * 0.03)  # Each past conflict adds 3%

    def calculate_tension_factor(self, relationship: BilateralRelationship) -> float:
        """
        Factor based on current tensions.

        Active tensions increase probability.
        """
        num_tensions = len(relationship.current_tensions)
        return min(0.15, num_tensions * 0.03)  # Each tension adds 3%

    def calculate_sanction_factor(self, relationship: BilateralRelationship) -> float:
        """
        Factor based on active sanctions.

        Sanctions increase hostility and war risk.
        """
        if not relationship.has_active_sanctions:
            return 0.0

        num_sanctions = len(relationship.sanctions_a_on_b) + len(
            relationship.sanctions_b_on_a
        )
        return min(0.1, num_sanctions * 0.02)  # Each sanction adds 2%

    def calculate_trade_dampening(self, relationship: BilateralRelationship) -> float:
        """
        Trade interdependence reduces war probability.

        Austrian insight: "If goods don't cross borders, soldiers will."
        - Frédéric Bastiat
        """
        trade = float(relationship.trade_volume_usd)

        # High trade = low war probability
        if trade > 500e9:  # $500B+
            return 0.6  # Reduce probability by 60%
        elif trade > 100e9:  # $100B+
            return 0.4  # Reduce by 40%
        elif trade > 10e9:  # $10B+
            return 0.2  # Reduce by 20%
        else:
            return 0.0  # No dampening

    def calculate_nuclear_dampening(self, country_a: str, country_b: str) -> float:
        """
        Nuclear deterrence (MAD) reduces war probability.

        If both countries have nukes, war is much less likely.
        """
        a_nuclear = country_a in self.NUCLEAR_POWERS
        b_nuclear = country_b in self.NUCLEAR_POWERS

        if a_nuclear and b_nuclear:
            return 0.7  # Reduce by 70% (MAD)
        elif a_nuclear or b_nuclear:
            return 0.3  # Reduce by 30% (one-sided deterrence)
        return 0.0

    def calculate_alliance_factor(self, country_a: str, country_b: str) -> float:
        """
        Alliance membership affects probability.

        NATO members are unlikely to fight each other.
        Attacking NATO member risks Article 5.
        """
        a_nato = country_a in self.NATO_MEMBERS
        b_nato = country_b in self.NATO_MEMBERS

        if a_nato and b_nato:
            return -0.1  # Same alliance, very unlikely
        elif a_nato or b_nato:
            return 0.02  # Attacking alliance risks escalation
        return 0.0

    def calculate_ideological_factor(self, country_a: str, country_b: str) -> float:
        """
        Ideological differences increase conflict probability.

        Democratic peace theory: democracies rarely fight each other.
        But totalitarians vs democracies have higher conflict risk.
        """
        regime_a = self.COUNTRY_REGIMES.get(country_a, RegimeType.DEMOCRACY_LIBERAL)
        regime_b = self.COUNTRY_REGIMES.get(country_b, RegimeType.DEMOCRACY_LIBERAL)

        # Both democracies = low conflict
        democracies = {RegimeType.DEMOCRACY_LIBERAL, RegimeType.DEMOCRACY_SOCIALIST}
        if regime_a in democracies and regime_b in democracies:
            return -0.03  # Democracies rarely fight

        # Totalitarian vs democracy = higher conflict
        if regime_a == RegimeType.TOTALITARIAN or regime_b == RegimeType.TOTALITARIAN:
            if regime_a != regime_b:
                return 0.05  # Ideological conflict

        return 0.0

    def calculate_territorial_factor(
        self, country_a: str, country_b: str
    ) -> tuple[float, list[str]]:
        """
        Territorial disputes significantly increase war probability.

        Returns:
            Tuple of (probability_factor, list_of_disputes)
        """
        pair = tuple(sorted([country_a, country_b]))
        disputes = self.TERRITORIAL_DISPUTES.get(pair, [])

        if not disputes:
            return 0.0, []

        factor = min(0.15, len(disputes) * 0.05)  # Each dispute adds 5%
        return factor, disputes

    def calculate_war_probability(
        self, relationship: BilateralRelationship
    ) -> WarRiskAssessment:
        """
        Calculate comprehensive war probability assessment.

        Returns:
            WarRiskAssessment with probability and analysis
        """
        country_a = relationship.country_a
        country_b = relationship.country_b

        # Start with base probability
        base = self.calculate_base_probability(relationship)

        # Add factors that increase probability
        historical = self.calculate_historical_factor(relationship)
        tensions = self.calculate_tension_factor(relationship)
        sanctions = self.calculate_sanction_factor(relationship)
        territorial, disputes = self.calculate_territorial_factor(country_a, country_b)
        ideological = self.calculate_ideological_factor(country_a, country_b)
        alliance = self.calculate_alliance_factor(country_a, country_b)

        # Calculate raw probability
        raw_prob = (
            base
            + historical
            + tensions
            + sanctions
            + territorial
            + ideological
            + alliance
        )

        # Apply dampening factors
        trade_damp = self.calculate_trade_dampening(relationship)
        nuclear_damp = self.calculate_nuclear_dampening(country_a, country_b)

        # Final probability with dampening
        dampened_prob = raw_prob * (1 - trade_damp) * (1 - nuclear_damp)

        # Add existing escalation
        final_prob = min(0.5, dampened_prob + relationship.escalation_level * 0.2)

        # Determine risk level
        if final_prob < 0.02:
            risk_level = "low"
        elif final_prob < 0.05:
            risk_level = "moderate"
        elif final_prob < 0.10:
            risk_level = "high"
        else:
            risk_level = "critical"

        # Determine triggers
        triggers = []
        if disputes:
            triggers.append(WarTrigger.TERRITORIAL_DISPUTE)
        if relationship.has_active_sanctions:
            triggers.append(WarTrigger.ECONOMIC_SANCTIONS)
        if ideological > 0:
            triggers.append(WarTrigger.IDEOLOGICAL_CONFLICT)
        if historical > 0:
            triggers.append(WarTrigger.NATIONALIST_INCIDENT)
        if not triggers:
            triggers.append(WarTrigger.PROXY_CONFLICT)

        # Determine most likely war type
        if country_a in self.NUCLEAR_POWERS and country_b in self.NUCLEAR_POWERS:
            if final_prob > 0.1:
                war_type = WarType.NUCLEAR  # Unlikely but catastrophic
            else:
                war_type = WarType.PROXY  # More likely to be proxy
        elif trade_damp > 0.3:
            war_type = WarType.ECONOMIC  # Trade war more likely
        else:
            war_type = WarType.CONVENTIONAL

        # Escalation factors
        escalation_factors = []
        if tensions > 0:
            escalation_factors.extend(relationship.current_tensions)
        if sanctions > 0:
            escalation_factors.append("active_sanctions")
        if territorial > 0:
            escalation_factors.extend(disputes)

        # De-escalation factors
        de_escalation_factors = []
        if trade_damp > 0:
            de_escalation_factors.append(
                f"trade_interdependence_${float(relationship.trade_volume_usd) / 1e9:.0f}B"
            )
        if nuclear_damp > 0:
            de_escalation_factors.append("nuclear_deterrence")
        if country_a in self.NATO_MEMBERS or country_b in self.NATO_MEMBERS:
            de_escalation_factors.append("alliance_obligations")

        # Estimate economic impact (rough)
        combined_gdp = float(relationship.trade_volume_usd) * 20  # Very rough estimate
        impact = Decimal(str(combined_gdp * final_prob))

        assessment = WarRiskAssessment(
            country_a=country_a,
            country_b=country_b,
            probability=final_prob,
            risk_level=risk_level,
            primary_triggers=triggers,
            most_likely_type=war_type,
            escalation_factors=escalation_factors,
            de_escalation_factors=de_escalation_factors,
            economic_impact_estimate=impact,
            nuclear_risk=country_a in self.NUCLEAR_POWERS
            and country_b in self.NUCLEAR_POWERS,
        )

        self._assessments[(country_a, country_b)] = assessment
        return assessment

    def get_all_high_risk_pairs(
        self,
        relationships: dict[tuple[str, str], BilateralRelationship],
        threshold: float = 0.03,
    ) -> list[WarRiskAssessment]:
        """
        Get all country pairs above a war probability threshold.

        Returns:
            List of WarRiskAssessments sorted by probability (descending)
        """
        high_risk = []
        seen = set()

        for (a, b), rel in relationships.items():
            pair = tuple(sorted([a, b]))
            if pair in seen:
                continue
            seen.add(pair)

            assessment = self.calculate_war_probability(rel)
            if assessment.probability >= threshold:
                high_risk.append(assessment)

        return sorted(high_risk, key=lambda x: x.probability, reverse=True)

    def simulate_escalation_event(
        self,
        relationship: BilateralRelationship,
        event_type: WarTrigger,
        severity: float = 0.1,
    ) -> WarRiskAssessment:
        """
        Simulate an escalation event and recalculate probability.

        Args:
            relationship: The bilateral relationship
            event_type: Type of escalation event
            severity: How severe (0-1)

        Returns:
            Updated WarRiskAssessment
        """
        # Add tension
        relationship.current_tensions.append(event_type.value)
        relationship.escalation_level = min(
            1.0, relationship.escalation_level + severity
        )

        # Recalculate
        return self.calculate_war_probability(relationship)

    def get_global_war_risk(
        self, relationships: dict[tuple[str, str], BilateralRelationship]
    ) -> dict[str, Any]:
        """
        Calculate global war risk metrics.

        Returns:
            Dict with global risk indicators
        """
        all_assessments = []
        seen = set()

        for (a, b), rel in relationships.items():
            pair = tuple(sorted([a, b]))
            if pair in seen:
                continue
            seen.add(pair)
            all_assessments.append(self.calculate_war_probability(rel))

        if not all_assessments:
            return {"error": "No relationships to analyze"}

        probs = [a.probability for a in all_assessments]
        avg_prob = sum(probs) / len(probs)
        max_prob = max(probs)

        critical = [a for a in all_assessments if a.risk_level == "critical"]
        high = [a for a in all_assessments if a.risk_level == "high"]
        nuclear = [
            a for a in all_assessments if a.nuclear_risk and a.probability > 0.05
        ]

        return {
            "average_war_probability": avg_prob,
            "maximum_war_probability": max_prob,
            "critical_risk_pairs": len(critical),
            "high_risk_pairs": len(high),
            "nuclear_risk_pairs": len(nuclear),
            "total_pairs_analyzed": len(all_assessments),
            "global_risk_level": "critical"
            if critical
            else ("high" if high else "moderate"),
            "most_dangerous_pair": max(all_assessments, key=lambda x: x.probability)
            if all_assessments
            else None,
        }


# Convenience functions
def calculate_war_probability(relationship: BilateralRelationship) -> float:
    """Calculate war probability for a relationship"""
    calc = WarProbabilityCalculator()
    assessment = calc.calculate_war_probability(relationship)
    return assessment.probability


def get_war_risk_assessment(relationship: BilateralRelationship) -> WarRiskAssessment:
    """Get full war risk assessment"""
    calc = WarProbabilityCalculator()
    return calc.calculate_war_probability(relationship)
