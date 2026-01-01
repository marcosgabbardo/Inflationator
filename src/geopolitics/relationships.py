"""
Bilateral Relationships Module

Defines relationships between countries for multi-country simulation.

Austrian Theory Relevance:
- Trade interdependence reduces conflict (peace through commerce)
- Sanctions distort markets and harm both parties
- Wars destroy capital and increase government power
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from src.countries.base import RelationType, BilateralRelationship


class SanctionType(str, Enum):
    """Types of economic sanctions"""
    TRADE_EMBARGO = "trade_embargo"         # Full trade ban
    ASSET_FREEZE = "asset_freeze"           # Freeze foreign assets
    FINANCIAL_BAN = "financial_ban"         # SWIFT/banking exclusion
    ARMS_EMBARGO = "arms_embargo"           # Military trade ban
    TRAVEL_BAN = "travel_ban"               # Visa restrictions
    SECTOR_TARGETED = "sector_targeted"     # Specific sector sanctions
    SECONDARY = "secondary"                 # Sanctions on third parties


# Initial relationships between all 20 countries
# Format: (country_a, country_b): {relationship_data}
INITIAL_RELATIONSHIPS: Dict[Tuple[str, str], Dict[str, Any]] = {
    # ============================================
    # USA RELATIONSHIPS
    # ============================================
    ("USA", "CHN"): {
        "type": RelationType.RIVAL,
        "strength": -0.3,
        "trade_volume": Decimal("700000000000"),  # $700B
        "tensions": ["taiwan", "trade_war", "tech_war", "south_china_sea"],
        "war_prob": 0.08,
    },
    ("USA", "RUS"): {
        "type": RelationType.ENEMY,
        "strength": -0.7,
        "trade_volume": Decimal("30000000000"),  # $30B
        "tensions": ["ukraine", "nato_expansion", "nuclear_arms"],
        "sanctions_a_on_b": [SanctionType.FINANCIAL_BAN, SanctionType.ASSET_FREEZE, SanctionType.SECTOR_TARGETED],
        "war_prob": 0.12,
    },
    ("USA", "GBR"): {
        "type": RelationType.ALLY,
        "strength": 0.85,
        "trade_volume": Decimal("150000000000"),
        "war_prob": 0.0,
    },
    ("USA", "CAN"): {
        "type": RelationType.ALLY,
        "strength": 0.9,
        "trade_volume": Decimal("750000000000"),
        "war_prob": 0.0,
    },
    ("USA", "MEX"): {
        "type": RelationType.ALLY,
        "strength": 0.6,
        "trade_volume": Decimal("800000000000"),
        "war_prob": 0.0,
    },
    ("USA", "JPN"): {
        "type": RelationType.ALLY,
        "strength": 0.75,
        "trade_volume": Decimal("250000000000"),
        "war_prob": 0.0,
    },
    ("USA", "DEU"): {
        "type": RelationType.ALLY,
        "strength": 0.65,
        "trade_volume": Decimal("200000000000"),
        "war_prob": 0.0,
    },
    ("USA", "BRA"): {
        "type": RelationType.NEUTRAL,
        "strength": 0.2,
        "trade_volume": Decimal("80000000000"),
        "war_prob": 0.0,
    },
    ("USA", "IND"): {
        "type": RelationType.ALLY,
        "strength": 0.4,
        "trade_volume": Decimal("150000000000"),
        "tensions": ["trade_deficit"],
        "war_prob": 0.0,
    },
    ("USA", "CHE"): {
        "type": RelationType.ALLY,
        "strength": 0.5,
        "trade_volume": Decimal("80000000000"),
        "war_prob": 0.0,
    },
    ("USA", "SAU"): {
        "type": RelationType.ALLY,
        "strength": 0.5,
        "trade_volume": Decimal("40000000000"),
        "tensions": ["human_rights", "oil_policy"],
        "war_prob": 0.0,
    },
    ("USA", "TUR"): {
        "type": RelationType.NEUTRAL,
        "strength": 0.1,
        "trade_volume": Decimal("30000000000"),
        "tensions": ["s400_purchase", "kurdish_policy"],
        "war_prob": 0.01,
    },

    # ============================================
    # CHINA RELATIONSHIPS
    # ============================================
    ("CHN", "RUS"): {
        "type": RelationType.ALLY,
        "strength": 0.5,
        "trade_volume": Decimal("200000000000"),
        "war_prob": 0.01,
    },
    ("CHN", "BRA"): {
        "type": RelationType.ALLY,
        "strength": 0.45,
        "trade_volume": Decimal("150000000000"),  # Largest trade partner
        "war_prob": 0.0,
    },
    ("CHN", "SAU"): {
        "type": RelationType.ALLY,
        "strength": 0.35,
        "trade_volume": Decimal("80000000000"),
        "war_prob": 0.0,
    },
    ("CHN", "IND"): {
        "type": RelationType.RIVAL,
        "strength": -0.25,
        "trade_volume": Decimal("120000000000"),
        "tensions": ["border_disputes", "tibet", "pakistan_alliance"],
        "war_prob": 0.06,
    },
    ("CHN", "JPN"): {
        "type": RelationType.RIVAL,
        "strength": -0.3,
        "trade_volume": Decimal("350000000000"),
        "tensions": ["senkaku_islands", "historical_grievances", "taiwan"],
        "war_prob": 0.04,
    },
    ("CHN", "ARE"): {
        "type": RelationType.ALLY,
        "strength": 0.3,
        "trade_volume": Decimal("60000000000"),
        "war_prob": 0.0,
    },
    ("CHN", "IDN"): {
        "type": RelationType.NEUTRAL,
        "strength": 0.2,
        "trade_volume": Decimal("100000000000"),
        "tensions": ["south_china_sea"],
        "war_prob": 0.02,
    },

    # ============================================
    # BRAZIL-ARGENTINA (RIVALS per user request)
    # ============================================
    ("BRA", "ARG"): {
        "type": RelationType.RIVAL,
        "strength": -0.2,
        "trade_volume": Decimal("25000000000"),
        "tensions": ["mercosur_disputes", "currency_competition", "regional_leadership"],
        "war_prob": 0.02,
    },

    # ============================================
    # EUROPE RELATIONSHIPS
    # ============================================
    ("DEU", "FRA"): {
        "type": RelationType.ALLY,
        "strength": 0.8,
        "trade_volume": Decimal("180000000000"),
        "war_prob": 0.0,
    },
    ("DEU", "RUS"): {
        "type": RelationType.NEUTRAL,
        "strength": -0.35,
        "trade_volume": Decimal("40000000000"),
        "tensions": ["energy_dependency", "ukraine", "nato"],
        "sanctions_a_on_b": [SanctionType.SECTOR_TARGETED],
        "war_prob": 0.02,
    },
    ("GBR", "FRA"): {
        "type": RelationType.ALLY,
        "strength": 0.6,
        "trade_volume": Decimal("90000000000"),
        "tensions": ["brexit", "fishing_rights"],
        "war_prob": 0.0,
    },
    ("SWE", "NOR"): {
        "type": RelationType.ALLY,
        "strength": 0.9,
        "trade_volume": Decimal("30000000000"),
        "war_prob": 0.0,
    },
    ("SWE", "RUS"): {
        "type": RelationType.RIVAL,
        "strength": -0.35,
        "trade_volume": Decimal("5000000000"),
        "tensions": ["baltic_security", "nato_membership", "gotland"],
        "war_prob": 0.04,
    },
    ("CHE", "LIE"): {
        "type": RelationType.ALLY,
        "strength": 0.95,
        "trade_volume": Decimal("5000000000"),
        "war_prob": 0.0,
    },
    ("CHE", "DEU"): {
        "type": RelationType.ALLY,
        "strength": 0.7,
        "trade_volume": Decimal("100000000000"),
        "war_prob": 0.0,
    },

    # ============================================
    # MIDDLE EAST
    # ============================================
    ("SAU", "ARE"): {
        "type": RelationType.ALLY,
        "strength": 0.7,
        "trade_volume": Decimal("20000000000"),
        "war_prob": 0.0,
    },
    ("SAU", "TUR"): {
        "type": RelationType.RIVAL,
        "strength": -0.25,
        "trade_volume": Decimal("10000000000"),
        "tensions": ["regional_hegemony", "muslim_brotherhood", "qatar"],
        "war_prob": 0.03,
    },

    # ============================================
    # RUSSIA CONFLICTS
    # ============================================
    ("RUS", "TUR"): {
        "type": RelationType.RIVAL,
        "strength": -0.15,
        "trade_volume": Decimal("50000000000"),
        "tensions": ["syria", "caucasus", "black_sea", "libya"],
        "war_prob": 0.05,
    },
    ("RUS", "JPN"): {
        "type": RelationType.RIVAL,
        "strength": -0.2,
        "trade_volume": Decimal("20000000000"),
        "tensions": ["kuril_islands", "us_alliance"],
        "war_prob": 0.02,
    },
    ("RUS", "NOR"): {
        "type": RelationType.RIVAL,
        "strength": -0.3,
        "trade_volume": Decimal("3000000000"),
        "tensions": ["arctic", "nato", "svalbard"],
        "war_prob": 0.03,
    },

    # ============================================
    # LATIN AMERICA
    # ============================================
    ("MEX", "CAN"): {
        "type": RelationType.ALLY,
        "strength": 0.5,
        "trade_volume": Decimal("40000000000"),
        "war_prob": 0.0,
    },
    ("ARG", "GBR"): {
        "type": RelationType.RIVAL,
        "strength": -0.4,
        "trade_volume": Decimal("2000000000"),
        "tensions": ["falklands_malvinas"],
        "historical_conflicts": 1,  # 1982 war
        "war_prob": 0.01,
    },

    # ============================================
    # ASIA
    # ============================================
    ("JPN", "IND"): {
        "type": RelationType.ALLY,
        "strength": 0.4,
        "trade_volume": Decimal("20000000000"),
        "war_prob": 0.0,
    },
    ("IND", "IDN"): {
        "type": RelationType.NEUTRAL,
        "strength": 0.2,
        "trade_volume": Decimal("25000000000"),
        "war_prob": 0.0,
    },
}


class RelationshipManager:
    """
    Manages all bilateral relationships between countries.

    Provides methods to:
    - Query relationships
    - Calculate trade impacts
    - Apply sanctions effects
    - Track escalation
    """

    def __init__(self):
        self._relationships: Dict[Tuple[str, str], BilateralRelationship] = {}
        self._initialize_relationships()

    def _initialize_relationships(self):
        """Initialize all relationships from INITIAL_RELATIONSHIPS"""
        for (country_a, country_b), data in INITIAL_RELATIONSHIPS.items():
            rel = BilateralRelationship(
                country_a=country_a,
                country_b=country_b,
                relationship_type=data.get("type", RelationType.NEUTRAL),
                strength=data.get("strength", 0.0),
                trade_volume_usd=data.get("trade_volume", Decimal("0")),
                tariff_a_to_b=data.get("tariff_a_to_b", 0.0),
                tariff_b_to_a=data.get("tariff_b_to_a", 0.0),
                sanctions_a_on_b=[s.value for s in data.get("sanctions_a_on_b", [])],
                sanctions_b_on_a=[s.value for s in data.get("sanctions_b_on_a", [])],
                historical_conflicts=data.get("historical_conflicts", 0),
                current_tensions=data.get("tensions", []),
                war_probability=data.get("war_prob", 0.0),
            )
            # Store both directions
            self._relationships[(country_a, country_b)] = rel
            self._relationships[(country_b, country_a)] = self._create_inverse(rel)

    def _create_inverse(self, rel: BilateralRelationship) -> BilateralRelationship:
        """Create inverse relationship (B→A from A→B)"""
        return BilateralRelationship(
            country_a=rel.country_b,
            country_b=rel.country_a,
            relationship_type=rel.relationship_type,
            strength=rel.strength,
            trade_volume_usd=rel.trade_volume_usd,
            trade_balance=-rel.trade_balance,
            tariff_a_to_b=rel.tariff_b_to_a,
            tariff_b_to_a=rel.tariff_a_to_b,
            sanctions_a_on_b=rel.sanctions_b_on_a,
            sanctions_b_on_a=rel.sanctions_a_on_b,
            historical_conflicts=rel.historical_conflicts,
            current_tensions=rel.current_tensions,
            a_dependency_on_b=rel.b_dependency_on_a,
            b_dependency_on_a=rel.a_dependency_on_b,
            war_probability=rel.war_probability,
            war_triggers=rel.war_triggers,
            escalation_level=rel.escalation_level,
        )

    def get_relationship(
        self,
        country_a: str,
        country_b: str
    ) -> Optional[BilateralRelationship]:
        """Get relationship between two countries"""
        return self._relationships.get((country_a, country_b))

    def get_all_relationships_for(
        self,
        country: str
    ) -> List[BilateralRelationship]:
        """Get all relationships for a country"""
        return [
            rel for (a, b), rel in self._relationships.items()
            if a == country
        ]

    def get_allies(self, country: str) -> List[str]:
        """Get list of allied countries"""
        return [
            rel.country_b for rel in self.get_all_relationships_for(country)
            if rel.relationship_type == RelationType.ALLY
        ]

    def get_enemies(self, country: str) -> List[str]:
        """Get list of enemy countries"""
        return [
            rel.country_b for rel in self.get_all_relationships_for(country)
            if rel.relationship_type == RelationType.ENEMY
        ]

    def get_rivals(self, country: str) -> List[str]:
        """Get list of rival countries"""
        return [
            rel.country_b for rel in self.get_all_relationships_for(country)
            if rel.relationship_type == RelationType.RIVAL
        ]

    def get_trade_partners(
        self,
        country: str,
        min_volume: Decimal = Decimal("10000000000")  # $10B
    ) -> List[Tuple[str, Decimal]]:
        """Get major trade partners (sorted by volume)"""
        partners = []
        for rel in self.get_all_relationships_for(country):
            if rel.trade_volume_usd >= min_volume:
                partners.append((rel.country_b, rel.trade_volume_usd))
        return sorted(partners, key=lambda x: x[1], reverse=True)

    def get_high_war_risk_pairs(
        self,
        threshold: float = 0.03
    ) -> List[Tuple[str, str, float]]:
        """Get country pairs with high war probability"""
        seen = set()
        high_risk = []

        for (a, b), rel in self._relationships.items():
            pair = tuple(sorted([a, b]))
            if pair not in seen and rel.war_probability >= threshold:
                seen.add(pair)
                high_risk.append((a, b, rel.war_probability))

        return sorted(high_risk, key=lambda x: x[2], reverse=True)

    def get_sanctioned_pairs(self) -> List[Tuple[str, str, List[str]]]:
        """Get all country pairs with active sanctions"""
        sanctioned = []
        seen = set()

        for (a, b), rel in self._relationships.items():
            pair = tuple(sorted([a, b]))
            if pair not in seen and rel.has_active_sanctions:
                seen.add(pair)
                all_sanctions = rel.sanctions_a_on_b + rel.sanctions_b_on_a
                sanctioned.append((a, b, all_sanctions))

        return sanctioned

    def apply_tariff(
        self,
        imposer: str,
        target: str,
        rate: float
    ):
        """Apply tariff from one country to another"""
        rel = self._relationships.get((imposer, target))
        if rel:
            rel.tariff_a_to_b = rate
            # Update inverse
            inv = self._relationships.get((target, imposer))
            if inv:
                inv.tariff_b_to_a = rate

    def apply_sanction(
        self,
        imposer: str,
        target: str,
        sanction_type: SanctionType
    ):
        """Apply sanction from one country to another"""
        rel = self._relationships.get((imposer, target))
        if rel:
            if sanction_type.value not in rel.sanctions_a_on_b:
                rel.sanctions_a_on_b.append(sanction_type.value)
            # Increase war probability
            rel.war_probability = min(0.5, rel.war_probability + 0.02)
            rel.escalation_level = min(1.0, rel.escalation_level + 0.1)
            # Update inverse
            inv = self._relationships.get((target, imposer))
            if inv:
                if sanction_type.value not in inv.sanctions_b_on_a:
                    inv.sanctions_b_on_a.append(sanction_type.value)
                inv.war_probability = rel.war_probability
                inv.escalation_level = rel.escalation_level

    def escalate_tension(
        self,
        country_a: str,
        country_b: str,
        tension: str,
        amount: float = 0.1
    ):
        """Escalate tension between two countries"""
        rel = self._relationships.get((country_a, country_b))
        if rel:
            if tension not in rel.current_tensions:
                rel.current_tensions.append(tension)
            rel.escalation_level = min(1.0, rel.escalation_level + amount)
            rel.war_probability = min(0.5, rel.war_probability + amount * 0.5)
            # Worsen relationship
            rel.strength = max(-1.0, rel.strength - amount * 0.2)
            # Update inverse
            inv = self._relationships.get((country_b, country_a))
            if inv:
                inv.current_tensions = rel.current_tensions
                inv.escalation_level = rel.escalation_level
                inv.war_probability = rel.war_probability
                inv.strength = rel.strength

    def de_escalate(
        self,
        country_a: str,
        country_b: str,
        amount: float = 0.1
    ):
        """De-escalate tension between two countries"""
        rel = self._relationships.get((country_a, country_b))
        if rel:
            rel.escalation_level = max(0.0, rel.escalation_level - amount)
            rel.war_probability = max(0.0, rel.war_probability - amount * 0.3)
            # Improve relationship slightly
            rel.strength = min(1.0, rel.strength + amount * 0.1)
            # Update inverse
            inv = self._relationships.get((country_b, country_a))
            if inv:
                inv.escalation_level = rel.escalation_level
                inv.war_probability = rel.war_probability
                inv.strength = rel.strength

    def calculate_trade_impact(
        self,
        country: str
    ) -> Dict[str, Any]:
        """
        Calculate total trade impact for a country.

        Returns:
            Dict with trade volumes, tariff costs, sanction effects
        """
        total_trade = Decimal("0")
        total_tariff_paid = Decimal("0")
        sanctioned_trade = Decimal("0")
        blocked_trade = Decimal("0")

        for rel in self.get_all_relationships_for(country):
            total_trade += rel.trade_volume_usd

            # Tariff costs
            tariff_cost = rel.trade_volume_usd * Decimal(str(rel.tariff_b_to_a)) / 2
            total_tariff_paid += tariff_cost

            # Sanction effects
            if rel.has_active_sanctions:
                sanctioned_trade += rel.trade_volume_usd
                if SanctionType.TRADE_EMBARGO.value in rel.sanctions_b_on_a:
                    blocked_trade += rel.trade_volume_usd

        return {
            "total_trade_volume": total_trade,
            "total_tariff_cost": total_tariff_paid,
            "sanctioned_trade_volume": sanctioned_trade,
            "blocked_trade_volume": blocked_trade,
            "trade_freedom": float(1 - (blocked_trade / total_trade)) if total_trade > 0 else 1.0,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all relationships"""
        unique_pairs = set()
        allies = 0
        rivals = 0
        enemies = 0
        neutral = 0
        sanctioned = 0
        high_risk = 0

        for (a, b), rel in self._relationships.items():
            pair = tuple(sorted([a, b]))
            if pair in unique_pairs:
                continue
            unique_pairs.add(pair)

            if rel.relationship_type == RelationType.ALLY:
                allies += 1
            elif rel.relationship_type == RelationType.RIVAL:
                rivals += 1
            elif rel.relationship_type == RelationType.ENEMY:
                enemies += 1
            else:
                neutral += 1

            if rel.has_active_sanctions:
                sanctioned += 1
            if rel.war_probability >= 0.03:
                high_risk += 1

        return {
            "total_relationships": len(unique_pairs),
            "allies": allies,
            "rivals": rivals,
            "enemies": enemies,
            "neutral": neutral,
            "sanctioned_pairs": sanctioned,
            "high_war_risk_pairs": high_risk,
        }
