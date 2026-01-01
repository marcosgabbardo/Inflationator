"""
Government Agent - THE POLITICAL VILLAIN

The government causes economic distortions through:
1. Taxation (extraction from productive economy)
2. Regulation (barriers to entry, compliance costs)
3. Spending (crowding out private investment)
4. Wars (destruction of capital)
5. Trade restrictions (tariffs, sanctions)

Austrian/Libertarian Theory (Hoppe, Rothbard):
- Government is a monopoly on violence
- Taxation is theft
- Regulation creates deadweight loss
- Democracy is worse than monarchy (higher time preference)
- Ancap is the ideal (no government)
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Any, List, Optional, Set
from enum import Enum
import random

from .base import Agent, AgentType, AgentState


class RegimeType(str, Enum):
    """
    Political regime types ordered from WORST to BEST
    according to Hoppe's analysis.
    """
    TOTALITARIAN = "totalitarian"      # 100% - Worst
    DEMOCRACY_SOCIALIST = "democracy_socialist"  # 80%
    DEMOCRACY_LIBERAL = "democracy_liberal"      # 50%
    MONARCHY = "monarchy"              # 30%
    MINARCHY = "minarchy"              # 10%
    ANCAP = "ancap"                    # 0% - Best (Rothbard ideal)


# Regime parameters based on Hoppe's analysis
REGIME_PARAMS = {
    RegimeType.TOTALITARIAN: {
        "intervention_level": 1.0,
        "tax_income": 0.90,
        "tax_capital": 0.90,
        "tax_consumption": 0.50,
        "regulation": 1.0,
        "time_preference": 0.95,  # Very short-term (steal everything now)
        "description": "Total state control - Venezuela/Cuba/USSR model"
    },
    RegimeType.DEMOCRACY_SOCIALIST: {
        "intervention_level": 0.80,
        "tax_income": 0.60,
        "tax_capital": 0.40,
        "tax_consumption": 0.25,
        "regulation": 0.80,
        "time_preference": 0.80,  # Short electoral cycles
        "description": "High intervention democracy - Scandinavian model"
    },
    RegimeType.DEMOCRACY_LIBERAL: {
        "intervention_level": 0.50,
        "tax_income": 0.35,
        "tax_capital": 0.20,
        "tax_consumption": 0.10,
        "regulation": 0.50,
        "time_preference": 0.70,
        "description": "Moderate democracy - USA/UK model"
    },
    RegimeType.MONARCHY: {
        "intervention_level": 0.30,
        "tax_income": 0.20,
        "tax_capital": 0.10,
        "tax_consumption": 0.05,
        "regulation": 0.30,
        "time_preference": 0.40,  # Longer-term (dynasty thinking)
        "description": "Traditional monarchy - Liechtenstein model"
    },
    RegimeType.MINARCHY: {
        "intervention_level": 0.10,
        "tax_income": 0.08,
        "tax_capital": 0.02,
        "tax_consumption": 0.02,
        "regulation": 0.10,
        "time_preference": 0.30,
        "description": "Minimal state - Night watchman state"
    },
    RegimeType.ANCAP: {
        "intervention_level": 0.0,
        "tax_income": 0.0,
        "tax_capital": 0.0,
        "tax_consumption": 0.0,
        "regulation": 0.0,
        "time_preference": 0.0,  # Natural market preference
        "description": "Anarcho-capitalism - Rothbard ideal"
    },
}


@dataclass
class GovernmentState(AgentState):
    """State for government villain"""
    # Revenue extraction
    tax_revenue: Decimal = Decimal("0")
    tariff_revenue: Decimal = Decimal("0")

    # Spending (waste)
    spending: Decimal = Decimal("0")
    deficit: Decimal = Decimal("0")
    debt: Decimal = Decimal("0")

    # Damage metrics
    deadweight_loss: Decimal = Decimal("0")
    compliance_costs: Decimal = Decimal("0")
    capital_destroyed: Decimal = Decimal("0")
    trade_disruption: Decimal = Decimal("0")


class Government(Agent):
    """
    Government Agent - The Political Villain.

    "The State is that great fiction by which everyone tries
     to live at the expense of everyone else."
    - Frédéric Bastiat

    Hoppe's Analysis:
    - Democracy has HIGH time preference (politicians don't own)
    - Monarchy has LOWER time preference (king owns)
    - No government (Ancap) is ideal
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        country: str = "USA",
        regime_type: RegimeType = RegimeType.DEMOCRACY_LIBERAL,
        name: Optional[str] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            country=country,
            time_preference=REGIME_PARAMS[regime_type]["time_preference"],
            risk_tolerance=1.0,  # Plays with other people's money
            initial_wealth=Decimal("0"),
        )

        self.agent_type = AgentType.GOVERNMENT
        self.name = name or f"Government of {country}"
        self.regime_type = regime_type

        # Get parameters for this regime
        params = REGIME_PARAMS[regime_type]
        self.intervention_level = params["intervention_level"]

        # Taxation rates (extraction)
        self.tax_rate_income = params["tax_income"]
        self.tax_rate_capital = params["tax_capital"]
        self.tax_rate_consumption = params["tax_consumption"]
        self.tariff_rate = 0.05

        # Regulation
        self.regulation_level = params["regulation"]
        self.bureaucracy_cost = params["regulation"] * 0.15  # % of GDP

        # Debt
        self.debt_gdp_ratio = 0.0
        self.spending_gdp_ratio = params["intervention_level"] * 0.5

        # International relations
        self.at_war_with: Set[str] = set()
        self.sanctions_on: Set[str] = set()
        self.allies: Set[str] = set()

        # State
        self.state = GovernmentState()

        # Freedom index (inverse of intervention - higher = better)
        self.freedom_index = (1 - self.intervention_level) * 100

        # Policies
        self.policies: Dict[str, Any] = {}

    @property
    def total_tax_burden(self) -> float:
        """Combined tax burden"""
        return (
            self.tax_rate_income * 0.5 +
            self.tax_rate_capital * 0.3 +
            self.tax_rate_consumption * 0.2
        )

    @property
    def is_at_war(self) -> bool:
        return len(self.at_war_with) > 0

    # ===========================================
    # TAXATION (Theft, according to Rothbard)
    # ===========================================

    def collect_income_tax(self, income: Decimal) -> Decimal:
        """
        Collect income tax.

        Austrian/Libertarian View:
        - Taxation is theft
        - Discourages work and production
        - Creates deadweight loss
        """
        tax = income * Decimal(str(self.tax_rate_income))
        self.state.tax_revenue += tax

        # Calculate deadweight loss (economic damage beyond tax)
        # Higher taxes = more distortion
        deadweight = tax * Decimal(str(self.tax_rate_income * 0.3))
        self.state.deadweight_loss += deadweight

        return tax

    def collect_capital_tax(self, capital_gains: Decimal) -> Decimal:
        """
        Collect capital gains tax.

        Austrian View:
        - Punishes saving and investment
        - Reduces capital formation
        - Lowers future productivity
        """
        tax = capital_gains * Decimal(str(self.tax_rate_capital))
        self.state.tax_revenue += tax

        # Capital tax is especially harmful
        deadweight = tax * Decimal(str(self.tax_rate_capital * 0.5))
        self.state.deadweight_loss += deadweight

        return tax

    def collect_consumption_tax(self, consumption: Decimal) -> Decimal:
        """Collect consumption/sales tax"""
        tax = consumption * Decimal(str(self.tax_rate_consumption))
        self.state.tax_revenue += tax
        return tax

    def collect_tariff(self, import_value: Decimal) -> Decimal:
        """
        Collect import tariff.

        Austrian View:
        - Protectionism hurts consumers
        - Reduces trade benefits
        - Invites retaliation
        """
        tariff = import_value * Decimal(str(self.tariff_rate))
        self.state.tariff_revenue += tariff
        self.state.trade_disruption += tariff * Decimal("0.5")
        return tariff

    # ===========================================
    # REGULATION (Barriers to economic activity)
    # ===========================================

    def calculate_compliance_cost(self, business_revenue: Decimal) -> Decimal:
        """
        Calculate regulatory compliance cost.

        Austrian View:
        - Regulations create barriers to entry
        - Favor large corporations over small businesses
        - Add cost without creating value
        """
        cost = business_revenue * Decimal(str(self.bureaucracy_cost))
        self.state.compliance_costs += cost
        return cost

    def calculate_regulatory_burden(self) -> Dict[str, float]:
        """Calculate the burden of regulation"""
        return {
            "barriers_to_entry": self.regulation_level,
            "compliance_cost_pct": self.bureaucracy_cost,
            "licensing_burden": self.regulation_level * 0.5,
            "labor_law_burden": self.regulation_level * 0.3,
            "environmental_burden": self.regulation_level * 0.2,
        }

    # ===========================================
    # SPENDING (Crowding out private sector)
    # ===========================================

    def spend(self, amount: Decimal, category: str):
        """
        Government spending.

        Austrian View:
        - Crowds out private investment
        - Misallocates resources (no profit/loss signal)
        - Creates dependency
        """
        self.state.spending += amount

        # Calculate if we need to borrow
        if amount > self.state.tax_revenue:
            deficit = amount - self.state.tax_revenue
            self.state.deficit += deficit
            self.state.debt += deficit

    def calculate_crowding_out(self, gdp: Decimal) -> Decimal:
        """
        Calculate crowding out effect.

        Government spending "crowds out" private investment
        by consuming resources and raising interest rates.
        """
        spending_ratio = float(self.state.spending / max(gdp, Decimal("1")))
        crowding_out = gdp * Decimal(str(spending_ratio * 0.3))
        return crowding_out

    # ===========================================
    # WAR & SANCTIONS (Destruction of capital)
    # ===========================================

    def declare_war(self, target_country: str):
        """
        Declare war.

        Austrian/Libertarian View:
        - War is the health of the state
        - Destroys capital and lives
        - Leads to more government power
        - "War is a racket" - Smedley Butler
        """
        self.at_war_with.add(target_country)

    def end_war(self, target_country: str):
        """End a war"""
        self.at_war_with.discard(target_country)

    def calculate_war_cost(self, intensity: float = 0.5) -> Decimal:
        """
        Calculate economic cost of war.

        War destroys:
        - Physical capital
        - Human capital (lives)
        - Trade relationships
        - Savings (inflation to fund)
        """
        if not self.is_at_war:
            return Decimal("0")

        base_cost = Decimal("10000000000")  # $10B per war per week
        cost = base_cost * Decimal(str(len(self.at_war_with) * intensity))

        self.state.capital_destroyed += cost
        return cost

    def impose_sanctions(self, target_country: str):
        """
        Impose economic sanctions.

        Austrian View:
        - Hurts civilian population
        - Restricts mutually beneficial trade
        - Often counterproductive
        """
        self.sanctions_on.add(target_country)

    def calculate_sanction_cost(self) -> Decimal:
        """Sanctions hurt both sides"""
        cost = Decimal(str(len(self.sanctions_on) * 1000000000))  # $1B per sanction
        self.state.trade_disruption += cost
        return cost

    # ===========================================
    # DAMAGE METRICS
    # ===========================================

    def get_damage_report(self) -> Dict[str, Any]:
        """Get comprehensive damage report"""
        return {
            "regime_type": self.regime_type.value,
            "intervention_level": self.intervention_level,
            "freedom_index": self.freedom_index,
            "total_tax_burden": self.total_tax_burden,
            "deadweight_loss": str(self.state.deadweight_loss),
            "compliance_costs": str(self.state.compliance_costs),
            "capital_destroyed": str(self.state.capital_destroyed),
            "trade_disruption": str(self.state.trade_disruption),
            "debt": str(self.state.debt),
            "debt_gdp_ratio": self.debt_gdp_ratio,
            "at_war_with": list(self.at_war_with),
            "sanctions_on": list(self.sanctions_on),
            "hoppe_analysis": self._hoppe_analysis(),
        }

    def _hoppe_analysis(self) -> Dict[str, Any]:
        """
        Analysis based on Hoppe's Democracy: The God That Failed
        """
        regime_analysis = {
            RegimeType.TOTALITARIAN: {
                "verdict": "WORST - Total extraction",
                "explanation": "No property rights, total control, fastest decline"
            },
            RegimeType.DEMOCRACY_SOCIALIST: {
                "verdict": "VERY BAD - High extraction, short-term thinking",
                "explanation": "Politicians don't own, so they extract maximum now"
            },
            RegimeType.DEMOCRACY_LIBERAL: {
                "verdict": "BAD - Moderate extraction, still short-term",
                "explanation": "Better than socialist, but still democracy problems"
            },
            RegimeType.MONARCHY: {
                "verdict": "LESS BAD - Owner mentality",
                "explanation": "King preserves capital value for dynasty"
            },
            RegimeType.MINARCHY: {
                "verdict": "ACCEPTABLE - Minimal intervention",
                "explanation": "Night watchman state, limited damage"
            },
            RegimeType.ANCAP: {
                "verdict": "IDEAL - No forced extraction",
                "explanation": "All interactions voluntary, maximum freedom"
            },
        }

        return {
            "current_regime": self.regime_type.value,
            "analysis": regime_analysis[self.regime_type],
            "time_preference": self.time_preference,
            "expected_debt_trend": "increasing" if self.regime_type in [
                RegimeType.DEMOCRACY_SOCIALIST, RegimeType.DEMOCRACY_LIBERAL
            ] else "stable",
        }

    # ===========================================
    # MAIN STEP FUNCTION
    # ===========================================

    def step(self, world_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute one simulation step"""
        actions = []

        gdp = Decimal(str(world_state.get("gdp", 1e12)))

        # Reset period revenue
        self.state.tax_revenue = Decimal("0")
        self.state.tariff_revenue = Decimal("0")
        self.state.spending = Decimal("0")

        # 1. Collect taxes (based on economic activity)
        total_income = world_state.get("total_income", 0)
        total_capital_gains = world_state.get("total_capital_gains", 0)
        total_consumption = world_state.get("total_consumption", 0)
        total_imports = world_state.get("total_imports", 0)

        income_tax = self.collect_income_tax(Decimal(str(total_income)))
        capital_tax = self.collect_capital_tax(Decimal(str(total_capital_gains)))
        consumption_tax = self.collect_consumption_tax(Decimal(str(total_consumption)))
        tariffs = self.collect_tariff(Decimal(str(total_imports)))

        actions.append({
            "type": "tax_collection",
            "income_tax": str(income_tax),
            "capital_tax": str(capital_tax),
            "consumption_tax": str(consumption_tax),
            "tariffs": str(tariffs),
            "total": str(self.state.tax_revenue + self.state.tariff_revenue),
            "damage": "Wealth extracted from productive economy"
        })

        # 2. Government spending (waste)
        target_spending = gdp * Decimal(str(self.spending_gdp_ratio))
        self.spend(target_spending, "general")

        actions.append({
            "type": "spending",
            "amount": str(target_spending),
            "deficit": str(self.state.deficit),
            "damage": "Resources misallocated, no profit/loss signal"
        })

        # 3. War costs
        if self.is_at_war:
            war_cost = self.calculate_war_cost()
            actions.append({
                "type": "war_cost",
                "amount": str(war_cost),
                "wars": list(self.at_war_with),
                "damage": "Capital and lives destroyed"
            })

        # 4. Sanctions costs
        if self.sanctions_on:
            sanction_cost = self.calculate_sanction_cost()
            actions.append({
                "type": "sanction_cost",
                "amount": str(sanction_cost),
                "targets": list(self.sanctions_on),
                "damage": "Trade disruption hurts all parties"
            })

        # 5. Update debt ratio
        if gdp > 0:
            self.debt_gdp_ratio = float(self.state.debt / gdp)

        # 6. Democracy deterioration (Hoppe's prediction)
        if self.regime_type in [RegimeType.DEMOCRACY_SOCIALIST, RegimeType.DEMOCRACY_LIBERAL]:
            # Debt tends to grow in democracies
            self.state.debt *= Decimal("1.001")  # Compound growth

            # Taxes tend to increase
            if random.random() < 0.01:  # 1% chance per tick
                self.tax_rate_income *= 1.01

        # Add damage report
        actions.append({
            "type": "damage_report",
            "report": self.get_damage_report()
        })

        return actions

    def receive_income(self, amount: Decimal, source: str):
        """Government receives tax revenue"""
        self.state.tax_revenue += amount

    def pay_expense(self, amount: Decimal, destination: str) -> bool:
        """Government can always spend (prints/borrows)"""
        self.spend(amount, destination)
        return True

    # ===========================================
    # REGIME CHANGE
    # ===========================================

    def change_regime(self, new_regime: RegimeType):
        """
        Change political regime.

        For scenario testing (e.g., "What if USA became Ancap?")
        """
        old_regime = self.regime_type
        self.regime_type = new_regime

        params = REGIME_PARAMS[new_regime]
        self.intervention_level = params["intervention_level"]
        self.tax_rate_income = params["tax_income"]
        self.tax_rate_capital = params["tax_capital"]
        self.tax_rate_consumption = params["tax_consumption"]
        self.regulation_level = params["regulation"]
        self.time_preference = params["time_preference"]
        self.freedom_index = (1 - self.intervention_level) * 100

        return {
            "old_regime": old_regime.value,
            "new_regime": new_regime.value,
            "new_freedom_index": self.freedom_index,
            "new_intervention": self.intervention_level,
        }

    # ===========================================
    # FACTORY METHODS
    # ===========================================

    @classmethod
    def create_for_country(
        cls,
        country: str,
        regime_type: Optional[RegimeType] = None
    ) -> "Government":
        """Create government for a specific country"""

        # Default regimes by country (approximate)
        default_regimes = {
            "USA": RegimeType.DEMOCRACY_LIBERAL,
            "GBR": RegimeType.DEMOCRACY_LIBERAL,
            "DEU": RegimeType.DEMOCRACY_SOCIALIST,
            "FRA": RegimeType.DEMOCRACY_SOCIALIST,
            "SWE": RegimeType.DEMOCRACY_SOCIALIST,
            "CHN": RegimeType.TOTALITARIAN,
            "RUS": RegimeType.TOTALITARIAN,
            "PRK": RegimeType.TOTALITARIAN,
            "VEN": RegimeType.TOTALITARIAN,
            "CUB": RegimeType.TOTALITARIAN,
            "LIE": RegimeType.MONARCHY,  # Liechtenstein
            "MCO": RegimeType.MONARCHY,  # Monaco
            "SGP": RegimeType.MINARCHY,  # Close to minarchy
            "HKG": RegimeType.MINARCHY,  # Was close to minarchy
        }

        if regime_type is None:
            regime_type = default_regimes.get(country, RegimeType.DEMOCRACY_LIBERAL)

        return cls(
            country=country,
            regime_type=regime_type,
        )
