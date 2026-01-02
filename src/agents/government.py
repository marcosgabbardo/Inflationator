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

import random
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

from .base import Agent, AgentState, AgentType


class RegimeType(str, Enum):
    """
    Political regime types ordered from WORST to BEST
    according to Hoppe's analysis.
    """

    TOTALITARIAN = "totalitarian"  # 100% - Worst
    DEMOCRACY_SOCIALIST = "democracy_socialist"  # 80%
    DEMOCRACY_LIBERAL = "democracy_liberal"  # 50%
    MONARCHY = "monarchy"  # 30%
    MINARCHY = "minarchy"  # 10%
    ANCAP = "ancap"  # 0% - Best (Rothbard ideal)


# Regime parameters based on Hoppe's analysis
REGIME_PARAMS = {
    RegimeType.TOTALITARIAN: {
        "intervention_level": 1.0,
        "tax_income": 0.90,
        "tax_capital": 0.90,
        "tax_consumption": 0.50,
        "regulation": 1.0,
        "time_preference": 0.95,  # Very short-term (steal everything now)
        "description": "Total state control - Venezuela/Cuba/USSR model",
    },
    RegimeType.DEMOCRACY_SOCIALIST: {
        "intervention_level": 0.80,
        "tax_income": 0.60,
        "tax_capital": 0.40,
        "tax_consumption": 0.25,
        "regulation": 0.80,
        "time_preference": 0.80,  # Short electoral cycles
        "description": "High intervention democracy - Scandinavian model",
    },
    RegimeType.DEMOCRACY_LIBERAL: {
        "intervention_level": 0.50,
        "tax_income": 0.35,
        "tax_capital": 0.20,
        "tax_consumption": 0.10,
        "regulation": 0.50,
        "time_preference": 0.70,
        "description": "Moderate democracy - USA/UK model",
    },
    RegimeType.MONARCHY: {
        "intervention_level": 0.30,
        "tax_income": 0.20,
        "tax_capital": 0.10,
        "tax_consumption": 0.05,
        "regulation": 0.30,
        "time_preference": 0.40,  # Longer-term (dynasty thinking)
        "description": "Traditional monarchy - Liechtenstein model",
    },
    RegimeType.MINARCHY: {
        "intervention_level": 0.10,
        "tax_income": 0.08,
        "tax_capital": 0.02,
        "tax_consumption": 0.02,
        "regulation": 0.10,
        "time_preference": 0.30,
        "description": "Minimal state - Night watchman state",
    },
    RegimeType.ANCAP: {
        "intervention_level": 0.0,
        "tax_income": 0.0,
        "tax_capital": 0.0,
        "tax_consumption": 0.0,
        "regulation": 0.0,
        "time_preference": 0.0,  # Natural market preference
        "description": "Anarcho-capitalism - Rothbard ideal",
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

    # Damage metrics (Austrian view: ALL government action destroys value)
    deadweight_loss: Decimal = Decimal("0")  # Economic loss beyond taxes collected
    compliance_costs: Decimal = Decimal("0")  # Regulatory burden
    capital_destroyed: Decimal = Decimal("0")  # Wars, confiscation
    trade_disruption: Decimal = Decimal("0")  # Tariffs, sanctions
    spending_waste: Decimal = Decimal("0")  # Misallocation (no profit signal)


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
        agent_id: str | None = None,
        country: str = "USA",
        regime_type: RegimeType = RegimeType.DEMOCRACY_LIBERAL,
        name: str | None = None,
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

        # Tariffs (trade barriers)
        self.tariff_rate = 0.05  # Default 5%
        self.tariff_sectors: dict[str, float] = {}  # Sector-specific tariffs
        self.tariff_mode = "normal"  # normal, trade_war, protectionist

        # Regulation
        self.regulation_level = params["regulation"]
        self.bureaucracy_cost = params["regulation"] * 0.15  # % of GDP

        # Debt
        self.debt_gdp_ratio = 0.0
        self.spending_gdp_ratio = params["intervention_level"] * 0.5

        # International relations
        self.at_war_with: set[str] = set()
        self.sanctions_on: set[str] = set()
        self.allies: set[str] = set()

        # ===========================================
        # POLITICAL BUSINESS CYCLE (Democracies only)
        # ===========================================
        # Democratic governments ease policy before elections
        # to boost the economy and get re-elected
        # Austrian view: Another reason democracy is bad

        self.election_cycle_years = 4  # Years between elections (US: 4, some EU: 5)
        self.current_year_in_cycle = random.randint(1, 4)  # Start at random point
        self.is_election_year = False
        self.election_easing_active = False

        # State
        self.state = GovernmentState()

        # Freedom index (inverse of intervention - higher = better)
        self.freedom_index = (1 - self.intervention_level) * 100

        # Policies
        self.policies: dict[str, Any] = {}

    @property
    def total_tax_burden(self) -> float:
        """Combined tax burden"""
        return (
            self.tax_rate_income * 0.5
            + self.tax_rate_capital * 0.3
            + self.tax_rate_consumption * 0.2
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
        # Austrian View: Taxes destroy FAR more value than they collect
        # - Behavioral distortions (people work less, hide income)
        # - Capital flight
        # - Reduced investment
        # - Administrative burden on both sides
        # Estimate: 30-50% of tax amount is additional deadweight loss
        deadweight_multiplier = 0.3 + (
            self.tax_rate_income * 0.5
        )  # 30-65% based on rate
        deadweight = tax * Decimal(str(deadweight_multiplier))
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

        # Capital tax is especially harmful (Austrian View)
        # - Punishes saving and investment
        # - Reduces future capital stock
        # - Compounds over time (lost future productivity)
        # - Estimate: 50-100% additional deadweight loss
        deadweight_multiplier = 0.5 + (
            self.tax_rate_capital * 1.0
        )  # 50-100% based on rate
        deadweight = tax * Decimal(str(deadweight_multiplier))
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

    def set_tariff_rate(self, rate: float, mode: str = "normal"):
        """
        Set tariff rate (like Trump tariffs).

        Modes:
        - normal: Standard tariffs (5-10%)
        - trade_war: Aggressive tariffs (25-50%)
        - protectionist: Extreme tariffs (50-100%)

        Austrian View:
        - All tariffs hurt consumers
        - Trade wars destroy wealth
        - Free trade is always better
        """
        self.tariff_rate = max(0, min(1.0, rate))
        self.tariff_mode = mode

    def set_sector_tariff(self, sector: str, rate: float):
        """
        Set tariff for specific sector.

        Example: Steel tariffs, auto tariffs, electronics tariffs
        """
        self.tariff_sectors[sector] = max(0, min(1.0, rate))

    def calculate_tariff_price_impact(self) -> dict[str, float]:
        """
        Calculate how tariffs affect consumer prices.

        Tariffs are passed through to consumers:
        - Directly raises import prices
        - Reduces competition → domestic prices rise too
        - Supply chain effects multiply the impact

        Returns multipliers for different categories.
        """
        base_impact = self.tariff_rate

        if self.tariff_mode == "trade_war":
            # Trade war: additional disruption costs
            base_impact *= 1.5
        elif self.tariff_mode == "protectionist":
            # Extreme: massive supply chain disruption
            base_impact *= 2.0

        return {
            # Consumer goods heavily impacted (many are imported)
            "consumer_goods": 1 + base_impact * 0.3,  # 30% pass-through
            # Capital goods (machinery often imported)
            "capital_goods": 1 + base_impact * 0.4,  # 40% pass-through
            # Commodities (some affected by tariffs)
            "commodities": 1 + base_impact * 0.2,  # 20% pass-through
            # Labor (indirectly affected)
            "labor": 1 + base_impact * 0.05,  # 5% through wage pressure
            # Crypto (unaffected by tariffs)
            "crypto": 1.0,  # No tariff impact on BTC
        }

    def initiate_trade_war(self, target_rate: float = 0.25):
        """
        Initiate trade war (like US-China 2018-2020).

        Austrian View:
        - Nobody wins trade wars
        - Consumers pay higher prices
        - Retaliation hurts exports
        - Capital misallocated to protected industries
        """
        self.set_tariff_rate(target_rate, "trade_war")
        self.state.trade_disruption += Decimal(str(target_rate * 1e10))

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

    def calculate_regulatory_burden(self) -> dict[str, float]:
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
        - Has NO feedback mechanism (profit/loss)
        """
        self.state.spending += amount

        # Calculate if we need to borrow
        if amount > self.state.tax_revenue:
            deficit = amount - self.state.tax_revenue
            self.state.deficit += deficit
            self.state.debt += deficit

        # Austrian View: Government spending is inherently wasteful
        # - No market signals (profit/loss) to guide allocation
        # - Politicians spend other people's money on other people
        # - Estimated waste: 30-60% based on regime type
        waste_rate = {
            RegimeType.TOTALITARIAN: 0.7,
            RegimeType.DEMOCRACY_SOCIALIST: 0.5,
            RegimeType.DEMOCRACY_LIBERAL: 0.4,
            RegimeType.MONARCHY: 0.3,
            RegimeType.MINARCHY: 0.2,
            RegimeType.ANCAP: 0.0,
        }.get(self.regime_type, 0.4)

        self.state.spending_waste += amount * Decimal(str(waste_rate))

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

    @property
    def total_damage_caused(self) -> Decimal:
        """
        Total economic damage caused by government intervention.

        Austrian View: Government destroys value through:
        1. Deadweight loss from taxes
        2. Compliance costs from regulations
        3. Capital destroyed (wars, confiscation)
        4. Trade disruption (tariffs, sanctions)
        5. Spending waste (misallocation)
        """
        return (
            self.state.deadweight_loss
            + self.state.compliance_costs
            + self.state.capital_destroyed
            + self.state.trade_disruption
            + self.state.spending_waste
        )

    def get_damage_report(self) -> dict[str, Any]:
        """Get comprehensive damage report"""
        return {
            "regime_type": self.regime_type.value,
            "intervention_level": self.intervention_level,
            "freedom_index": self.freedom_index,
            "total_tax_burden": self.total_tax_burden,
            "total_damage": str(self.total_damage_caused),
            "deadweight_loss": str(self.state.deadweight_loss),
            "compliance_costs": str(self.state.compliance_costs),
            "capital_destroyed": str(self.state.capital_destroyed),
            "trade_disruption": str(self.state.trade_disruption),
            "spending_waste": str(self.state.spending_waste),
            "debt": str(self.state.debt),
            "debt_gdp_ratio": self.debt_gdp_ratio,
            "at_war_with": list(self.at_war_with),
            "sanctions_on": list(self.sanctions_on),
            "hoppe_analysis": self._hoppe_analysis(),
        }

    def _hoppe_analysis(self) -> dict[str, Any]:
        """
        Analysis based on Hoppe's Democracy: The God That Failed
        """
        regime_analysis = {
            RegimeType.TOTALITARIAN: {
                "verdict": "WORST - Total extraction",
                "explanation": "No property rights, total control, fastest decline",
            },
            RegimeType.DEMOCRACY_SOCIALIST: {
                "verdict": "VERY BAD - High extraction, short-term thinking",
                "explanation": "Politicians don't own, so they extract maximum now",
            },
            RegimeType.DEMOCRACY_LIBERAL: {
                "verdict": "BAD - Moderate extraction, still short-term",
                "explanation": "Better than socialist, but still democracy problems",
            },
            RegimeType.MONARCHY: {
                "verdict": "LESS BAD - Owner mentality",
                "explanation": "King preserves capital value for dynasty",
            },
            RegimeType.MINARCHY: {
                "verdict": "ACCEPTABLE - Minimal intervention",
                "explanation": "Night watchman state, limited damage",
            },
            RegimeType.ANCAP: {
                "verdict": "IDEAL - No forced extraction",
                "explanation": "All interactions voluntary, maximum freedom",
            },
        }

        return {
            "current_regime": self.regime_type.value,
            "analysis": regime_analysis[self.regime_type],
            "time_preference": self.time_preference,
            "expected_debt_trend": "increasing"
            if self.regime_type
            in [RegimeType.DEMOCRACY_SOCIALIST, RegimeType.DEMOCRACY_LIBERAL]
            else "stable",
        }

    # ===========================================
    # MAIN STEP FUNCTION
    # ===========================================

    def step(self, world_state: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute one simulation step"""
        actions = []

        gdp = Decimal(str(world_state.get("gdp", 1e12)))

        # Reset period revenue
        self.state.tax_revenue = Decimal("0")
        self.state.tariff_revenue = Decimal("0")
        self.state.spending = Decimal("0")

        # ===========================================
        # POLITICAL BUSINESS CYCLE UPDATE
        # ===========================================
        self.update_election_cycle(months_elapsed=1)
        easing_factors = self.get_election_easing_multiplier()

        # 1. Collect taxes (based on economic activity)
        total_income = world_state.get("total_income", 0)
        total_capital_gains = world_state.get("total_capital_gains", 0)
        total_consumption = world_state.get("total_consumption", 0)
        total_imports = world_state.get("total_imports", 0)

        # Apply election year tax collection efficiency
        tax_efficiency = easing_factors["tax_collection_efficiency"]

        income_tax = self.collect_income_tax(
            Decimal(str(total_income)) * Decimal(str(tax_efficiency))
        )
        capital_tax = self.collect_capital_tax(
            Decimal(str(total_capital_gains)) * Decimal(str(tax_efficiency))
        )
        consumption_tax = self.collect_consumption_tax(Decimal(str(total_consumption)))
        tariffs = self.collect_tariff(Decimal(str(total_imports)))

        actions.append(
            {
                "type": "tax_collection",
                "income_tax": str(income_tax),
                "capital_tax": str(capital_tax),
                "consumption_tax": str(consumption_tax),
                "tariffs": str(tariffs),
                "total": str(self.state.tax_revenue + self.state.tariff_revenue),
                "election_year_efficiency": tax_efficiency,
                "damage": "Wealth extracted from productive economy",
            }
        )

        # 2. Government spending (waste) - INCREASED before elections!
        base_spending = gdp * Decimal(str(self.spending_gdp_ratio))
        spending_multiplier = Decimal(str(easing_factors["spending_multiplier"]))
        target_spending = base_spending * spending_multiplier
        self.spend(target_spending, "general")

        actions.append(
            {
                "type": "spending",
                "amount": str(target_spending),
                "base_amount": str(base_spending),
                "election_multiplier": float(spending_multiplier),
                "deficit": str(self.state.deficit),
                "damage": "Resources misallocated, no profit/loss signal",
            }
        )

        # 3. Regulatory compliance costs (burden on businesses)
        # Austrian View: Regulations are hidden taxes that destroy value
        total_business_revenue = world_state.get("total_business_revenue", 0)
        if total_business_revenue > 0:
            compliance = self.calculate_compliance_cost(
                Decimal(str(total_business_revenue))
            )
            actions.append(
                {
                    "type": "compliance_cost",
                    "amount": str(compliance),
                    "regulation_level": self.regulation_level,
                    "damage": "Hidden tax on business, barriers to entry",
                }
            )

        # 5. War costs
        if self.is_at_war:
            war_cost = self.calculate_war_cost()
            actions.append(
                {
                    "type": "war_cost",
                    "amount": str(war_cost),
                    "wars": list(self.at_war_with),
                    "damage": "Capital and lives destroyed",
                }
            )

        # 6. Sanctions costs
        if self.sanctions_on:
            sanction_cost = self.calculate_sanction_cost()
            actions.append(
                {
                    "type": "sanction_cost",
                    "amount": str(sanction_cost),
                    "targets": list(self.sanctions_on),
                    "damage": "Trade disruption hurts all parties",
                }
            )

        # 7. Update debt ratio
        if gdp > 0:
            self.debt_gdp_ratio = float(self.state.debt / gdp)

        # 8. Democracy deterioration (Hoppe's prediction)
        if self.regime_type in [
            RegimeType.DEMOCRACY_SOCIALIST,
            RegimeType.DEMOCRACY_LIBERAL,
        ]:
            # Debt tends to grow in democracies
            self.state.debt *= Decimal("1.001")  # Compound growth

            # Taxes tend to increase (but less likely during election year)
            tax_increase_prob = 0.01 if not self.election_easing_active else 0.001
            if random.random() < tax_increase_prob:
                self.tax_rate_income *= 1.01

        # 9. Election year easing report
        if self.election_easing_active:
            actions.append(
                {
                    "type": "election_easing",
                    "is_election_year": self.is_election_year,
                    "current_year_in_cycle": round(self.current_year_in_cycle, 2),
                    "easing_active": self.election_easing_active,
                    "spending_boost": f"{(easing_factors['spending_multiplier'] - 1) * 100:.1f}%",
                    "cb_pressure": easing_factors["cb_pressure"],
                    "tax_relief": f"{(1 - easing_factors['tax_collection_efficiency']) * 100:.1f}%",
                    "damage": "Politicians buying votes with fiscal stimulus",
                }
            )

        # Add damage report
        actions.append({"type": "damage_report", "report": self.get_damage_report()})

        return actions

    def receive_income(self, amount: Decimal, source: str):
        """Government receives tax revenue"""
        self.state.tax_revenue += amount

    def pay_expense(self, amount: Decimal, destination: str) -> bool:
        """Government can always spend (prints/borrows)"""
        self.spend(amount, destination)
        return True

    # ===========================================
    # POLITICAL BUSINESS CYCLE (Democracies)
    # ===========================================

    def update_election_cycle(self, months_elapsed: int = 1):
        """
        Update the election cycle based on time passed.

        Austrian/Public Choice View:
        - Politicians manipulate economy before elections
        - Short-term stimulus for votes, long-term damage
        - Another reason democracy is inferior to monarchy
        """
        # Only democracies have election cycles
        if self.regime_type not in [
            RegimeType.DEMOCRACY_LIBERAL,
            RegimeType.DEMOCRACY_SOCIALIST,
        ]:
            self.is_election_year = False
            self.election_easing_active = False
            return

        # Convert months to years progress (12 months = 1 year)
        year_progress = months_elapsed / 12.0

        # Update position in cycle
        self.current_year_in_cycle += year_progress

        # Check if we've completed a cycle
        while self.current_year_in_cycle > self.election_cycle_years:
            self.current_year_in_cycle -= self.election_cycle_years

        # Determine if we're in election year (year 4 of 4, or last year of cycle)
        years_until_election = self.election_cycle_years - self.current_year_in_cycle
        self.is_election_year = years_until_election < 1.0

        # Pre-election easing starts 6-12 months before election
        self.election_easing_active = years_until_election < 0.75  # ~9 months before

    def get_election_easing_multiplier(self) -> dict[str, float]:
        """
        Calculate how much the government eases policy before elections.

        Effects:
        - Increased spending (fiscal stimulus)
        - Pressure on Central Bank for easy money
        - Tax "relief" or delayed collection
        - Regulatory forbearance

        Austrian View:
        - This creates artificial booms before elections
        - Bust comes after election (not incumbent's problem)
        - Democracy's short time preference in action
        """
        if not self.election_easing_active:
            return {
                "spending_multiplier": 1.0,
                "cb_pressure": 0.0,
                "tax_collection_efficiency": 1.0,
                "regulatory_forbearance": 1.0,
            }

        # How close to election? (more easing as election approaches)
        years_until = self.election_cycle_years - self.current_year_in_cycle
        proximity_factor = max(0, 1 - years_until)  # 0-1, higher = closer to election

        # Socialist democracies ease more aggressively
        if self.regime_type == RegimeType.DEMOCRACY_SOCIALIST:
            intensity = 0.3 + proximity_factor * 0.4  # 30-70% extra
        else:  # Liberal democracy
            intensity = 0.2 + proximity_factor * 0.3  # 20-50% extra

        return {
            # Increase government spending before election
            "spending_multiplier": 1.0 + intensity,
            # Pressure on Central Bank to lower rates/print money
            "cb_pressure": intensity * 0.5,  # 0-0.35 rate pressure
            # Collect taxes less aggressively
            "tax_collection_efficiency": 1.0 - intensity * 0.2,
            # Be lenient on regulations
            "regulatory_forbearance": 1.0 - intensity * 0.3,
        }

    def pressure_central_bank(self, central_bank, world_state: dict[str, Any]):
        """
        Democratic governments pressure central banks before elections.

        Even "independent" central banks face political pressure:
        - Threat to change mandate
        - Public criticism
        - Appointment of dovish governors

        Austrian View:
        - CB "independence" is a myth
        - Politicians always want easy money
        - Gold standard would prevent this
        """
        if not self.election_easing_active:
            return None

        easing_factors = self.get_election_easing_multiplier()
        pressure = easing_factors["cb_pressure"]

        if pressure > 0.1:
            return {
                "type": "political_pressure",
                "source": "government",
                "pressure_level": pressure,
                "message": "Government pressuring CB for easier policy before election",
                "suggested_rate_cut": pressure * 0.01,  # Up to 35bps cut
            }
        return None

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
        cls, country: str, regime_type: RegimeType | None = None
    ) -> "Government":
        """Create government for a specific country"""

        # Default regimes by country (based on Hoppe hierarchy)
        # 20 countries for multi-country simulation
        default_regimes = {
            # Americas
            "USA": RegimeType.DEMOCRACY_LIBERAL,
            "CAN": RegimeType.DEMOCRACY_SOCIALIST,
            "MEX": RegimeType.DEMOCRACY_SOCIALIST,
            "BRA": RegimeType.DEMOCRACY_SOCIALIST,
            "ARG": RegimeType.DEMOCRACY_SOCIALIST,
            # Europe
            "GBR": RegimeType.DEMOCRACY_SOCIALIST,
            "DEU": RegimeType.DEMOCRACY_SOCIALIST,
            "FRA": RegimeType.DEMOCRACY_SOCIALIST,
            "SWE": RegimeType.DEMOCRACY_SOCIALIST,
            "NOR": RegimeType.DEMOCRACY_SOCIALIST,
            "CHE": RegimeType.MINARCHY,  # Switzerland - low intervention
            "LIE": RegimeType.MONARCHY,  # Liechtenstein - hereditary
            # Asia
            "CHN": RegimeType.TOTALITARIAN,
            "JPN": RegimeType.DEMOCRACY_LIBERAL,
            "IND": RegimeType.DEMOCRACY_SOCIALIST,
            "IDN": RegimeType.DEMOCRACY_SOCIALIST,
            # Middle East
            "SAU": RegimeType.MONARCHY,  # Saudi Arabia
            "ARE": RegimeType.MONARCHY,  # UAE
            "TUR": RegimeType.TOTALITARIAN,  # Erdogan autocracy
            # Eurasia
            "RUS": RegimeType.TOTALITARIAN,
            # Legacy entries (kept for backwards compatibility)
            "PRK": RegimeType.TOTALITARIAN,
            "VEN": RegimeType.TOTALITARIAN,
            "CUB": RegimeType.TOTALITARIAN,
            "MCO": RegimeType.MONARCHY,
            "SGP": RegimeType.MINARCHY,
            "HKG": RegimeType.MINARCHY,
        }

        if regime_type is None:
            regime_type = default_regimes.get(country, RegimeType.DEMOCRACY_LIBERAL)

        gov = cls(
            country=country,
            regime_type=regime_type,
        )

        # Set election cycle based on country's actual political calendar
        # USA: Trump inaugurated Jan 2025, so Jan 2026 = Year 2
        # (Year 1 = first year of term, Year 4 = election year)
        from datetime import datetime

        current_year = datetime.now().year

        # Calculate years since last presidential inauguration
        usa_inauguration_years = [
            2025,
            2021,
            2017,
            2013,
            2009,
            2005,
            2001,
        ]  # Recent ones
        brazil_inauguration_years = [2023, 2019, 2015, 2011]  # Brazil 4-year cycle
        mexico_inauguration_years = [2024, 2018, 2012, 2006]  # Mexico 6-year but use 4

        if country == "USA":
            # Find most recent inauguration
            for year in usa_inauguration_years:
                if current_year >= year:
                    years_since = current_year - year
                    # Year 1 starts at inauguration (Jan), we track 0-based
                    gov.current_year_in_cycle = years_since % 4
                    break
        elif country == "BRA":
            for year in brazil_inauguration_years:
                if current_year >= year:
                    years_since = current_year - year
                    gov.current_year_in_cycle = years_since % 4
                    break
        elif country == "MEX":
            for year in mexico_inauguration_years:
                if current_year >= year:
                    years_since = current_year - year
                    gov.current_year_in_cycle = years_since % 4
                    break
        # Other countries keep the random initialization

        return gov
