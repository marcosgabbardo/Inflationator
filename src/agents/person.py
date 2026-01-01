"""
Person Agent

Individual economic actor in the simulation.
Makes decisions about:
- Labor (work, job seeking)
- Consumption (what to buy)
- Savings (how much to save)
- Investment (where to invest)
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Any, List, Optional
import random

from .base import Agent, AgentType, AgentState


@dataclass
class PersonState(AgentState):
    """Extended state for person agents"""
    income: Decimal = Decimal("0")
    employed: bool = True
    employer_id: Optional[str] = None
    skill_level: float = 0.5
    consumption_rate: float = 0.7
    age_group: str = "adult"  # young, adult, senior


class Person(Agent):
    """
    Person agent - individual economic actor.

    Austrian Economics Behavior:
    - Acts purposefully to maximize utility (subjective)
    - Time preference determines savings/consumption split
    - Responds to price signals
    - May hold Bitcoin as inflation hedge
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        country: str = "USA",
        time_preference: float = 0.5,
        risk_tolerance: float = 0.5,
        initial_wealth: Decimal = Decimal("10000"),
        skill_level: float = 0.5,
        age_group: str = "adult",
    ):
        super().__init__(
            agent_id=agent_id,
            country=country,
            time_preference=time_preference,
            risk_tolerance=risk_tolerance,
            initial_wealth=initial_wealth,
        )

        self.agent_type = AgentType.PERSON

        # Person-specific state
        self.state = PersonState(
            wealth=initial_wealth,
            skill_level=self._validate_range(skill_level),
            age_group=age_group,
            consumption_rate=0.5 + (time_preference * 0.4),  # Higher TP = more consumption
        )

        # Employment
        self.employer_id: Optional[str] = None
        self.wage: Decimal = Decimal("0")

    @property
    def employed(self) -> bool:
        return self.employer_id is not None

    @property
    def skill_level(self) -> float:
        return self.state.skill_level

    # ===========================================
    # LABOR DECISIONS
    # ===========================================

    def seek_employment(self, available_jobs: List[Dict]) -> Optional[Dict]:
        """
        Look for employment based on wage offers.

        Austrian Theory: Labor is priced by marginal productivity
        Workers seek highest wage matching their skills
        """
        if not available_jobs:
            return None

        # Filter jobs by skill requirement
        suitable_jobs = [
            job for job in available_jobs
            if job.get("required_skill", 0) <= self.skill_level
        ]

        if not suitable_jobs:
            return None

        # Choose highest paying job (rational actor)
        best_job = max(suitable_jobs, key=lambda j: j.get("wage", 0))

        # Reservation wage based on current wealth and time preference
        reservation_wage = self._calculate_reservation_wage()

        if Decimal(str(best_job.get("wage", 0))) >= reservation_wage:
            return best_job

        return None

    def _calculate_reservation_wage(self) -> Decimal:
        """
        Minimum wage to accept employment.

        Based on:
        - Current wealth (higher wealth = higher reservation)
        - Time preference (higher TP = need money now = lower reservation)
        - Skill level (higher skill = higher expectations)
        """
        base_wage = Decimal("1000")  # Minimum subsistence
        wealth_factor = Decimal(str(1 + float(self.wealth) / 100000))
        tp_factor = Decimal(str(1 - self.time_preference * 0.3))
        skill_factor = Decimal(str(1 + self.skill_level))

        return base_wage * wealth_factor * tp_factor * skill_factor

    def accept_job(self, employer_id: str, wage: Decimal):
        """Accept a job offer"""
        self.employer_id = employer_id
        self.wage = wage
        self.state.employed = True

    def lose_job(self):
        """Become unemployed"""
        self.employer_id = None
        self.wage = Decimal("0")
        self.state.employed = False

    # ===========================================
    # CONSUMPTION DECISIONS
    # ===========================================

    def decide_consumption(self, prices: Dict[str, Decimal]) -> Dict[str, Decimal]:
        """
        Decide how much to consume of each good.

        Austrian Theory:
        - Marginal utility determines consumption
        - Higher prices = less consumption (demand curve)
        - Time preference affects overall consumption level
        """
        savings_rate = self.calculate_savings_rate()
        available_for_consumption = self.wealth * Decimal(str(1 - savings_rate))

        consumption = {}

        # Basic needs first (inelastic demand)
        essential_spending = min(available_for_consumption, Decimal("500"))
        consumption["essentials"] = essential_spending

        # Discretionary spending
        discretionary = available_for_consumption - essential_spending
        if discretionary > 0:
            # Split between goods based on prices and preferences
            for good, price in prices.items():
                if good == "essentials":
                    continue
                # Simple demand: lower price = more quantity
                if price > 0:
                    price_dec = Decimal(str(price))
                    demand = discretionary * Decimal("0.2") / price_dec
                    consumption[good] = min(demand * price_dec, discretionary * Decimal("0.2"))

        return consumption

    def consume(self, amount: Decimal) -> bool:
        """Execute consumption (reduce wealth)"""
        if amount <= self.wealth:
            self.wealth -= amount
            return True
        return False

    # ===========================================
    # SAVINGS & INVESTMENT
    # ===========================================

    def decide_savings_allocation(
        self,
        bitcoin_price: Decimal,
        gold_price: Decimal,
        expected_inflation: float
    ) -> Dict[str, Decimal]:
        """
        Decide how to allocate savings.

        Austrian Theory:
        - Save more when time preference is low
        - Hedge inflation with hard assets (gold, Bitcoin)
        - Risk tolerance affects allocation
        """
        savings = self.wealth * Decimal(str(self.calculate_savings_rate()))

        allocation = {
            "cash": Decimal("0"),
            "bitcoin": Decimal("0"),
            "gold": Decimal("0"),
        }

        # If expecting high inflation, move to hard assets
        inflation_fear = min(1.0, expected_inflation * 10)  # Scale inflation expectation

        # Bitcoin allocation (higher risk, higher potential return)
        btc_allocation = inflation_fear * self.risk_tolerance * 0.3
        allocation["bitcoin"] = savings * Decimal(str(btc_allocation))

        # Gold allocation (lower risk, stable)
        gold_allocation = inflation_fear * (1 - self.risk_tolerance) * 0.3
        allocation["gold"] = savings * Decimal(str(gold_allocation))

        # Rest in cash
        allocation["cash"] = savings - allocation["bitcoin"] - allocation["gold"]

        return allocation

    def buy_bitcoin(self, amount: Decimal, price: Decimal) -> bool:
        """Purchase Bitcoin"""
        cost = amount * price
        if cost <= self.wealth:
            self.wealth -= cost
            self.state.bitcoin += amount
            return True
        return False

    def buy_gold(self, amount: Decimal, price: Decimal) -> bool:
        """Purchase gold"""
        cost = amount * price
        if cost <= self.wealth:
            self.wealth -= cost
            self.state.gold += amount
            return True
        return False

    # ===========================================
    # MAIN STEP FUNCTION
    # ===========================================

    def step(self, world_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute one simulation step.

        Returns list of actions taken.
        """
        actions = []

        # 1. Receive income if employed
        if self.employed and self.wage > 0:
            # Account for taxes (government extraction)
            tax_rate = world_state.get("tax_rate_income", 0.35)
            net_income = self.wage * Decimal(str(1 - tax_rate))
            self.receive_income(net_income, "wage")
            actions.append({
                "type": "income",
                "amount": str(net_income),
                "gross": str(self.wage),
                "tax_paid": str(self.wage - net_income)
            })

        # 2. Make consumption decisions
        prices = world_state.get("prices", {})
        consumption = self.decide_consumption(prices)
        total_consumed = sum(consumption.values())
        if total_consumed > 0:
            self.consume(total_consumed)
            actions.append({
                "type": "consumption",
                "amount": str(total_consumed),
                "breakdown": {k: str(v) for k, v in consumption.items()}
            })

        # 3. Update expectations
        actual_inflation = world_state.get("inflation_rate", 0.02)
        expected_inflation = self.state.expectations.get("inflation", 0.02)
        self.update_expectations(actual_inflation, expected_inflation)
        self.state.expectations["inflation"] = self.get_expected_inflation()

        # 4. Adjust savings allocation (periodically)
        if random.random() < 0.1:  # 10% chance to rebalance
            btc_price = Decimal(str(world_state.get("bitcoin_price", 50000)))
            gold_price = Decimal(str(world_state.get("gold_price", 2000)))
            allocation = self.decide_savings_allocation(
                btc_price, gold_price, self.state.expectations.get("inflation", 0.02)
            )
            actions.append({
                "type": "rebalance",
                "allocation": {k: str(v) for k, v in allocation.items()}
            })

        return actions

    def receive_income(self, amount: Decimal, source: str):
        """Receive income"""
        self.wealth += amount
        self.state.income = amount

    def pay_expense(self, amount: Decimal, destination: str) -> bool:
        """Pay an expense"""
        if amount <= self.wealth:
            self.wealth -= amount
            return True
        return False

    # ===========================================
    # FACTORY METHODS
    # ===========================================

    @classmethod
    def create_random(cls, country: str = "USA") -> "Person":
        """Create a person with random characteristics"""
        # Time preference distribution (most people moderate, some extremes)
        time_preference = random.betavariate(2, 2)  # Bell curve around 0.5

        # Risk tolerance
        risk_tolerance = random.betavariate(2, 3)  # Slightly risk-averse

        # Skill level (log-normal distribution)
        skill_level = min(1.0, random.lognormvariate(-0.5, 0.5))

        # Age group
        age_group = random.choices(
            ["young", "adult", "senior"],
            weights=[0.25, 0.55, 0.20]
        )[0]

        # Initial wealth (log-normal - most poor, some rich)
        initial_wealth = Decimal(str(max(100, random.lognormvariate(9, 1.5))))

        return cls(
            country=country,
            time_preference=time_preference,
            risk_tolerance=risk_tolerance,
            initial_wealth=initial_wealth,
            skill_level=skill_level,
            age_group=age_group,
        )
