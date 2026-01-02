"""
Company Agent

Business entity in the simulation.
Makes decisions about:
- Production (what and how much to produce)
- Hiring (how many workers, at what wage)
- Investment (expand or contract)
- Pricing (based on costs and demand)
"""

import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .base import Agent, AgentState, AgentType


class ProductionType:
    """Order of goods in Hayekian structure"""

    CONSUMER_GOODS = "consumer_goods"  # First order (direct consumption)
    CAPITAL_GOODS_1 = "capital_goods_1"  # Second order
    CAPITAL_GOODS_2 = "capital_goods_2"  # Third order (more removed)
    RAW_MATERIALS = "raw_materials"  # Higher order


@dataclass
class CompanyState(AgentState):
    """Extended state for company agents"""

    capital_stock: Decimal = Decimal("0")
    revenue: Decimal = Decimal("0")
    costs: Decimal = Decimal("0")
    debt: Decimal = Decimal("0")
    inventory: Decimal = Decimal("0")
    production_capacity: Decimal = Decimal("100")


class Company(Agent):
    """
    Company agent - business entity.

    Austrian Economics Behavior:
    - Produces goods based on expected profit
    - Hires labor at marginal productivity
    - Investment decisions based on interest rates
    - Responds to price signals
    - Can be malinvested due to central bank manipulation
    """

    def __init__(
        self,
        agent_id: str | None = None,
        country: str = "USA",
        time_preference: float = 0.3,  # Companies usually longer-term
        risk_tolerance: float = 0.5,
        initial_capital: Decimal = Decimal("100000"),
        sector: str = "manufacturing",
        production_type: str = ProductionType.CONSUMER_GOODS,
        order_of_goods: int = 1,
        name: str | None = None,
    ):
        super().__init__(
            agent_id=agent_id,
            country=country,
            time_preference=time_preference,
            risk_tolerance=risk_tolerance,
            initial_wealth=initial_capital,
        )

        self.agent_type = AgentType.COMPANY
        self.name = name or f"Company_{self.id[:8]}"
        self.sector = sector
        self.production_type = production_type
        self.order_of_goods = order_of_goods  # Hayekian production structure

        # State
        self.state = CompanyState(
            wealth=initial_capital,
            capital_stock=initial_capital * Decimal("0.7"),
        )

        # Employment
        self.employees: list[str] = []  # List of person IDs
        self.wage_rate: Decimal = Decimal("1000")

        # Financial
        self.interest_rate: float = 0.05
        self.profit_margin: float = 0.1

    @property
    def num_employees(self) -> int:
        return len(self.employees)

    @property
    def capital_stock(self) -> Decimal:
        return self.state.capital_stock

    # ===========================================
    # PRODUCTION DECISIONS
    # ===========================================

    def calculate_optimal_production(
        self, price: Decimal, input_costs: dict[str, Decimal]
    ) -> Decimal:
        """
        Calculate optimal production quantity.

        Austrian Theory:
        - Produce where marginal revenue = marginal cost
        - Higher order goods more sensitive to interest rates
        """
        # Simple production function
        labor_cost = self.wage_rate * Decimal(self.num_employees)
        material_cost = sum(input_costs.values())
        total_variable_cost = labor_cost + material_cost

        if price > 0:
            # Marginal cost approximation
            marginal_cost = total_variable_cost / max(
                Decimal("1"), self.state.production_capacity
            )

            # Produce up to capacity if profitable
            if price > marginal_cost:
                return self.state.production_capacity
            else:
                # Reduce production proportionally
                ratio = price / marginal_cost
                return self.state.production_capacity * ratio

        return Decimal("0")

    def produce(self, quantity: Decimal) -> Decimal:
        """
        Execute production.

        Returns actual quantity produced.
        """
        # Can't produce more than capacity
        actual = min(quantity, self.state.production_capacity)

        # Update inventory
        self.state.inventory += actual

        return actual

    # ===========================================
    # HIRING DECISIONS
    # ===========================================

    def calculate_labor_demand(self, market_wage: Decimal) -> int:
        """
        Calculate how many workers to hire.

        Austrian Theory:
        - Hire where marginal productivity = wage
        - Lower interest rates cause over-hiring (malinvestment)
        """
        # Marginal productivity of labor (simplified)
        marginal_product = (
            self.capital_stock * Decimal("0.01") / max(1, self.num_employees + 1)
        )

        # Demand more labor if marginal product > wage
        if marginal_product > market_wage:
            # How many more can we profitably hire?
            max_new_hires = int((marginal_product / market_wage) * 10)
            return max_new_hires
        elif marginal_product < market_wage * Decimal("0.8"):
            # Need to lay off
            return -max(1, self.num_employees // 10)

        return 0  # Keep current workforce

    def hire_worker(self, worker_id: str, wage: Decimal):
        """Hire a new worker"""
        if worker_id not in self.employees:
            self.employees.append(worker_id)
            self.wage_rate = wage

    def fire_worker(self, worker_id: str):
        """Fire a worker"""
        if worker_id in self.employees:
            self.employees.remove(worker_id)

    def pay_wages(self) -> Decimal:
        """Pay all employees"""
        total_wages = self.wage_rate * Decimal(self.num_employees)
        if total_wages <= self.wealth:
            self.wealth -= total_wages
            self.state.costs += total_wages
            return total_wages
        return Decimal("0")  # Can't pay - will need to fire

    # ===========================================
    # INVESTMENT DECISIONS
    # ===========================================

    def evaluate_investment_project(
        self,
        project_cost: Decimal,
        expected_return: float,
        duration: int,
        interest_rate: float,
    ) -> bool:
        """
        Evaluate whether to invest in a project.

        Austrian Theory:
        - Interest rate signals scarcity of capital
        - Artificially low rates cause malinvestment
        - Higher order goods more sensitive to rates
        """
        # Discount future returns
        present_value = Decimal("0")
        annual_return = project_cost * Decimal(str(expected_return))

        for year in range(1, duration + 1):
            # Discount more heavily for higher order goods
            discount_modifier = 1 + (self.order_of_goods - 1) * 0.1
            discount_rate = interest_rate * discount_modifier
            pv = annual_return / Decimal(str((1 + discount_rate) ** year))
            present_value += pv

        # Invest if NPV > 0
        npv = present_value - project_cost
        return npv > 0

    def invest_in_capital(self, amount: Decimal):
        """Invest in capital stock"""
        if amount <= self.wealth:
            self.wealth -= amount
            self.state.capital_stock += amount
            # Increase production capacity
            self.state.production_capacity += amount * Decimal("0.1")

    def take_loan(self, amount: Decimal, interest_rate: float):
        """Take a loan from bank"""
        self.wealth += amount
        self.state.debt += amount
        self.interest_rate = interest_rate

    def pay_interest(self) -> Decimal:
        """Pay interest on debt"""
        interest_payment = self.state.debt * Decimal(
            str(self.interest_rate / 52)
        )  # Weekly
        if interest_payment <= self.wealth:
            self.wealth -= interest_payment
            self.state.costs += interest_payment
            return interest_payment
        return Decimal("0")

    # ===========================================
    # PRICING DECISIONS
    # ===========================================

    def set_price(
        self, production_cost: Decimal, market_price: Decimal, demand_level: float
    ) -> Decimal:
        """
        Set price for output.

        Austrian Theory:
        - Prices are discovered, not set
        - Cost-plus pricing as approximation
        - Adjust based on market conditions
        """
        # Base: cost plus margin
        cost_price = production_cost * Decimal(str(1 + self.profit_margin))

        # Adjust based on demand
        if demand_level > 1.0:  # High demand
            price = cost_price * Decimal(str(1 + (demand_level - 1) * 0.2))
        elif demand_level < 1.0:  # Low demand
            price = cost_price * Decimal(str(max(0.8, demand_level)))
        else:
            price = cost_price

        # Don't deviate too much from market
        if market_price > 0:
            if price > market_price * Decimal("1.3"):
                price = market_price * Decimal("1.3")
            elif price < market_price * Decimal("0.7"):
                price = market_price * Decimal("0.7")

        return price

    # ===========================================
    # MAIN STEP FUNCTION
    # ===========================================

    def step(self, world_state: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute one simulation step"""
        actions = []

        interest_rate = world_state.get("interest_rate", 0.05)
        self.interest_rate = interest_rate

        # 1. Pay wages
        wages_paid = self.pay_wages()
        if wages_paid > 0:
            actions.append({"type": "wages_paid", "amount": str(wages_paid)})

        # 2. Pay interest on debt
        interest_paid = self.pay_interest()
        if interest_paid > 0:
            actions.append({"type": "interest_paid", "amount": str(interest_paid)})

        # 3. Pay taxes (government extraction)
        tax_rate = world_state.get("tax_rate_capital", 0.20)
        profit = self.state.revenue - self.state.costs
        if profit > 0:
            tax = profit * Decimal(str(tax_rate))
            if self.pay_expense(tax, "government"):
                actions.append({"type": "tax_paid", "amount": str(tax)})

        # 4. Production decision
        price = Decimal(
            str(world_state.get("prices", {}).get(self.production_type, 100))
        )
        input_costs = {"materials": Decimal(str(world_state.get("material_costs", 50)))}
        optimal_quantity = self.calculate_optimal_production(price, input_costs)
        produced = self.produce(optimal_quantity)
        if produced > 0:
            actions.append({"type": "production", "quantity": str(produced)})

        # 5. Sell inventory
        if self.state.inventory > 0:
            revenue = self.state.inventory * price
            self.state.revenue = revenue
            self.wealth += revenue
            sold = self.state.inventory
            self.state.inventory = Decimal("0")
            actions.append(
                {"type": "sales", "quantity": str(sold), "revenue": str(revenue)}
            )

        # 6. Hiring/firing decisions
        market_wage = Decimal(str(world_state.get("market_wage", 1000)))
        labor_demand = self.calculate_labor_demand(market_wage)
        if labor_demand != 0:
            actions.append({"type": "labor_adjustment", "demand": labor_demand})

        # 7. Investment decisions (if interest rate is attractive)
        if self.wealth > self.state.capital_stock * Decimal("0.2"):
            project_cost = self.wealth * Decimal("0.1")
            if self.evaluate_investment_project(
                project_cost,
                expected_return=0.15,
                duration=5,
                interest_rate=interest_rate,
            ):
                self.invest_in_capital(project_cost)
                actions.append({"type": "investment", "amount": str(project_cost)})

        # Reset period costs
        self.state.costs = Decimal("0")

        return actions

    def receive_income(self, amount: Decimal, source: str):
        """Receive income (revenue)"""
        self.wealth += amount
        self.state.revenue += amount

    def pay_expense(self, amount: Decimal, destination: str) -> bool:
        """Pay an expense"""
        if amount <= self.wealth:
            self.wealth -= amount
            self.state.costs += amount
            return True
        return False

    # ===========================================
    # FACTORY METHODS
    # ===========================================

    @classmethod
    def create_random(
        cls, country: str = "USA", sectors: list[str] | None = None
    ) -> "Company":
        """Create a company with random characteristics"""
        if sectors is None:
            sectors = [
                "manufacturing",
                "services",
                "agriculture",
                "technology",
                "finance",
            ]

        sector = random.choice(sectors)

        # Production type based on sector
        if sector in ["manufacturing", "agriculture"]:
            production_type = random.choice(
                [
                    ProductionType.CONSUMER_GOODS,
                    ProductionType.CAPITAL_GOODS_1,
                ]
            )
            order_of_goods = (
                1 if production_type == ProductionType.CONSUMER_GOODS else 2
            )
        elif sector == "technology":
            production_type = ProductionType.CAPITAL_GOODS_2
            order_of_goods = 3
        else:
            production_type = ProductionType.CONSUMER_GOODS
            order_of_goods = 1

        # Initial capital (log-normal distribution)
        initial_capital = Decimal(str(max(10000, random.lognormvariate(11, 1.5))))

        # Time preference (companies vary)
        time_preference = random.betavariate(2, 4)  # Usually low (long-term)

        return cls(
            country=country,
            time_preference=time_preference,
            risk_tolerance=random.betavariate(2, 2),
            initial_capital=initial_capital,
            sector=sector,
            production_type=production_type,
            order_of_goods=order_of_goods,
        )
