"""
Simulation Engine

The main simulation loop that orchestrates all agents,
markets, and data collection.

Austrian Economics Simulation:
- Agents act purposefully based on subjective values
- Prices emerge from market interactions
- Central bank/government create distortions
- Track the damage caused by interventions
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import asyncio
import random
from enum import Enum

from src.agents.person import Person
from src.agents.company import Company
from src.agents.bank import Bank
from src.agents.central_bank import CentralBank
from src.agents.government import Government, RegimeType
from src.economy.market import MarketManager, MarketType
from src.economy.labor_market import LaborMarket
from src.economy.austrian.business_cycle import BusinessCycle, CyclePhase
from src.data.collectors.bitcoin import BitcoinCollector
from src.data.collectors.commodities import CommoditiesCollector


class SimulationState(str, Enum):
    """Simulation states"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SimulationMetrics:
    """Key metrics tracked during simulation"""
    tick: int = 0
    real_time_weeks: int = 0

    # Economic indicators (our calculation, not government's)
    inflation_rate: float = 0.0
    money_supply: Decimal = Decimal("0")
    credit_expansion: Decimal = Decimal("0")
    unemployment_rate: float = 0.0
    gdp: Decimal = Decimal("0")

    # Price indicators (real assets)
    bitcoin_price: Decimal = Decimal("0")
    gold_price: Decimal = Decimal("0")

    # Damage metrics
    central_bank_damage: Decimal = Decimal("0")
    government_damage: Decimal = Decimal("0")
    total_malinvestment: Decimal = Decimal("0")

    # Freedom metrics
    freedom_index: float = 50.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "real_time_weeks": self.real_time_weeks,
            "inflation_rate": self.inflation_rate,
            "money_supply": str(self.money_supply),
            "credit_expansion": str(self.credit_expansion),
            "unemployment_rate": self.unemployment_rate,
            "gdp": str(self.gdp),
            "bitcoin_price": str(self.bitcoin_price),
            "gold_price": str(self.gold_price),
            "central_bank_damage": str(self.central_bank_damage),
            "government_damage": str(self.government_damage),
            "total_malinvestment": str(self.total_malinvestment),
            "freedom_index": self.freedom_index,
        }


@dataclass
class SimulationConfig:
    """Configuration for a simulation run"""
    country: str = "USA"
    num_persons: int = 100_000
    num_companies: int = 10_000
    num_banks: int = 500
    regime_type: RegimeType = RegimeType.DEMOCRACY_LIBERAL
    central_bank_intervention: float = 0.5
    ticks_per_run: int = 52  # 1 year by default
    use_real_data: bool = True


class SimulationEngine:
    """
    Main simulation engine.

    Orchestrates the Austrian Economics simulation:
    1. Initialize agents with realistic distributions
    2. Run market clearing each tick
    3. Agents make decisions based on prices
    4. Track distortions from CB/Government
    5. Calculate true inflation (not CPI)
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.state = SimulationState.IDLE
        self.current_tick = 0

        # Agents
        self.persons: List[Person] = []
        self.companies: List[Company] = []
        self.banks: List[Bank] = []
        self.central_bank: Optional[CentralBank] = None
        self.government: Optional[Government] = None

        # Markets
        self.market_manager = MarketManager(self.config.country)
        self.labor_market = LaborMarket(self.config.country)

        # Austrian Business Cycle tracker
        self.business_cycle = BusinessCycle()

        # Metrics
        self.metrics = SimulationMetrics()
        self.metrics_history: List[SimulationMetrics] = []

        # Callbacks for UI updates
        self.on_tick_complete: Optional[Callable[[SimulationMetrics], None]] = None
        self.on_simulation_complete: Optional[Callable[[List[SimulationMetrics]], None]] = None

        # External data (real prices)
        self.external_data: Dict[str, Any] = {}
        self.bitcoin_collector = BitcoinCollector() if self.config.use_real_data else None
        self.commodities_collector = CommoditiesCollector() if self.config.use_real_data else None

    # ===========================================
    # INITIALIZATION
    # ===========================================

    def initialize(self):
        """Initialize all agents and markets"""
        print(f"Initializing simulation for {self.config.country}...")

        # Create markets
        self._create_markets()

        # Create agents
        self._create_persons()
        self._create_companies()
        self._create_banks()

        # Create villains
        self._create_central_bank()
        self._create_government()

        # Initialize labor market (connect persons to companies)
        self._initialize_labor_market()

        # Fetch real data for initial prices
        if self.config.use_real_data:
            print("Fetching real-world prices...")
            self._fetch_real_data()

        # Initialize metrics
        self._update_metrics()

        self.state = SimulationState.IDLE
        print(f"Initialized: {len(self.persons)} persons, "
              f"{len(self.companies)} companies, "
              f"{len(self.banks)} banks")

    def _create_markets(self):
        """Create default markets"""
        self.market_manager.setup_default_markets()
        print(f"Created {len(self.market_manager.markets)} markets")

    def _create_persons(self):
        """Create person agents with realistic distribution"""
        print("Creating persons...")
        for _ in range(self.config.num_persons):
            person = Person.create_random(self.config.country)
            self.persons.append(person)

    def _create_companies(self):
        """Create company agents"""
        print("Creating companies...")
        for _ in range(self.config.num_companies):
            company = Company.create_random(self.config.country)
            self.companies.append(company)

    def _create_banks(self):
        """Create private banks"""
        print("Creating banks...")
        for _ in range(self.config.num_banks):
            bank = Bank.create_random(self.config.country)
            self.banks.append(bank)

    def _create_central_bank(self):
        """Create the central bank villain"""
        self.central_bank = CentralBank.create_for_country(self.config.country)
        self.central_bank.intervention_level = self.config.central_bank_intervention
        print(f"Created Central Bank: {self.central_bank.name}")

    def _create_government(self):
        """Create the government villain"""
        self.government = Government.create_for_country(
            self.config.country,
            self.config.regime_type
        )
        print(f"Created Government: {self.government.name} "
              f"(Regime: {self.government.regime_type.value})")

    def _fetch_real_data(self):
        """
        Fetch real-world price data from APIs.

        Updates Bitcoin, Gold, and commodity markets with real prices.
        Only called if use_real_data is True.
        """
        if not self.config.use_real_data:
            return

        try:
            # Fetch Bitcoin price
            if self.bitcoin_collector:
                btc_prices = asyncio.run(self.bitcoin_collector.get_current_price())
                if "usd" in btc_prices:
                    btc_market = self.market_manager.get_market_by_name("Bitcoin")
                    if btc_market:
                        btc_market.current_price = btc_prices["usd"]
                        self.external_data["bitcoin_usd"] = btc_prices["usd"]
                        print(f"  Real BTC price: ${btc_prices['usd']:,.0f}")

            # Fetch commodity prices
            if self.commodities_collector:
                comm_prices = self.commodities_collector.get_current_prices()

                # Update Gold market
                if "gold" in comm_prices:
                    gold_market = self.market_manager.get_market_by_name("Gold (oz)")
                    if gold_market:
                        gold_market.current_price = comm_prices["gold"]
                        self.external_data["gold_usd"] = comm_prices["gold"]

                # Update Silver market
                if "silver" in comm_prices:
                    silver_market = self.market_manager.get_market_by_name("Silver (oz)")
                    if silver_market:
                        silver_market.current_price = comm_prices["silver"]

                # Update Oil market
                if "oil" in comm_prices:
                    oil_market = self.market_manager.get_market_by_name("Oil (barrel)")
                    if oil_market:
                        oil_market.current_price = comm_prices["oil"]

        except Exception as e:
            print(f"  Warning: Could not fetch real data: {e}")

    def _initialize_labor_market(self):
        """
        Initialize labor market - connect persons to companies.

        This is critical for the simulation to work properly.
        Without this, everyone starts unemployed.
        """
        # Apply government distortions to labor market
        if self.government:
            # Minimum wage based on regime
            min_wage_by_regime = {
                RegimeType.ANCAP: Decimal("0"),
                RegimeType.MINARCHY: Decimal("0"),
                RegimeType.MONARCHY: Decimal("200"),
                RegimeType.DEMOCRACY_LIBERAL: Decimal("500"),
                RegimeType.DEMOCRACY_SOCIALIST: Decimal("800"),
                RegimeType.TOTALITARIAN: Decimal("300"),
            }
            self.labor_market.set_minimum_wage(
                min_wage_by_regime.get(self.government.regime_type, Decimal("500"))
            )

            # Payroll tax
            self.labor_market.set_payroll_tax(self.government.tax_rate_income * 0.3)

            # Labor regulations
            regulation_by_regime = {
                RegimeType.ANCAP: 0.0,
                RegimeType.MINARCHY: 0.05,
                RegimeType.MONARCHY: 0.15,
                RegimeType.DEMOCRACY_LIBERAL: 0.3,
                RegimeType.DEMOCRACY_SOCIALIST: 0.6,
                RegimeType.TOTALITARIAN: 0.9,
            }
            self.labor_market.set_labor_regulations(
                regulation_by_regime.get(self.government.regime_type, 0.3)
            )

        # Initialize employment
        result = self.labor_market.initialize_employment(
            persons=self.persons,
            companies=self.companies,
            initial_employment_rate=0.85  # Start with 85% employed
        )

        print(f"Labor Market Initialized:")
        print(f"  - Employed: {result['employed']} ({result['employment_rate']:.1%})")
        print(f"  - Unemployed: {result['unemployed']} ({result['unemployment_rate']:.1%})")

    # ===========================================
    # SIMULATION LOOP
    # ===========================================

    def run(self, ticks: Optional[int] = None) -> List[SimulationMetrics]:
        """
        Run the simulation for specified ticks.

        Each tick represents 1 week of economic activity.
        """
        ticks = ticks or self.config.ticks_per_run
        self.state = SimulationState.RUNNING

        print(f"\nRunning simulation for {ticks} weeks...")

        for _ in range(ticks):
            if self.state != SimulationState.RUNNING:
                break

            self._run_tick()
            self.current_tick += 1

        self.state = SimulationState.COMPLETED

        if self.on_simulation_complete:
            self.on_simulation_complete(self.metrics_history)

        return self.metrics_history

    def _run_tick(self):
        """Execute one simulation tick"""
        # Fetch real data periodically (every 10 ticks to avoid rate limiting)
        if self.current_tick % 10 == 0 and self.config.use_real_data:
            self._fetch_real_data()

        # Build world state for agents
        world_state = self._build_world_state()

        # 1. Central Bank actions (monetary distortions)
        if self.central_bank:
            cb_actions = self.central_bank.step(world_state)

        # 2. Government actions (fiscal distortions)
        if self.government:
            gov_actions = self.government.step(world_state)

        # 3. Banks process (credit creation)
        for bank in self.banks:
            bank.step(world_state)

        # 4. Companies produce and submit sell orders
        self._process_company_orders(world_state)

        # 5. Persons consume, save, invest - submit buy orders
        self._process_person_orders(world_state)

        # 6. Labor market dynamics
        self._process_labor_market()

        # 7. Credit system - loans and deposits
        self._process_credit_system(world_state)

        # 8. Clear all markets and discover prices
        trades = self.market_manager.clear_all_markets(self.current_tick)

        # 9. Update Austrian Business Cycle
        self._update_business_cycle(world_state)

        # 10. Update metrics
        self._update_metrics()

        # Store metrics history
        self.metrics_history.append(SimulationMetrics(
            tick=self.metrics.tick,
            real_time_weeks=self.metrics.real_time_weeks,
            inflation_rate=self.metrics.inflation_rate,
            money_supply=self.metrics.money_supply,
            credit_expansion=self.metrics.credit_expansion,
            unemployment_rate=self.metrics.unemployment_rate,
            gdp=self.metrics.gdp,
            bitcoin_price=self.metrics.bitcoin_price,
            gold_price=self.metrics.gold_price,
            central_bank_damage=self.metrics.central_bank_damage,
            government_damage=self.metrics.government_damage,
            total_malinvestment=self.metrics.total_malinvestment,
            freedom_index=self.metrics.freedom_index,
        ))

        # Callback
        if self.on_tick_complete:
            self.on_tick_complete(self.metrics)

        # Progress indicator
        if self.current_tick % 10 == 0:
            print(f"  Week {self.current_tick}: Inflation={self.metrics.inflation_rate:.2%}, "
                  f"BTC=${self.metrics.bitcoin_price}")

    def _build_world_state(self) -> Dict[str, Any]:
        """Build the world state that agents use for decisions"""
        prices = self.market_manager.get_all_prices()

        # Calculate aggregates
        total_income = sum(p.wage for p in self.persons if p.employed)
        total_consumption = sum(p.wealth * Decimal(str(1 - p.calculate_savings_rate()))
                               for p in self.persons[:1000]) * 100  # Scale up
        employed_count = sum(1 for p in self.persons if p.employed)
        unemployment = 1 - (employed_count / max(1, len(self.persons)))

        # Central bank rate
        policy_rate = self.central_bank.policy_rate if self.central_bank else 0.05

        return {
            "tick": self.current_tick,
            "prices": {k: float(v) for k, v in prices.items()},
            "inflation_rate": self.metrics.inflation_rate,
            "policy_rate": policy_rate,
            "interest_rate": policy_rate + 0.02,  # Bank spread
            "market_wage": float(prices.get("Labor", Decimal("1000"))),
            "unemployment": unemployment,
            "bitcoin_price": float(self.metrics.bitcoin_price),
            "gold_price": float(self.metrics.gold_price),
            "tax_rate_income": self.government.tax_rate_income if self.government else 0.35,
            "tax_rate_capital": self.government.tax_rate_capital if self.government else 0.20,
            "total_income": float(total_income),
            "total_consumption": float(total_consumption),
            "total_capital_gains": float(total_consumption * Decimal("0.1")),
            "total_imports": float(total_consumption * Decimal("0.2")),
            "avg_time_preference": 0.5,
            "gdp": float(self.metrics.gdp),
            "failing_banks": [],
            "gdp_growth": 0.02,
            "material_costs": 50,
        }

    def _process_company_orders(self, world_state: Dict[str, Any]):
        """
        Companies produce goods and submit sell orders to markets.

        Austrian Theory:
        - Production decisions based on price signals
        - Higher order goods more sensitive to interest rates
        - Malinvestment when prices distorted by central bank
        """
        # Get consumer goods markets
        consumer_markets = self.market_manager.get_markets_by_type(MarketType.CONSUMER_GOODS)
        capital_markets = self.market_manager.get_markets_by_type(MarketType.CAPITAL_GOODS)

        # Sample companies for performance
        sample_size = min(1000, len(self.companies))
        sampled_companies = random.sample(self.companies, sample_size)

        for company in sampled_companies:
            # Let company make its decisions
            company.step(world_state)

            # Submit sell orders based on production
            production_value = company.capital_stock * Decimal("0.02")  # Weekly output

            if production_value > 0:
                # Choose market based on production type
                if company.production_type == "consumer_goods":
                    target_markets = consumer_markets
                else:
                    target_markets = capital_markets if capital_markets else consumer_markets

                if target_markets:
                    # Pick random market (could be more sophisticated)
                    market = random.choice(target_markets)

                    # Calculate quantity and price
                    quantity = production_value / market.current_price
                    # Price slightly above current price to make profit
                    min_price = market.current_price * Decimal("0.95")

                    if quantity > 0:
                        market.submit_sell_order(
                            agent_id=company.id,
                            quantity=quantity,
                            min_price=min_price,
                            tick=self.current_tick
                        )

    def _process_person_orders(self, world_state: Dict[str, Any]):
        """
        Persons make consumption and investment decisions, submit buy orders.

        Austrian Theory:
        - Consumption based on time preference
        - Savings rate affects capital accumulation
        - Bitcoin/Gold as inflation hedge
        """
        consumer_markets = self.market_manager.get_markets_by_type(MarketType.CONSUMER_GOODS)
        bitcoin_market = self.market_manager.get_market_by_name("Bitcoin")
        gold_market = self.market_manager.get_market_by_name("Gold (oz)")

        # Sample persons for performance
        sample_size = min(5000, len(self.persons))
        sampled_persons = random.sample(self.persons, sample_size)

        for person in sampled_persons:
            # Let person make decisions
            person.step(world_state)

            # Skip if no wealth
            if person.wealth <= 0:
                continue

            # Calculate weekly budget
            savings_rate = person.calculate_savings_rate()
            weekly_income = person.wage if person.employed else person.wealth * Decimal("0.01")
            consumption_budget = weekly_income * Decimal(str(1 - savings_rate))
            investment_budget = weekly_income * Decimal(str(savings_rate * 0.3))  # Part of savings goes to assets

            # Submit buy orders for consumer goods
            if consumption_budget > 0 and consumer_markets:
                market = random.choice(consumer_markets)
                quantity = consumption_budget / market.current_price
                # Willing to pay slightly above current price
                max_price = market.current_price * Decimal("1.05")

                if quantity > 0:
                    market.submit_buy_order(
                        agent_id=person.id,
                        quantity=quantity,
                        max_price=max_price,
                        tick=self.current_tick
                    )

            # Investment in Bitcoin (based on risk tolerance and inflation expectations)
            if investment_budget > 0 and bitcoin_market:
                btc_allocation = person.risk_tolerance * float(investment_budget) * 0.5
                if btc_allocation > 100:  # Minimum investment
                    quantity = Decimal(str(btc_allocation)) / bitcoin_market.current_price
                    max_price = bitcoin_market.current_price * Decimal("1.02")

                    bitcoin_market.submit_buy_order(
                        agent_id=person.id,
                        quantity=quantity,
                        max_price=max_price,
                        tick=self.current_tick
                    )

            # Investment in Gold (more conservative)
            if investment_budget > 0 and gold_market:
                gold_allocation = (1 - person.risk_tolerance) * float(investment_budget) * 0.3
                if gold_allocation > 50:  # Minimum investment
                    quantity = Decimal(str(gold_allocation)) / gold_market.current_price
                    max_price = gold_market.current_price * Decimal("1.01")

                    gold_market.submit_buy_order(
                        agent_id=person.id,
                        quantity=quantity,
                        max_price=max_price,
                        tick=self.current_tick
                    )

    def _process_labor_market(self):
        """
        Process labor market dynamics - hiring, firing, job search.

        Austrian Theory:
        - Wages determined by marginal productivity
        - Unemployment from wage rigidity (minimum wage)
        - Labor market clears at market wage
        """
        # Collect job postings from companies needing workers
        for company in self.companies[:500]:  # Sample
            current_employees = len(company.employees)
            optimal_employees = max(1, int(float(company.capital_stock) / 15000))

            if current_employees < optimal_employees:
                # Need to hire
                wage_offer = company.wage_rate
                positions_needed = optimal_employees - current_employees

                self.labor_market.post_job(
                    company_id=company.id,
                    wage=wage_offer,
                    required_skill=0.3,  # Base skill requirement
                    positions=min(positions_needed, 5),  # Max 5 per tick
                    sector=company.sector
                )

        # Collect job applications from unemployed
        unemployed = [p for p in self.persons if not p.employed][:1000]  # Sample
        for person in unemployed:
            # Reservation wage based on skills and expectations
            reservation_wage = Decimal("500") + Decimal(str(person.skill_level * 500))

            self.labor_market.apply_for_job(
                person_id=person.id,
                skill_level=person.skill_level,
                reservation_wage=reservation_wage
            )

        # Clear labor market
        result = self.labor_market.clear_market(self.current_tick)

        # Update person states based on matches
        for person in self.persons:
            if self.labor_market.is_employed(person.id):
                person.employer_id = self.labor_market.get_employer(person.id)
                person.wage = self.labor_market.get_wage(person.id)
                person.state.employed = True
            else:
                if person.employer_id and not self.labor_market.is_employed(person.id):
                    # Person lost job
                    person.employer_id = None
                    person.wage = Decimal("0")
                    person.state.employed = False

    def _process_credit_system(self, world_state: Dict[str, Any]):
        """
        Process credit operations - loans and deposits.

        Austrian Business Cycle Theory:
        - Credit expansion lowers interest rates artificially
        - Companies take loans for capital goods (higher order)
        - Malinvestment occurs when rates don't reflect savings
        - Eventually bust corrects the malinvestment
        """
        # Assign banks to agents if not done
        if not hasattr(self, '_bank_assignments'):
            self._bank_assignments = {}
            for person in self.persons:
                self._bank_assignments[person.id] = random.choice(self.banks)
            for company in self.companies:
                self._bank_assignments[company.id] = random.choice(self.banks)

        # 1. Persons deposit savings
        for person in random.sample(self.persons, min(2000, len(self.persons))):
            if person.wealth > Decimal("1000") and person.employed:
                savings_rate = person.calculate_savings_rate()
                deposit_amount = person.wealth * Decimal(str(savings_rate * 0.1))

                if deposit_amount > Decimal("100"):
                    bank = self._bank_assignments.get(person.id)
                    if bank:
                        bank.accept_deposit(person.id, deposit_amount)
                        person.wealth -= deposit_amount

        # 2. Companies request loans for investment
        for company in random.sample(self.companies, min(500, len(self.companies))):
            # Check if company needs capital
            desired_capital = Decimal(str(len(company.employees) * 15000))
            current_capital = company.capital_stock

            if current_capital < desired_capital:
                # Need to invest
                loan_amount = (desired_capital - current_capital) * Decimal("0.3")

                if loan_amount > Decimal("10000"):
                    bank = self._bank_assignments.get(company.id)
                    if bank and bank.loanable_funds >= loan_amount:
                        # Evaluate and make loan
                        creditworthiness = min(1.0, float(company.capital_stock) / 100000)

                        if bank.evaluate_loan_application(
                            company.id,
                            loan_amount,
                            company.capital_stock,  # Collateral
                            creditworthiness
                        ):
                            if bank.make_loan(
                                company.id,
                                loan_amount,
                                term_weeks=52,  # 1 year loan
                                collateral=company.capital_stock
                            ):
                                # Company receives funds
                                company.state.capital_stock += loan_amount
                                company.state.debt += loan_amount

        # 3. Loan payments
        for bank in self.banks:
            for company in self.companies[:200]:  # Sample
                if company.id in bank.loans:
                    payment_due = bank.collect_loan_payment(company.id)
                    if payment_due and company.wealth >= payment_due:
                        bank.receive_loan_payment(company.id, payment_due)
                        company.wealth -= payment_due

        # 4. Track credit expansion (key Austrian metric)
        total_credit = sum(b.state.credit_created for b in self.banks)
        if total_credit > self.metrics.credit_expansion:
            # Credit is expanding - signal of potential malinvestment
            expansion_rate = (total_credit - self.metrics.credit_expansion) / max(Decimal("1"), self.metrics.credit_expansion)
            if float(expansion_rate) > 0.05:  # 5% expansion per tick is concerning
                # This could trigger business cycle effects
                if self.central_bank:
                    self.central_bank.state.malinvestment_induced += total_credit * Decimal("0.01")

    def _update_business_cycle(self, world_state: Dict[str, Any]):
        """
        Update the Austrian Business Cycle state.

        Tracks the artificial boom-bust caused by credit expansion.
        """
        # Calculate inputs for cycle update
        market_rate = world_state.get("policy_rate", 0.05)
        avg_time_preference = world_state.get("avg_time_preference", 0.5)

        # Credit metrics
        current_credit = sum(b.state.credit_created for b in self.banks)
        previous_credit = self.metrics.credit_expansion

        # Investment metrics (from markets)
        capital_markets = self.market_manager.get_markets_by_type(MarketType.CAPITAL_GOODS)
        consumer_markets = self.market_manager.get_markets_by_type(MarketType.CONSUMER_GOODS)

        capital_investment = sum(m.volume for m in capital_markets)
        consumer_investment = sum(m.volume for m in consumer_markets)

        # Malinvestment from central bank
        malinvestment = Decimal("0")
        if self.central_bank:
            malinvestment = self.central_bank.state.malinvestment_induced

        # Count liquidations (companies going bankrupt)
        liquidations = sum(1 for c in self.companies if c.wealth < Decimal("0"))

        # Update business cycle
        cycle_signals = self.business_cycle.update(
            market_rate=market_rate,
            avg_time_preference=avg_time_preference,
            credit_expansion=current_credit,
            previous_credit=previous_credit,
            capital_goods_investment=capital_investment,
            consumer_goods_investment=consumer_investment,
            total_malinvestment=malinvestment,
            liquidations=liquidations
        )

        # Apply cycle effects to agents
        self._apply_cycle_effects(cycle_signals)

    def _apply_cycle_effects(self, cycle_signals: Dict[str, Any]):
        """
        Apply business cycle effects to agents.

        Austrian Theory:
        - During boom: entrepreneurs are misled
        - During bust: correction is healthy
        """
        phase = cycle_signals["phase"]
        boom_intensity = cycle_signals["boom_intensity"]

        if phase == "boom":
            # Companies are misled into expanding
            for company in random.sample(self.companies, min(100, len(self.companies))):
                # Increase desire to invest (will lead to malinvestment)
                company.time_preference *= (1 - boom_intensity * 0.1)

        elif phase == "bust":
            # Liquidation of malinvestment
            correction_rate = 0.05  # 5% of unsustainable projects fail per tick

            # Companies with high debt/capital ratio more likely to fail
            for company in self.companies:
                if company.state.debt > company.capital_stock * Decimal("0.8"):
                    # High leverage - vulnerable to bust
                    if random.random() < correction_rate:
                        # Company goes bankrupt - this is healthy!
                        company.wealth = Decimal("-1")  # Mark as failed

                        # Fire all employees
                        for emp_id in company.employees:
                            self.labor_market.fire(emp_id, self.current_tick)
                        company.employees = []

                        # Reduce malinvestment (the correction)
                        if self.central_bank:
                            correction = company.state.debt * Decimal("0.1")
                            self.central_bank.state.malinvestment_induced = max(
                                Decimal("0"),
                                self.central_bank.state.malinvestment_induced - correction
                            )

        elif phase == "trough":
            # Good time to start new businesses
            pass  # Future: could add new company creation

        elif phase == "recovery":
            # Normal operations resume
            pass

    def _update_metrics(self):
        """Update simulation metrics"""
        self.metrics.tick = self.current_tick
        self.metrics.real_time_weeks = self.current_tick

        # Calculate inflation from market prices
        self.metrics.inflation_rate = self.market_manager.get_inflation_index()

        # Money supply from central bank
        if self.central_bank:
            self.metrics.money_supply = self.central_bank.state.base_money

        # Credit expansion from banks
        total_credit = sum(b.state.credit_created for b in self.banks)
        self.metrics.credit_expansion = total_credit

        # Unemployment from labor market
        self.metrics.unemployment_rate = self.labor_market.unemployment_rate

        # GDP approximation
        total_wealth = sum(p.wealth for p in self.persons)
        total_capital = sum(c.capital_stock for c in self.companies)
        self.metrics.gdp = total_wealth + total_capital

        # Get Bitcoin and Gold prices from markets
        for market in self.market_manager.markets.values():
            if market.name == "Bitcoin":
                self.metrics.bitcoin_price = market.current_price
            elif market.name == "Gold (oz)":
                self.metrics.gold_price = market.current_price

        # Damage metrics
        if self.central_bank:
            self.metrics.central_bank_damage = self.central_bank.total_damage_caused
            self.metrics.total_malinvestment = self.central_bank.state.malinvestment_induced

        if self.government:
            self.metrics.government_damage = (
                self.government.state.deadweight_loss +
                self.government.state.compliance_costs +
                self.government.state.capital_destroyed
            )
            self.metrics.freedom_index = self.government.freedom_index

    # ===========================================
    # CONTROL
    # ===========================================

    def pause(self):
        """Pause simulation"""
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.PAUSED

    def resume(self):
        """Resume simulation"""
        if self.state == SimulationState.PAUSED:
            self.state = SimulationState.RUNNING

    def stop(self):
        """Stop simulation"""
        self.state = SimulationState.COMPLETED

    def reset(self):
        """Reset simulation to initial state"""
        self.current_tick = 0
        self.metrics = SimulationMetrics()
        self.metrics_history = []
        self.state = SimulationState.IDLE

        # Reinitialize
        self.persons = []
        self.companies = []
        self.banks = []
        self.central_bank = None
        self.government = None
        self.labor_market = LaborMarket(self.config.country)
        self.business_cycle = BusinessCycle()

        self.initialize()

    # ===========================================
    # SCENARIOS
    # ===========================================

    def apply_scenario(self, scenario_name: str, parameters: Dict[str, Any]):
        """
        Apply a what-if scenario.

        Examples:
        - "fed_doubles_money": Central bank doubles base money
        - "ancap_transition": Government becomes ancap
        - "hyperinflation": Extreme money printing
        """
        if scenario_name == "fed_doubles_money":
            if self.central_bank:
                current_base = self.central_bank.state.base_money
                self.central_bank.print_money(current_base)
                print(f"Scenario: FED doubled money supply to "
                      f"${self.central_bank.state.base_money}")

        elif scenario_name == "ancap_transition":
            if self.government:
                result = self.government.change_regime(RegimeType.ANCAP)
                print(f"Scenario: Transitioned to Ancap. "
                      f"Freedom Index: {result['new_freedom_index']}")

        elif scenario_name == "increase_taxes":
            tax_increase = parameters.get("increase", 0.1)
            if self.government:
                self.government.tax_rate_income += tax_increase
                print(f"Scenario: Increased income tax to "
                      f"{self.government.tax_rate_income:.0%}")

        elif scenario_name == "regime_change":
            new_regime = parameters.get("regime", RegimeType.MINARCHY)
            if self.government:
                result = self.government.change_regime(new_regime)
                print(f"Scenario: Changed to {new_regime.value}")

        elif scenario_name == "zero_intervention":
            if self.central_bank:
                self.central_bank.reset_to_zero_intervention()
            if self.government:
                self.government.change_regime(RegimeType.ANCAP)
            print("Scenario: Zero intervention (Austrian ideal)")

    def get_damage_summary(self) -> Dict[str, Any]:
        """Get summary of all damage caused by interventions"""
        cb_report = self.central_bank.get_damage_report() if self.central_bank else {}
        gov_report = self.government.get_damage_report() if self.government else {}

        return {
            "central_bank": cb_report,
            "government": gov_report,
            "total_intervention_damage": str(
                self.metrics.central_bank_damage + self.metrics.government_damage
            ),
            "freedom_index": self.metrics.freedom_index,
            "recommendation": self._get_recommendation(),
        }

    def _get_recommendation(self) -> str:
        """Get Austrian recommendation based on current state"""
        if self.metrics.freedom_index >= 90:
            return "Excellent! Near-optimal free market conditions."
        elif self.metrics.freedom_index >= 70:
            return "Good. Reduce remaining interventions for better outcomes."
        elif self.metrics.freedom_index >= 50:
            return "Moderate intervention. Significant improvement possible."
        elif self.metrics.freedom_index >= 30:
            return "High intervention. Major reforms needed."
        else:
            return "Critical! Extreme intervention destroying economy."

    # ===========================================
    # REPORTING
    # ===========================================

    def get_summary(self) -> Dict[str, Any]:
        """Get simulation summary"""
        # Business cycle signals
        cycle_signals = self.business_cycle._get_cycle_signals()

        return {
            "config": {
                "country": self.config.country,
                "num_persons": self.config.num_persons,
                "num_companies": self.config.num_companies,
                "regime": self.config.regime_type.value,
            },
            "state": self.state.value,
            "current_tick": self.current_tick,
            "metrics": self.metrics.to_dict(),
            "damage_summary": self.get_damage_summary(),
            "business_cycle": {
                "phase": cycle_signals["phase"],
                "boom_intensity": cycle_signals["boom_intensity"],
                "rate_distortion": cycle_signals["rate_distortion"],
                "credit_signal": cycle_signals["credit_signal"],
                "investment_signal": cycle_signals["investment_signal"],
                "recommendation": cycle_signals["recommendation"],
            },
            "labor_market": self.labor_market.get_statistics(),
        }
