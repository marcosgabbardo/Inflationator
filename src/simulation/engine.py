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
import copy
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
from src.data.real_world_conditions import (
    RealWorldConditionsCollector,
    RealWorldInitializer,
    EconomicConditions,
    print_conditions_summary,
)


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
    real_time_months: int = 0  # Changed from weeks to months

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
            "real_time_months": self.real_time_months,
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
    num_persons: int = 100_000  # 100k persons default
    num_companies: int = 10_000  # 10k companies default
    num_banks: int = 500
    regime_type: RegimeType = RegimeType.DEMOCRACY_LIBERAL
    central_bank_intervention: float = 0.5
    ticks_per_run: int = 12  # 1 year by default (12 months instead of 52 weeks)
    use_real_data: bool = True
    # Historical data (set from RealWorldConditionsCollector)
    historical_gdp_growth: float = 2.0  # 10-year average GDP growth %
    historical_unemployment: float = 7.0  # Austrian estimate (includes underemployment)


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

        # Real-world conditions (comprehensive economic state)
        self.real_conditions: Optional[EconomicConditions] = None
        self.conditions_collector = RealWorldConditionsCollector(self.config.country) if self.config.use_real_data else None
        self.real_world_initializer: Optional[RealWorldInitializer] = None

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
        Fetch comprehensive real-world economic conditions from APIs.

        Updates:
        - Bitcoin, Gold, commodities with real prices
        - Market sentiment and fear levels
        - Interest rates and yield curve
        - Dollar strength and volatility

        Austrian Theory:
        - Initial conditions must reflect TODAY's actual economy
        - Market prices contain distributed information (Hayek)
        - Government statistics lie; use market data instead
        """
        if not self.config.use_real_data:
            return

        def _set_market_price(market, price):
            """Helper to set price without creating artificial change."""
            if market and price:
                market.current_price = price
                market.previous_price = price  # No initial jump
                market.price_history = [price]  # Fresh history

        try:
            # Fetch comprehensive real-world conditions
            if self.conditions_collector:
                print("Fetching comprehensive real-world conditions...")
                self.real_conditions = asyncio.run(
                    self.conditions_collector.fetch_all_conditions()
                )

                # Create initializer for agent behavior
                self.real_world_initializer = RealWorldInitializer(
                    self.real_conditions, self.config.country
                )

                # Set market prices from conditions
                btc_market = self.market_manager.get_market_by_name("Bitcoin")
                _set_market_price(btc_market, self.real_conditions.btc_price_usd)
                self.external_data["bitcoin_usd"] = self.real_conditions.btc_price_usd

                gold_market = self.market_manager.get_market_by_name("Gold (oz)")
                _set_market_price(gold_market, self.real_conditions.gold_price_usd)
                self.external_data["gold_usd"] = self.real_conditions.gold_price_usd

                silver_market = self.market_manager.get_market_by_name("Silver (oz)")
                _set_market_price(silver_market, self.real_conditions.silver_price_usd)

                oil_market = self.market_manager.get_market_by_name("Oil (barrel)")
                _set_market_price(oil_market, self.real_conditions.oil_price_usd)

                # Store all conditions for reference
                self.external_data["real_conditions"] = {
                    "btc_price": float(self.real_conditions.btc_price_usd),
                    "gold_price": float(self.real_conditions.gold_price_usd),
                    "sp500": self.real_conditions.sp500_value,
                    "vix": self.real_conditions.vix_value,
                    "dxy": self.real_conditions.dxy_value,
                    "treasury_10y": self.real_conditions.treasury_10y,
                    "yield_inverted": self.real_conditions.yield_curve_inverted,
                    "fear_level": self.real_conditions.market_fear_level,
                    "sentiment": self.real_conditions.market_sentiment,
                    "inflation_estimate": self.real_conditions.inflation_estimate,
                    "recession_prob": self.real_conditions.recession_probability,
                    "geopolitical_risk": self.real_conditions.geopolitical_risk_level,
                }

                # Update config with historical data from real conditions
                self.config.historical_gdp_growth = self.real_conditions.historical_gdp_growth
                self.config.historical_unemployment = self.real_conditions.historical_unemployment

                # Print summary
                print_conditions_summary(self.real_conditions)

                # Apply conditions to central bank
                if self.central_bank:
                    self._apply_conditions_to_central_bank()

                # Apply conditions to agents (time preference, risk tolerance)
                self._apply_conditions_to_agents()

                print(f"  Real BTC price: ${self.real_conditions.btc_price_usd:,.0f}")

        except Exception as e:
            print(f"  Warning: Could not fetch real data: {e}")
            # Fallback to basic data fetch
            self._fetch_basic_real_data()

    def _fetch_basic_real_data(self):
        """Fallback: fetch basic price data if comprehensive fetch fails."""
        def _set_market_price(market, price):
            if market and price:
                market.current_price = price
                market.previous_price = price
                market.price_history = [price]

        try:
            if self.bitcoin_collector:
                btc_prices = asyncio.run(self.bitcoin_collector.get_current_price())
                if "usd" in btc_prices:
                    btc_market = self.market_manager.get_market_by_name("Bitcoin")
                    _set_market_price(btc_market, btc_prices["usd"])
                    self.external_data["bitcoin_usd"] = btc_prices["usd"]
                    print(f"  Real BTC price: ${btc_prices['usd']:,.0f}")

            if self.commodities_collector:
                comm_prices = self.commodities_collector.get_current_prices()
                if "gold" in comm_prices:
                    gold_market = self.market_manager.get_market_by_name("Gold (oz)")
                    _set_market_price(gold_market, comm_prices["gold"])
                if "silver" in comm_prices:
                    silver_market = self.market_manager.get_market_by_name("Silver (oz)")
                    _set_market_price(silver_market, comm_prices["silver"])
                if "oil" in comm_prices:
                    oil_market = self.market_manager.get_market_by_name("Oil (barrel)")
                    _set_market_price(oil_market, comm_prices["oil"])
        except Exception as e:
            print(f"  Warning: Fallback data fetch also failed: {e}")

    def _apply_conditions_to_central_bank(self):
        """
        Apply real-world conditions to central bank initial state.

        Uses market-derived interest rates instead of defaults.
        """
        if not self.real_conditions or not self.central_bank:
            return

        # Set policy rate based on market rates (not CB's fake target)
        # Use 2Y treasury as proxy for short-term rates
        market_rate = self.real_conditions.treasury_2y / 100
        self.central_bank.policy_rate = market_rate

        # Set inflation target (CB's stated goal, not reality)
        self.central_bank.inflation_target = 0.02  # Always claims 2%

        # Determine if QE/QT active based on conditions
        if self.real_conditions.monetary_expansion_signal == "loose":
            self.central_bank.start_qe(monthly_amount=Decimal("50000000000"))  # $50B/month
        elif self.real_conditions.monetary_expansion_signal == "tight":
            self.central_bank.start_qt(monthly_reduction=Decimal("30000000000"))  # $30B/month

    def _apply_conditions_to_agents(self):
        """
        Apply real-world conditions to agents' initial behavior.

        Austrian Theory:
        - Fear increases time preference (want things NOW)
        - High inflation expectations increase time preference
        - Market conditions affect risk tolerance
        """
        if not self.real_conditions:
            return

        # Calculate shifts from conditions collector
        collector = self.conditions_collector
        sim_config = collector.to_simulation_config(self.real_conditions)
        agent_modifiers = sim_config.get("agent_modifiers", {})

        fear_shift = agent_modifiers.get("fear_level", 0.5)
        time_pref_shift = agent_modifiers.get("time_preference_shift", 0)
        risk_shift = agent_modifiers.get("risk_tolerance_shift", 0)
        inflation_exp = agent_modifiers.get("inflation_expectations", 0.03)

        # Apply to persons
        for person in self.persons:
            # Shift time preference based on conditions
            person.time_preference = max(0.1, min(0.9,
                person.time_preference + time_pref_shift
            ))

            # Shift risk tolerance
            person.risk_tolerance = max(0.1, min(0.9,
                person.risk_tolerance + risk_shift
            ))

            # Set inflation expectations
            person.inflation_expectation = inflation_exp

        # Apply to companies
        for company in self.companies:
            # Companies also affected by conditions
            company.time_preference = max(0.1, min(0.9,
                company.time_preference + time_pref_shift * 0.5  # Less reactive
            ))

        # Update employment rate based on conditions
        if self.real_world_initializer:
            target_employment = self.real_world_initializer.get_employment_rate()
            current_employed = sum(1 for p in self.persons if p.employed)
            current_rate = current_employed / len(self.persons)

            # Adjust if significantly different
            if abs(current_rate - target_employment) > 0.05:
                print(f"  Adjusting employment: {current_rate:.1%} -> {target_employment:.1%}")

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

        # Initialize employment with historical unemployment rate
        # Convert historical unemployment (%) to employment rate (1 - unemployment)
        # Historical unemployment already reflects Austrian analysis (includes underemployment)
        historical_unemployment_rate = self.config.historical_unemployment / 100
        initial_employment = max(0.5, min(0.95, 1 - historical_unemployment_rate))

        result = self.labor_market.initialize_employment(
            persons=self.persons,
            companies=self.companies,
            initial_employment_rate=initial_employment
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

        Each tick represents 1 month of economic activity.
        """
        ticks = ticks or self.config.ticks_per_run
        self.state = SimulationState.RUNNING

        print(f"\nRunning simulation for {ticks} months...")

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
        # Note: Real data is only fetched at initialization, not during simulation
        # This allows the simulation to evolve prices naturally based on:
        # - Supply/demand dynamics
        # - Monetary effects (Central Bank money printing)
        # - Business cycle phases
        # - Tariff effects (Government trade policy)
        # - Political business cycle (election year easing)
        # Real-world prices serve as initial anchor, not continuous override

        # Build world state for agents
        world_state = self._build_world_state()

        # 1. Government actions (fiscal distortions) - FIRST to get election easing
        if self.government:
            gov_actions = self.government.step(world_state)

            # Check for political pressure on Central Bank (election year)
            cb_pressure = self.government.pressure_central_bank(
                self.central_bank, world_state
            )
            if cb_pressure:
                world_state["political_pressure"] = cb_pressure

        # 2. Central Bank actions (monetary distortions)
        if self.central_bank:
            # Apply political pressure if election year
            if world_state.get("political_pressure"):
                pressure_level = world_state["political_pressure"]["pressure_level"]
                # CB succumbs to pressure - lowers rates more than needed
                # Austrian view: This is why CB "independence" is a myth
                self.central_bank.state.rate_cuts_cumulative += pressure_level * 0.002

            cb_actions = self.central_bank.step(world_state)

        # 3. Banks process (credit creation)
        for bank in self.banks:
            bank.step(world_state)

        # 4. Apply monetary effects to prices (Austrian: inflation is monetary)
        self._apply_monetary_effects()

        # 5. Apply tariff effects to prices (Government trade policy)
        self._apply_tariff_effects()

        # 6. Companies produce and submit sell orders
        self._process_company_orders(world_state)

        # 7. Persons consume, save, invest - submit buy orders
        self._process_person_orders(world_state)

        # 8. Labor market dynamics
        self._process_labor_market()

        # 9. Credit system - loans and deposits
        self._process_credit_system(world_state)

        # 10. Clear all markets and discover prices
        trades = self.market_manager.clear_all_markets(self.current_tick)

        # 11. Clear order books for next tick (fresh orders each month)
        self._clear_order_books()

        # 12. Update Austrian Business Cycle
        self._update_business_cycle(world_state)

        # 13. Update metrics
        self._update_metrics()

        # Store metrics history
        self.metrics_history.append(SimulationMetrics(
            tick=self.metrics.tick,
            real_time_months=self.metrics.real_time_months,
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

        # Progress indicator (only for single-country runs, multi-country has its own)
        # Suppress if running as part of multi-country simulation
        if not hasattr(self, '_multi_country_mode') and self.current_tick % 6 == 0:
            print(f"  Month {self.current_tick}: Inflation={self.metrics.inflation_rate:.1%} (annualized), "
                  f"BTC=${self.metrics.bitcoin_price:,.0f}")

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
            # Use historical GDP growth rate from real-world data
            "gdp_growth": self.config.historical_gdp_growth / 100,  # Convert % to decimal
            "historical_unemployment": self.config.historical_unemployment / 100,
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

        # Calculate supply pressure (more companies = more supply = lower prices)
        supply_pressure = len(sampled_companies) / max(1, self.config.num_companies)

        for company in sampled_companies:
            # Let company make its decisions
            company.step(world_state)

            # Submit sell orders based on production
            production_value = company.capital_stock * Decimal("0.08")  # Monthly output (~4x weekly)

            if production_value > 0:
                # Choose market based on production type
                if company.production_type == "consumer_goods":
                    target_markets = consumer_markets
                else:
                    target_markets = capital_markets if capital_markets else consumer_markets

                if target_markets:
                    market = random.choice(target_markets)

                    # Calculate quantity and price with VARIABILITY
                    quantity = production_value / market.current_price

                    # Price based on company's situation:
                    # - Lower margin if high supply pressure
                    # - Higher margin if company is profitable
                    # - Random variability for market dynamics
                    base_margin = 0.95  # Base: sell at 95% of market

                    # Adjust based on supply pressure (more supply = compete harder)
                    supply_adjustment = -0.05 * supply_pressure  # Up to -5%

                    # Adjust based on company profitability (profitable = less desperate)
                    profit_adjustment = 0.03 if company.wealth > Decimal("10000") else -0.02

                    # Random variability for realistic market dynamics (smaller for stability)
                    random_adjustment = random.uniform(-0.03, 0.02)

                    price_factor = base_margin + supply_adjustment + profit_adjustment + random_adjustment
                    price_factor = max(0.90, min(1.02, price_factor))  # Tighter clamp to 90%-102%

                    min_price = market.current_price * Decimal(str(price_factor))

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

        # Demand pressure (more buyers = higher prices)
        demand_pressure = len(sampled_persons) / max(1, self.config.num_persons)

        # Inflation expectations affect willingness to pay
        inflation_expectation = world_state.get("inflation_rate", 0.0)

        for person in sampled_persons:
            # Let person make decisions
            person.step(world_state)

            # Skip if no wealth
            if person.wealth <= 0:
                continue

            # Calculate monthly budget (4x weekly)
            savings_rate = person.calculate_savings_rate()
            monthly_income = (person.wage if person.employed else person.wealth * Decimal("0.01")) * Decimal("4")
            consumption_budget = monthly_income * Decimal(str(1 - savings_rate))
            investment_budget = monthly_income * Decimal(str(savings_rate * 0.3))  # Part of savings goes to assets

            # Submit buy orders for consumer goods with MONTE CARLO VARIABILITY
            if consumption_budget > 0 and consumer_markets:
                market = random.choice(consumer_markets)
                quantity = consumption_budget / market.current_price

                # Price willingness based on:
                # - Base: willing to pay above market
                # - Higher time preference = more willing to pay now
                # - Higher demand pressure = compete harder
                # - Higher inflation expectation = buy now before prices rise
                # - Monte Carlo random variability
                base_premium = 1.02

                # High time preference = pay more now
                time_pref_adjustment = person.time_preference * 0.05  # Up to +5%

                # Demand pressure
                demand_adjustment = demand_pressure * 0.03  # Up to +3%

                # Inflation expectation (Austrian: expect money to lose value)
                inflation_adjustment = min(0.10, inflation_expectation * 0.5)  # Up to +10%

                # Monte Carlo random variability (smaller for stability)
                random_adjustment = random.gauss(0, 0.015)  # Normal distribution, σ=1.5%

                price_factor = base_premium + time_pref_adjustment + demand_adjustment + inflation_adjustment + random_adjustment
                price_factor = max(0.98, min(1.08, price_factor))  # Tighter clamp 98%-108%

                max_price = market.current_price * Decimal(str(price_factor))

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

                # Higher allocation if expecting inflation (Bitcoin as hedge)
                if inflation_expectation > 0.05:
                    btc_allocation *= (1 + inflation_expectation)

                if btc_allocation > 100:  # Minimum investment
                    quantity = Decimal(str(btc_allocation)) / bitcoin_market.current_price

                    # Monte Carlo for BTC price willingness
                    # Crypto buyers more volatile but still reasonable
                    btc_premium = 1.01 + random.gauss(0, 0.03)  # Moderate volatility
                    btc_premium = max(0.95, min(1.10, btc_premium))
                    max_price = bitcoin_market.current_price * Decimal(str(btc_premium))

                    bitcoin_market.submit_buy_order(
                        agent_id=person.id,
                        quantity=quantity,
                        max_price=max_price,
                        tick=self.current_tick
                    )

            # Investment in Gold (more conservative)
            if investment_budget > 0 and gold_market:
                gold_allocation = (1 - person.risk_tolerance) * float(investment_budget) * 0.3

                # Higher allocation if expecting inflation (Gold as hedge)
                if inflation_expectation > 0.05:
                    gold_allocation *= (1 + inflation_expectation * 0.5)

                if gold_allocation > 50:  # Minimum investment
                    quantity = Decimal(str(gold_allocation)) / gold_market.current_price

                    # Monte Carlo for Gold price willingness
                    # Gold buyers more conservative
                    gold_premium = 1.005 + random.gauss(0, 0.01)  # Very low volatility
                    gold_premium = max(0.99, min(1.03, gold_premium))
                    max_price = gold_market.current_price * Decimal(str(gold_premium))

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
                                term_months=12,  # 1 year loan
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

    def _apply_monetary_effects(self):
        """
        Apply monetary effects to all prices based on CB policy.

        Austrian Theory (Mises, Hayek):
        - Inflation is ALWAYS a monetary phenomenon
        - More money chasing same goods = higher prices
        - New money enters economy unevenly (Cantillon effect)
        - Central bank money printing is the root cause of inflation

        Key insight for simulation:
        - During QE: BTC/Gold appreciate as people flee fiat
        - During QT: BTC/Gold stabilize, consumer prices drop
        - The "controlled" inflation is the illusion
        """
        if not self.central_bank:
            return

        # Calculate money supply growth rate
        current_money = self.central_bank.state.base_money
        previous_money = getattr(self, '_previous_money_supply', current_money)

        if previous_money > 0:
            money_growth_rate = float((current_money - previous_money) / previous_money)
        else:
            money_growth_rate = 0.0

        self._previous_money_supply = current_money

        # Get current monetary policy stance
        policy = self.central_bank.current_policy
        qe_active = self.central_bank.qe_active
        qt_active = self.central_bank.qt_active

        # ===========================================
        # DIFFERENT EFFECTS BY POLICY STANCE
        # ===========================================
        # Tuned for realistic inflation dynamics (MONTHLY):
        # - Annual inflation target: 2-3%
        # - Monthly equivalent: ~0.17-0.25% base (2-3% / 12)
        # - QE effects are gradual, not immediate
        # - Hard assets (BTC, Gold) react more, but still realistic

        # Dampen money growth rate to monthly equivalent
        # CB prints money gradually, not all at once
        # For realistic inflation: 2-5% annual = 0.17-0.42% monthly
        # More conservative: only 3% of money growth affects prices per month
        monthly_money_effect = money_growth_rate * 0.03  # 3% of growth affects prices per month
        # Cap the effect to prevent runaway inflation in early ticks
        monthly_money_effect = max(-0.02, min(0.02, monthly_money_effect))  # ±2% max per month

        for market in self.market_manager.markets.values():
            old_price = market.current_price
            market.previous_price = old_price

            # CRYPTO: Uses presidential cycle as PRIMARY driver (independent of monetary policy)
            if market.market_type == MarketType.CRYPTO:
                # Bitcoin: Follows US Presidential Cycle
                # Research shows BTC correlates with traditional market cycles:
                # - Year 1 (new president): Bull run - liquidity optimism
                # - Year 2: Bear market - policy uncertainty, tightening
                # - Year 3: Recovery - mid-term clarity
                # - Year 4 (election): Moderate bull - election stimulus

                # Get presidential cycle year (1-4)
                cycle_year = 2  # Default to year 2 (bear) if no government
                if self.government:
                    # current_year_in_cycle is 0-4, convert to 1-4
                    cycle_year = int(self.government.current_year_in_cycle) + 1
                    if cycle_year > 4:
                        cycle_year = 1


                if cycle_year == 1:
                    # YEAR 1: BULL RUN (new president)
                    # Historical: ~80-150% annual = 5-8% monthly
                    btc_monthly = 0.05 + random.gauss(0, 0.025)
                    btc_monthly = max(0.0, min(0.08, btc_monthly))  # 0-8% (never negative in bull)
                elif cycle_year == 2:
                    # YEAR 2: BEAR MARKET
                    # Historical: -20% to -40% annual = -2% to -4% monthly
                    btc_monthly = -0.025 + random.gauss(0, 0.015)
                    btc_monthly = max(-0.05, min(0.0, btc_monthly))  # -5% to 0% (never positive in bear)
                elif cycle_year == 3:
                    # YEAR 3: RECOVERY
                    # Historical: +20-40% annual = 1.5-3% monthly
                    btc_monthly = 0.02 + random.gauss(0, 0.015)
                    btc_monthly = max(-0.02, min(0.035, btc_monthly))  # -2% to +3.5%
                else:  # cycle_year == 4
                    # YEAR 4: ELECTION YEAR (moderate bull)
                    # Historical: +30-50% annual = 2-4% monthly
                    btc_monthly = 0.025 + random.gauss(0, 0.015)
                    btc_monthly = max(-0.01, min(0.04, btc_monthly))  # -1% to +4%

                # QE/QT modifies the cycle effect slightly
                if qe_active:
                    btc_monthly += 0.005  # QE adds 0.5% boost
                elif qt_active:
                    btc_monthly -= 0.005  # QT subtracts 0.5%

                price_factor = Decimal(str(1 + btc_monthly))
                market.current_price = old_price * price_factor
                market.price_history.append(market.current_price)
                if len(market.price_history) > 1000:
                    market.price_history.pop(0)
                continue  # Skip the rest for crypto

            # OTHER MARKETS: Use monetary policy as driver
            if monthly_money_effect > 0:
                # EXPANSION: Prices rise (but controlled by CB)
                # Consumer goods: dampened effect (the "controlled" inflation)
                # Hard assets: amplified effect (people flee to safety)

                if market.market_type == MarketType.COMMODITIES:
                    # Gold/Silver: Also benefit from QE (traditional safe haven)
                    # Realistic: ~10-20% annual appreciation during QE
                    if qe_active:
                        gold_multiplier = 0.8 + random.gauss(0, 0.1)
                        price_factor = Decimal(str(1 + monthly_money_effect * gold_multiplier))
                    else:
                        price_factor = Decimal(str(1 + monthly_money_effect * 0.4))

                else:
                    # Consumer goods: CB tries to control this (the "2% target")
                    # Dampened pass-through to maintain the illusion
                    damping = 0.2 if qt_active else 0.4
                    price_factor = Decimal(str(1 + monthly_money_effect * damping))

                # Monte Carlo variability (smaller for stability)
                price_factor *= Decimal(str(random.uniform(0.995, 1.005)))

            elif monthly_money_effect < 0:
                # CONTRACTION (QT): Prices should fall, but...
                # Consumer goods: sticky downward (wages don't fall)
                # Hard assets: can fall during QT
                # Note: CRYPTO is handled separately above via continue

                if market.market_type == MarketType.COMMODITIES:
                    # Gold more stable during QT
                    gold_factor = 1 + monthly_money_effect * 0.5
                    price_factor = Decimal(str(max(0.98, gold_factor)))

                else:
                    # Consumer goods: sticky - don't fall much
                    # This is why QT "works" to control measured inflation
                    price_factor = Decimal(str(max(0.995, 1 + monthly_money_effect * 0.1)))

            else:
                # Neutral monetary policy: small random walk for non-crypto markets
                # Note: CRYPTO is handled separately above via continue
                price_factor = Decimal(str(1 + random.gauss(0, 0.001)))

            # Apply price change
            market.current_price = old_price * price_factor

            # Track in history
            market.price_history.append(market.current_price)
            if len(market.price_history) > 1000:
                market.price_history.pop(0)

    def _apply_tariff_effects(self):
        """
        Apply government tariff effects to prices.

        Austrian/Free Trade Theory:
        - Tariffs are taxes on consumers (not foreign producers)
        - Raise prices of imported goods
        - Cause supply chain disruptions
        - Invite retaliation (trade wars)
        - Protect inefficient domestic producers at consumer expense

        Current example: Trump tariffs raising prices on consumer goods
        """
        if not self.government:
            return

        # Skip if no significant tariffs
        if self.government.tariff_rate < 0.01:
            return

        # Get tariff price impact multipliers
        tariff_impacts = self.government.calculate_tariff_price_impact()

        for market in self.market_manager.markets.values():
            # Determine which category this market falls into
            if market.market_type == MarketType.CONSUMER_GOODS:
                impact_multiplier = tariff_impacts.get("consumer_goods", 1.0)
            elif market.market_type == MarketType.CAPITAL_GOODS:
                impact_multiplier = tariff_impacts.get("capital_goods", 1.0)
            elif market.market_type == MarketType.COMMODITIES:
                impact_multiplier = tariff_impacts.get("commodities", 1.0)
            elif market.market_type == MarketType.CRYPTO:
                # Crypto unaffected by tariffs (stateless money)
                impact_multiplier = 1.0
            else:
                impact_multiplier = 1.0

            # Apply tariff effect if significant
            if impact_multiplier > 1.001:
                old_price = market.current_price

                # Tariff effect is gradual (pass-through takes time)
                # Each tick applies a portion of the tariff impact
                monthly_impact = 1 + (impact_multiplier - 1) * 0.3  # 30% pass-through per month (~10%/week * 4)

                # Add some randomness for realistic market dynamics
                monthly_impact *= random.uniform(0.98, 1.02)

                market.current_price = old_price * Decimal(str(monthly_impact))

                # Track trade disruption damage
                if impact_multiplier > 1.05:  # Significant tariff
                    disruption = (market.current_price - old_price) * market.volume
                    self.government.state.trade_disruption += disruption

    def _clear_order_books(self):
        """
        Clear all order books for the next tick.

        Each tick represents a new trading month with fresh orders.
        Stale orders don't carry over.
        """
        for market in self.market_manager.markets.values():
            market.order_book.clear()

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
        self.metrics.real_time_months = self.current_tick

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
            # Use total_damage_caused which includes ALL damage types:
            # deadweight_loss, compliance_costs, capital_destroyed,
            # trade_disruption, and spending_waste
            self.metrics.government_damage = self.government.total_damage_caused
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

        elif scenario_name == "trade_war":
            # Simulate trade war (like US-China or Trump tariffs)
            tariff_rate = parameters.get("tariff_rate", 0.25)  # 25% default
            if self.government:
                self.government.initiate_trade_war(target_rate=tariff_rate)
                print(f"Scenario: Trade War initiated with {tariff_rate:.0%} tariffs")

        elif scenario_name == "trump_tariffs":
            # Simulate current Trump tariff policy (2025+)
            if self.government:
                # General tariffs on imports
                self.government.set_tariff_rate(0.20, mode="protectionist")
                # Sector-specific tariffs
                self.government.set_sector_tariff("electronics", 0.35)
                self.government.set_sector_tariff("steel", 0.50)
                self.government.set_sector_tariff("automotive", 0.25)
                print("Scenario: Trump-style tariffs activated")
                print(f"  - General: 20%, Electronics: 35%, Steel: 50%, Auto: 25%")

        elif scenario_name == "election_year":
            # Force election year dynamics
            if self.government:
                self.government.current_year_in_cycle = 3.5  # 6 months before election
                self.government.is_election_year = True
                self.government.election_easing_active = True
                print("Scenario: Election year activated - expect fiscal easing")

        elif scenario_name == "hyperinflation":
            # Central bank prints massively
            if self.central_bank:
                current_base = self.central_bank.state.base_money
                # Print 10x the money supply
                for _ in range(10):
                    self.central_bank.print_money(current_base)
                print(f"Scenario: Hyperinflation - money supply increased 10x")

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


# ===========================================
# MULTI-COUNTRY SIMULATION ENGINE
# ===========================================

@dataclass
class MultiCountryConfig:
    """Configuration for multi-country simulation"""
    countries: List[str] = field(default_factory=lambda: ["USA"])
    base_num_persons: int = 100_000  # USA baseline, others scaled by GDP
    base_num_companies: int = 10_000
    base_num_banks: int = 500
    ticks_per_run: int = 12  # 1 year
    use_real_data: bool = True
    scale_by_gdp: bool = True  # Scale agent count by GDP


@dataclass
class MultiCountryMetrics:
    """Metrics for multi-country simulation"""
    tick: int = 0
    country_metrics: Dict[str, SimulationMetrics] = field(default_factory=dict)
    global_metrics: Dict[str, Any] = field(default_factory=dict)
    war_risks: List[Dict[str, Any]] = field(default_factory=list)
    influence_effects: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "countries": {k: v.to_dict() for k, v in self.country_metrics.items()},
            "global": self.global_metrics,
            "war_risks": self.war_risks,
            "influence_effects": self.influence_effects,
        }


class MultiCountrySimulationEngine:
    """
    Multi-country simulation engine.

    Orchestrates multiple country simulations with:
    - Geopolitical relationships
    - Trade flows and tariffs
    - Monetary influence transmission
    - War probability tracking
    - Crisis contagion

    Austrian Theory Application:
    - Trade creates interdependence (peace through commerce)
    - Monetary policy spills over internationally
    - Government interventions affect trading partners
    - Wars destroy capital and increase state power
    """

    def __init__(self, config: Optional[MultiCountryConfig] = None):
        from src.countries.registry import get_country_config, get_all_countries
        from src.geopolitics.relationships import RelationshipManager
        from src.geopolitics.war_probability import WarProbabilityCalculator
        from src.geopolitics.influence import InfluenceCalculator
        from src.data.collectors.country_collector import MultiCountryCollector

        self.config = config or MultiCountryConfig()
        self.state = SimulationState.IDLE
        self.current_tick = 0

        # Country engines (one per country)
        self.country_engines: Dict[str, SimulationEngine] = {}

        # Geopolitics
        self.relationship_manager = RelationshipManager()
        self.war_calculator = WarProbabilityCalculator()
        self.influence_calculator = InfluenceCalculator()

        # Data collectors
        self.data_collector = MultiCountryCollector(self.config.countries)

        # Metrics
        self.metrics = MultiCountryMetrics()
        self.metrics_history: List[MultiCountryMetrics] = []

        # Country configs
        self._country_configs: Dict[str, Any] = {}
        for country in self.config.countries:
            try:
                self._country_configs[country] = get_country_config(country)
            except Exception:
                pass

    def initialize(self):
        """Initialize all country simulations"""
        from src.countries.registry import get_country_config
        from src.data.real_world_conditions import RealWorldConditionsCollector

        print(f"Initializing multi-country simulation for {len(self.config.countries)} countries...")

        for country in self.config.countries:
            print(f"\n--- Initializing {country} ---")

            # Get country config for regime and scale
            country_config = self._country_configs.get(country)

            # Get historical data for this country
            conditions_collector = RealWorldConditionsCollector(country)
            historical_gdp_growth = conditions_collector.HISTORICAL_GDP_GROWTH.get(country, 2.0)
            historical_unemployment = conditions_collector.HISTORICAL_UNEMPLOYMENT.get(country, 7.0)

            # Calculate scale based on GDP (USA = 1.0)
            if self.config.scale_by_gdp and country_config:
                usa_gdp = Decimal("27000000000000")  # $27T
                country_gdp = country_config.gdp_nominal_usd
                scale = float(country_gdp / usa_gdp)
                scale = max(0.1, min(1.0, scale))  # Clamp between 10% and 100%
            else:
                scale = 1.0

            # Create simulation config for this country with historical data
            sim_config = SimulationConfig(
                country=country,
                num_persons=int(self.config.base_num_persons * scale),
                num_companies=int(self.config.base_num_companies * scale),
                num_banks=max(10, int(self.config.base_num_banks * scale)),
                regime_type=country_config.regime_type if country_config else RegimeType.DEMOCRACY_LIBERAL,
                central_bank_intervention=country_config.intervention_level if country_config else 0.5,
                ticks_per_run=self.config.ticks_per_run,
                use_real_data=self.config.use_real_data,
                historical_gdp_growth=historical_gdp_growth,
                historical_unemployment=historical_unemployment,
            )

            # Create and initialize engine
            engine = SimulationEngine(sim_config)
            engine._multi_country_mode = True  # Suppress individual progress output
            engine.initialize()
            self.country_engines[country] = engine

            # Print historical data used
            print(f"  Historical GDP Growth: {historical_gdp_growth:.1f}%")
            print(f"  Historical Unemployment: {historical_unemployment:.1f}%")

        print(f"\nMulti-country simulation initialized with {len(self.country_engines)} countries")
        self.state = SimulationState.IDLE

    def run(self, ticks: Optional[int] = None) -> List[MultiCountryMetrics]:
        """
        Run multi-country simulation.

        Each tick:
        1. Calculate inter-country influences
        2. Run each country's simulation step
        3. Apply cross-border effects
        4. Update war probabilities
        5. Track global metrics
        """
        ticks = ticks or self.config.ticks_per_run
        self.state = SimulationState.RUNNING

        print(f"\nRunning multi-country simulation for {ticks} months...")

        for _ in range(ticks):
            if self.state != SimulationState.RUNNING:
                break

            self._run_tick()
            self.current_tick += 1

        self.state = SimulationState.COMPLETED
        return self.metrics_history

    def _run_tick(self):
        """Execute one multi-country simulation tick"""
        # 1. Calculate influence matrix
        influence_effects = self._calculate_influences()

        # 2. Run each country's tick with influences applied
        for country, engine in self.country_engines.items():
            # Apply external influences to world state
            world_state = engine._build_world_state()
            world_state["external_influences"] = influence_effects.get(country, {})

            # Sync tick counter with multi-country engine
            engine.current_tick = self.current_tick

            # Run the tick
            engine._run_tick()

        # 3. Process international trade
        self._process_international_trade()

        # 4. Update war probabilities
        war_risks = self._update_war_probabilities()

        # 5. Apply crisis contagion if any country in crisis
        self._check_crisis_contagion()

        # 6. Update metrics
        self._update_metrics(war_risks, influence_effects)

        # Progress indicator
        if self.current_tick % 3 == 0:
            self._print_progress()

    def _calculate_influences(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate how each country influences others this tick.

        Returns:
            Dict[target_country][influence_type] = magnitude
        """
        influences: Dict[str, Dict[str, float]] = {c: {} for c in self.config.countries}

        # Get relationships for influence calculation
        relationships = {}
        for country_a in self.config.countries:
            for country_b in self.config.countries:
                if country_a != country_b:
                    rel = self.relationship_manager.get_relationship(country_a, country_b)
                    if rel:
                        relationships[(country_a, country_b)] = rel

        # Calculate influence matrix
        matrix = self.influence_calculator.calculate_influence_matrix(
            self.config.countries,
            relationships
        )

        # Convert to per-country influences
        for target in self.config.countries:
            total_influence = 0.0
            for source in self.config.countries:
                if source != target:
                    total_influence += matrix.get(source, {}).get(target, 0.0)
            influences[target]["total_external"] = total_influence

        return influences

    def _process_international_trade(self):
        """
        Process trade between countries.

        Austrian insight:
        - Free trade benefits all parties
        - Tariffs harm consumers
        - Sanctions disrupt markets
        """
        # For each country pair with trade
        for country_a in self.config.countries:
            for country_b in self.config.countries:
                if country_a >= country_b:  # Avoid double processing
                    continue

                rel = self.relationship_manager.get_relationship(country_a, country_b)
                if not rel:
                    continue

                # Apply tariff effects
                if rel.tariff_a_to_b > 0 or rel.tariff_b_to_a > 0:
                    # Tariffs reduce trade and raise prices
                    avg_tariff = (rel.tariff_a_to_b + rel.tariff_b_to_a) / 2

                    # Reduce trade volume by tariff effect
                    trade_reduction = 1 - (avg_tariff * 0.5)  # 50% tariff = 25% trade reduction
                    rel.trade_volume_usd *= Decimal(str(trade_reduction))

                    # Track damage in both governments
                    engine_a = self.country_engines.get(country_a)
                    engine_b = self.country_engines.get(country_b)

                    if engine_a and engine_a.government:
                        engine_a.government.state.trade_disruption += rel.trade_volume_usd * Decimal(str(avg_tariff * 0.01))
                    if engine_b and engine_b.government:
                        engine_b.government.state.trade_disruption += rel.trade_volume_usd * Decimal(str(avg_tariff * 0.01))

                # Sanctions effect
                if rel.has_active_sanctions:
                    # Severe trade reduction
                    rel.trade_volume_usd *= Decimal("0.95")  # 5% reduction per tick

    def _update_war_probabilities(self) -> List[Dict[str, Any]]:
        """
        Update war probabilities for all country pairs.

        Returns list of high-risk pairs for display.
        """
        high_risk = []
        seen = set()

        for country_a in self.config.countries:
            for country_b in self.config.countries:
                if country_a >= country_b:
                    continue

                pair = tuple(sorted([country_a, country_b]))
                if pair in seen:
                    continue
                seen.add(pair)

                rel = self.relationship_manager.get_relationship(country_a, country_b)
                if not rel:
                    continue

                # Recalculate war probability
                assessment = self.war_calculator.calculate_war_probability(rel)

                # Update relationship
                rel.war_probability = assessment.probability

                # Track high risk pairs
                if assessment.probability >= 0.03:  # 3%+
                    high_risk.append({
                        "countries": [country_a, country_b],
                        "probability": assessment.probability,
                        "risk_level": assessment.risk_level,
                        "triggers": [t.value for t in assessment.primary_triggers[:2]],
                    })

        return sorted(high_risk, key=lambda x: x["probability"], reverse=True)

    def _check_crisis_contagion(self):
        """
        Check if any country is in crisis and propagate effects.

        Austrian insight:
        - Crises spread through interconnected markets
        - Bailouts create moral hazard
        - Sound money countries more resilient
        """
        for country, engine in self.country_engines.items():
            # Check for crisis conditions
            is_crisis = False

            if engine.metrics.inflation_rate > 0.5:  # 50%+ inflation
                is_crisis = True
            if engine.metrics.unemployment_rate > 0.2:  # 20%+ unemployment
                is_crisis = True

            if is_crisis:
                # Propagate crisis to connected countries
                relationships = {}
                for other in self.config.countries:
                    if other != country:
                        rel = self.relationship_manager.get_relationship(country, other)
                        if rel:
                            relationships[(country, other)] = rel

                # Simulate shock propagation
                impacts = self.influence_calculator.simulate_shock_propagation(
                    source=country,
                    shock_magnitude=0.5,  # 50% crisis
                    countries=self.config.countries,
                    relationships=relationships,
                )

                # Apply impacts to other countries
                for target, impact in impacts.items():
                    if target != country and impact > 0.05:  # 5%+ impact
                        target_engine = self.country_engines.get(target)
                        if target_engine:
                            # Increase fear/time preference
                            for person in target_engine.persons[:100]:
                                person.time_preference = min(0.9, person.time_preference + impact * 0.1)

    def _update_metrics(
        self,
        war_risks: List[Dict[str, Any]],
        influences: Dict[str, Dict[str, float]]
    ):
        """Update multi-country metrics"""
        self.metrics.tick = self.current_tick
        self.metrics.war_risks = war_risks

        # Collect metrics from each country (deep copy to preserve history)
        for country, engine in self.country_engines.items():
            # Create a snapshot of the metrics, not a reference
            self.metrics.country_metrics[country] = copy.deepcopy(engine.metrics)

        # Calculate global metrics
        total_gdp = sum(e.metrics.gdp for e in self.country_engines.values())
        avg_inflation = sum(e.metrics.inflation_rate for e in self.country_engines.values()) / len(self.country_engines)
        avg_freedom = sum(e.metrics.freedom_index for e in self.country_engines.values()) / len(self.country_engines)
        total_cb_damage = sum(e.metrics.central_bank_damage for e in self.country_engines.values())
        total_gov_damage = sum(e.metrics.government_damage for e in self.country_engines.values())

        self.metrics.global_metrics = {
            "total_gdp": str(total_gdp),
            "avg_inflation": avg_inflation,
            "avg_freedom_index": avg_freedom,
            "total_cb_damage": str(total_cb_damage),
            "total_gov_damage": str(total_gov_damage),
            "high_war_risk_pairs": len(war_risks),
        }

        # Store influence effects
        self.metrics.influence_effects = {
            k: v.get("total_external", 0.0)
            for k, v in influences.items()
        }

        # Store history (deep copy to ensure historical values are preserved)
        self.metrics_history.append(MultiCountryMetrics(
            tick=self.metrics.tick,
            country_metrics=copy.deepcopy(self.metrics.country_metrics),
            global_metrics=self.metrics.global_metrics.copy(),
            war_risks=self.metrics.war_risks.copy() if self.metrics.war_risks else [],
            influence_effects=self.metrics.influence_effects.copy() if self.metrics.influence_effects else {},
        ))

    def _print_progress(self):
        """Print progress indicator"""
        print(f"\n--- Month {self.current_tick} ---")
        for country, engine in list(self.country_engines.items())[:5]:  # Top 5
            print(f"  {country}: Inflation={engine.metrics.inflation_rate:.1%}, "
                  f"Freedom={engine.metrics.freedom_index:.0f}")

        if self.metrics.war_risks:
            highest = self.metrics.war_risks[0]
            print(f"  Highest war risk: {highest['countries'][0]}-{highest['countries'][1]} "
                  f"({highest['probability']:.1%})")

    def get_summary(self) -> Dict[str, Any]:
        """Get multi-country simulation summary"""
        country_summaries = {}
        for country, engine in self.country_engines.items():
            country_summaries[country] = {
                "inflation": engine.metrics.inflation_rate,
                "unemployment": engine.metrics.unemployment_rate,
                "freedom_index": engine.metrics.freedom_index,
                "cb_damage": str(engine.metrics.central_bank_damage),
                "gov_damage": str(engine.metrics.government_damage),
                "btc_price": str(engine.metrics.bitcoin_price),
            }

        return {
            "config": {
                "countries": self.config.countries,
                "ticks": self.current_tick,
            },
            "state": self.state.value,
            "countries": country_summaries,
            "global_metrics": self.metrics.global_metrics,
            "war_risks": self.metrics.war_risks[:5],  # Top 5
            "relationship_summary": self.relationship_manager.get_summary(),
        }

    def get_war_risk_report(self) -> Dict[str, Any]:
        """Get detailed war risk report"""
        relationships = {}
        for country_a in self.config.countries:
            for country_b in self.config.countries:
                if country_a != country_b:
                    rel = self.relationship_manager.get_relationship(country_a, country_b)
                    if rel:
                        relationships[(country_a, country_b)] = rel

        global_risk = self.war_calculator.get_global_war_risk(relationships)

        return {
            "global_risk": global_risk,
            "high_risk_pairs": self.metrics.war_risks,
            "sanctioned_pairs": self.relationship_manager.get_sanctioned_pairs(),
        }

    def compare_countries(
        self,
        countries: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Compare metrics across countries"""
        countries = countries or list(self.country_engines.keys())

        comparison = {}
        for country in countries:
            engine = self.country_engines.get(country)
            if engine:
                comparison[country] = {
                    "regime": engine.config.regime_type.value,
                    "freedom_index": engine.metrics.freedom_index,
                    "inflation": engine.metrics.inflation_rate,
                    "unemployment": engine.metrics.unemployment_rate,
                    "cb_damage": float(engine.metrics.central_bank_damage),
                    "gov_damage": float(engine.metrics.government_damage),
                    "btc_price": float(engine.metrics.bitcoin_price),
                }

        # Sort by freedom index (Austrian preference)
        sorted_comparison = dict(sorted(
            comparison.items(),
            key=lambda x: x[1]["freedom_index"],
            reverse=True
        ))

        return sorted_comparison
