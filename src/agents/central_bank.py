"""
Central Bank Agent - THE MONETARY VILLAIN

The central bank causes economic distortions through:
1. Artificial interest rate manipulation
2. Money printing (Quantitative Easing)
3. Purchasing toxic assets
4. Bailouts of "too big to fail" institutions

Austrian Theory (Mises, Hayek, Rothbard):
- Central banks cause business cycles through credit expansion
- Artificial low rates cause malinvestment
- Money printing is theft through inflation
- No central planning can replace market price signals
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Any, List, Optional
from enum import Enum
import random

from .base import Agent, AgentType, AgentState


class MonetaryPolicy(str, Enum):
    """Current monetary policy stance"""
    TIGHT = "tight"           # QT - reducing money supply
    NEUTRAL = "neutral"       # Stable
    EASY = "easy"             # QE - expanding money supply
    EMERGENCY = "emergency"   # Crisis mode - massive QE


@dataclass
class CentralBankState(AgentState):
    """State for central bank villain"""
    base_money: Decimal = Decimal("0")
    money_printed: Decimal = Decimal("0")
    money_destroyed: Decimal = Decimal("0")  # QT
    assets_purchased: Decimal = Decimal("0")
    assets_sold: Decimal = Decimal("0")  # QT
    bailouts_given: Decimal = Decimal("0")

    # Damage metrics (tracking the harm caused)
    inflation_caused: Decimal = Decimal("0")
    malinvestment_induced: Decimal = Decimal("0")
    bubbles_created: int = 0
    cycles_caused: int = 0

    # Policy tracking
    qe_rounds: int = 0
    qt_rounds: int = 0
    current_policy: str = "neutral"
    rate_cuts_cumulative: float = 0.0  # Track cumulative rate changes


class CentralBank(Agent):
    """
    Central Bank - The Monetary Villain.

    "The Fed is the great enabler of big government."
    - Ron Paul

    Austrian Economics View:
    - Central banks are unnecessary and harmful
    - They transfer wealth from savers to borrowers
    - They cause boom-bust cycles
    - They enable government deficit spending
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        country: str = "USA",
        name: str = "Federal Reserve",
        initial_base_money: Decimal = Decimal("1000000000000"),  # $1 trillion
        intervention_level: float = 0.5,
    ):
        super().__init__(
            agent_id=agent_id,
            country=country,
            time_preference=0.0,  # Irrelevant - doesn't follow market signals
            risk_tolerance=1.0,  # No risk for them - taxpayers pay
            initial_wealth=Decimal("0"),  # Creates money from nothing
        )

        self.agent_type = AgentType.CENTRAL_BANK
        self.name = name

        # Intervention level (0 = no intervention, 1 = maximum)
        self.intervention_level = intervention_level

        # ===========================================
        # INFLATION TARGETING (The "Fiction")
        # ===========================================
        # The Fed pretends to control inflation through these tools
        # Austrian view: This is impossible - they cause inflation, they can't control it
        self.inflation_target = 0.02  # 2% target (Fed's "mandate")
        self.inflation_tolerance = 0.01  # ±1% band

        # Policy rates
        self.policy_rate = 0.05  # Target interest rate
        self.natural_rate = 0.05  # What rate SHOULD be (time preference)

        # Rate bounds (since 1971 fiat era)
        self.rate_floor = 0.0  # ZIRP (zero interest rate policy)
        self.rate_ceiling = 0.20  # Volcker-style extreme tightening

        # Current monetary policy stance
        self.current_policy = MonetaryPolicy.NEUTRAL

        # State
        self.state = CentralBankState(
            base_money=initial_base_money,
        )

        # Balance sheet
        self.treasuries_held: Decimal = initial_base_money * Decimal("0.5")
        self.mortgage_backed_securities: Decimal = initial_base_money * Decimal("0.3")
        self.toxic_assets: Decimal = Decimal("0")

        # QE/QT tracking
        self.qe_active = False
        self.qt_active = False
        self.consecutive_qe_weeks = 0
        self.consecutive_qt_weeks = 0

        # Cumulative damage tracking
        self._total_damage: Decimal = Decimal("0")

    @property
    def balance_sheet_size(self) -> Decimal:
        """Total assets on balance sheet"""
        return (
            self.treasuries_held +
            self.mortgage_backed_securities +
            self.toxic_assets
        )

    @property
    def rate_manipulation(self) -> float:
        """How much the rate is being manipulated below natural rate"""
        return max(0, self.natural_rate - self.policy_rate)

    @property
    def total_damage_caused(self) -> Decimal:
        """Total economic damage caused by interventions"""
        return (
            self.state.inflation_caused +
            self.state.malinvestment_induced +
            self.state.bailouts_given
        )

    # ===========================================
    # MONETARY POLICY (All harmful according to Austrians)
    # ===========================================

    def set_policy_rate(self, rate: float, natural_rate: float):
        """
        Set the policy rate.

        Austrian Theory:
        - Rate should equal the natural rate (time preference)
        - Any deviation causes distortions
        - Lower rates = more malinvestment
        """
        self.policy_rate = max(0, rate)  # Can go to zero (ZIRP)
        self.natural_rate = natural_rate

        # Calculate damage from manipulation
        rate_gap = max(0, natural_rate - rate)
        manipulation_damage = rate_gap * float(self.state.base_money) * 0.01

        self.state.malinvestment_induced += Decimal(str(manipulation_damage))

    def lower_rates(self, amount: float):
        """
        Lower interest rates - THE CLASSIC MISTAKE

        Austrian Theory:
        - This sends false signals to entrepreneurs
        - Makes unprofitable projects seem profitable
        - Creates boom that must end in bust
        """
        new_rate = max(0, self.policy_rate - amount)
        self.set_policy_rate(new_rate, self.natural_rate)

    def print_money(self, amount: Decimal) -> Decimal:
        """
        Create money from nothing - QUANTITATIVE EASING

        Austrian Theory:
        - This is legalized counterfeiting
        - Transfers wealth from cash holders to first receivers
        - Causes price inflation (eventually)
        - Enables government deficit spending

        "Inflation is taxation without legislation."
        - Milton Friedman
        """
        self.state.base_money += amount
        self.state.money_printed += amount

        # Calculate inflation caused (simplified)
        # New money dilutes existing money
        inflation_effect = amount / max(self.state.base_money, Decimal("1"))
        self.state.inflation_caused += amount * inflation_effect

        return amount

    def quantitative_easing(
        self,
        amount: Decimal,
        asset_type: str = "treasuries"
    ) -> Decimal:
        """
        Buy assets with printed money - QE

        Austrian Theory:
        - Distorts asset prices
        - Creates asset bubbles
        - Benefits the wealthy (Cantillon effect)
        - Punishes savers
        """
        # Print the money
        printed = self.print_money(amount)

        # "Buy" the assets
        if asset_type == "treasuries":
            self.treasuries_held += amount
        elif asset_type == "mbs":
            self.mortgage_backed_securities += amount
        elif asset_type == "toxic":
            self.toxic_assets += amount

        self.state.assets_purchased += amount

        # Check for bubble creation
        if self.state.assets_purchased > self.state.base_money * Decimal("0.3"):
            self.state.bubbles_created += 1

        # Track QE activity
        self.qe_active = True
        self.qt_active = False
        self.consecutive_qe_weeks += 1
        self.consecutive_qt_weeks = 0
        self.state.qe_rounds += 1
        self.current_policy = MonetaryPolicy.EASY

        return printed

    def quantitative_tightening(
        self,
        amount: Decimal,
        asset_type: str = "treasuries"
    ) -> Decimal:
        """
        Sell assets and destroy money - QT (Quantitative Tightening)

        The Fed's attempt to "unwind" QE and control inflation.

        Austrian Theory:
        - Too little, too late - damage is done
        - Can't undo malinvestment
        - Will cause market crashes when liquidity removed
        - The "cure" causes another recession

        Reality:
        - Fed always chickens out when markets drop ("Fed put")
        - QT never equals previous QE
        - Net effect is always expansionary long-term
        """
        # Limit QT to available assets
        if asset_type == "treasuries":
            sellable = min(amount, self.treasuries_held)
            self.treasuries_held -= sellable
        elif asset_type == "mbs":
            sellable = min(amount, self.mortgage_backed_securities)
            self.mortgage_backed_securities -= sellable
        else:
            sellable = Decimal("0")

        if sellable > 0:
            # "Destroy" the money received from sale
            self.state.base_money -= sellable
            self.state.money_destroyed += sellable
            self.state.assets_sold += sellable

            # Track QT activity
            self.qt_active = True
            self.qe_active = False
            self.consecutive_qt_weeks += 1
            self.consecutive_qe_weeks = 0
            self.state.qt_rounds += 1
            self.current_policy = MonetaryPolicy.TIGHT

        return sellable

    def raise_rates(self, amount: float):
        """
        Raise interest rates - Fighting inflation

        Austrian Theory:
        - This is trying to fix the problem they caused
        - Will cause recession (the bust that was inevitable)
        - Better to have never lowered rates artificially
        """
        new_rate = min(self.rate_ceiling, self.policy_rate + amount)
        self.policy_rate = new_rate

    # ===========================================
    # BAILOUTS (Moral hazard)
    # ===========================================

    def bailout(self, institution_id: str, amount: Decimal) -> Decimal:
        """
        Bail out a failing institution.

        Austrian Theory:
        - Creates moral hazard
        - Rewards bad behavior
        - Socializes losses, privatizes gains
        - "Too big to fail" is too big to exist

        "Capitalism without bankruptcy is like Christianity
         without hell." - Frank Borman
        """
        # Print money for bailout
        bailout_funds = self.print_money(amount)
        self.state.bailouts_given += amount

        # Take toxic assets in return (usually worthless)
        self.toxic_assets += amount

        # Increase damage metrics
        self.state.malinvestment_induced += amount * Decimal("0.5")

        return bailout_funds

    # ===========================================
    # DAMAGE METRICS (Austrian perspective)
    # ===========================================

    def calculate_cantillon_effect(self) -> Dict[str, Decimal]:
        """
        Calculate wealth transfer from money printing.

        Cantillon Effect:
        - First receivers of new money benefit
        - Last receivers (poor, savers) are harmed
        - Banks and government benefit most
        """
        total_printed = self.state.money_printed

        return {
            "benefited_banks": total_printed * Decimal("0.4"),
            "benefited_government": total_printed * Decimal("0.3"),
            "benefited_wealthy": total_printed * Decimal("0.2"),
            "harmed_savers": total_printed * Decimal("0.5"),
            "harmed_poor": total_printed * Decimal("0.3"),
            "harmed_fixed_income": total_printed * Decimal("0.4"),
        }

    def calculate_business_cycle_damage(self) -> Dict[str, Any]:
        """
        Calculate damage from artificial credit expansion.

        Austrian Business Cycle Theory (ABCT):
        - Low rates → credit expansion
        - Credit expansion → malinvestment
        - Malinvestment → boom
        - Reality catches up → bust
        - Bust is NECESSARY correction
        """
        rate_manipulation = self.rate_manipulation
        credit_expanded = self.state.money_printed

        # Malinvestment proportional to rate manipulation and credit
        malinvestment = credit_expanded * Decimal(str(rate_manipulation))

        return {
            "rate_manipulation": rate_manipulation,
            "credit_expansion": str(credit_expanded),
            "malinvestment_estimate": str(malinvestment),
            "bust_severity_estimate": str(malinvestment * Decimal("0.7")),
            "recovery_time_weeks": int(float(malinvestment) / 1e9 * 52),
        }

    def get_damage_report(self) -> Dict[str, Any]:
        """Get comprehensive damage report"""
        return {
            "total_money_printed": str(self.state.money_printed),
            "total_assets_purchased": str(self.state.assets_purchased),
            "total_bailouts": str(self.state.bailouts_given),
            "inflation_caused": str(self.state.inflation_caused),
            "malinvestment_induced": str(self.state.malinvestment_induced),
            "bubbles_created": self.state.bubbles_created,
            "cycles_caused": self.state.cycles_caused,
            "balance_sheet_size": str(self.balance_sheet_size),
            "rate_manipulation": self.rate_manipulation,
            "cantillon_effect": self.calculate_cantillon_effect(),
            "business_cycle_damage": self.calculate_business_cycle_damage(),
        }

    # ===========================================
    # MAIN STEP FUNCTION - INFLATION TARGETING
    # ===========================================

    def step(self, world_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute one simulation step with INFLATION TARGETING.

        The Fed's dual mandate (the "fiction"):
        1. Price stability (2% inflation target)
        2. Maximum employment

        Reality (Austrian view):
        - They CAUSE inflation, then pretend to fight it
        - QE during recessions → QT when inflation rises
        - Net effect: always expansionary long-term
        - Real assets (Gold, BTC) benefit during QE periods
        """
        actions = []

        # Get economic indicators
        inflation_rate = world_state.get("inflation_rate", 0.02)
        unemployment = world_state.get("unemployment", 0.05)
        gdp_growth = world_state.get("gdp_growth", 0.02)

        # Calculate "natural rate" from market time preferences
        avg_time_preference = world_state.get("avg_time_preference", 0.5)
        self.natural_rate = 0.02 + avg_time_preference * 0.08

        # No intervention if level is 0 (Austrian ideal)
        if self.intervention_level == 0:
            self.current_policy = MonetaryPolicy.NEUTRAL
            self.state.current_policy = "neutral"
            return actions

        # ===========================================
        # INFLATION TARGETING LOGIC
        # ===========================================
        # This is how the Fed actually operates (post-1971)

        inflation_gap = inflation_rate - self.inflation_target
        unemployment_threshold = 0.06  # "Natural" unemployment
        recession_risk = gdp_growth < 0 or unemployment > 0.08

        # Determine policy stance
        if inflation_rate > self.inflation_target + self.inflation_tolerance:
            # ===========================================
            # INFLATION TOO HIGH → TIGHTENING (QT)
            # ===========================================
            # This is what they do when inflation spikes (like 2022-2023)

            actions.extend(self._execute_tightening(inflation_gap, world_state))

        elif recession_risk or unemployment > unemployment_threshold:
            # ===========================================
            # RECESSION RISK → EASING (QE)
            # ===========================================
            # This is the "Fed put" - always bail out the economy

            actions.extend(self._execute_easing(unemployment, gdp_growth, world_state))

        else:
            # ===========================================
            # NEUTRAL - But still slightly expansionary
            # ===========================================
            # Even "neutral" Fed policy is net expansionary

            self.current_policy = MonetaryPolicy.NEUTRAL
            self.state.current_policy = "neutral"

            # Small baseline expansion (the long-term inflationary bias)
            if self.intervention_level > 0.2:
                baseline_expansion = Decimal(str(self.intervention_level * 1e9))
                self.print_money(baseline_expansion)
                actions.append({
                    "type": "baseline_expansion",
                    "amount": str(baseline_expansion),
                    "note": "Even 'neutral' policy is expansionary"
                })

        # ===========================================
        # BAILOUTS (always available if needed)
        # ===========================================
        failing_banks = world_state.get("failing_banks", [])
        if failing_banks and self.intervention_level > 0.5:
            for bank_id in failing_banks[:3]:
                bailout_amount = Decimal("10000000000")  # $10B
                self.bailout(bank_id, bailout_amount)
                actions.append({
                    "type": "bailout",
                    "institution": bank_id,
                    "amount": str(bailout_amount),
                    "damage": "Creating moral hazard"
                })

        # Record cycle if conditions indicate bust
        if gdp_growth < -0.02 and self.state.malinvestment_induced > 0:
            self.state.cycles_caused += 1

        # Add damage report
        actions.append({
            "type": "policy_summary",
            "policy": self.current_policy.value,
            "inflation_rate": inflation_rate,
            "target": self.inflation_target,
            "policy_rate": self.policy_rate,
            "qe_active": self.qe_active,
            "qt_active": self.qt_active,
        })

        return actions

    def _execute_tightening(
        self,
        inflation_gap: float,
        world_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute tightening policy (QT + rate hikes).

        This is what the Fed does when inflation gets "out of control":
        1. Raise interest rates
        2. Sell assets (QT)
        3. Reduce money supply

        Austrian view: Too little, too late. Damage is done.
        """
        actions = []

        # How aggressive based on intervention level and inflation gap
        aggressiveness = self.intervention_level * min(2.0, abs(inflation_gap))

        # 1. RAISE RATES
        rate_hike = min(0.0075, aggressiveness * 0.005)  # Max 75bp per step
        if self.policy_rate < self.rate_ceiling:
            self.raise_rates(rate_hike)
            actions.append({
                "type": "rate_hike",
                "amount": rate_hike,
                "new_rate": self.policy_rate,
                "reason": f"Fighting inflation ({inflation_gap:.1%} above target)"
            })

        # 2. QUANTITATIVE TIGHTENING
        # Sell assets proportional to inflation gap
        qt_amount = Decimal(str(aggressiveness * 5e9))  # Up to $10B per step
        if qt_amount > 0:
            sold = self.quantitative_tightening(qt_amount, "treasuries")
            if sold > 0:
                actions.append({
                    "type": "quantitative_tightening",
                    "amount": str(sold),
                    "reason": "Reducing money supply to fight inflation"
                })

        self.current_policy = MonetaryPolicy.TIGHT
        self.state.current_policy = "tight"

        return actions

    def _execute_easing(
        self,
        unemployment: float,
        gdp_growth: float,
        world_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute easing policy (QE + rate cuts).

        This is the "Fed put" - always rescue the economy:
        1. Cut interest rates
        2. Buy assets (QE)
        3. Print money

        Austrian view: This CAUSES the next boom-bust cycle.
        """
        actions = []

        # How aggressive based on recession severity
        recession_severity = max(0, -gdp_growth) + max(0, unemployment - 0.06)
        aggressiveness = self.intervention_level * (1 + recession_severity * 5)

        # 1. CUT RATES
        rate_cut = min(0.005, aggressiveness * 0.003)  # Max 50bp per step
        if self.policy_rate > self.rate_floor:
            self.lower_rates(rate_cut)
            actions.append({
                "type": "rate_cut",
                "amount": rate_cut,
                "new_rate": self.policy_rate,
                "reason": "Stimulating economy",
                "damage": "Sending false signals to entrepreneurs"
            })

        # 2. QUANTITATIVE EASING
        # More aggressive if rates already at floor (ZIRP)
        if self.policy_rate <= 0.01:
            qe_multiplier = 2.0  # Double QE when at zero rates
        else:
            qe_multiplier = 1.0

        qe_amount = Decimal(str(aggressiveness * 1e10 * qe_multiplier))
        if qe_amount > 0:
            printed = self.quantitative_easing(qe_amount)
            actions.append({
                "type": "quantitative_easing",
                "amount": str(printed),
                "reason": "Supporting the economy",
                "damage": "Debasing currency, stealing from savers"
            })

        # Emergency mode if severe recession
        if gdp_growth < -0.05 or unemployment > 0.10:
            self.current_policy = MonetaryPolicy.EMERGENCY
            self.state.current_policy = "emergency"
            # Extra emergency stimulus
            emergency_stimulus = Decimal(str(self.intervention_level * 5e10))
            self.quantitative_easing(emergency_stimulus, "mbs")
            actions.append({
                "type": "emergency_stimulus",
                "amount": str(emergency_stimulus),
                "reason": "Crisis response"
            })
        else:
            self.current_policy = MonetaryPolicy.EASY
            self.state.current_policy = "easy"

        return actions

    def receive_income(self, amount: Decimal, source: str):
        """Central bank doesn't need income - it prints money"""
        pass

    def pay_expense(self, amount: Decimal, destination: str) -> bool:
        """Central bank can always pay - just prints more"""
        self.print_money(amount)
        return True

    # ===========================================
    # SCENARIO METHODS
    # ===========================================

    def set_intervention_level(self, level: float):
        """Set intervention level (for scenarios)"""
        self.intervention_level = max(0, min(1, level))

    def reset_to_zero_intervention(self):
        """
        Reset to no intervention (Austrian ideal).

        What SHOULD happen:
        - Rates set by market
        - No money printing
        - No bailouts
        - Sound money
        """
        self.intervention_level = 0
        self.policy_rate = self.natural_rate
        # Note: damage already caused cannot be undone

    @classmethod
    def create_for_country(cls, country: str) -> "CentralBank":
        """Create central bank for a specific country (20 countries supported)"""

        # Central bank names for all 20 countries
        names = {
            # Americas
            "USA": "Federal Reserve",
            "CAN": "Bank of Canada",
            "MEX": "Banco de México",
            "BRA": "Banco Central do Brasil",
            "ARG": "Banco Central de la República Argentina",

            # Europe
            "GBR": "Bank of England",
            "DEU": "Deutsche Bundesbank",  # via ECB
            "FRA": "Banque de France",     # via ECB
            "SWE": "Sveriges Riksbank",
            "NOR": "Norges Bank",
            "CHE": "Swiss National Bank",
            "LIE": "Swiss National Bank",  # Uses CHF

            # Asia
            "CHN": "People's Bank of China",
            "JPN": "Bank of Japan",
            "IND": "Reserve Bank of India",
            "IDN": "Bank Indonesia",

            # Middle East
            "ARE": "Central Bank of UAE",
            "SAU": "Saudi Central Bank (SAMA)",

            # Eurasia
            "RUS": "Central Bank of Russia",
            "TUR": "Central Bank of the Republic of Turkey",
        }

        # Base money supply in local currency (approximate)
        base_money = {
            # Americas (in USD or local)
            "USA": Decimal("6000000000000"),     # ~$6T USD
            "CAN": Decimal("300000000000"),      # ~$300B CAD
            "MEX": Decimal("2500000000000"),     # ~2.5T MXN
            "BRA": Decimal("500000000000"),      # ~500B BRL
            "ARG": Decimal("10000000000000"),    # ~10T ARS (high due to inflation)

            # Europe
            "GBR": Decimal("1000000000000"),     # ~£1T
            "DEU": Decimal("3000000000000"),     # ~€3T (Eurozone share)
            "FRA": Decimal("2500000000000"),     # ~€2.5T (Eurozone share)
            "SWE": Decimal("400000000000"),      # ~400B SEK
            "NOR": Decimal("300000000000"),      # ~300B NOK
            "CHE": Decimal("800000000000"),      # ~800B CHF
            "LIE": Decimal("5000000000"),        # ~5B CHF (tiny)

            # Asia
            "CHN": Decimal("40000000000000"),    # ~40T CNY
            "JPN": Decimal("700000000000000"),   # ~¥700T
            "IND": Decimal("45000000000000"),    # ~45T INR
            "IDN": Decimal("3000000000000000"),  # ~3000T IDR

            # Middle East
            "ARE": Decimal("500000000000"),      # ~500B AED
            "SAU": Decimal("2000000000000"),     # ~2T SAR

            # Eurasia
            "RUS": Decimal("20000000000000"),    # ~20T RUB
            "TUR": Decimal("5000000000000"),     # ~5T TRY
        }

        # Intervention levels by regime type (Hoppe hierarchy)
        # MINARCHY: 0.25, MONARCHY: 0.30, DEMOCRACY_LIBERAL: 0.50
        # DEMOCRACY_SOCIALIST: 0.65, TOTALITARIAN: 0.80
        intervention_levels = {
            # Low intervention (minarchy)
            "CHE": 0.25,  # Switzerland - low intervention
            "LIE": 0.20,  # Liechtenstein - minimal (uses CHF)

            # Moderate-low (monarchy)
            "ARE": 0.30,  # UAE - monarchy
            "SAU": 0.30,  # Saudi Arabia - monarchy

            # Moderate (liberal democracy)
            "USA": 0.50,  # USA - liberal democracy
            "JPN": 0.50,  # Japan - liberal democracy

            # Moderate-high (socialist democracy)
            "GBR": 0.65,  # UK - socialist democracy
            "CAN": 0.65,  # Canada - socialist democracy
            "MEX": 0.65,  # Mexico - socialist democracy
            "BRA": 0.65,  # Brazil - socialist democracy
            "ARG": 0.70,  # Argentina - high intervention history
            "DEU": 0.60,  # Germany - via ECB, moderate
            "FRA": 0.65,  # France - via ECB, higher
            "SWE": 0.65,  # Sweden - socialist democracy
            "NOR": 0.60,  # Norway - moderate, oil wealth
            "IND": 0.65,  # India - socialist democracy
            "IDN": 0.60,  # Indonesia - moderate

            # High intervention (totalitarian)
            "CHN": 0.80,  # China - totalitarian
            "RUS": 0.75,  # Russia - totalitarian
            "TUR": 0.80,  # Turkey - Erdogan's unorthodox policies
        }

        return cls(
            country=country,
            name=names.get(country, f"Central Bank of {country}"),
            initial_base_money=base_money.get(country, Decimal("1000000000000")),
            intervention_level=intervention_levels.get(country, 0.5),
        )
