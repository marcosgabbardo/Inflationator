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
import random

from .base import Agent, AgentType, AgentState


@dataclass
class CentralBankState(AgentState):
    """State for central bank villain"""
    base_money: Decimal = Decimal("0")
    money_printed: Decimal = Decimal("0")
    assets_purchased: Decimal = Decimal("0")
    bailouts_given: Decimal = Decimal("0")

    # Damage metrics (tracking the harm caused)
    inflation_caused: Decimal = Decimal("0")
    malinvestment_induced: Decimal = Decimal("0")
    bubbles_created: int = 0
    cycles_caused: int = 0


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

        # Policy rates
        self.policy_rate = 0.05  # Target interest rate
        self.natural_rate = 0.05  # What rate SHOULD be (time preference)

        # State
        self.state = CentralBankState(
            base_money=initial_base_money,
        )

        # Balance sheet
        self.treasuries_held: Decimal = Decimal("0")
        self.mortgage_backed_securities: Decimal = Decimal("0")
        self.toxic_assets: Decimal = Decimal("0")

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

        return printed

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
    # MAIN STEP FUNCTION
    # ===========================================

    def step(self, world_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute one simulation step.

        The central bank takes actions based on its intervention level.
        Higher intervention = more damage to the economy.
        """
        actions = []

        # Get economic indicators
        inflation_rate = world_state.get("inflation_rate", 0.02)
        unemployment = world_state.get("unemployment", 0.05)
        gdp_growth = world_state.get("gdp_growth", 0.02)

        # Calculate "natural rate" from market time preferences
        avg_time_preference = world_state.get("avg_time_preference", 0.5)
        self.natural_rate = 0.02 + avg_time_preference * 0.08

        # ===========================================
        # INTERVENTION BASED ON intervention_level
        # ===========================================

        if self.intervention_level > 0:
            # 1. Rate manipulation
            if unemployment > 0.06:  # "Too high" unemployment
                rate_cut = self.intervention_level * 0.005
                self.lower_rates(rate_cut)
                actions.append({
                    "type": "rate_cut",
                    "amount": rate_cut,
                    "new_rate": self.policy_rate,
                    "damage": "Sending false signals to entrepreneurs"
                })

            # 2. Money printing (QE) if rates near zero
            if self.policy_rate < 0.01 and self.intervention_level > 0.3:
                qe_amount = Decimal(str(self.intervention_level * 1e10))
                self.quantitative_easing(qe_amount)
                actions.append({
                    "type": "quantitative_easing",
                    "amount": str(qe_amount),
                    "damage": "Debasing currency, stealing from savers"
                })

            # 3. Bailouts if there are failing banks
            failing_banks = world_state.get("failing_banks", [])
            if failing_banks and self.intervention_level > 0.5:
                for bank_id in failing_banks[:3]:  # Limit bailouts
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

        # Add damage report to actions
        actions.append({
            "type": "damage_report",
            "report": self.get_damage_report()
        })

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
        """Create central bank for a specific country"""
        names = {
            "USA": "Federal Reserve",
            "EUR": "European Central Bank",
            "GBR": "Bank of England",
            "JPN": "Bank of Japan",
            "CHN": "People's Bank of China",
            "BRA": "Banco Central do Brasil",
        }

        base_money = {
            "USA": Decimal("6000000000000"),   # ~$6 trillion
            "EUR": Decimal("7000000000000"),   # ~€7 trillion
            "JPN": Decimal("700000000000000"), # ~¥700 trillion
        }

        return cls(
            country=country,
            name=names.get(country, f"Central Bank of {country}"),
            initial_base_money=base_money.get(country, Decimal("1000000000000")),
            intervention_level=0.5,  # Default moderate intervention
        )
