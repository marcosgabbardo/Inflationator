"""
Private Bank Agent

Private banking entity in the simulation.
Makes decisions about:
- Deposit rates (attracting savers)
- Lending rates (to borrowers)
- Reserve ratios
- Credit creation

Austrian Theory: Banks create credit through fractional reserves,
which can cause malinvestment when done excessively.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Any, List, Optional
import random

from .base import Agent, AgentType, AgentState


@dataclass
class BankState(AgentState):
    """Extended state for bank agents"""
    reserves: Decimal = Decimal("0")
    total_deposits: Decimal = Decimal("0")
    total_loans: Decimal = Decimal("0")
    credit_created: Decimal = Decimal("0")


class Bank(Agent):
    """
    Private bank agent.

    Austrian Economics Behavior:
    - Creates credit through fractional reserve banking
    - Credit creation expands money supply (inflationary)
    - Interest rates should reflect time preference of savers
    - Excessive credit leads to malinvestment
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        country: str = "USA",
        initial_capital: Decimal = Decimal("1000000"),
        reserve_ratio: float = 0.10,
        name: Optional[str] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            country=country,
            time_preference=0.3,  # Banks are long-term oriented
            risk_tolerance=0.4,
            initial_wealth=initial_capital,
        )

        self.agent_type = AgentType.BANK
        self.name = name or f"Bank_{self.id[:8]}"

        # Reserve settings
        self.reserve_ratio = reserve_ratio  # Fractional reserve

        # Interest rates
        self.deposit_rate = 0.02  # Rate paid to depositors
        self.lending_rate = 0.05  # Rate charged to borrowers

        # State
        self.state = BankState(
            wealth=initial_capital,
            reserves=initial_capital,
        )

        # Loan tracking
        self.loans: Dict[str, Dict] = {}  # borrower_id -> loan details
        self.deposits: Dict[str, Decimal] = {}  # depositor_id -> amount

    @property
    def reserves(self) -> Decimal:
        return self.state.reserves

    @property
    def excess_reserves(self) -> Decimal:
        """Reserves above required minimum"""
        required = self.state.total_deposits * Decimal(str(self.reserve_ratio))
        return max(Decimal("0"), self.reserves - required)

    @property
    def loanable_funds(self) -> Decimal:
        """Amount available for lending"""
        # Can lend excess reserves
        # Credit creation happens when loans > deposits
        return self.excess_reserves

    @property
    def credit_multiplier(self) -> float:
        """Money multiplier from fractional reserve"""
        if self.reserve_ratio > 0:
            return 1 / self.reserve_ratio
        return float("inf")

    # ===========================================
    # DEPOSIT OPERATIONS
    # ===========================================

    def accept_deposit(self, depositor_id: str, amount: Decimal) -> bool:
        """Accept a deposit from a customer"""
        self.state.reserves += amount
        self.state.total_deposits += amount

        if depositor_id in self.deposits:
            self.deposits[depositor_id] += amount
        else:
            self.deposits[depositor_id] = amount

        return True

    def process_withdrawal(self, depositor_id: str, amount: Decimal) -> bool:
        """Process a withdrawal request"""
        if depositor_id not in self.deposits:
            return False

        available = self.deposits[depositor_id]
        if amount > available:
            return False

        if amount > self.state.reserves:
            # Bank run situation!
            return False

        self.deposits[depositor_id] -= amount
        self.state.reserves -= amount
        self.state.total_deposits -= amount

        return True

    def pay_deposit_interest(self) -> Decimal:
        """Pay interest to all depositors (weekly)"""
        weekly_rate = self.deposit_rate / 52
        total_interest = self.state.total_deposits * Decimal(str(weekly_rate))

        if total_interest <= self.state.reserves:
            self.state.reserves -= total_interest
            return total_interest

        return Decimal("0")

    # ===========================================
    # LENDING OPERATIONS
    # ===========================================

    def evaluate_loan_application(
        self,
        borrower_id: str,
        amount: Decimal,
        collateral: Decimal,
        borrower_creditworthiness: float
    ) -> bool:
        """
        Evaluate whether to approve a loan.

        Austrian Theory:
        - Loans should match genuine savings
        - Credit creation beyond savings causes distortions
        """
        # Basic checks
        if amount > self.loanable_funds:
            return False

        if amount <= Decimal("0"):
            return False

        # Risk assessment
        ltv_ratio = amount / max(collateral, Decimal("1"))  # Loan-to-value
        if ltv_ratio > Decimal("0.8"):  # Max 80% LTV
            return False

        # Credit score check
        if borrower_creditworthiness < 0.3:
            return False

        # Profit check
        expected_profit = amount * Decimal(str(self.lending_rate - self.deposit_rate))
        default_risk = 1 - borrower_creditworthiness
        expected_loss = amount * Decimal(str(default_risk * 0.5))

        return expected_profit > expected_loss

    def make_loan(
        self,
        borrower_id: str,
        amount: Decimal,
        term_weeks: int,
        collateral: Decimal = Decimal("0")
    ) -> bool:
        """
        Create a loan.

        Austrian Theory: This is where credit creation happens.
        The bank creates money by making a loan entry.
        """
        if amount > self.loanable_funds:
            return False

        # Create the loan
        self.loans[borrower_id] = {
            "principal": amount,
            "remaining": amount,
            "rate": self.lending_rate,
            "term": term_weeks,
            "weeks_remaining": term_weeks,
            "collateral": collateral,
        }

        # Credit creation: loan appears as deposit
        self.state.total_loans += amount
        self.state.credit_created += amount

        # Reduce reserves (partial - fractional reserve magic)
        reserve_use = amount * Decimal(str(self.reserve_ratio))
        self.state.reserves -= reserve_use

        return True

    def collect_loan_payment(self, borrower_id: str) -> Optional[Decimal]:
        """Collect weekly loan payment"""
        if borrower_id not in self.loans:
            return None

        loan = self.loans[borrower_id]
        if loan["weeks_remaining"] <= 0:
            return None

        # Calculate payment (simple amortization)
        weekly_rate = loan["rate"] / 52
        principal_payment = loan["principal"] / Decimal(str(loan["term"]))
        interest_payment = loan["remaining"] * Decimal(str(weekly_rate))
        total_payment = principal_payment + interest_payment

        return total_payment

    def receive_loan_payment(self, borrower_id: str, amount: Decimal) -> bool:
        """Receive payment on a loan"""
        if borrower_id not in self.loans:
            return False

        loan = self.loans[borrower_id]

        # Apply payment
        weekly_rate = loan["rate"] / 52
        interest_portion = loan["remaining"] * Decimal(str(weekly_rate))
        principal_portion = amount - interest_portion

        loan["remaining"] -= principal_portion
        loan["weeks_remaining"] -= 1

        # Add to reserves
        self.state.reserves += amount
        self.state.wealth += interest_portion  # Profit

        # Check if loan is paid off
        if loan["remaining"] <= 0 or loan["weeks_remaining"] <= 0:
            del self.loans[borrower_id]
            self.state.total_loans -= loan["principal"]

        return True

    def handle_default(self, borrower_id: str):
        """Handle loan default"""
        if borrower_id not in self.loans:
            return

        loan = self.loans[borrower_id]

        # Seize collateral (if any)
        recovery = min(loan["collateral"], loan["remaining"])
        loss = loan["remaining"] - recovery

        # Write off the loan
        self.state.wealth -= loss
        self.state.total_loans -= loan["principal"]
        self.state.credit_created -= loan["remaining"]  # Destroyed credit

        del self.loans[borrower_id]

    # ===========================================
    # INTEREST RATE DECISIONS
    # ===========================================

    def adjust_rates(self, central_bank_rate: float, market_conditions: Dict):
        """
        Adjust deposit and lending rates.

        Austrian Theory:
        - Rates should reflect time preference
        - Central bank manipulation distorts this signal
        """
        # Base on central bank rate (forced to follow)
        base_rate = central_bank_rate

        # Spread based on risk
        default_rate = market_conditions.get("default_rate", 0.02)
        risk_premium = default_rate * 2

        # Competition (simplified)
        competition = market_conditions.get("bank_competition", 0.5)

        # Set rates
        self.deposit_rate = base_rate - 0.01 * competition
        self.lending_rate = base_rate + 0.02 + risk_premium - 0.01 * competition

        # Ensure positive spread
        if self.lending_rate <= self.deposit_rate:
            self.lending_rate = self.deposit_rate + 0.01

    # ===========================================
    # MAIN STEP FUNCTION
    # ===========================================

    def step(self, world_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute one simulation step"""
        actions = []

        # 1. Adjust interest rates based on central bank
        cb_rate = world_state.get("policy_rate", 0.05)
        self.adjust_rates(cb_rate, world_state)
        actions.append({
            "type": "rate_adjustment",
            "deposit_rate": self.deposit_rate,
            "lending_rate": self.lending_rate
        })

        # 2. Pay deposit interest
        interest_paid = self.pay_deposit_interest()
        if interest_paid > 0:
            actions.append({
                "type": "deposit_interest_paid",
                "amount": str(interest_paid)
            })

        # 3. Collect loan payments
        for borrower_id in list(self.loans.keys()):
            payment_due = self.collect_loan_payment(borrower_id)
            if payment_due:
                actions.append({
                    "type": "loan_payment_due",
                    "borrower": borrower_id,
                    "amount": str(payment_due)
                })

        # 4. Report credit creation (Austrian focus)
        actions.append({
            "type": "credit_status",
            "total_credit_created": str(self.state.credit_created),
            "credit_multiplier": self.credit_multiplier,
            "loanable_funds": str(self.loanable_funds)
        })

        return actions

    def receive_income(self, amount: Decimal, source: str):
        """Receive income"""
        self.state.reserves += amount
        self.wealth += amount

    def pay_expense(self, amount: Decimal, destination: str) -> bool:
        """Pay an expense"""
        if amount <= self.state.reserves:
            self.state.reserves -= amount
            return True
        return False

    # ===========================================
    # FACTORY METHODS
    # ===========================================

    @classmethod
    def create_random(cls, country: str = "USA") -> "Bank":
        """Create a bank with random characteristics"""
        # Initial capital (log-normal)
        initial_capital = Decimal(str(max(100000, random.lognormvariate(14, 1))))

        # Reserve ratio (legal minimum + buffer)
        reserve_ratio = 0.10 + random.uniform(0, 0.05)

        return cls(
            country=country,
            initial_capital=initial_capital,
            reserve_ratio=reserve_ratio,
        )
