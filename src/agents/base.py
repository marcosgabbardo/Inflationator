"""
Base Agent Class

Foundation for all economic agents in the simulation.
Based on Austrian Economics principles:
- Subjective value theory
- Time preference
- Economic calculation
- Spontaneous order
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from decimal import Decimal
from enum import Enum
import uuid
import random


class AgentType(str, Enum):
    """Types of agents in the simulation"""
    PERSON = "person"
    COMPANY = "company"
    BANK = "bank"
    CENTRAL_BANK = "central_bank"
    GOVERNMENT = "government"


@dataclass
class AgentState:
    """Current state of an agent"""
    wealth: Decimal = Decimal("0")
    bitcoin: Decimal = Decimal("0")
    gold: Decimal = Decimal("0")
    is_active: bool = True
    knowledge: Dict[str, Any] = field(default_factory=dict)
    expectations: Dict[str, float] = field(default_factory=dict)


class Agent(ABC):
    """
    Base class for all economic agents.

    Austrian Economics Principles Applied:
    1. Agents act purposefully to improve their situation
    2. Agents have subjective preferences
    3. Agents have time preferences (present vs future)
    4. Agents make decisions based on available knowledge (imperfect)
    5. Prices emerge from individual actions (not set centrally)
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        country: str = "USA",
        time_preference: float = 0.5,
        risk_tolerance: float = 0.5,
        initial_wealth: Decimal = Decimal("0"),
    ):
        self.id = agent_id or str(uuid.uuid4())
        self.country = country
        self.agent_type = AgentType.PERSON  # Override in subclasses

        # Austrian Economics Parameters
        self.time_preference = self._validate_range(time_preference)
        self.risk_tolerance = self._validate_range(risk_tolerance)

        # State
        self.state = AgentState(wealth=initial_wealth)

        # History for learning
        self._decision_history: List[Dict] = []
        self._error_history: List[float] = []  # For adaptive expectations

    @staticmethod
    def _validate_range(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Ensure value is within valid range"""
        return max(min_val, min(max_val, value))

    @property
    def wealth(self) -> Decimal:
        """Current monetary wealth"""
        return self.state.wealth

    @wealth.setter
    def wealth(self, value: Decimal):
        self.state.wealth = max(Decimal("0"), value)

    @property
    def total_assets(self) -> Decimal:
        """Total assets including crypto and gold"""
        # Will be priced using market data
        return self.state.wealth + self.state.bitcoin + self.state.gold

    # ===========================================
    # AUSTRIAN ECONOMICS DECISION FRAMEWORK
    # ===========================================

    def calculate_savings_rate(self) -> float:
        """
        Calculate how much to save vs consume.

        Austrian Theory: Lower time preference = more savings
        - Low time preference agents save more (think long-term)
        - High time preference agents consume more (present-oriented)
        """
        base_savings = 1.0 - self.time_preference
        # Add some noise for realistic behavior
        noise = random.gauss(0, 0.05)
        return self._validate_range(base_savings + noise)

    def evaluate_investment(
        self,
        expected_return: float,
        time_horizon: int,
        risk_level: float
    ) -> bool:
        """
        Decide whether to make an investment.

        Austrian Theory: Investment decisions based on:
        1. Time preference (discount future returns)
        2. Risk tolerance
        3. Expected return
        """
        # Discount future returns based on time preference
        discount_factor = (1 - self.time_preference) ** time_horizon

        # Adjust for risk
        risk_adjusted_return = expected_return * (1 - risk_level * (1 - self.risk_tolerance))

        # Present value of investment
        present_value = risk_adjusted_return * discount_factor

        # Invest if present value exceeds threshold
        threshold = self.time_preference * 0.5  # Higher time preference = higher threshold
        return present_value > threshold

    def update_expectations(self, actual_value: float, expected_value: float):
        """
        Update expectations based on errors (adaptive expectations).

        Austrian Theory: Agents learn from their mistakes
        """
        error = actual_value - expected_value
        self._error_history.append(error)

        # Keep only recent history
        if len(self._error_history) > 10:
            self._error_history.pop(0)

    def get_expected_inflation(self) -> float:
        """
        Form inflation expectations.

        Austrian Theory: Inflation is a monetary phenomenon
        - Agents observe money supply expansion
        - Agents adjust expectations based on past errors
        """
        # Base expectation from knowledge
        known_inflation = self.state.knowledge.get("observed_inflation", 0.02)

        # Adaptive adjustment from errors
        if self._error_history:
            avg_error = sum(self._error_history) / len(self._error_history)
            adjustment = avg_error * 0.3  # Partial adjustment
        else:
            adjustment = 0

        return known_inflation + adjustment

    # ===========================================
    # ACTION METHODS (to be implemented by subclasses)
    # ===========================================

    @abstractmethod
    def step(self, world_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute one simulation step.

        Args:
            world_state: Current state of the world (prices, rates, etc.)

        Returns:
            List of actions to take (transactions, decisions, etc.)
        """
        pass

    @abstractmethod
    def receive_income(self, amount: Decimal, source: str):
        """Receive income from some source"""
        pass

    @abstractmethod
    def pay_expense(self, amount: Decimal, destination: str) -> bool:
        """Pay an expense if possible"""
        pass

    # ===========================================
    # UTILITY METHODS
    # ===========================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent to dictionary"""
        return {
            "id": self.id,
            "type": self.agent_type.value,
            "country": self.country,
            "time_preference": self.time_preference,
            "risk_tolerance": self.risk_tolerance,
            "wealth": str(self.state.wealth),
            "bitcoin": str(self.state.bitcoin),
            "gold": str(self.state.gold),
            "is_active": self.state.is_active,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id[:8]}, wealth={self.wealth})"
