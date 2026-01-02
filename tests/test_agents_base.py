"""
Unit tests for the base Agent class.

Tests the Austrian Economics decision framework including:
- Time preference calculations
- Investment evaluations
- Expectation updates
"""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from src.agents.base import Agent, AgentType, AgentState


class ConcreteAgent(Agent):
    """Concrete implementation of Agent for testing."""

    def step(self, world_state):
        """Execute one simulation step."""
        return []

    def receive_income(self, amount, source):
        """Receive income from source."""
        self.wealth += amount

    def pay_expense(self, amount, destination):
        """Pay an expense."""
        if amount <= self.wealth:
            self.wealth -= amount
            return True
        return False


class TestAgentInitialization:
    """Test Agent initialization."""

    def test_default_initialization(self) -> None:
        """Test agent creation with default parameters."""
        agent = ConcreteAgent()

        assert agent.id is not None
        assert agent.country == "USA"
        assert agent.time_preference == 0.5
        assert agent.risk_tolerance == 0.5
        assert agent.wealth == Decimal("0")

    def test_custom_initialization(self) -> None:
        """Test agent creation with custom parameters."""
        agent = ConcreteAgent(
            agent_id="test-123",
            country="BRA",
            time_preference=0.8,
            risk_tolerance=0.3,
            initial_wealth=Decimal("10000"),
        )

        assert agent.id == "test-123"
        assert agent.country == "BRA"
        assert agent.time_preference == 0.8
        assert agent.risk_tolerance == 0.3
        assert agent.wealth == Decimal("10000")

    def test_time_preference_clamping(self) -> None:
        """Test that time preference is clamped to valid range."""
        agent_high = ConcreteAgent(time_preference=1.5)
        agent_low = ConcreteAgent(time_preference=-0.5)

        assert agent_high.time_preference == 1.0
        assert agent_low.time_preference == 0.0


class TestSavingsRate:
    """Test savings rate calculations based on Austrian Economics."""

    def test_low_time_preference_high_savings(self) -> None:
        """Low time preference should result in higher savings rate."""
        agent = ConcreteAgent(time_preference=0.2)
        savings_rate = agent.calculate_savings_rate()

        # Should be around 0.8 (1 - time_preference) with some noise
        assert 0.6 <= savings_rate <= 1.0

    def test_high_time_preference_low_savings(self) -> None:
        """High time preference should result in lower savings rate."""
        agent = ConcreteAgent(time_preference=0.9)
        savings_rate = agent.calculate_savings_rate()

        # Should be around 0.1 (1 - time_preference) with some noise
        assert 0.0 <= savings_rate <= 0.4


class TestInvestmentEvaluation:
    """Test investment decision-making."""

    def test_high_return_low_risk_accepted(self) -> None:
        """High return, low risk investment should be accepted."""
        agent = ConcreteAgent(time_preference=0.1, risk_tolerance=0.9)

        result = agent.evaluate_investment(
            expected_return=0.80,  # Very high return
            time_horizon=1,  # Short horizon
            risk_level=0.05,  # Very low risk
        )

        assert result is True

    def test_low_return_high_risk_rejected(self) -> None:
        """Low return, high risk investment should be rejected."""
        agent = ConcreteAgent(time_preference=0.7, risk_tolerance=0.2)

        result = agent.evaluate_investment(
            expected_return=0.02,
            time_horizon=10,
            risk_level=0.8,
        )

        assert result is False

    def test_time_preference_affects_decision(self) -> None:
        """Higher time preference should require higher returns."""
        params = {
            "expected_return": 0.08,
            "time_horizon": 5,
            "risk_level": 0.3,
        }

        agent_low_tp = ConcreteAgent(time_preference=0.2, risk_tolerance=0.5)
        agent_high_tp = ConcreteAgent(time_preference=0.8, risk_tolerance=0.5)

        # Low time preference agent should be more willing to invest long-term
        result_low = agent_low_tp.evaluate_investment(**params)
        result_high = agent_high_tp.evaluate_investment(**params)

        # At least one should differ, or low TP should be more accepting
        assert result_low or not result_high


class TestExpectations:
    """Test adaptive expectations mechanism."""

    def test_update_expectations_records_error(self) -> None:
        """Updating expectations should record prediction error."""
        agent = ConcreteAgent()

        agent.update_expectations(actual_value=0.05, expected_value=0.03)

        assert len(agent._error_history) == 1
        assert agent._error_history[0] == pytest.approx(0.02)

    def test_inflation_expectations_adapt(self) -> None:
        """Inflation expectations should adapt based on errors."""
        agent = ConcreteAgent()
        agent.state.knowledge["observed_inflation"] = 0.03

        # Simulate series of underestimations
        for _ in range(5):
            agent.update_expectations(actual_value=0.05, expected_value=0.03)

        expected_inflation = agent.get_expected_inflation()

        # Should adjust upward from base of 0.03
        assert expected_inflation > 0.03


class TestWealth:
    """Test wealth property handling."""

    def test_wealth_cannot_go_negative(self) -> None:
        """Wealth setter should prevent negative values."""
        agent = ConcreteAgent(initial_wealth=Decimal("1000"))
        agent.wealth = Decimal("-500")

        assert agent.wealth == Decimal("0")

    def test_total_assets_calculation(self) -> None:
        """Total assets should include wealth, bitcoin, and gold."""
        agent = ConcreteAgent(initial_wealth=Decimal("1000"))
        agent.state.bitcoin = Decimal("500")
        agent.state.gold = Decimal("300")

        assert agent.total_assets == Decimal("1800")


class TestSerialization:
    """Test agent serialization."""

    def test_to_dict_contains_required_fields(self) -> None:
        """to_dict should include all required fields."""
        agent = ConcreteAgent(
            agent_id="test-id",
            country="USA",
            time_preference=0.5,
            risk_tolerance=0.5,
            initial_wealth=Decimal("1000"),
        )

        result = agent.to_dict()

        assert result["id"] == "test-id"
        assert result["country"] == "USA"
        assert result["time_preference"] == 0.5
        assert result["risk_tolerance"] == 0.5
        assert result["wealth"] == "1000"
        assert result["is_active"] is True
