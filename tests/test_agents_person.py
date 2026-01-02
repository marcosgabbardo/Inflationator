"""
Unit tests for the Person agent.

Tests individual economic actor behavior based on Austrian Economics:
- Labor decisions
- Consumption decisions
- Savings and investment allocation
"""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from src.agents.person import Person, PersonState


class TestPersonInitialization:
    """Test Person agent initialization."""

    def test_default_initialization(self) -> None:
        """Test person creation with default parameters."""
        person = Person()

        assert person.id is not None
        assert person.country == "USA"
        assert person.time_preference == 0.5
        assert person.risk_tolerance == 0.5
        assert person.wealth == Decimal("10000")

    def test_custom_initialization(self) -> None:
        """Test person creation with custom parameters."""
        person = Person(
            agent_id="person-123",
            country="BRA",
            time_preference=0.3,
            risk_tolerance=0.7,
            initial_wealth=Decimal("50000"),
            skill_level=0.8,
            age_group="senior",
        )

        assert person.id == "person-123"
        assert person.country == "BRA"
        assert person.time_preference == 0.3
        assert person.risk_tolerance == 0.7
        assert person.wealth == Decimal("50000")
        assert person.skill_level == 0.8
        assert person.state.age_group == "senior"

    def test_employed_property(self) -> None:
        """Employment status should depend on employer_id."""
        person = Person()

        assert person.employed is False

        person.employer_id = "company-123"
        assert person.employed is True


class TestEmployment:
    """Test employment-related behavior."""

    def test_seek_employment_filters_by_skill(self) -> None:
        """Job seeking should filter by skill requirements."""
        # Use low wealth and high time preference to lower reservation wage
        person = Person(
            skill_level=0.5,
            initial_wealth=Decimal("100"),
            time_preference=0.9,
        )

        jobs = [
            {"id": "job1", "wage": 5000, "required_skill": 0.3},
            {"id": "job2", "wage": 8000, "required_skill": 0.8},  # Too high skill
            {"id": "job3", "wage": 6000, "required_skill": 0.5},
        ]

        best_job = person.seek_employment(jobs)

        # Should pick job3 (highest wage within skill requirement)
        assert best_job is not None
        assert best_job["id"] == "job3"

    def test_seek_employment_returns_none_for_empty_list(self) -> None:
        """Should return None when no jobs available."""
        person = Person()

        result = person.seek_employment([])

        assert result is None

    def test_accept_job(self) -> None:
        """Accepting job should update employment state."""
        person = Person()

        person.accept_job("company-123", Decimal("1500"))

        assert person.employer_id == "company-123"
        assert person.wage == Decimal("1500")
        assert person.employed is True

    def test_lose_job(self) -> None:
        """Losing job should reset employment state."""
        person = Person()
        person.accept_job("company-123", Decimal("1500"))

        person.lose_job()

        assert person.employer_id is None
        assert person.wage == Decimal("0")
        assert person.employed is False


class TestConsumption:
    """Test consumption decisions."""

    def test_decide_consumption_respects_savings_rate(self) -> None:
        """Consumption should respect time preference based savings rate."""
        person = Person(time_preference=0.3, initial_wealth=Decimal("10000"))

        consumption = person.decide_consumption({"goods": Decimal("100")})

        total = sum(consumption.values())
        # With low time preference (0.3), savings rate is high (~0.7)
        # So consumption should be limited
        assert total < Decimal("5000")

    def test_consume_reduces_wealth(self) -> None:
        """Consuming should reduce wealth."""
        person = Person(initial_wealth=Decimal("10000"))

        success = person.consume(Decimal("500"))

        assert success is True
        assert person.wealth == Decimal("9500")

    def test_consume_fails_when_insufficient_funds(self) -> None:
        """Consumption should fail when insufficient wealth."""
        person = Person(initial_wealth=Decimal("100"))

        success = person.consume(Decimal("500"))

        assert success is False
        assert person.wealth == Decimal("100")


class TestSavingsAllocation:
    """Test savings and investment allocation."""

    def test_high_inflation_increases_bitcoin_allocation(self) -> None:
        """High inflation expectation should increase Bitcoin allocation."""
        person = Person(
            risk_tolerance=0.5,
            initial_wealth=Decimal("10000"),
        )

        low_inflation = person.decide_savings_allocation(
            bitcoin_price=Decimal("50000"),
            gold_price=Decimal("2000"),
            expected_inflation=0.02,
        )

        high_inflation = person.decide_savings_allocation(
            bitcoin_price=Decimal("50000"),
            gold_price=Decimal("2000"),
            expected_inflation=0.10,
        )

        # Higher inflation should increase Bitcoin allocation
        assert high_inflation["bitcoin"] > low_inflation["bitcoin"]

    def test_low_risk_tolerance_favors_gold(self) -> None:
        """Low risk tolerance should favor gold over Bitcoin."""
        person = Person(
            risk_tolerance=0.1,  # Very low
            initial_wealth=Decimal("10000"),
        )

        allocation = person.decide_savings_allocation(
            bitcoin_price=Decimal("50000"),
            gold_price=Decimal("2000"),
            expected_inflation=0.05,
        )

        # Low risk tolerance should favor gold (less volatile)
        assert allocation["gold"] >= allocation["bitcoin"]

    def test_buy_bitcoin_reduces_wealth(self) -> None:
        """Buying Bitcoin should reduce fiat wealth."""
        person = Person(initial_wealth=Decimal("10000"))

        success = person.buy_bitcoin(
            amount=Decimal("0.1"),
            price=Decimal("50000"),
        )

        assert success is True
        assert person.wealth == Decimal("5000")
        assert person.state.bitcoin == Decimal("0.1")

    def test_buy_bitcoin_fails_insufficient_funds(self) -> None:
        """Bitcoin purchase should fail with insufficient funds."""
        person = Person(initial_wealth=Decimal("1000"))

        success = person.buy_bitcoin(
            amount=Decimal("0.1"),
            price=Decimal("50000"),
        )

        assert success is False
        assert person.wealth == Decimal("1000")
        assert person.state.bitcoin == Decimal("0")


class TestStep:
    """Test the main step function."""

    def test_step_with_employment_income(self) -> None:
        """Step should add income for employed persons."""
        person = Person(initial_wealth=Decimal("1000"))
        person.accept_job("company-123", Decimal("1000"))

        world_state = {
            "tax_rate_income": 0.20,
            "prices": {},
            "inflation_rate": 0.03,
            "bitcoin_price": 50000,
            "gold_price": 2000,
        }

        actions = person.step(world_state)

        # Should have income action
        income_action = next((a for a in actions if a["type"] == "income"), None)
        assert income_action is not None
        # Net income should be 1000 * 0.8 = 800
        assert Decimal(income_action["amount"]) == Decimal("800")


class TestRandomCreation:
    """Test random person creation."""

    def test_create_random(self) -> None:
        """Should create person with random but valid values."""
        person = Person.create_random(country="USA")

        assert person.country == "USA"
        assert 0.0 <= person.time_preference <= 1.0
        assert 0.0 <= person.risk_tolerance <= 1.0
        assert 0.0 <= person.skill_level <= 1.0
        assert person.wealth > Decimal("0")
        assert person.state.age_group in ["young", "adult", "senior"]

    def test_create_random_different_persons(self) -> None:
        """Multiple random persons should have different attributes."""
        persons = [Person.create_random() for _ in range(10)]

        # At least some should have different time preferences
        time_prefs = [p.time_preference for p in persons]
        assert len(set(time_prefs)) > 1
