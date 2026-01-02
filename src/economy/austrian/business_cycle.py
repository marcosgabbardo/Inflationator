"""
Austrian Business Cycle Theory (ABCT)

Implementation of Mises-Hayek business cycle theory:
- Credit expansion artificially lowers interest rates
- Lower rates signal increased savings (falsely)
- Companies invest in higher-order capital goods
- Malinvestment accumulates during boom
- Eventually reality corrects the distortion (bust)
- Longer/bigger boom = worse bust

Key concepts:
- Natural rate of interest (from time preference)
- Market rate of interest (manipulated by CB)
- Roundaboutness of production (capital goods orders)
- Cluster of errors (malinvestment)
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any


class CyclePhase(str, Enum):
    """Phases of the business cycle"""

    RECOVERY = "recovery"  # Post-bust, rebuilding
    BOOM = "boom"  # Credit expansion, growth
    PEAK = "peak"  # Maximum distortion
    BUST = "bust"  # Correction, liquidation
    TROUGH = "trough"  # Bottom of recession


@dataclass
class CycleState:
    """Current state of the business cycle"""

    phase: CyclePhase = CyclePhase.RECOVERY
    phase_duration: int = 0  # Ticks in current phase

    # Credit metrics
    credit_growth_rate: float = 0.0
    cumulative_credit_expansion: Decimal = Decimal("0")

    # Interest rate distortion
    natural_rate: float = 0.05  # From time preference
    market_rate: float = 0.05  # Set by central bank
    rate_distortion: float = 0.0  # Gap between them

    # Malinvestment tracking
    malinvestment_accumulated: Decimal = Decimal("0")
    capital_goods_ratio: float = 0.5  # Higher order vs consumer goods

    # Boom indicators
    boom_intensity: float = 0.0  # 0-1, how artificial the boom is
    unsustainable_projects: int = 0

    # Bust indicators
    correction_needed: Decimal = Decimal("0")
    liquidations: int = 0


class BusinessCycle:
    """
    Austrian Business Cycle manager.

    Tracks the artificial boom-bust cycle caused by credit expansion.
    The central bank's intervention creates the cycle that wouldn't
    exist in a free market with sound money.
    """

    def __init__(self):
        self.state = CycleState()
        self.history: list[CycleState] = []

        # Phase transition thresholds
        self.boom_threshold = 0.3  # Rate distortion to enter boom
        self.peak_threshold = 0.6  # Intensity to reach peak
        self.bust_threshold = 0.8  # Malinvestment ratio to trigger bust

        # Cycle parameters
        self.min_phase_duration = 4  # Minimum weeks in a phase
        self.correction_speed = 0.1  # How fast liquidation happens

    def calculate_natural_rate(
        self, avg_time_preference: float, risk_premium: float = 0.02
    ) -> float:
        """
        Calculate the natural rate of interest.

        Austrian Theory:
        - Natural rate = social time preference
        - Reflects society's preference for present vs future goods
        - Higher time preference = higher natural rate
        """
        # Natural rate based on aggregate time preference
        base_rate = avg_time_preference * 0.15  # 0-15% range

        # Add risk premium
        natural = base_rate + risk_premium

        # Ensure reasonable bounds
        return max(0.01, min(0.20, natural))

    def update(
        self,
        market_rate: float,
        avg_time_preference: float,
        credit_expansion: Decimal,
        previous_credit: Decimal,
        capital_goods_investment: Decimal,
        consumer_goods_investment: Decimal,
        total_malinvestment: Decimal,
        liquidations: int = 0,
    ) -> dict[str, Any]:
        """
        Update business cycle state based on current economy.

        Returns signals about the cycle phase and recommendations.
        """
        # Calculate natural rate
        self.state.natural_rate = self.calculate_natural_rate(avg_time_preference)
        self.state.market_rate = market_rate

        # Rate distortion (key Austrian metric)
        self.state.rate_distortion = max(0, self.state.natural_rate - market_rate)

        # Credit growth
        if previous_credit > 0:
            self.state.credit_growth_rate = float(
                (credit_expansion - previous_credit) / previous_credit
            )
        self.state.cumulative_credit_expansion = credit_expansion

        # Capital goods ratio (roundaboutness)
        total_investment = capital_goods_investment + consumer_goods_investment
        if total_investment > 0:
            self.state.capital_goods_ratio = float(
                capital_goods_investment / total_investment
            )

        # Malinvestment accumulation
        self.state.malinvestment_accumulated = total_malinvestment

        # Boom intensity
        self.state.boom_intensity = self._calculate_boom_intensity()

        # Update phase
        self._update_phase(liquidations)

        # Record history
        self.history.append(
            CycleState(
                phase=self.state.phase,
                phase_duration=self.state.phase_duration,
                credit_growth_rate=self.state.credit_growth_rate,
                cumulative_credit_expansion=self.state.cumulative_credit_expansion,
                natural_rate=self.state.natural_rate,
                market_rate=self.state.market_rate,
                rate_distortion=self.state.rate_distortion,
                malinvestment_accumulated=self.state.malinvestment_accumulated,
                capital_goods_ratio=self.state.capital_goods_ratio,
                boom_intensity=self.state.boom_intensity,
                unsustainable_projects=self.state.unsustainable_projects,
                correction_needed=self.state.correction_needed,
                liquidations=self.state.liquidations,
            )
        )

        # Keep history manageable
        if len(self.history) > 520:  # 10 years of weekly data
            self.history.pop(0)

        return self._get_cycle_signals()

    def _calculate_boom_intensity(self) -> float:
        """
        Calculate how artificial/unsustainable the current boom is.

        Based on:
        - Rate distortion (bigger gap = more artificial)
        - Credit growth rate (faster = more unsustainable)
        - Capital goods ratio (higher = more roundabout production)
        """
        # Rate distortion component
        rate_component = min(
            1.0, self.state.rate_distortion / 0.05
        )  # Max at 5% distortion

        # Credit growth component
        credit_component = min(
            1.0, self.state.credit_growth_rate / 0.10
        )  # Max at 10% growth

        # Capital goods component (deviation from natural ~50%)
        capital_deviation = abs(self.state.capital_goods_ratio - 0.5)
        capital_component = min(1.0, capital_deviation / 0.3)  # Max at 30% deviation

        # Weighted average
        intensity = (
            rate_component * 0.4 + credit_component * 0.35 + capital_component * 0.25
        )

        return min(1.0, intensity)

    def _update_phase(self, liquidations: int):
        """Update the current cycle phase."""
        self.state.phase_duration += 1
        self.state.liquidations = liquidations

        current_phase = self.state.phase

        if current_phase == CyclePhase.RECOVERY:
            # Move to boom when credit expansion starts
            if (
                self.state.rate_distortion > self.boom_threshold * 0.5
                and self.state.credit_growth_rate > 0.02
                and self.state.phase_duration >= self.min_phase_duration
            ):
                self._transition_to(CyclePhase.BOOM)

        elif current_phase == CyclePhase.BOOM:
            # Move to peak when intensity is high
            if (
                self.state.boom_intensity > self.peak_threshold
                and self.state.phase_duration >= self.min_phase_duration * 2
            ):
                self._transition_to(CyclePhase.PEAK)
            # Or back to recovery if distortion disappears
            elif (
                self.state.rate_distortion < 0.01
                and self.state.phase_duration >= self.min_phase_duration
            ):
                self._transition_to(CyclePhase.RECOVERY)

        elif current_phase == CyclePhase.PEAK:
            # Bust is inevitable once at peak
            if self.state.phase_duration >= self.min_phase_duration:
                self._transition_to(CyclePhase.BUST)

        elif current_phase == CyclePhase.BUST:
            # Calculate correction needed
            self.state.correction_needed = self.state.malinvestment_accumulated

            # Move to trough when liquidation is substantial
            if (
                liquidations > 0
                and self.state.phase_duration >= self.min_phase_duration * 2
            ):
                self._transition_to(CyclePhase.TROUGH)

        elif current_phase == CyclePhase.TROUGH:
            # Recovery starts when malinvestment is cleared
            remaining_malinvestment = self.state.malinvestment_accumulated
            if (
                remaining_malinvestment < self.state.correction_needed * Decimal("0.3")
                and self.state.phase_duration >= self.min_phase_duration
            ):
                self._transition_to(CyclePhase.RECOVERY)

    def _transition_to(self, new_phase: CyclePhase):
        """Transition to a new phase."""
        self.state.phase = new_phase
        self.state.phase_duration = 0

        if new_phase == CyclePhase.BUST:
            # Calculate unsustainable projects that will fail
            self.state.unsustainable_projects = int(
                float(self.state.malinvestment_accumulated) / 50000
            )

    def _get_cycle_signals(self) -> dict[str, Any]:
        """
        Get signals about the current cycle state.

        Used by agents and the simulation engine.
        """
        phase = self.state.phase

        # Investment signals
        if phase == CyclePhase.BOOM:
            investment_signal = "caution"  # Smart investors see the distortion
            credit_signal = "expanding"
        elif phase == CyclePhase.PEAK:
            investment_signal = "danger"
            credit_signal = "maximum"
        elif phase == CyclePhase.BUST:
            investment_signal = "opportunity"  # Buy distressed assets
            credit_signal = "contracting"
        elif phase == CyclePhase.TROUGH:
            investment_signal = "accumulate"
            credit_signal = "minimal"
        else:  # RECOVERY
            investment_signal = "normal"
            credit_signal = "stable"

        # Austrian recommendation
        if self.state.rate_distortion > 0.02:
            recommendation = "Central bank is distorting rates. Expect correction."
        elif phase in [CyclePhase.BOOM, CyclePhase.PEAK]:
            recommendation = (
                "Artificial boom in progress. Hold real assets (gold, BTC)."
            )
        elif phase == CyclePhase.BUST:
            recommendation = "Correction underway. Liquidation is healthy."
        else:
            recommendation = "Economy relatively stable."

        return {
            "phase": phase.value,
            "phase_duration_weeks": self.state.phase_duration,
            "boom_intensity": self.state.boom_intensity,
            "rate_distortion": self.state.rate_distortion,
            "natural_rate": self.state.natural_rate,
            "market_rate": self.state.market_rate,
            "credit_growth": self.state.credit_growth_rate,
            "malinvestment": str(self.state.malinvestment_accumulated),
            "investment_signal": investment_signal,
            "credit_signal": credit_signal,
            "recommendation": recommendation,
            "unsustainable_projects": self.state.unsustainable_projects,
            "correction_needed": str(self.state.correction_needed),
        }

    def get_phase_description(self) -> str:
        """Get human-readable description of current phase."""
        descriptions = {
            CyclePhase.RECOVERY: (
                "Recovery: Economy rebuilding after correction. "
                "Malinvestment has been liquidated. New sustainable investments beginning."
            ),
            CyclePhase.BOOM: (
                "Boom: Credit expansion underway. Interest rates artificially low. "
                "Entrepreneurs misled into unsustainable projects. "
                "Higher-order capital goods favored."
            ),
            CyclePhase.PEAK: (
                "Peak: Maximum distortion reached. Malinvestment at highest. "
                "Resource constraints becoming apparent. "
                "Bust is imminent."
            ),
            CyclePhase.BUST: (
                "Bust: Reality asserts itself. Malinvestments revealed as errors. "
                "Liquidation of unsustainable projects. "
                "Necessary correction, not a disease."
            ),
            CyclePhase.TROUGH: (
                "Trough: Bottom of recession. Most malinvestment cleared. "
                "Resources being reallocated to sustainable uses. "
                "Recovery approaching."
            ),
        }
        return descriptions.get(self.state.phase, "Unknown phase")

    def calculate_damage_from_cycle(self) -> dict[str, Decimal]:
        """
        Calculate economic damage caused by the artificial cycle.

        Austrian Theory:
        - All damage is caused by initial distortion (CB)
        - Bust is not damage, it's cure
        - Real damage is resources wasted in malinvestment
        """
        # Malinvestment is pure waste
        wasted_capital = self.state.malinvestment_accumulated

        # Higher-order goods overinvestment
        if self.state.capital_goods_ratio > 0.6:
            excess = Decimal(str(self.state.capital_goods_ratio - 0.5))
            overinvestment = self.state.cumulative_credit_expansion * excess
        else:
            overinvestment = Decimal("0")

        # Distortion damage (opportunity cost)
        distortion_damage = self.state.cumulative_credit_expansion * Decimal(
            str(self.state.rate_distortion)
        )

        return {
            "wasted_capital": wasted_capital,
            "overinvestment": overinvestment,
            "distortion_damage": distortion_damage,
            "total_cycle_damage": wasted_capital + overinvestment + distortion_damage,
        }
