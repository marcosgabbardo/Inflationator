"""
Inflationator Configuration Settings

All configurable parameters for the simulation.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from enum import Enum


class RegimeType(str, Enum):
    """Political regime types (worst to best according to Hoppe)"""
    TOTALITARIAN = "totalitarian"      # 100% intervention
    DEMOCRACY_SOCIALIST = "democracy_socialist"  # 80%
    DEMOCRACY_LIBERAL = "democracy_liberal"      # 50%
    MONARCHY = "monarchy"              # 30%
    MINARCHY = "minarchy"              # 10%
    ANCAP = "ancap"                    # 0% - ideal


class Settings(BaseSettings):
    """Main application settings"""

    # Database
    db_host: str = Field(default="localhost", description="MySQL host")
    db_port: int = Field(default=3306, description="MySQL port")
    db_name: str = Field(default="inflationator", description="Database name")
    db_user: str = Field(default="root", description="Database user")
    db_password: str = Field(default="", description="Database password")

    # Simulation Parameters
    simulation_tick_weeks: int = Field(
        default=1,
        description="Each tick represents N weeks"
    )

    # Agent Scale (MVP - USA)
    num_persons: int = Field(default=100_000, description="Number of person agents")
    num_companies: int = Field(default=10_000, description="Number of company agents")
    num_banks: int = Field(default=500, description="Number of private banks")

    # Initial Country
    initial_country: str = Field(default="USA", description="MVP country code")

    # Regime Settings (USA defaults to democracy_liberal)
    default_regime: RegimeType = Field(
        default=RegimeType.DEMOCRACY_LIBERAL,
        description="Default political regime"
    )

    # Central Bank (Villain) Settings
    central_bank_intervention: float = Field(
        default=0.5,  # 50% intervention level
        ge=0.0, le=1.0,
        description="Central bank intervention level (0=none, 1=maximum)"
    )

    # Government (Villain) Settings
    government_tax_income: float = Field(default=0.35, ge=0.0, le=1.0)
    government_tax_capital: float = Field(default=0.20, ge=0.0, le=1.0)
    government_tax_consumption: float = Field(default=0.08, ge=0.0, le=1.0)
    government_regulation_level: float = Field(default=0.50, ge=0.0, le=1.0)

    # Data Collection
    use_real_apis: bool = Field(
        default=True,
        description="Use real APIs (Bitcoin, commodities) or mocks"
    )
    coingecko_api_key: Optional[str] = Field(default=None)
    alpha_vantage_key: Optional[str] = Field(default=None)

    # Performance
    batch_size: int = Field(default=1000, description="Batch size for DB operations")
    use_parallel: bool = Field(default=True, description="Use parallel processing")

    class Config:
        env_prefix = "INFLATIONATOR_"
        env_file = ".env"


# Intervention levels by regime (Hoppe hierarchy)
REGIME_PARAMETERS = {
    RegimeType.TOTALITARIAN: {
        "intervention_level": 1.0,
        "tax_income": 0.90,
        "tax_capital": 0.90,
        "regulation": 1.0,
        "time_preference_modifier": 2.0,  # Very short-term thinking
        "description": "Total state control - worst case"
    },
    RegimeType.DEMOCRACY_SOCIALIST: {
        "intervention_level": 0.80,
        "tax_income": 0.70,
        "tax_capital": 0.50,
        "regulation": 0.80,
        "time_preference_modifier": 1.5,  # Short electoral cycles
        "description": "High intervention democracy"
    },
    RegimeType.DEMOCRACY_LIBERAL: {
        "intervention_level": 0.50,
        "tax_income": 0.35,
        "tax_capital": 0.20,
        "regulation": 0.50,
        "time_preference_modifier": 1.3,
        "description": "Moderate intervention democracy"
    },
    RegimeType.MONARCHY: {
        "intervention_level": 0.30,
        "tax_income": 0.25,
        "tax_capital": 0.10,
        "regulation": 0.30,
        "time_preference_modifier": 0.8,  # Longer-term (dynasty)
        "description": "Traditional monarchy - owner mentality"
    },
    RegimeType.MINARCHY: {
        "intervention_level": 0.10,
        "tax_income": 0.08,
        "tax_capital": 0.02,
        "regulation": 0.10,
        "time_preference_modifier": 0.5,
        "description": "Minimal state - night watchman"
    },
    RegimeType.ANCAP: {
        "intervention_level": 0.0,
        "tax_income": 0.0,
        "tax_capital": 0.0,
        "regulation": 0.0,
        "time_preference_modifier": 0.0,  # Natural market preference
        "description": "Anarcho-capitalism - Rothbard ideal"
    },
}


# Global settings instance
settings = Settings()
