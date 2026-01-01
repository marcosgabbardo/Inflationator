"""
Country Configuration Module

Provides configuration for all 20 countries in the multi-country simulation.
Each country has economic parameters, regime type, central bank info, and
historical context that influences their behavior in the simulation.
"""

from src.countries.base import CountryConfig, RelationType
from src.countries.registry import (
    COUNTRY_REGISTRY,
    get_country_config,
    get_all_countries,
    get_countries_by_regime,
)

__all__ = [
    "CountryConfig",
    "RelationType",
    "COUNTRY_REGISTRY",
    "get_country_config",
    "get_all_countries",
    "get_countries_by_regime",
]
