"""
Country Registry

Central registry of all 20 countries in the multi-country simulation.
Provides access to country configurations and helper functions.
"""

from decimal import Decimal
from typing import Dict, List, Optional

from src.agents.government import RegimeType
from src.countries.base import CountryConfig, BilateralRelationship, RelationType


# =============================================================================
# COUNTRY CONFIGURATIONS (20 countries)
# =============================================================================

COUNTRY_REGISTRY: Dict[str, CountryConfig] = {
    # ---------------------------------------------------------------------
    # AMERICAS
    # ---------------------------------------------------------------------
    "USA": CountryConfig(
        code="USA",
        name="United States",
        currency="USD",
        regime_type=RegimeType.DEMOCRACY_LIBERAL,
        intervention_level=0.50,
        gdp_nominal_usd=Decimal("27000000000000"),  # $27T
        population=335_000_000,
        unemployment_rate=0.04,
        inflation_rate=0.03,
        real_inflation_estimate=0.08,
        central_bank_name="Federal Reserve",
        base_money=Decimal("6000000000000"),  # $6T
        policy_rate=0.055,
        inflation_target=0.02,
        debt_to_gdp=1.23,
        tax_burden=0.27,
        trade_openness=0.28,
        main_exports=["machinery", "electronics", "aircraft", "oil", "pharmaceuticals"],
        main_imports=["electronics", "machinery", "vehicles", "oil", "pharmaceuticals"],
        stock_index_ticker="^GSPC",
        currency_ticker=None,  # Base currency
        bond_yield_ticker="^TNX",
        historical_context={
            "reserve_currency": True,
            "military_hegemony": True,
            "tech_leader": True,
            "energy_producer": True,
        },
        usd_sensitivity=0.0,  # Is the USD
        inflation_memory=0.3,
        commodity_exposure=0.3,
        geopolitical_volatility=0.2,
    ),

    "CAN": CountryConfig(
        code="CAN",
        name="Canada",
        currency="CAD",
        regime_type=RegimeType.DEMOCRACY_SOCIALIST,
        intervention_level=0.65,
        gdp_nominal_usd=Decimal("2100000000000"),  # $2.1T
        population=40_000_000,
        unemployment_rate=0.055,
        inflation_rate=0.03,
        real_inflation_estimate=0.07,
        central_bank_name="Bank of Canada",
        base_money=Decimal("300000000000"),  # CAD 300B
        policy_rate=0.05,
        inflation_target=0.02,
        debt_to_gdp=1.06,
        tax_burden=0.33,
        trade_openness=0.65,
        main_exports=["oil", "vehicles", "machinery", "gold", "lumber"],
        main_imports=["vehicles", "machinery", "electronics", "plastics"],
        stock_index_ticker="^GSPTSE",
        currency_ticker="CADUSD=X",
        historical_context={
            "us_integration": True,
            "resource_economy": True,
            "stable_banking": True,
        },
        usd_sensitivity=0.7,
        inflation_memory=0.2,
        commodity_exposure=0.7,
        geopolitical_volatility=0.1,
    ),

    "MEX": CountryConfig(
        code="MEX",
        name="Mexico",
        currency="MXN",
        regime_type=RegimeType.DEMOCRACY_SOCIALIST,
        intervention_level=0.60,
        gdp_nominal_usd=Decimal("1300000000000"),  # $1.3T
        population=130_000_000,
        unemployment_rate=0.03,
        inflation_rate=0.045,
        real_inflation_estimate=0.10,
        central_bank_name="Banco de Mexico",
        base_money=Decimal("2500000000000"),  # MXN 2.5T
        policy_rate=0.11,
        inflation_target=0.03,
        debt_to_gdp=0.55,
        tax_burden=0.17,
        trade_openness=0.80,
        main_exports=["vehicles", "electronics", "machinery", "oil", "vegetables"],
        main_imports=["electronics", "machinery", "steel", "plastics", "vehicles"],
        stock_index_ticker="^MXX",
        currency_ticker="MXNUSD=X",
        historical_context={
            "us_trade_partner": True,
            "peso_crisis_1994": True,
            "nearshoring_beneficiary": True,
        },
        usd_sensitivity=0.8,
        inflation_memory=0.6,
        commodity_exposure=0.5,
        geopolitical_volatility=0.4,
    ),

    "BRA": CountryConfig(
        code="BRA",
        name="Brazil",
        currency="BRL",
        regime_type=RegimeType.DEMOCRACY_SOCIALIST,
        intervention_level=0.65,
        gdp_nominal_usd=Decimal("2100000000000"),  # $2.1T
        population=215_000_000,
        unemployment_rate=0.08,
        inflation_rate=0.045,
        real_inflation_estimate=0.12,
        central_bank_name="Banco Central do Brasil",
        base_money=Decimal("500000000000"),  # BRL 500B
        policy_rate=0.1075,
        inflation_target=0.03,
        debt_to_gdp=0.75,
        tax_burden=0.33,
        trade_openness=0.32,
        main_exports=["soybeans", "iron_ore", "oil", "beef", "sugar"],
        main_imports=["electronics", "machinery", "chemicals", "oil_products"],
        stock_index_ticker="^BVSP",
        currency_ticker="BRLUSD=X",
        historical_context={
            "hyperinflation_history": True,
            "plano_real_1994": True,
            "commodity_dependent": True,
            "china_trade_dependent": True,
            "political_instability": "moderate",
        },
        usd_sensitivity=0.6,
        inflation_memory=0.7,
        commodity_exposure=0.8,
        geopolitical_volatility=0.5,
    ),

    "ARG": CountryConfig(
        code="ARG",
        name="Argentina",
        currency="ARS",
        regime_type=RegimeType.DEMOCRACY_SOCIALIST,
        intervention_level=0.75,
        gdp_nominal_usd=Decimal("630000000000"),  # $630B
        population=46_000_000,
        unemployment_rate=0.07,
        inflation_rate=0.25,  # Official understates reality
        real_inflation_estimate=0.50,
        central_bank_name="Banco Central de Argentina",
        base_money=Decimal("10000000000000"),  # ARS 10T (hyperinflated)
        policy_rate=0.40,
        inflation_target=0.05,  # Never achieved
        debt_to_gdp=0.85,
        tax_burden=0.30,
        trade_openness=0.35,
        main_exports=["soybeans", "corn", "beef", "wheat", "lithium"],
        main_imports=["machinery", "vehicles", "chemicals", "electronics"],
        stock_index_ticker="^MERV",
        currency_ticker="ARSUSD=X",
        historical_context={
            "serial_defaulter": True,
            "hyperinflation_multiple": True,
            "imf_history": True,
            "capital_controls": True,
            "dollarization_informal": True,
            "peronism": True,
        },
        usd_sensitivity=0.95,
        inflation_memory=0.98,
        commodity_exposure=0.7,
        geopolitical_volatility=0.7,
    ),

    # ---------------------------------------------------------------------
    # EUROPE
    # ---------------------------------------------------------------------
    "GBR": CountryConfig(
        code="GBR",
        name="United Kingdom",
        currency="GBP",
        regime_type=RegimeType.DEMOCRACY_SOCIALIST,
        intervention_level=0.60,
        gdp_nominal_usd=Decimal("3100000000000"),  # $3.1T
        population=67_000_000,
        unemployment_rate=0.04,
        inflation_rate=0.04,
        real_inflation_estimate=0.08,
        central_bank_name="Bank of England",
        base_money=Decimal("900000000000"),  # GBP 900B
        policy_rate=0.0525,
        inflation_target=0.02,
        debt_to_gdp=1.01,
        tax_burden=0.35,
        trade_openness=0.60,
        main_exports=["machinery", "pharmaceuticals", "vehicles", "gold", "oil"],
        main_imports=["machinery", "vehicles", "electronics", "gold", "pharmaceuticals"],
        stock_index_ticker="^FTSE",
        currency_ticker="GBPUSD=X",
        historical_context={
            "former_empire": True,
            "financial_center": True,
            "brexit": True,
            "pound_crises": True,
        },
        usd_sensitivity=0.4,
        inflation_memory=0.4,
        commodity_exposure=0.3,
        geopolitical_volatility=0.3,
    ),

    "DEU": CountryConfig(
        code="DEU",
        name="Germany",
        currency="EUR",
        regime_type=RegimeType.DEMOCRACY_SOCIALIST,
        intervention_level=0.65,
        gdp_nominal_usd=Decimal("4200000000000"),  # $4.2T
        population=84_000_000,
        unemployment_rate=0.03,
        inflation_rate=0.025,
        real_inflation_estimate=0.06,
        central_bank_name="Bundesbank (via ECB)",
        base_money=Decimal("3000000000000"),  # EUR 3T
        policy_rate=0.045,
        inflation_target=0.02,
        debt_to_gdp=0.66,
        tax_burden=0.40,
        trade_openness=0.88,
        main_exports=["vehicles", "machinery", "chemicals", "electronics", "pharmaceuticals"],
        main_imports=["machinery", "oil", "vehicles", "chemicals", "electronics"],
        stock_index_ticker="^GDAXI",
        currency_ticker="EURUSD=X",
        historical_context={
            "hyperinflation_1920s": True,
            "export_powerhouse": True,
            "eu_leader": True,
            "energy_transition": True,
            "russia_energy_dependent": True,
        },
        usd_sensitivity=0.4,
        inflation_memory=0.8,  # Weimar memory
        commodity_exposure=0.4,
        geopolitical_volatility=0.2,
    ),

    "FRA": CountryConfig(
        code="FRA",
        name="France",
        currency="EUR",
        regime_type=RegimeType.DEMOCRACY_SOCIALIST,
        intervention_level=0.70,
        gdp_nominal_usd=Decimal("2800000000000"),  # $2.8T
        population=68_000_000,
        unemployment_rate=0.07,
        inflation_rate=0.03,
        real_inflation_estimate=0.07,
        central_bank_name="Banque de France (via ECB)",
        base_money=Decimal("2500000000000"),  # EUR 2.5T
        policy_rate=0.045,
        inflation_target=0.02,
        debt_to_gdp=1.12,
        tax_burden=0.47,
        trade_openness=0.65,
        main_exports=["aircraft", "pharmaceuticals", "vehicles", "wine", "machinery"],
        main_imports=["machinery", "vehicles", "oil", "electronics", "chemicals"],
        stock_index_ticker="^FCHI",
        currency_ticker="EURUSD=X",
        historical_context={
            "nuclear_power": True,
            "high_social_spending": True,
            "labor_rigidity": True,
            "eu_founder": True,
        },
        usd_sensitivity=0.4,
        inflation_memory=0.3,
        commodity_exposure=0.3,
        geopolitical_volatility=0.3,
    ),

    "SWE": CountryConfig(
        code="SWE",
        name="Sweden",
        currency="SEK",
        regime_type=RegimeType.DEMOCRACY_SOCIALIST,
        intervention_level=0.70,
        gdp_nominal_usd=Decimal("580000000000"),  # $580B
        population=10_500_000,
        unemployment_rate=0.08,
        inflation_rate=0.03,
        real_inflation_estimate=0.06,
        central_bank_name="Sveriges Riksbank",
        base_money=Decimal("500000000000"),  # SEK 500B
        policy_rate=0.04,
        inflation_target=0.02,
        debt_to_gdp=0.33,
        tax_burden=0.44,
        trade_openness=0.90,
        main_exports=["machinery", "vehicles", "wood", "iron", "electronics"],
        main_imports=["machinery", "oil", "vehicles", "chemicals", "electronics"],
        stock_index_ticker="^OMX",
        currency_ticker="SEKUSD=X",
        historical_context={
            "nordic_model": True,
            "high_taxes": True,
            "innovation_leader": True,
            "nato_new_member": True,
        },
        usd_sensitivity=0.5,
        inflation_memory=0.2,
        commodity_exposure=0.4,
        geopolitical_volatility=0.3,
    ),

    "NOR": CountryConfig(
        code="NOR",
        name="Norway",
        currency="NOK",
        regime_type=RegimeType.DEMOCRACY_SOCIALIST,
        intervention_level=0.65,
        gdp_nominal_usd=Decimal("480000000000"),  # $480B
        population=5_500_000,
        unemployment_rate=0.035,
        inflation_rate=0.035,
        real_inflation_estimate=0.06,
        central_bank_name="Norges Bank",
        base_money=Decimal("200000000000"),  # NOK 200B
        policy_rate=0.045,
        inflation_target=0.02,
        debt_to_gdp=0.42,
        tax_burden=0.42,
        trade_openness=0.75,
        main_exports=["oil", "natural_gas", "fish", "metals", "machinery"],
        main_imports=["machinery", "vehicles", "electronics", "metals", "food"],
        stock_index_ticker="^OSEAX",
        currency_ticker="NOKUSD=X",
        historical_context={
            "oil_wealth": True,
            "sovereign_wealth_fund": True,
            "not_eu_member": True,
            "high_living_standard": True,
        },
        usd_sensitivity=0.5,
        inflation_memory=0.2,
        commodity_exposure=0.8,  # Oil dependent
        geopolitical_volatility=0.1,
    ),

    "CHE": CountryConfig(
        code="CHE",
        name="Switzerland",
        currency="CHF",
        regime_type=RegimeType.MINARCHY,
        intervention_level=0.25,
        gdp_nominal_usd=Decimal("810000000000"),  # $810B
        population=8_800_000,
        unemployment_rate=0.02,
        inflation_rate=0.015,
        real_inflation_estimate=0.03,
        central_bank_name="Swiss National Bank",
        base_money=Decimal("700000000000"),  # CHF 700B
        policy_rate=0.0175,
        inflation_target=0.02,
        debt_to_gdp=0.40,
        tax_burden=0.28,
        trade_openness=1.20,  # Very open
        main_exports=["pharmaceuticals", "watches", "machinery", "chemicals", "gold"],
        main_imports=["machinery", "chemicals", "vehicles", "gold", "pharmaceuticals"],
        stock_index_ticker="^SSMI",
        currency_ticker="CHFUSD=X",
        historical_context={
            "safe_haven": True,
            "banking_secrecy": True,
            "neutrality": True,
            "direct_democracy": True,
            "low_taxes": True,
            "canton_autonomy": True,
        },
        usd_sensitivity=0.3,
        inflation_memory=0.1,
        commodity_exposure=0.2,
        geopolitical_volatility=0.05,
    ),

    "LIE": CountryConfig(
        code="LIE",
        name="Liechtenstein",
        currency="CHF",  # Uses Swiss Franc
        regime_type=RegimeType.MONARCHY,
        intervention_level=0.20,
        gdp_nominal_usd=Decimal("7000000000"),  # $7B
        population=40_000,
        unemployment_rate=0.015,
        inflation_rate=0.015,
        real_inflation_estimate=0.03,
        central_bank_name="Uses Swiss National Bank",
        base_money=Decimal("0"),  # No own currency
        policy_rate=0.0175,  # SNB rate
        inflation_target=0.02,
        debt_to_gdp=0.00,  # No public debt
        tax_burden=0.20,
        trade_openness=1.50,
        main_exports=["machinery", "dental_products", "stamps", "hardware"],
        main_imports=["machinery", "food", "vehicles", "metals"],
        stock_index_ticker="^SSMI",  # Use Swiss index as proxy
        currency_ticker="CHFUSD=X",
        historical_context={
            "hereditary_monarchy": True,
            "tax_haven": True,
            "financial_services": True,
            "smallest_german_speaking": True,
            "hoppe_ideal_small_state": True,
        },
        usd_sensitivity=0.3,
        inflation_memory=0.1,
        commodity_exposure=0.1,
        geopolitical_volatility=0.02,
    ),

    # ---------------------------------------------------------------------
    # ASIA
    # ---------------------------------------------------------------------
    "CHN": CountryConfig(
        code="CHN",
        name="China",
        currency="CNY",
        regime_type=RegimeType.TOTALITARIAN,
        intervention_level=0.90,
        gdp_nominal_usd=Decimal("18000000000000"),  # $18T
        population=1_400_000_000,
        unemployment_rate=0.05,
        inflation_rate=0.02,
        real_inflation_estimate=0.08,
        central_bank_name="People's Bank of China",
        base_money=Decimal("300000000000000"),  # CNY 300T
        policy_rate=0.035,
        inflation_target=0.03,
        debt_to_gdp=0.77,  # Official, actual much higher
        tax_burden=0.22,
        trade_openness=0.38,
        main_exports=["electronics", "machinery", "furniture", "textiles", "plastics"],
        main_imports=["semiconductors", "oil", "iron_ore", "soybeans", "machinery"],
        stock_index_ticker="000001.SS",
        currency_ticker="CNYUSD=X",
        historical_context={
            "state_capitalism": True,
            "manufacturing_hub": True,
            "belt_road_initiative": True,
            "taiwan_conflict": True,
            "tech_competition": True,
            "property_bubble": True,
        },
        usd_sensitivity=0.5,
        inflation_memory=0.3,
        commodity_exposure=0.6,
        geopolitical_volatility=0.6,
    ),

    "JPN": CountryConfig(
        code="JPN",
        name="Japan",
        currency="JPY",
        regime_type=RegimeType.DEMOCRACY_LIBERAL,
        intervention_level=0.55,
        gdp_nominal_usd=Decimal("4200000000000"),  # $4.2T
        population=124_000_000,
        unemployment_rate=0.025,
        inflation_rate=0.03,
        real_inflation_estimate=0.05,
        central_bank_name="Bank of Japan",
        base_money=Decimal("700000000000000"),  # JPY 700T
        policy_rate=0.001,  # Near zero for decades
        inflation_target=0.02,
        debt_to_gdp=2.60,  # Highest in world
        tax_burden=0.32,
        trade_openness=0.37,
        main_exports=["vehicles", "machinery", "electronics", "steel", "chemicals"],
        main_imports=["oil", "natural_gas", "electronics", "machinery", "food"],
        stock_index_ticker="^N225",
        currency_ticker="JPYUSD=X",
        historical_context={
            "lost_decades": True,
            "deflation_history": True,
            "aging_population": True,
            "yield_curve_control": True,
            "us_alliance": True,
            "china_rival": True,
        },
        usd_sensitivity=0.6,
        inflation_memory=0.2,  # Deflation memory
        commodity_exposure=0.5,
        geopolitical_volatility=0.3,
    ),

    "IND": CountryConfig(
        code="IND",
        name="India",
        currency="INR",
        regime_type=RegimeType.DEMOCRACY_SOCIALIST,
        intervention_level=0.60,
        gdp_nominal_usd=Decimal("3700000000000"),  # $3.7T
        population=1_400_000_000,
        unemployment_rate=0.08,
        inflation_rate=0.05,
        real_inflation_estimate=0.10,
        central_bank_name="Reserve Bank of India",
        base_money=Decimal("30000000000000"),  # INR 30T
        policy_rate=0.065,
        inflation_target=0.04,
        debt_to_gdp=0.82,
        tax_burden=0.18,
        trade_openness=0.42,
        main_exports=["software", "pharmaceuticals", "jewelry", "textiles", "rice"],
        main_imports=["oil", "gold", "electronics", "machinery", "chemicals"],
        stock_index_ticker="^BSESN",
        currency_ticker="INRUSD=X",
        historical_context={
            "tech_services": True,
            "young_population": True,
            "china_border_disputes": True,
            "growing_middle_class": True,
            "bureaucracy": True,
        },
        usd_sensitivity=0.6,
        inflation_memory=0.5,
        commodity_exposure=0.6,
        geopolitical_volatility=0.5,
    ),

    "IDN": CountryConfig(
        code="IDN",
        name="Indonesia",
        currency="IDR",
        regime_type=RegimeType.DEMOCRACY_SOCIALIST,
        intervention_level=0.55,
        gdp_nominal_usd=Decimal("1400000000000"),  # $1.4T
        population=275_000_000,
        unemployment_rate=0.055,
        inflation_rate=0.04,
        real_inflation_estimate=0.08,
        central_bank_name="Bank Indonesia",
        base_money=Decimal("8000000000000000"),  # IDR 8,000T
        policy_rate=0.06,
        inflation_target=0.03,
        debt_to_gdp=0.40,
        tax_burden=0.12,
        trade_openness=0.40,
        main_exports=["palm_oil", "coal", "nickel", "rubber", "textiles"],
        main_imports=["machinery", "chemicals", "oil", "steel", "electronics"],
        stock_index_ticker="^JKSE",
        currency_ticker="IDRUSD=X",
        historical_context={
            "asian_crisis_1997": True,
            "commodity_exporter": True,
            "largest_muslim_country": True,
            "archipelago_challenges": True,
        },
        usd_sensitivity=0.7,
        inflation_memory=0.6,
        commodity_exposure=0.7,
        geopolitical_volatility=0.4,
    ),

    # ---------------------------------------------------------------------
    # MIDDLE EAST
    # ---------------------------------------------------------------------
    "SAU": CountryConfig(
        code="SAU",
        name="Saudi Arabia",
        currency="SAR",
        regime_type=RegimeType.MONARCHY,
        intervention_level=0.40,
        gdp_nominal_usd=Decimal("1100000000000"),  # $1.1T
        population=35_000_000,
        unemployment_rate=0.05,
        inflation_rate=0.025,
        real_inflation_estimate=0.05,
        central_bank_name="Saudi Central Bank",
        base_money=Decimal("500000000000"),  # SAR 500B
        policy_rate=0.055,  # Pegged to USD
        inflation_target=0.02,
        debt_to_gdp=0.25,
        tax_burden=0.05,  # No income tax
        trade_openness=0.70,
        main_exports=["oil", "petrochemicals", "plastics"],
        main_imports=["machinery", "vehicles", "food", "electronics", "textiles"],
        stock_index_ticker="^TASI",
        currency_ticker="SARUSD=X",
        historical_context={
            "oil_kingdom": True,
            "opec_leader": True,
            "vision_2030": True,
            "us_alliance": True,
            "iran_rival": True,
            "sar_usd_peg": True,
        },
        usd_sensitivity=0.2,  # Pegged to USD
        inflation_memory=0.2,
        commodity_exposure=0.95,  # Almost entirely oil
        geopolitical_volatility=0.5,
    ),

    "ARE": CountryConfig(
        code="ARE",
        name="United Arab Emirates",
        currency="AED",
        regime_type=RegimeType.MONARCHY,
        intervention_level=0.35,
        gdp_nominal_usd=Decimal("500000000000"),  # $500B
        population=10_000_000,
        unemployment_rate=0.03,
        inflation_rate=0.03,
        real_inflation_estimate=0.05,
        central_bank_name="Central Bank of UAE",
        base_money=Decimal("200000000000"),  # AED 200B
        policy_rate=0.055,  # Pegged to USD
        inflation_target=0.02,
        debt_to_gdp=0.35,
        tax_burden=0.10,
        trade_openness=1.80,  # Huge trading hub
        main_exports=["oil", "gold", "diamonds", "machinery", "aluminum"],
        main_imports=["gold", "diamonds", "machinery", "vehicles", "electronics"],
        stock_index_ticker="^DFMGI",
        currency_ticker="AEDUSD=X",
        historical_context={
            "trading_hub": True,
            "diversified_economy": True,
            "tourism": True,
            "low_taxes": True,
            "aed_usd_peg": True,
        },
        usd_sensitivity=0.2,
        inflation_memory=0.2,
        commodity_exposure=0.6,
        geopolitical_volatility=0.3,
    ),

    "TUR": CountryConfig(
        code="TUR",
        name="Turkey",
        currency="TRY",
        regime_type=RegimeType.TOTALITARIAN,  # Erdogan autocracy
        intervention_level=0.75,
        gdp_nominal_usd=Decimal("900000000000"),  # $900B
        population=85_000_000,
        unemployment_rate=0.10,
        inflation_rate=0.50,  # Severe inflation
        real_inflation_estimate=0.80,
        central_bank_name="Central Bank of Turkey",
        base_money=Decimal("3000000000000"),  # TRY 3T
        policy_rate=0.50,
        inflation_target=0.05,  # Laughable
        debt_to_gdp=0.35,
        tax_burden=0.25,
        trade_openness=0.60,
        main_exports=["vehicles", "machinery", "textiles", "steel", "food"],
        main_imports=["oil", "gold", "machinery", "electronics", "chemicals"],
        stock_index_ticker="XU100.IS",
        currency_ticker="TRYUSD=X",
        historical_context={
            "erdogan_economics": True,
            "lira_crisis": True,
            "inflation_spiral": True,
            "nato_member": True,
            "geopolitical_pivot": True,
            "cb_not_independent": True,
        },
        usd_sensitivity=0.9,
        inflation_memory=0.85,
        commodity_exposure=0.5,
        geopolitical_volatility=0.8,
    ),

    # ---------------------------------------------------------------------
    # RUSSIA
    # ---------------------------------------------------------------------
    "RUS": CountryConfig(
        code="RUS",
        name="Russia",
        currency="RUB",
        regime_type=RegimeType.TOTALITARIAN,
        intervention_level=0.85,
        gdp_nominal_usd=Decimal("2200000000000"),  # $2.2T
        population=145_000_000,
        unemployment_rate=0.03,
        inflation_rate=0.08,
        real_inflation_estimate=0.15,
        central_bank_name="Central Bank of Russia",
        base_money=Decimal("15000000000000"),  # RUB 15T
        policy_rate=0.16,
        inflation_target=0.04,
        debt_to_gdp=0.20,
        tax_burden=0.28,
        trade_openness=0.45,
        main_exports=["oil", "natural_gas", "metals", "wheat", "weapons"],
        main_imports=["machinery", "vehicles", "electronics", "pharmaceuticals", "food"],
        stock_index_ticker="IMOEX.ME",
        currency_ticker="RUBUSD=X",
        historical_context={
            "sanctions_target": True,
            "oil_gas_dependent": True,
            "ukraine_war": True,
            "china_pivot": True,
            "soviet_legacy": True,
            "reserve_fortress": True,
        },
        usd_sensitivity=0.7,
        inflation_memory=0.8,
        commodity_exposure=0.85,
        geopolitical_volatility=0.9,
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_country_config(code: str) -> Optional[CountryConfig]:
    """Get configuration for a specific country by ISO code"""
    return COUNTRY_REGISTRY.get(code.upper())


def get_all_countries() -> List[str]:
    """Get list of all country codes"""
    return list(COUNTRY_REGISTRY.keys())


def get_countries_by_regime(regime_type: RegimeType) -> List[str]:
    """Get all countries with a specific regime type"""
    return [
        code for code, config in COUNTRY_REGISTRY.items()
        if config.regime_type == regime_type
    ]


def get_countries_by_region(region: str) -> List[str]:
    """Get countries by region"""
    regions = {
        "americas": ["USA", "CAN", "MEX", "BRA", "ARG"],
        "europe": ["GBR", "DEU", "FRA", "SWE", "NOR", "CHE", "LIE"],
        "asia": ["CHN", "JPN", "IND", "IDN"],
        "middle_east": ["SAU", "ARE", "TUR"],
        "eurasia": ["RUS"],
    }
    return regions.get(region.lower(), [])


def get_high_intervention_countries() -> List[str]:
    """Get countries with intervention level > 0.6"""
    return [
        code for code, config in COUNTRY_REGISTRY.items()
        if config.intervention_level > 0.6
    ]


def get_low_intervention_countries() -> List[str]:
    """Get countries with intervention level < 0.4 (Hoppe's better ones)"""
    return [
        code for code, config in COUNTRY_REGISTRY.items()
        if config.intervention_level < 0.4
    ]
