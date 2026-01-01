"""
SQLAlchemy Database Models

All database tables for the Inflationator simulation.
Based on Austrian Economics principles.
"""

from sqlalchemy import (
    Column, String, Float, Integer, BigInteger, Boolean,
    DateTime, Text, Enum, ForeignKey, JSON, Index,
    DECIMAL
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum as PyEnum
import uuid

Base = declarative_base()


def generate_uuid() -> str:
    """Generate a UUID string"""
    return str(uuid.uuid4())


# ============================================
# ENUMS
# ============================================

class AgentType(str, PyEnum):
    PERSON = "person"
    COMPANY = "company"
    BANK = "bank"
    CENTRAL_BANK = "central_bank"
    GOVERNMENT = "government"


class RegimeType(str, PyEnum):
    ANCAP = "ancap"
    MINARCHY = "minarchy"
    MONARCHY = "monarchy"
    DEMOCRACY_LIBERAL = "democracy_liberal"
    DEMOCRACY_SOCIALIST = "democracy_socialist"
    TOTALITARIAN = "totalitarian"


class ConflictType(str, PyEnum):
    TRADE_WAR = "trade_war"
    SANCTIONS = "sanctions"
    WAR = "war"
    COLD_WAR = "cold_war"


class MarketType(str, PyEnum):
    LABOR = "labor"
    CONSUMER_GOODS = "consumer_goods"
    CAPITAL_GOODS = "capital_goods"
    COMMODITIES = "commodities"
    CRYPTO = "crypto"


# ============================================
# AGENTS
# ============================================

class Agent(Base):
    """Base agent table for all economic actors"""
    __tablename__ = "agents"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    type = Column(Enum(AgentType), nullable=False, index=True)
    country = Column(String(3), nullable=False, index=True)

    # Wealth & Holdings
    wealth = Column(DECIMAL(20, 8), default=0)  # In monetary units
    bitcoin = Column(DECIMAL(20, 8), default=0)  # BTC holdings
    gold = Column(DECIMAL(20, 8), default=0)    # Gold oz

    # Austrian Economics Parameters
    time_preference = Column(Float, default=0.5)  # 0=low (saves), 1=high (consumes)
    risk_tolerance = Column(Float, default=0.5)   # 0=conservative, 1=aggressive

    # State
    is_active = Column(Boolean, default=True)
    knowledge = Column(JSON, default=dict)      # Available information
    expectations = Column(JSON, default=dict)   # Future expectations

    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Indexes for performance
    __table_args__ = (
        Index('idx_agent_country_type', 'country', 'type'),
    )


class Person(Base):
    """Person agent - individual economic actor"""
    __tablename__ = "persons"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    agent_id = Column(String(36), ForeignKey("agents.id"), nullable=False)

    # Income & Labor
    income = Column(DECIMAL(20, 8), default=0)
    employed = Column(Boolean, default=True)
    employer_id = Column(String(36), ForeignKey("companies.id"), nullable=True)
    skill_level = Column(Float, default=0.5)  # 0-1

    # Consumption
    consumption_rate = Column(Float, default=0.7)  # % of income consumed
    savings_rate = Column(Float, default=0.3)      # % saved

    # Demographics (affects behavior)
    age_group = Column(String(20), default="adult")  # young, adult, senior

    # Relationships
    agent = relationship("Agent", backref="person_data")


class Company(Base):
    """Company agent - business entity"""
    __tablename__ = "companies"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    agent_id = Column(String(36), ForeignKey("agents.id"), nullable=False)

    name = Column(String(200))

    # Production
    sector = Column(String(50))  # agriculture, manufacturing, services, etc.
    production_type = Column(String(50))  # consumer_goods, capital_goods

    # Capital Structure (Hayekian)
    capital_stock = Column(DECIMAL(20, 8), default=0)
    order_of_goods = Column(Integer, default=1)  # 1=consumer, higher=capital

    # Labor
    num_employees = Column(Integer, default=0)
    wage_rate = Column(DECIMAL(20, 8), default=0)

    # Financial
    revenue = Column(DECIMAL(20, 8), default=0)
    costs = Column(DECIMAL(20, 8), default=0)
    debt = Column(DECIMAL(20, 8), default=0)
    interest_rate = Column(Float, default=0.05)

    # Relationships
    agent = relationship("Agent", backref="company_data")
    employees = relationship("Person", backref="employer", foreign_keys=[Person.employer_id])


class Bank(Base):
    """Private bank agent"""
    __tablename__ = "banks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    agent_id = Column(String(36), ForeignKey("agents.id"), nullable=False)

    name = Column(String(200))

    # Reserves & Lending
    reserves = Column(DECIMAL(20, 8), default=0)
    reserve_ratio = Column(Float, default=0.10)  # Fractional reserve
    total_deposits = Column(DECIMAL(20, 8), default=0)
    total_loans = Column(DECIMAL(20, 8), default=0)

    # Interest Rates (market-determined)
    deposit_rate = Column(Float, default=0.02)
    lending_rate = Column(Float, default=0.05)

    # Credit Creation (Austrian focus)
    credit_created = Column(DECIMAL(20, 8), default=0)  # Money created from nothing

    # Relationships
    agent = relationship("Agent", backref="bank_data")


class CentralBank(Base):
    """Central Bank - THE VILLAIN (monetary distortions)"""
    __tablename__ = "central_banks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    agent_id = Column(String(36), ForeignKey("agents.id"), nullable=False)
    country = Column(String(3), unique=True, nullable=False)

    name = Column(String(200), default="Federal Reserve")

    # Monetary Policy (All EVIL according to Austrians)
    base_money = Column(DECIMAL(20, 8), default=0)
    money_printed = Column(DECIMAL(20, 8), default=0)  # Total QE
    policy_rate = Column(Float, default=0.05)  # Artificial rate manipulation

    # Intervention Level (0=none, 1=maximum)
    intervention_level = Column(Float, default=0.5)

    # Balance Sheet (toxic assets bought)
    assets_purchased = Column(DECIMAL(20, 8), default=0)
    bailouts_given = Column(DECIMAL(20, 8), default=0)

    # DAMAGE METRICS (tracking the harm caused)
    inflation_caused = Column(DECIMAL(20, 8), default=0)
    malinvestment_induced = Column(DECIMAL(20, 8), default=0)
    bubbles_created = Column(Integer, default=0)
    economic_cycles_caused = Column(Integer, default=0)

    # Relationships
    agent = relationship("Agent", backref="central_bank_data")


class Government(Base):
    """Government - THE VILLAIN (political distortions)"""
    __tablename__ = "governments"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    agent_id = Column(String(36), ForeignKey("agents.id"), nullable=False)
    country = Column(String(3), unique=True, nullable=False)

    # Regime Type (Hoppe hierarchy: ancap=best, totalitarian=worst)
    regime_type = Column(Enum(RegimeType), default=RegimeType.DEMOCRACY_LIBERAL)
    intervention_level = Column(Float, default=0.5)  # 0-1

    # Taxation (extraction from productive economy)
    tax_rate_income = Column(Float, default=0.35)
    tax_rate_capital = Column(Float, default=0.20)
    tax_rate_consumption = Column(Float, default=0.08)
    tariff_rate = Column(Float, default=0.05)

    # Regulation & Bureaucracy
    regulation_level = Column(Float, default=0.50)  # 0-1
    bureaucracy_cost = Column(Float, default=0.10)  # % of GDP

    # Debt & Spending (always growing in democracy - Hoppe)
    debt_gdp_ratio = Column(Float, default=1.20)
    government_spending_gdp = Column(Float, default=0.40)
    deficit = Column(DECIMAL(20, 8), default=0)

    # International Relations
    at_war_with = Column(JSON, default=list)     # List of country codes
    sanctions_on = Column(JSON, default=list)    # List of country codes
    allies = Column(JSON, default=list)          # List of country codes

    # Policies & Metrics
    policies = Column(JSON, default=dict)        # Active policies
    damage_metrics = Column(JSON, default=dict)  # Harm caused

    # Freedom Index (inverse of intervention - higher=better)
    freedom_index = Column(Float, default=50.0)  # 0-100

    # Timestamps
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    agent = relationship("Agent", backref="government_data")


# ============================================
# MARKETS
# ============================================

class Market(Base):
    """Market for exchange of goods/services"""
    __tablename__ = "markets"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False)
    country = Column(String(3), nullable=False, index=True)
    market_type = Column(Enum(MarketType), nullable=False)

    # Pricing (emergent from supply/demand)
    current_price = Column(DECIMAL(20, 8), default=1.0)
    price_change_pct = Column(Float, default=0.0)

    # Volume
    volume_24h = Column(DECIMAL(20, 8), default=0)
    total_supply = Column(DECIMAL(20, 8), default=0)
    total_demand = Column(DECIMAL(20, 8), default=0)

    # Market Health
    liquidity = Column(Float, default=1.0)
    volatility = Column(Float, default=0.1)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Transaction(Base):
    """Record of economic transactions"""
    __tablename__ = "transactions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    simulation_tick = Column(BigInteger, nullable=False, index=True)

    buyer_id = Column(String(36), ForeignKey("agents.id"), nullable=False)
    seller_id = Column(String(36), ForeignKey("agents.id"), nullable=False)
    market_id = Column(String(36), ForeignKey("markets.id"), nullable=True)

    quantity = Column(DECIMAL(20, 8), nullable=False)
    price = Column(DECIMAL(20, 8), nullable=False)
    total_value = Column(DECIMAL(20, 8), nullable=False)

    transaction_type = Column(String(50))  # purchase, wage, loan, etc.

    created_at = Column(DateTime, default=func.now())

    # Indexes
    __table_args__ = (
        Index('idx_transaction_tick', 'simulation_tick'),
        Index('idx_transaction_buyer', 'buyer_id'),
        Index('idx_transaction_seller', 'seller_id'),
    )


# ============================================
# CONFLICTS & GEOPOLITICS
# ============================================

class Conflict(Base):
    """International conflicts and their economic impact"""
    __tablename__ = "conflicts"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    country_a = Column(String(3), nullable=False)
    country_b = Column(String(3), nullable=False)

    conflict_type = Column(Enum(ConflictType), nullable=False)
    severity = Column(Float, default=0.5)  # 0-1

    started_tick = Column(BigInteger, nullable=False)
    ended_tick = Column(BigInteger, nullable=True)
    is_active = Column(Boolean, default=True)

    # Economic Impact
    economic_impact = Column(JSON, default=dict)
    trade_disruption = Column(Float, default=0.0)  # % reduction in trade
    capital_destroyed = Column(DECIMAL(20, 8), default=0)

    created_at = Column(DateTime, default=func.now())


# ============================================
# SCENARIOS & SIMULATION
# ============================================

class Scenario(Base):
    """What-if scenarios for simulation"""
    __tablename__ = "scenarios"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(200), nullable=False)
    description = Column(Text)

    # Scenario Configuration
    parameters = Column(JSON, default=dict)
    is_baseline = Column(Boolean, default=False)

    # Branching
    parent_scenario_id = Column(String(36), ForeignKey("scenarios.id"), nullable=True)

    created_at = Column(DateTime, default=func.now())

    # Relationships
    parent = relationship("Scenario", remote_side=[id], backref="children")


class WorldSnapshot(Base):
    """Snapshot of world state at a simulation tick"""
    __tablename__ = "world_snapshots"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    scenario_id = Column(String(36), ForeignKey("scenarios.id"), nullable=False)
    simulation_tick = Column(BigInteger, nullable=False)

    # Aggregated State
    state = Column(JSON, default=dict)

    # Key Metrics (Austrian-focused)
    metrics = Column(JSON, default=dict)
    """
    Expected metrics structure:
    {
        "inflation_rate": float,  # Our calculation, not government's
        "money_supply": float,
        "credit_expansion": float,
        "bitcoin_price": float,
        "gold_price": float,
        "unemployment": float,
        "central_bank_damage": float,
        "government_damage": float,
        "freedom_index": float,
        "malinvestment_level": float
    }
    """

    created_at = Column(DateTime, default=func.now())

    # Indexes
    __table_args__ = (
        Index('idx_snapshot_scenario_tick', 'scenario_id', 'simulation_tick'),
    )


# ============================================
# EXTERNAL DATA
# ============================================

class ExternalData(Base):
    """Data from external sources (APIs)"""
    __tablename__ = "external_data"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    source = Column(String(100), nullable=False)  # coingecko, yahoo, etc.
    data_type = Column(String(50), nullable=False)  # bitcoin_price, gold_price, etc.

    timestamp = Column(DateTime, nullable=False, index=True)
    value = Column(DECIMAL(20, 8), nullable=False)

    meta_info = Column(JSON, default=dict)  # renamed from 'metadata' (reserved)

    created_at = Column(DateTime, default=func.now())

    # Indexes
    __table_args__ = (
        Index('idx_external_source_type', 'source', 'data_type'),
    )
