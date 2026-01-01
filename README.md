# Inflationator

**Austrian Economics World Simulator**

An agent-based economic world simulator grounded in the Austrian School of Economics (Hayek, Rothbard, Hoppe). Focused on predicting real inflation and measuring the damage caused by central banks and governments.

## Philosophy

> "The State is that great fiction by which everyone tries to live at the expense of everyone else." - Frédéric Bastiat

This simulator is built on the following principles:
- **Inflation is a monetary phenomenon** - caused by expanding the money supply
- **Prices are signals** - they emerge from individual valuations, not centrally defined
- **Central banks cause business cycles** - through artificial interest rate manipulation
- **Governments extract wealth** - via taxes, regulations, and wars
- **Less intervention = better outcomes** - following Hoppe's hierarchy

## Regime Hierarchy (Hoppe)

From WORST to BEST:

| Regime | Intervention | Description |
|--------|--------------|-------------|
| Totalitarian | 100% | Total control - Venezuela/Cuba model |
| Socialist Democracy | 80% | High intervention - Scandinavian model |
| Liberal Democracy | 50% | Moderate - USA/UK model |
| Monarchy | 30% | Traditional - Liechtenstein model |
| Minarchy | 10% | Night-watchman state |
| Anarcho-Capitalism | 0% | Rothbard's ideal |

## Installation

```bash
# Clone the repository
cd /path/to/Inflationator

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## CLI Commands

### 1. Run Simulation

```bash
# Basic simulation (12 months, liberal democracy regime)
python -m src.cli.main run

# Custom simulation
python -m src.cli.main run --months 24 --regime ancap --persons 50000

# Totalitarian regime (for comparison)
python -m src.cli.main run --regime totalitarian --months 12

# Full options
python -m src.cli.main run \
  --months 12 \
  --country USA \
  --regime democracy_liberal \
  --persons 10000 \
  --companies 1000 \
  --intervention 0.5
```

### 2. View Real-World Conditions (NEW!)

```bash
# Fetch TODAY's economy from private market sources
python -m src.cli.main conditions

# For a specific country
python -m src.cli.main conditions --country USA
```

This shows:
- **Asset prices**: Bitcoin, Gold, Silver, Oil (real-time)
- **Market indices**: S&P 500, VIX, DXY
- **Interest rates**: Treasury yields, yield curve status
- **Sentiment**: Market fear level, Crypto Fear & Greed
- **Derived metrics**: Real inflation estimate (commodity-based), dollar debasement, recession probability

### 3. Fetch Real Prices

```bash
# Fetch Bitcoin and commodity prices from real APIs
python -m src.cli.main prices
```

### 4. What-If Scenarios

```bash
# Monetary Scenarios
python -m src.cli.main scenario fed_doubles_money      # FED doubles money supply
python -m src.cli.main scenario hyperinflation         # Extreme money printing (10x)
python -m src.cli.main scenario zero_intervention      # No CB or government

# Political Scenarios
python -m src.cli.main scenario ancap_transition       # Transition to anarcho-capitalism
python -m src.cli.main scenario election_year          # Simulate election year fiscal easing

# Trade War Scenarios (NEW!)
python -m src.cli.main scenario trade_war --tariff 0.25   # 25% tariffs
python -m src.cli.main scenario trump_tariffs             # Trump-style tariffs (20% general + sector-specific)
```

### 5. Compare Regimes

```bash
# Compare democracy vs monarchy (test Hoppe's thesis)
python -m src.cli.main compare democracy_liberal monarchy

# Compare ancap vs totalitarian
python -m src.cli.main compare ancap totalitarian

# Compare with custom duration
python -m src.cli.main compare minarchy democracy_socialist --months 24
```

### 6. List Available Regimes

```bash
python -m src.cli.main regimes
```

### 7. Initialize Database (Optional)

```bash
python -m src.cli.main init-db
```

## Project Structure

```
inflationator/
├── src/
│   ├── agents/              # Economic agents
│   │   ├── base.py          # Base class
│   │   ├── person.py        # Individual persons
│   │   ├── company.py       # Companies
│   │   ├── bank.py          # Private banks
│   │   ├── central_bank.py  # Central Bank (villain)
│   │   └── government.py    # Government (villain)
│   │
│   ├── economy/             # Economic system
│   │   ├── market.py        # Markets and price discovery
│   │   ├── labor_market.py  # Employment and wages
│   │   └── austrian/
│   │       └── business_cycle.py  # ABCT implementation
│   │
│   ├── data/                # Data collection
│   │   ├── real_world_conditions.py  # TODAY's economy aggregator
│   │   └── collectors/
│   │       ├── bitcoin.py        # CoinGecko API
│   │       ├── commodities.py    # Yahoo Finance
│   │       └── market_sentiment.py  # Indices, VIX, yields
│   │
│   ├── simulation/          # Simulation engine
│   │   └── engine.py        # Main loop
│   │
│   ├── database/            # Persistence (optional)
│   │   ├── connection.py    # MySQL connection
│   │   └── models.py        # SQLAlchemy models
│   │
│   └── cli/                 # Interface
│       └── main.py          # Typer commands
│
├── config/
│   └── settings.py          # Configuration
│
├── requirements.txt
└── README.md
```

## Data Sources (Private Only - No Government Data)

| Category | Source | API |
|----------|--------|-----|
| Bitcoin | CoinGecko | Free API |
| Gold/Silver/Oil | Yahoo Finance | yfinance |
| S&P 500 / VIX / DXY | Yahoo Finance | yfinance |
| Treasury Yields | Yahoo Finance | yfinance |
| Crypto Fear & Greed | Alternative.me | Free API |

## Metrics Calculated

### Real Inflation (Not CPI!)
- Calculated from commodity prices (Gold, Oil, etc.)
- Current estimate: **~30% annually** (vs government's "2-3%")
- Dollar debasement over 10 years: **260%+** (vs gold)

### Central Bank Damage
- Money printed (QE)
- Money destroyed (QT)
- Malinvestment induced
- Bubbles created
- Bailouts given
- QE/QT cycles tracked

### Government Damage
- Deadweight loss from taxes
- Compliance costs
- Capital destroyed (wars)
- Trade distortion (tariffs)
- Election year manipulation

### Political Business Cycle (Democracies)
- Election cycle tracking (4 years)
- Pre-election fiscal easing
- Central bank pressure
- Tax collection efficiency drops

### Freedom Index
0-100 scale, inverse of intervention level.

### Business Cycle Phase
- **Recovery**: Post-bust rebuilding
- **Boom**: Credit expansion (artificial growth)
- **Peak**: Maximum distortion
- **Bust**: Healthy correction
- **Trough**: Bottom, ready for recovery

## Economic Theory

### Austrian School
- **Mises**: Economic calculation impossible under socialism
- **Hayek**: Dispersed knowledge, spontaneous order
- **Rothbard**: Ethics of liberty, anarcho-capitalism
- **Hoppe**: Democracy vs monarchy, time preference theory

### Austrian Business Cycle Theory (ABCT)
1. Central bank artificially lowers interest rates
2. Credit expands beyond real savings
3. Malinvestment in capital goods (higher order)
4. Unsustainable boom
5. Bust is the necessary correction

## Key Features

- **10,000+ agents** simulating individual economic decisions
- **Real-time data** from private APIs (BTC, Gold, VIX, yields)
- **TODAY's economy** as initial conditions (not mock data)
- **Business cycle tracking** based on ABCT
- **Inflation targeting** simulation (QE/QT cycles)
- **Trade war scenarios** (tariffs, protectionism)
- **Political business cycle** (election year manipulation)
- **Regime comparison** to test Hoppe's thesis
- **Credit system** with fractional reserve banking
- **Labor market** with employment dynamics

## Example Output

```
REAL WORLD CONDITIONS - TODAY'S ECONOMY
==================================================

                   ASSET PRICES
--------------------------------------------------
  Bitcoin:  $88,194
  Gold:     $4,326
  Oil:      $57

                  DERIVED METRICS
--------------------------------------------------
  Real Inflation Est:   30.0% (commodity-based)
  Dollar Debasement:    260.5% (10 years)
  Recession Prob:       15%

Austrian Interpretation:
  - Real inflation (30.0%) significantly higher than CPI claims
  - Dollar has lost 260% of purchasing power in 10 years
```

```
       Simulation Results
┌────────────────┬──────────────┐
│ Duration       │ 12 months    │
│ Inflation Rate │ 13.03%       │
│ Bitcoin Price  │ $273,666     │
│ Gold Price     │ $13,944.71   │
└────────────────┴──────────────┘
          Intervention Damage
┌─────────────────────┬────────────────┐
│ Central Bank Damage │ $1,713,568,267 │
│ Government Damage   │ $31,908,023    │
│ Freedom Index       │ 50.0/100       │
└─────────────────────┴────────────────┘
```

---

*"Inflation is taxation without legislation."* - Milton Friedman

*"The first panacea for a mismanaged nation is inflation of the currency; the second is war. Both bring a temporary prosperity; both bring a permanent ruin."* - Ernest Hemingway
