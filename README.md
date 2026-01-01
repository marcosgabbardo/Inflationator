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

## Usage

### Run Simulation

```bash
# Basic simulation (52 weeks, liberal democracy regime)
python -m src.cli.main run

# Custom simulation
python -m src.cli.main run --weeks 104 --regime ancap --persons 50000

# Totalitarian regime (for comparison)
python -m src.cli.main run --regime totalitarian --weeks 52
```

### Fetch Real Prices

```bash
# Fetch Bitcoin and commodity prices from real APIs
python -m src.cli.main prices
```

### What-If Scenarios

```bash
# FED doubles the monetary base
python -m src.cli.main scenario fed_doubles_money

# Transition to anarcho-capitalism
python -m src.cli.main scenario ancap_transition

# Zero intervention (Austrian ideal)
python -m src.cli.main scenario zero_intervention
```

### Compare Regimes

```bash
# Compare democracy vs monarchy (test Hoppe's thesis)
python -m src.cli.main compare democracy_liberal monarchy

# Compare ancap vs totalitarian
python -m src.cli.main compare ancap totalitarian
```

### List Available Regimes

```bash
python -m src.cli.main regimes
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
│   │   └── collectors/
│   │       ├── bitcoin.py   # CoinGecko API
│   │       └── commodities.py # Yahoo Finance
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

## Data Sources

The simulator uses **only private sources**, never government data:

- **Bitcoin**: CoinGecko API (real-time price)
- **Gold/Silver/Oil**: Yahoo Finance
- **Internal prices**: Agent-based simulation with order book matching

## Metrics Calculated

### Real Inflation
Calculated from market prices, not government CPI.

### Central Bank Damage
- Money printed (QE)
- Malinvestment induced
- Bubbles created
- Bailouts given

### Government Damage
- Deadweight loss from taxes
- Compliance costs
- Capital destroyed (wars)
- Trade distortion (tariffs)

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

- **100,000+ agents** simulating individual economic decisions
- **Real-time Bitcoin/Gold prices** from private APIs
- **Business cycle tracking** based on ABCT
- **Regime comparison** to test Hoppe's thesis
- **Credit system** with fractional reserve banking
- **Labor market** with employment dynamics

## Example Results

```
=== REGIME COMPARISON ===
Metric           | Ancap          | Democracy
--------------------------------------------------
Freedom Index    |         100.0  |          50.0
Unemployment     |          6.6%  |          7.2%
CB Damage ($)    |        21,169  | 14,250,016,207

Ancap has higher freedom - as Rothbard predicted!
```

---

*"Inflation is taxation without legislation."* - Milton Friedman
