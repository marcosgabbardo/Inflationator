"""
Inflationator CLI

Command-line interface for the Austrian Economics simulator.
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from typing import Optional
from enum import Enum
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.simulation.engine import SimulationEngine, SimulationConfig
from src.agents.government import RegimeType
from src.data.collectors.bitcoin import get_bitcoin_price
from src.data.collectors.commodities import get_commodity_prices
from src.data.real_world_conditions import (
    get_real_world_conditions,
    print_conditions_summary,
)

app = typer.Typer(
    name="inflationator",
    help="Austrian Economics World Simulator - Predict real inflation, expose central bank damage",
    add_completion=False,
)

console = Console()


class RegimeChoice(str, Enum):
    """Regime choices for CLI"""
    ANCAP = "ancap"
    MINARCHY = "minarchy"
    MONARCHY = "monarchy"
    DEMOCRACY_LIBERAL = "democracy_liberal"
    DEMOCRACY_SOCIALIST = "democracy_socialist"
    TOTALITARIAN = "totalitarian"


@app.command()
def run(
    months: int = typer.Option(12, "--months", "-m", help="Number of months to simulate"),
    country: str = typer.Option("USA", "--country", "-c", help="Country code (USA, EUR, etc.)"),
    regime: RegimeChoice = typer.Option(
        RegimeChoice.DEMOCRACY_LIBERAL,
        "--regime", "-r",
        help="Government regime type"
    ),
    persons: int = typer.Option(10000, "--persons", "-p", help="Number of person agents"),
    companies: int = typer.Option(1000, "--companies", help="Number of companies"),
    intervention: float = typer.Option(
        0.5, "--intervention", "-i",
        help="Central bank intervention level (0-1)"
    ),
):
    """
    Run the economic simulation.

    Simulates an economy with Austrian Economics principles,
    tracking the damage caused by central banks and governments.
    """
    console.print(Panel.fit(
        "[bold blue]INFLATIONATOR[/bold blue]\n"
        "[dim]Austrian Economics World Simulator[/dim]",
        border_style="blue"
    ))

    # Map CLI regime to actual enum
    regime_map = {
        RegimeChoice.ANCAP: RegimeType.ANCAP,
        RegimeChoice.MINARCHY: RegimeType.MINARCHY,
        RegimeChoice.MONARCHY: RegimeType.MONARCHY,
        RegimeChoice.DEMOCRACY_LIBERAL: RegimeType.DEMOCRACY_LIBERAL,
        RegimeChoice.DEMOCRACY_SOCIALIST: RegimeType.DEMOCRACY_SOCIALIST,
        RegimeChoice.TOTALITARIAN: RegimeType.TOTALITARIAN,
    }

    config = SimulationConfig(
        country=country,
        num_persons=persons,
        num_companies=companies,
        regime_type=regime_map[regime],
        central_bank_intervention=intervention,
        ticks_per_run=months,
    )

    # Show config
    config_table = Table(title="Simulation Configuration", show_header=False)
    config_table.add_row("Country", country)
    config_table.add_row("Regime", regime.value)
    config_table.add_row("Persons", str(persons))
    config_table.add_row("Companies", str(companies))
    config_table.add_row("CB Intervention", f"{intervention:.0%}")
    config_table.add_row("Duration", f"{months} months")
    console.print(config_table)

    # Initialize
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing simulation...", total=None)
        engine = SimulationEngine(config)
        engine.initialize()
        progress.update(task, description="[green]Initialization complete!")

    # Run simulation
    console.print("\n[bold]Running simulation...[/bold]")
    metrics = engine.run(months)

    # Show results
    _display_results(engine)


@app.command()
def prices():
    """
    Fetch current real-world prices.

    Gets Bitcoin and commodity prices from private sources
    (not government statistics).
    """
    console.print(Panel.fit(
        "[bold]Real-Time Prices[/bold]\n"
        "[dim]From private sources, not government[/dim]",
        border_style="green"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching Bitcoin price...", total=None)
        try:
            btc_prices = get_bitcoin_price()
            progress.update(task, description="[green]Bitcoin: Done")
        except Exception as e:
            btc_prices = {"usd": "Error", "error": str(e)}

        progress.update(task, description="Fetching commodity prices...")
        try:
            commodities = get_commodity_prices()
            progress.update(task, description="[green]Commodities: Done")
        except Exception as e:
            commodities = {"error": str(e)}

    # Bitcoin
    btc_table = Table(title="Bitcoin", show_header=True)
    btc_table.add_column("Currency", style="cyan")
    btc_table.add_column("Price", style="green")
    for currency, price in btc_prices.items():
        if currency not in ["change_24h", "market_cap_usd"]:
            btc_table.add_row(currency.upper(), f"${price:,.2f}" if isinstance(price, (int, float)) else str(price))
    console.print(btc_table)

    # Commodities
    comm_table = Table(title="Commodities", show_header=True)
    comm_table.add_column("Commodity", style="cyan")
    comm_table.add_column("Price (USD)", style="green")
    for name, price in commodities.items():
        comm_table.add_row(name.replace("_", " ").title(), f"${price:,.2f}")
    console.print(comm_table)


@app.command()
def scenario(
    name: str = typer.Argument(..., help="Scenario name"),
    months: int = typer.Option(12, "--months", "-m", help="Months to simulate"),
    tariff_rate: float = typer.Option(0.25, "--tariff", "-t", help="Tariff rate for trade war (0-1)"),
):
    """
    Run a what-if scenario.

    Available scenarios:

    [bold]Monetary Scenarios:[/bold]
    - fed_doubles_money: FED doubles the money supply
    - hyperinflation: Extreme money printing (10x)
    - zero_intervention: No CB or government intervention

    [bold]Political Scenarios:[/bold]
    - ancap_transition: Government becomes anarcho-capitalist
    - election_year: Simulate election year fiscal easing

    [bold]Trade Scenarios:[/bold]
    - trade_war: Initiate trade war with tariffs (use --tariff to set rate)
    - trump_tariffs: Apply Trump-style tariff policy (20% general + sector-specific)
    """
    console.print(Panel.fit(
        f"[bold yellow]SCENARIO: {name.upper()}[/bold yellow]",
        border_style="yellow"
    ))

    config = SimulationConfig(ticks_per_run=months)
    engine = SimulationEngine(config)
    engine.initialize()

    # Apply scenario
    if name == "ancap_transition":
        engine.apply_scenario("ancap_transition", {})
    elif name == "fed_doubles_money":
        engine.apply_scenario("fed_doubles_money", {})
    elif name == "zero_intervention":
        engine.apply_scenario("zero_intervention", {})
    elif name == "hyperinflation":
        engine.apply_scenario("hyperinflation", {})
    elif name == "trade_war":
        engine.apply_scenario("trade_war", {"tariff_rate": tariff_rate})
    elif name == "trump_tariffs":
        engine.apply_scenario("trump_tariffs", {})
    elif name == "election_year":
        engine.apply_scenario("election_year", {})
    else:
        console.print(f"[red]Unknown scenario: {name}[/red]")
        console.print("[dim]Run 'inflationator scenario --help' for available scenarios[/dim]")
        raise typer.Exit(1)

    # Run
    engine.run(months)
    _display_results(engine)


@app.command()
def compare(
    regime1: RegimeChoice = typer.Argument(..., help="First regime"),
    regime2: RegimeChoice = typer.Argument(..., help="Second regime"),
    months: int = typer.Option(12, "--months", "-m", help="Months to simulate"),
):
    """
    Compare two different regimes.

    Run simulations with different government types and compare outcomes.
    Perfect for testing Hoppe's democracy vs monarchy thesis.
    """
    console.print(Panel.fit(
        f"[bold]Regime Comparison[/bold]\n"
        f"[cyan]{regime1.value}[/cyan] vs [cyan]{regime2.value}[/cyan]",
        border_style="blue"
    ))

    regime_map = {
        RegimeChoice.ANCAP: RegimeType.ANCAP,
        RegimeChoice.MINARCHY: RegimeType.MINARCHY,
        RegimeChoice.MONARCHY: RegimeType.MONARCHY,
        RegimeChoice.DEMOCRACY_LIBERAL: RegimeType.DEMOCRACY_LIBERAL,
        RegimeChoice.DEMOCRACY_SOCIALIST: RegimeType.DEMOCRACY_SOCIALIST,
        RegimeChoice.TOTALITARIAN: RegimeType.TOTALITARIAN,
    }

    results = []

    for regime in [regime1, regime2]:
        console.print(f"\n[bold]Running {regime.value}...[/bold]")
        config = SimulationConfig(
            regime_type=regime_map[regime],
            num_persons=5000,  # Smaller for comparison
            num_companies=500,
            ticks_per_run=months,
        )
        engine = SimulationEngine(config)
        engine.initialize()
        engine.run(months)
        results.append({
            "regime": regime.value,
            "metrics": engine.metrics,
            "damage": engine.get_damage_summary(),
        })

    # Compare
    compare_table = Table(title="Comparison Results")
    compare_table.add_column("Metric", style="cyan")
    compare_table.add_column(regime1.value, style="green")
    compare_table.add_column(regime2.value, style="yellow")

    r1, r2 = results[0], results[1]
    compare_table.add_row(
        "Freedom Index",
        f"{r1['metrics'].freedom_index:.1f}",
        f"{r2['metrics'].freedom_index:.1f}"
    )
    compare_table.add_row(
        "Inflation Rate",
        f"{r1['metrics'].inflation_rate:.2%}",
        f"{r2['metrics'].inflation_rate:.2%}"
    )
    compare_table.add_row(
        "Unemployment",
        f"{r1['metrics'].unemployment_rate:.2%}",
        f"{r2['metrics'].unemployment_rate:.2%}"
    )
    compare_table.add_row(
        "CB Damage",
        f"${r1['metrics'].central_bank_damage:,.0f}",
        f"${r2['metrics'].central_bank_damage:,.0f}"
    )
    compare_table.add_row(
        "Gov Damage",
        f"${r1['metrics'].government_damage:,.0f}",
        f"${r2['metrics'].government_damage:,.0f}"
    )

    console.print(compare_table)

    # Winner
    winner = regime1 if r1['metrics'].freedom_index > r2['metrics'].freedom_index else regime2
    console.print(f"\n[bold green]Winner: {winner.value}[/bold green] (Higher freedom index)")


@app.command()
def conditions(
    country: str = typer.Option("USA", "--country", "-c", help="Country code"),
):
    """
    Show current real-world economic conditions.

    Fetches TODAY's economy from private market sources:
    - Asset prices (BTC, Gold, Oil)
    - Market indices (S&P 500, VIX)
    - Interest rates (Treasury yields)
    - Sentiment indicators
    - Derived metrics (real inflation, recession probability)

    Austrian Theory: Use market data, not government statistics.
    """
    console.print(Panel.fit(
        f"[bold blue]TODAY'S ECONOMY - {country}[/bold blue]\n"
        "[dim]Real-time data from private sources[/dim]",
        border_style="blue"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching real-world conditions...", total=None)
        try:
            conditions = get_real_world_conditions(country)
            progress.update(task, description="[green]Conditions fetched!")
        except Exception as e:
            console.print(f"[red]Error fetching conditions: {e}[/red]")
            raise typer.Exit(1)

    # Print the comprehensive summary
    print_conditions_summary(conditions)

    # Additional Rich tables for key insights
    console.print("\n")

    # Investment signals
    signals_table = Table(title="Investment Signals (Austrian)", show_header=False)
    signals_table.add_column("Signal", style="cyan")
    signals_table.add_column("Value", style="green")

    signals_table.add_row(
        "BTC vs Gold Ratio",
        f"{float(conditions.btc_price_usd / conditions.gold_price_usd):.1f}x"
    )
    signals_table.add_row(
        "Market Sentiment",
        conditions.market_sentiment.upper()
    )
    signals_table.add_row(
        "Monetary Policy",
        conditions.monetary_expansion_signal.upper()
    )

    if conditions.yield_curve_inverted:
        signals_table.add_row(
            "[bold red]WARNING[/bold red]",
            "Yield curve inverted - recession signal!"
        )

    console.print(signals_table)

    # Austrian interpretation
    console.print("\n[bold]Austrian Interpretation:[/bold]")
    real_inflation = conditions.inflation_estimate
    if real_inflation > 10:
        console.print(f"[red]  - Real inflation ({real_inflation:.1f}%) significantly higher than CPI claims[/red]")
    else:
        console.print(f"[yellow]  - Real inflation ({real_inflation:.1f}%) - monitor for changes[/yellow]")

    if conditions.dollar_debasement_10y > 100:
        console.print(f"[red]  - Dollar has lost {conditions.dollar_debasement_10y:.0f}% of purchasing power in 10 years[/red]")

    if conditions.recession_probability > 0.4:
        console.print(f"[red]  - Recession probability ({conditions.recession_probability:.0%}) is elevated[/red]")

    console.print("\n[dim]'Inflation is always and everywhere a monetary phenomenon.' - Friedman[/dim]")
    console.print("[dim](But the Austrians knew it first!)[/dim]")


@app.command()
def regimes():
    """
    List all available regime types with descriptions.

    Based on Hoppe's analysis in "Democracy: The God That Failed".
    """
    table = Table(title="Political Regimes (Worst to Best)")
    table.add_column("Regime", style="cyan")
    table.add_column("Intervention", style="red")
    table.add_column("Description", style="dim")

    regimes_info = [
        ("Totalitarian", "100%", "Total state control - Venezuela/Cuba model"),
        ("Democracy Socialist", "80%", "High intervention - Scandinavian model"),
        ("Democracy Liberal", "50%", "Moderate - USA/UK model"),
        ("Monarchy", "30%", "Traditional - Liechtenstein model"),
        ("Minarchy", "10%", "Night watchman state"),
        ("Ancap", "0%", "Rothbard ideal - No government"),
    ]

    for regime, intervention, desc in regimes_info:
        table.add_row(regime, intervention, desc)

    console.print(table)

    console.print("\n[dim]Hoppe's insight: Democracy has higher time preference than monarchy.[/dim]")
    console.print("[dim]Politicians don't own, so they extract maximum value short-term.[/dim]")


@app.command()
def init_db():
    """
    Initialize the database.

    Creates all necessary tables in MySQL.
    """
    from src.database.connection import db

    console.print("[bold]Initializing database...[/bold]")

    if not db.create_database():
        console.print("[red]Failed to create database[/red]")
        raise typer.Exit(1)

    if not db.test_connection():
        console.print("[red]Cannot connect to database[/red]")
        raise typer.Exit(1)

    db.init_schema()
    console.print("[green]Database initialized successfully![/green]")


def _display_results(engine: SimulationEngine):
    """Display simulation results"""
    console.print("\n")

    # Summary
    summary = engine.get_summary()
    metrics = summary["metrics"]

    results_table = Table(title="Simulation Results", show_header=False)
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")

    results_table.add_row("Duration", f"{metrics['tick']} months")
    results_table.add_row("Inflation Rate", f"{metrics['inflation_rate']:.2%}")
    results_table.add_row("Unemployment", f"{metrics['unemployment_rate']:.2%}")
    results_table.add_row("GDP", f"${float(metrics['gdp']):,.0f}")
    results_table.add_row("Bitcoin Price", f"${float(metrics['bitcoin_price']):,.0f}")
    results_table.add_row("Gold Price", f"${float(metrics['gold_price']):,.2f}")
    console.print(results_table)

    # Damage report
    damage = summary["damage_summary"]

    damage_table = Table(title="Intervention Damage", show_header=False)
    damage_table.add_column("Source", style="yellow")
    damage_table.add_column("Damage", style="red")

    damage_table.add_row("Central Bank Damage", f"${float(metrics['central_bank_damage']):,.0f}")
    damage_table.add_row("Government Damage", f"${float(metrics['government_damage']):,.0f}")
    damage_table.add_row("Total Malinvestment", f"${float(metrics['total_malinvestment']):,.0f}")
    damage_table.add_row("Freedom Index", f"{metrics['freedom_index']:.1f}/100")
    console.print(damage_table)

    # Government Policy Info (tariffs, election cycle)
    if engine.government:
        policy_table = Table(title="Government Policy", show_header=False)
        policy_table.add_column("Policy", style="cyan")
        policy_table.add_column("Status", style="yellow")

        policy_table.add_row("Regime", engine.government.regime_type.value)
        policy_table.add_row("Tariff Rate", f"{engine.government.tariff_rate:.1%}")
        policy_table.add_row("Tariff Mode", engine.government.tariff_mode)

        if engine.government.regime_type.value in ["democracy_liberal", "democracy_socialist"]:
            cycle_year = engine.government.current_year_in_cycle
            policy_table.add_row("Election Cycle", f"Year {cycle_year:.1f} of {engine.government.election_cycle_years}")
            policy_table.add_row("Election Easing", "Yes" if engine.government.election_easing_active else "No")

        if engine.government.state.trade_disruption > 0:
            policy_table.add_row("Trade Disruption", f"${float(engine.government.state.trade_disruption):,.0f}")

        console.print(policy_table)

    # Business Cycle
    cycle = summary.get("business_cycle", {})
    if cycle:
        cycle_table = Table(title="Business Cycle (Austrian)", show_header=False)
        cycle_table.add_column("Indicator", style="cyan")
        cycle_table.add_column("Value", style="magenta")

        cycle_table.add_row("Phase", cycle.get("phase", "unknown"))
        cycle_table.add_row("Boom Intensity", f"{cycle.get('boom_intensity', 0):.2f}")
        cycle_table.add_row("Rate Distortion", f"{cycle.get('rate_distortion', 0):.2%}")
        cycle_table.add_row("Credit Signal", cycle.get("credit_signal", "unknown"))
        cycle_table.add_row("Investment Signal", cycle.get("investment_signal", "unknown"))
        console.print(cycle_table)

    # Recommendation
    console.print(f"\n[bold]Recommendation:[/bold] {damage['recommendation']}")


@app.callback()
def main():
    """
    INFLATIONATOR - Austrian Economics World Simulator

    Simulate economic worlds based on Austrian Economics principles.
    Track real inflation (not government CPI), measure central bank damage,
    and compare different political regimes.

    "The State is that great fiction by which everyone tries
    to live at the expense of everyone else." - Bastiat
    """
    pass


if __name__ == "__main__":
    app()
