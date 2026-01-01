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
    weeks: int = typer.Option(52, "--weeks", "-w", help="Number of weeks to simulate"),
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
        ticks_per_run=weeks,
    )

    # Show config
    config_table = Table(title="Simulation Configuration", show_header=False)
    config_table.add_row("Country", country)
    config_table.add_row("Regime", regime.value)
    config_table.add_row("Persons", str(persons))
    config_table.add_row("Companies", str(companies))
    config_table.add_row("CB Intervention", f"{intervention:.0%}")
    config_table.add_row("Duration", f"{weeks} weeks")
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
    metrics = engine.run(weeks)

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
    weeks: int = typer.Option(52, "--weeks", "-w", help="Weeks to simulate"),
):
    """
    Run a what-if scenario.

    Available scenarios:
    - fed_doubles_money: FED doubles the money supply
    - ancap_transition: Government becomes anarcho-capitalist
    - hyperinflation: Extreme money printing
    - zero_intervention: No CB or government intervention
    """
    console.print(Panel.fit(
        f"[bold yellow]SCENARIO: {name.upper()}[/bold yellow]",
        border_style="yellow"
    ))

    config = SimulationConfig(ticks_per_run=weeks)
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
        if engine.central_bank:
            engine.central_bank.intervention_level = 1.0
            for _ in range(10):
                engine.central_bank.print_money(engine.central_bank.state.base_money)
    else:
        console.print(f"[red]Unknown scenario: {name}[/red]")
        raise typer.Exit(1)

    # Run
    engine.run(weeks)
    _display_results(engine)


@app.command()
def compare(
    regime1: RegimeChoice = typer.Argument(..., help="First regime"),
    regime2: RegimeChoice = typer.Argument(..., help="Second regime"),
    weeks: int = typer.Option(52, "--weeks", "-w", help="Weeks to simulate"),
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
            ticks_per_run=weeks,
        )
        engine = SimulationEngine(config)
        engine.initialize()
        engine.run(weeks)
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

    results_table.add_row("Duration", f"{metrics['tick']} weeks")
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
