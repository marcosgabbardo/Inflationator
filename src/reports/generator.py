"""
Report Generator - Academic Whitepaper Style

Generates comprehensive PDF reports for economic simulations.
Output format inspired by academic papers and whitepapers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .charts import ChartGenerator
from .pdf_generator import (
    PDFReportBuilder,
    format_currency,
    format_percent,
)


@dataclass
class ReportConfig:
    """Configuration for report generation"""

    output_dir: Path = field(default_factory=lambda: Path("reports"))
    include_charts: bool = True
    include_methodology: bool = True
    include_references: bool = True
    include_austrian_notes: bool = True
    include_theory_section: bool = True


class ReportGenerator:
    """
    Generates academic-style PDF reports for economic simulations.

    Output format inspired by Satoshi Nakamoto's Bitcoin whitepaper:
    - Clean, minimal formatting
    - Numbered sections
    - Compact tables
    - Academic references
    """

    def __init__(self, config: ReportConfig | None = None):
        self.config = config or ReportConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.chart_generator = ChartGenerator(self.config.output_dir / "charts")
        self.table_counter = 0
        self.figure_counter = 0

    def _next_table(self) -> int:
        self.table_counter += 1
        return self.table_counter

    def _next_figure(self) -> int:
        self.figure_counter += 1
        return self.figure_counter

    def generate_single_country_report(
        self,
        simulation_summary: dict[str, Any],
        metrics_history: list[Any],
        output_filename: str | None = None,
    ) -> Path:
        """Generate academic-style report for single-country simulation."""

        # Reset counters
        self.table_counter = 0
        self.figure_counter = 0

        # Setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        country = simulation_summary.get("config", {}).get("country", "USA")
        regime = simulation_summary.get("config", {}).get("regime", "democracy_liberal")

        if output_filename is None:
            output_filename = f"simulation_report_{country}_{timestamp}.pdf"

        output_path = self.config.output_dir / output_filename

        # Initialize PDF
        pdf = PDFReportBuilder(output_path)

        # Title section
        pdf.add_title_page(
            title="Austrian Economics Simulation Analysis",
            subtitle=f"Economic Analysis of {country} under {regime.replace('_', ' ').title()} Regime",
        )

        # Abstract
        abstract = self._generate_single_country_abstract(
            simulation_summary, metrics_history, country
        )
        pdf.add_abstract(abstract)

        # Introduction
        pdf.add_section("Introduction")
        pdf.add_paragraph(
            f"This paper presents the results of an agent-based economic simulation of {country}, "
            f"conducted using the Inflationator simulation engine. The simulation employs Austrian "
            f"Economics principles to model economic behavior and track the impact of government "
            f"and central bank intervention on economic outcomes."
        )
        pdf.add_paragraph(
            "Unlike mainstream economic models that treat intervention as neutral or beneficial, "
            "this framework recognizes that all intervention creates economic distortions. "
            "The simulation quantifies these distortions as 'damage' - value destroyed through "
            "misallocation of resources, deadweight loss, and malinvestment."
        )
        pdf.add_paragraph(
            "The theoretical foundation draws from the Austrian School of Economics, particularly "
            "the works of Ludwig von Mises on business cycles, Friedrich Hayek on knowledge and prices, "
            "Murray Rothbard on central banking, and Hans-Hermann Hoppe on political regimes. "
            "This simulation provides empirical support for their theoretical predictions."
        )

        # Methodology
        if self.config.include_methodology:
            pdf.add_methodology_section()

        # Theoretical Framework
        if self.config.include_theory_section:
            pdf.add_austrian_framework_section()

        # Simulation Parameters
        self._add_parameters_section(pdf, simulation_summary)

        # Results: Price Evolution
        self._add_price_results_section(
            pdf, simulation_summary, metrics_history, country
        )

        # Results: Monetary Analysis
        self._add_monetary_results_section(pdf, simulation_summary, metrics_history)

        # Results: Labor Market
        self._add_labor_results_section(pdf, simulation_summary, metrics_history)

        # Intervention Damage Analysis
        self._add_damage_analysis_section(pdf, simulation_summary)

        # Business Cycle
        self._add_business_cycle_section(pdf, simulation_summary)

        # ===== AUSTRIAN THEORY SECTIONS =====
        final_metrics = metrics_history[-1] if metrics_history else None
        metrics = simulation_summary.get("metrics", {})
        business_cycle = simulation_summary.get("business_cycle", {})

        # Hayek - Knowledge and Prices
        if self.config.include_austrian_notes:
            inflation_rate = metrics.get("inflation_rate", 0)
            pdf.add_hayek_section(inflation_rate)

        # Rothbard - Central Bank Damage
        if self.config.include_austrian_notes:
            cb_damage = float(metrics.get("central_bank_damage", 0))
            pdf.add_rothbard_section(cb_damage)

        # Hoppe - Regime Analysis
        if self.config.include_austrian_notes:
            freedom_index = metrics.get("freedom_index", 50)
            gov_damage = float(metrics.get("government_damage", 0))
            pdf.add_hoppe_section(regime, freedom_index, gov_damage)

        # Mises - Business Cycle
        if self.config.include_austrian_notes and business_cycle:
            phase = business_cycle.get("phase", "unknown")
            rate_distortion = business_cycle.get("rate_distortion", 0)
            pdf.add_mises_section(phase, rate_distortion)

        # Bitcoin Analysis
        if self.config.include_austrian_notes and final_metrics:
            btc_price = float(final_metrics.bitcoin_price)
            initial_btc = (
                float(metrics_history[0].bitcoin_price)
                if metrics_history
                else btc_price
            )
            btc_change = (
                (btc_price - initial_btc) / initial_btc if initial_btc > 0 else 0
            )
            pdf.add_bitcoin_analysis_section(btc_price, btc_change)

        # Dashboard and Additional Charts
        if self.config.include_charts and metrics_history:
            self._add_comprehensive_charts_section(
                pdf, metrics_history, simulation_summary, country
            )

        # Conclusions
        freedom_index = final_metrics.freedom_index if final_metrics else 50
        total_damage = (
            final_metrics.central_bank_damage if final_metrics else Decimal("0")
        ) + (final_metrics.government_damage if final_metrics else Decimal("0"))
        pdf.add_conclusions_section(freedom_index, total_damage)

        # References
        if self.config.include_references:
            pdf.add_references_section()

        # Build
        pdf.build()
        return output_path

    def generate_multi_country_report(
        self,
        simulation_summary: dict[str, Any],
        metrics_history: list[Any],
        output_filename: str | None = None,
    ) -> Path:
        """Generate academic-style report for multi-country simulation."""

        # Reset counters
        self.table_counter = 0
        self.figure_counter = 0

        # Setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        countries = simulation_summary.get("config", {}).get("countries", [])
        num_countries = len(countries)

        if output_filename is None:
            output_filename = (
                f"multi_country_report_{num_countries}countries_{timestamp}.pdf"
            )

        output_path = self.config.output_dir / output_filename

        # Initialize PDF
        pdf = PDFReportBuilder(output_path)

        # Title
        pdf.add_title_page(
            title="Comparative Economic Analysis",
            subtitle=f"Multi-Country Simulation: {num_countries} Economies",
        )

        # Abstract
        abstract = self._generate_multi_country_abstract(
            simulation_summary, metrics_history
        )
        pdf.add_abstract(abstract)

        # Introduction
        pdf.add_section("Introduction")
        pdf.add_paragraph(
            f"This paper presents a comparative analysis of {num_countries} national economies "
            "simulated simultaneously using the Inflationator engine. Each country is modeled "
            "with its own government regime, central bank policy, and agent population. "
            "The simulation tracks inter-country relationships including trade flows, tariffs, "
            "sanctions, and conflict probabilities."
        )
        pdf.add_paragraph(
            "The Austrian Economics framework predicts that countries with greater economic "
            "freedom (less intervention) will outperform those with heavy state involvement. "
            "This simulation provides empirical support for this thesis by comparing outcomes "
            "across different regime types."
        )

        # Methodology
        if self.config.include_methodology:
            pdf.add_methodology_section()

        # Country Overview
        self._add_country_overview_section(pdf, simulation_summary)

        # Comparative Rankings
        self._add_rankings_section(pdf, simulation_summary)

        # Geopolitical Analysis
        self._add_geopolitical_section(pdf, simulation_summary)

        # Global Metrics
        self._add_global_metrics_section(pdf, simulation_summary)

        # Evolution Charts
        if self.config.include_charts and metrics_history:
            self._add_evolution_charts_section(pdf, metrics_history, simulation_summary)

        # Conclusions
        country_summary = simulation_summary.get("countries", {})
        sorted_countries = sorted(
            country_summary.items(),
            key=lambda x: x[1].get("freedom_index", 0),
            reverse=True,
        )

        best = sorted_countries[0][0] if sorted_countries else None
        worst = sorted_countries[-1][0] if sorted_countries else None
        best_freedom = (
            sorted_countries[0][1].get("freedom_index", 50) if sorted_countries else 50
        )

        global_metrics = simulation_summary.get("global_metrics", {})
        total_damage = float(global_metrics.get("total_cb_damage", 0)) + float(
            global_metrics.get("total_gov_damage", 0)
        )

        pdf.add_conclusions_section(best_freedom, total_damage, best, worst)

        # References
        if self.config.include_references:
            pdf.add_references_section()

        # Build
        pdf.build()
        return output_path

    # ===========================================
    # HELPER METHODS - ABSTRACTS
    # ===========================================

    def _generate_single_country_abstract(
        self, summary: dict[str, Any], history: list[Any], country: str
    ) -> str:
        config = summary.get("config", {})
        metrics = summary.get("metrics", {})

        duration = config.get("ticks", 12)
        inflation = metrics.get("inflation_rate", 0)
        freedom = metrics.get("freedom_index", 50)
        cb_damage = metrics.get("central_bank_damage", 0)
        gov_damage = metrics.get("government_damage", 0)

        return (
            f"We present results from a {duration}-month economic simulation of {country} using "
            f"agent-based modeling within an Austrian Economics framework. The simulation tracked "
            f"{config.get('num_persons', 100000):,} economic agents interacting across multiple markets. "
            f"Results show an annualized inflation rate of {format_percent(inflation)}, an economic "
            f"freedom index of {freedom:.1f}/100, and total intervention damage of "
            f"{format_currency(float(cb_damage) + float(gov_damage))}. Central bank interventions "
            f"(monetary expansion, rate manipulation) caused {format_currency(cb_damage)} in damage, "
            f"while government interventions (taxation, regulation, tariffs) caused {format_currency(gov_damage)}. "
            f"These findings support the Austrian prediction that intervention systematically destroys value."
        )

    def _generate_multi_country_abstract(
        self, summary: dict[str, Any], history: list[Any]
    ) -> str:
        countries = summary.get("config", {}).get("countries", [])
        global_metrics = summary.get("global_metrics", {})

        return (
            f"We present a comparative analysis of {len(countries)} national economies simulated "
            f"using agent-based modeling. Each country operates under a distinct political regime "
            f"with its own central bank and government intervention levels. The simulation tracks "
            f"inter-country trade, currency relationships, and conflict probabilities. "
            f"Results show average inflation of {format_percent(global_metrics.get('avg_inflation', 0))} "
            f"and average freedom index of {global_metrics.get('avg_freedom_index', 50):.1f}. "
            f"Total global intervention damage reached {format_currency(float(global_metrics.get('total_cb_damage', 0)) + float(global_metrics.get('total_gov_damage', 0)))}. "
            f"Countries with higher economic freedom consistently demonstrated superior economic outcomes, "
            f"validating the Austrian thesis that free markets outperform interventionist systems."
        )

    # ===========================================
    # HELPER METHODS - SINGLE COUNTRY SECTIONS
    # ===========================================

    def _add_parameters_section(self, pdf: PDFReportBuilder, summary: dict[str, Any]):
        """Add simulation parameters section"""
        pdf.add_section("Simulation Parameters")

        config = summary.get("config", {})
        params = {
            "Country": config.get("country", "USA"),
            "Regime Type": config.get("regime", "unknown").replace("_", " ").title(),
            "Duration": f"{config.get('ticks', 12)} months",
            "Agent Population": f"{config.get('num_persons', 100000):,} persons",
            "Companies": f"{config.get('num_companies', 10000):,}",
            "Banks": f"{config.get('num_banks', 500):,}",
            "CB Intervention Level": format_percent(
                config.get("central_bank_intervention", 0.5)
            ),
        }

        pdf.add_metrics_table(
            "", params, "Simulation configuration parameters", self._next_table()
        )

    def _add_price_results_section(
        self,
        pdf: PDFReportBuilder,
        summary: dict[str, Any],
        history: list[Any],
        country: str,
    ):
        """Add price evolution results section"""
        pdf.add_section("Results: Price Evolution")

        if not history:
            pdf.add_paragraph("No price history data available for this simulation.")
            return

        # Initial and final prices
        initial = history[0]
        final = history[-1]

        btc_change = (
            float(final.bitcoin_price) - float(initial.bitcoin_price)
        ) / float(initial.bitcoin_price)
        gold_change = (float(final.gold_price) - float(initial.gold_price)) / float(
            initial.gold_price
        )

        pdf.add_paragraph(
            f"During the simulation period, Bitcoin price moved from {format_currency(initial.bitcoin_price)} "
            f"to {format_currency(final.bitcoin_price)} ({format_percent(btc_change)} change). "
            f"Gold price moved from {format_currency(initial.gold_price)} to {format_currency(final.gold_price)} "
            f"({format_percent(gold_change)} change)."
        )

        pdf.add_note(
            "In Austrian theory, hard asset appreciation reflects fiat currency debasement rather than "
            "speculation. Bitcoin's fixed supply (21M) makes it an ideal measure of dollar purchasing power loss."
        )

        # Price table
        price_data = {
            "Initial BTC Price": initial.bitcoin_price,
            "Final BTC Price": final.bitcoin_price,
            "BTC Change": f"{btc_change * 100:+.1f}%",
            "Initial Gold Price": initial.gold_price,
            "Final Gold Price": final.gold_price,
            "Gold Change": f"{gold_change * 100:+.1f}%",
        }
        pdf.add_metrics_table(
            "", price_data, "Price evolution during simulation", self._next_table()
        )

        # Charts
        if self.config.include_charts:
            btc_prices = [m.bitcoin_price for m in history]
            btc_chart = self.chart_generator.create_bitcoin_price_chart(
                btc_prices, "Bitcoin Price Evolution"
            )
            pdf.add_chart(
                btc_chart,
                "Bitcoin price trajectory showing monetary conditions impact",
                figure_num=self._next_figure(),
            )

    def _add_monetary_results_section(
        self, pdf: PDFReportBuilder, summary: dict[str, Any], history: list[Any]
    ):
        """Add monetary analysis section"""
        pdf.add_section("Results: Monetary Analysis")

        if not history:
            pdf.add_paragraph("No monetary data available.")
            return

        initial = history[0]
        final = history[-1]

        pdf.add_paragraph(
            f"The simulation tracked monetary expansion and its effects. Base money supply "
            f"changed from {format_currency(initial.money_supply)} to {format_currency(final.money_supply)}. "
            f"Credit created through fractional reserve banking reached {format_currency(final.credit_expansion)}. "
            f"The resulting annualized inflation rate was {format_percent(final.inflation_rate)}."
        )

        pdf.add_note(
            "Austrian Business Cycle Theory predicts that credit expansion without corresponding savings "
            "leads to malinvestment. The 'boom' is unsustainable and must end in a 'bust' to correct "
            "the misallocation of resources."
        )

        # Monetary metrics
        monetary_data = {
            "Initial Money Supply": initial.money_supply,
            "Final Money Supply": final.money_supply,
            "Credit Created": final.credit_expansion,
            "Inflation Rate (Annualized)": final.inflation_rate,
            "Central Bank Damage": final.central_bank_damage,
        }
        pdf.add_metrics_table(
            "", monetary_data, "Monetary indicators", self._next_table()
        )

    def _add_labor_results_section(
        self, pdf: PDFReportBuilder, summary: dict[str, Any], history: list[Any]
    ):
        """Add labor market section"""
        pdf.add_section("Results: Labor Market")

        if not history:
            pdf.add_paragraph("No labor data available.")
            return

        final = history[-1]

        pdf.add_paragraph(
            f"The unemployment rate at simulation end was {format_percent(final.unemployment_rate)}. "
            f"In Austrian economics, involuntary unemployment results from intervention "
            f"(minimum wage, regulations, taxation) that prevents the labor market from clearing. "
            f"The natural unemployment rate in a free market would be lower."
        )

        labor_data = {
            "Unemployment Rate": final.unemployment_rate,
            "GDP": final.gdp,
            "Freedom Index": f"{final.freedom_index:.1f}/100",
        }
        pdf.add_metrics_table(
            "", labor_data, "Labor market indicators", self._next_table()
        )

    def _add_damage_analysis_section(
        self, pdf: PDFReportBuilder, summary: dict[str, Any]
    ):
        """Add intervention damage analysis section"""
        pdf.add_section("Intervention Damage Analysis")

        summary.get("damage_summary", {})
        metrics = summary.get("metrics", {})

        pdf.add_paragraph(
            "A core principle of Austrian economics is that intervention creates deadweight loss. "
            "This simulation quantifies the damage caused by government and central bank actions. "
            "Unlike mainstream models that ignore or justify intervention costs, this framework "
            "tracks every dollar of value destroyed."
        )

        pdf.add_subsection("Central Bank Damage")
        pdf.add_paragraph(
            f"Central bank interventions caused {format_currency(metrics.get('central_bank_damage', 0))} "
            "in economic damage. This includes malinvestment caused by artificial interest rates, "
            "purchasing power loss from monetary expansion, and asset bubbles from quantitative easing."
        )

        pdf.add_subsection("Government Damage")
        pdf.add_paragraph(
            f"Government interventions caused {format_currency(metrics.get('government_damage', 0))} "
            "in economic damage. This includes deadweight loss from taxation, compliance costs from "
            "regulation, and trade disruption from tariffs and sanctions."
        )

        total_damage = float(metrics.get("central_bank_damage", 0)) + float(
            metrics.get("government_damage", 0)
        )
        damage_data = {
            "Central Bank Damage": metrics.get("central_bank_damage", 0),
            "Government Damage": metrics.get("government_damage", 0),
            "Total Malinvestment": metrics.get("total_malinvestment", 0),
            "Total Intervention Damage": Decimal(str(total_damage)),
        }
        pdf.add_metrics_table(
            "", damage_data, "Intervention damage summary", self._next_table()
        )

    def _add_business_cycle_section(
        self, pdf: PDFReportBuilder, summary: dict[str, Any]
    ):
        """Add business cycle analysis section"""
        pdf.add_section("Business Cycle Analysis")

        cycle = summary.get("business_cycle", {})

        if not cycle:
            pdf.add_paragraph("No business cycle data available.")
            return

        phase = cycle.get("phase", "unknown")
        pdf.add_paragraph(
            f"The economy is currently in the <b>{phase}</b> phase of the business cycle. "
            f"Boom intensity is {cycle.get('boom_intensity', 0):.2f}, indicating "
            f"{'strong expansion' if cycle.get('boom_intensity', 0) > 0.7 else 'moderate activity' if cycle.get('boom_intensity', 0) > 0.4 else 'weak conditions'}. "
            f"Interest rate distortion is {format_percent(cycle.get('rate_distortion', 0))}, "
            f"measuring how far current rates are from the natural rate."
        )

        cycle_data = {
            "Current Phase": phase.title(),
            "Boom Intensity": f"{cycle.get('boom_intensity', 0):.2f}",
            "Rate Distortion": format_percent(cycle.get("rate_distortion", 0)),
            "Credit Signal": cycle.get("credit_signal", "unknown").title(),
            "Investment Signal": cycle.get("investment_signal", "unknown").title(),
        }
        pdf.add_metrics_table(
            "", cycle_data, "Business cycle indicators", self._next_table()
        )

    def _add_dashboard_section(
        self, pdf: PDFReportBuilder, history: list[Any], country: str
    ):
        """Add dashboard visualization section"""
        pdf.add_section("Economic Dashboard")

        pdf.add_paragraph(
            "The following dashboard provides a comprehensive view of key economic indicators "
            "throughout the simulation period."
        )

        dashboard_bytes = self.chart_generator.create_simulation_dashboard(
            history, f"{country} Economic Simulation"
        )
        pdf.add_chart(
            dashboard_bytes,
            "Comprehensive economic dashboard",
            width=16,
            height=10,
            figure_num=self._next_figure(),
        )

    def _add_comprehensive_charts_section(
        self,
        pdf: PDFReportBuilder,
        history: list[Any],
        summary: dict[str, Any],
        country: str,
    ):
        """Add comprehensive charts section for single-country report"""

        # Extract data
        btc_prices = [float(m.bitcoin_price) for m in history]
        gold_prices = [float(m.gold_price) for m in history]
        inflation_rates = [m.inflation_rate for m in history]
        unemployment_rates = [m.unemployment_rate for m in history]
        freedom_indices = [m.freedom_index for m in history]
        cb_damage = [float(m.central_bank_damage) for m in history]
        gov_damage = [float(m.government_damage) for m in history]
        money_supply = [m.money_supply for m in history]
        credit_expansion = [m.credit_expansion for m in history]

        # ===========================================
        # SECTION: COMPREHENSIVE DASHBOARD
        # ===========================================
        pdf.add_section("Comprehensive Dashboard")
        pdf.add_paragraph(
            "The following dashboard provides a visual overview of all key economic indicators "
            "tracked during the simulation, allowing pattern recognition across multiple metrics."
        )

        dashboard = self.chart_generator.create_simulation_dashboard(
            history, f"{country} Economic Simulation"
        )
        pdf.add_chart(
            dashboard,
            "Six-panel economic dashboard",
            width=16,
            height=10,
            figure_num=self._next_figure(),
        )

        # ===========================================
        # SECTION: HARD ASSETS EVOLUTION
        # ===========================================
        pdf.add_section("Hard Asset Performance")
        pdf.add_paragraph(
            "Bitcoin and gold serve as 'economic truth detectors' - their prices in fiat currency "
            "reveal the true extent of monetary debasement. When priced in sound money, real economic "
            "activity becomes visible free from central bank distortion."
        )

        # Price comparison chart
        if len(btc_prices) > 1 and len(gold_prices) > 1:
            comparison = self.chart_generator.create_price_comparison_chart(
                [Decimal(str(p)) for p in btc_prices],
                [Decimal(str(p)) for p in gold_prices],
                "Hard Asset Performance (Indexed to 100)",
            )
            pdf.add_chart(
                comparison,
                "Bitcoin and Gold performance indexed to starting value",
                figure_num=self._next_figure(),
            )

        # Bitcoin chart
        btc_chart = self.chart_generator.create_bitcoin_price_chart(
            [Decimal(str(p)) for p in btc_prices],
            f"Bitcoin Price Evolution ({country})",
        )
        pdf.add_chart(
            btc_chart,
            "Bitcoin price trajectory during simulation",
            figure_num=self._next_figure(),
        )

        # ===========================================
        # SECTION: MONETARY DISTORTION
        # ===========================================
        pdf.add_section("Monetary Distortion Analysis")
        pdf.add_paragraph(
            "The relationship between money supply and credit expansion reveals the extent of "
            "fractional reserve banking's multiplication effect. Credit created beyond actual "
            "savings represents potential malinvestment waiting to be liquidated."
        )

        # Money supply vs credit
        if money_supply and credit_expansion:
            money_chart = self.chart_generator.create_money_supply_chart(
                money_supply, credit_expansion, "Money Supply & Credit Expansion"
            )
            pdf.add_chart(
                money_chart,
                "Base money and credit expansion over simulation period",
                figure_num=self._next_figure(),
            )

        # Inflation chart
        if inflation_rates:
            inflation_chart = self.chart_generator.create_inflation_chart(
                inflation_rates, f"Inflation Rate ({country})"
            )
            pdf.add_chart(
                inflation_chart,
                "Annualized inflation rate showing price distortion",
                figure_num=self._next_figure(),
            )

        # ===========================================
        # SECTION: INTERVENTION DAMAGE
        # ===========================================
        pdf.add_section("Intervention Damage Visualization")
        pdf.add_paragraph(
            "The following charts visualize the cumulative damage caused by central bank and "
            "government intervention. This damage represents real wealth destroyed through "
            "misallocation, deadweight loss, and malinvestment."
        )

        # Damage stacked area
        if cb_damage and gov_damage:
            damage_chart = self.chart_generator.create_damage_chart(
                [Decimal(str(d)) for d in cb_damage],
                [Decimal(str(d)) for d in gov_damage],
                "Cumulative Intervention Damage",
            )
            pdf.add_chart(
                damage_chart,
                "Stacked area showing CB and Government damage accumulation",
                figure_num=self._next_figure(),
            )

        # Damage donut
        final_cb = cb_damage[-1] if cb_damage else 0
        final_gov = gov_damage[-1] if gov_damage else 0
        if final_cb > 0 or final_gov > 0:
            donut = self.chart_generator.create_damage_breakdown_donut(
                final_cb, final_gov, "Damage Distribution"
            )
            pdf.add_chart(
                donut,
                "Proportion of damage caused by Central Bank vs Government",
                width=10,
                height=10,
                figure_num=self._next_figure(),
            )

        # ===========================================
        # SECTION: LABOR MARKET
        # ===========================================
        pdf.add_section("Labor Market Dynamics")
        pdf.add_paragraph(
            "Unemployment in Austrian theory results from intervention - minimum wages, regulations, "
            "and taxation prevent labor markets from clearing. The 'natural' unemployment rate in a "
            "free market would be minimal, representing only voluntary job transitions."
        )

        if unemployment_rates:
            unemployment_chart = self.chart_generator.create_unemployment_chart(
                unemployment_rates, f"Unemployment Rate ({country})"
            )
            pdf.add_chart(
                unemployment_chart,
                "Unemployment rate evolution during simulation",
                figure_num=self._next_figure(),
            )

        # ===========================================
        # SECTION: FREEDOM INDEX
        # ===========================================
        pdf.add_section("Economic Freedom Tracking")
        pdf.add_paragraph(
            "The Economic Freedom Index measures the degree to which voluntary exchange is permitted. "
            "Higher values indicate less intervention. Austrian theory predicts that freedom correlates "
            "positively with prosperity and negatively with malinvestment."
        )

        if freedom_indices:
            freedom_chart = self.chart_generator.create_freedom_index_chart(
                freedom_indices, f"Freedom Index ({country})"
            )
            pdf.add_chart(
                freedom_chart,
                "Economic freedom index evolution",
                figure_num=self._next_figure(),
            )

        # ===========================================
        # SECTION: DISTRIBUTION ANALYSIS
        # ===========================================
        pdf.add_section("Distribution Analysis")
        pdf.add_paragraph(
            "The following histogram shows the distribution of key metrics across the simulation "
            "period, revealing central tendencies and variance in economic outcomes."
        )

        # Inflation distribution histogram
        if inflation_rates:
            inflation_pct = [r * 100 if r < 1 else r for r in inflation_rates]
            if len(inflation_pct) >= 3:
                histogram = self.chart_generator.create_histogram(
                    inflation_pct,
                    "Inflation Rate Distribution",
                    "Inflation (%)",
                    bins=min(10, len(inflation_pct)),
                )
                pdf.add_chart(
                    histogram,
                    "Histogram of inflation rates during simulation",
                    figure_num=self._next_figure(),
                )

    # ===========================================
    # HELPER METHODS - MULTI-COUNTRY SECTIONS
    # ===========================================

    def _add_country_overview_section(
        self, pdf: PDFReportBuilder, summary: dict[str, Any]
    ):
        """Add country overview section for multi-country report"""
        pdf.add_section("Country Overview")

        country_data = summary.get("countries", {})

        pdf.add_paragraph(
            f"The simulation includes {len(country_data)} countries with varying regime types "
            "and intervention levels. Each country operates its own central bank and government, "
            "with agents responding to local conditions while also engaging in international trade."
        )

        # Build comparison table
        headers = ["Country", "Regime", "Freedom", "Inflation", "Unemployment"]
        rows = []

        for code, data in country_data.items():
            regime = data.get("regime", "N/A")
            if regime == "N/A":
                # Try to get from config
                regime = "Mixed"
            rows.append(
                [
                    code,
                    str(regime).replace("_", " ").title() if regime else "N/A",
                    f"{data.get('freedom_index', 0):.0f}",
                    format_percent(data.get("inflation", 0)),
                    format_percent(data.get("unemployment", 0)),
                ]
            )

        pdf.add_comparison_table(
            headers, rows, "Overview of simulated countries", self._next_table()
        )

    def _add_rankings_section(self, pdf: PDFReportBuilder, summary: dict[str, Any]):
        """Add country rankings section"""
        pdf.add_section("Comparative Rankings")

        country_data = summary.get("countries", {})

        # Freedom ranking
        sorted_by_freedom = sorted(
            country_data.items(),
            key=lambda x: x[1].get("freedom_index", 0),
            reverse=True,
        )

        pdf.add_subsection("Economic Freedom Ranking")
        pdf.add_paragraph(
            "Countries ranked by economic freedom index, where higher values indicate less "
            "government intervention and freer markets."
        )

        freedom_rankings = [
            (c, f"{data.get('freedom_index', 0):.1f}") for c, data in sorted_by_freedom
        ]
        pdf.add_ranking_table(
            "", freedom_rankings, "Countries by freedom index", self._next_table()
        )

        # Inflation ranking
        sorted_by_inflation = sorted(
            country_data.items(), key=lambda x: x[1].get("inflation", 0)
        )

        pdf.add_subsection("Inflation Ranking")
        pdf.add_paragraph("Countries ranked by inflation rate (lower is better).")

        inflation_rankings = [
            (c, format_percent(data.get("inflation", 0)))
            for c, data in sorted_by_inflation
        ]
        pdf.add_ranking_table(
            "", inflation_rankings, "Countries by inflation rate", self._next_table()
        )

    def _add_geopolitical_section(self, pdf: PDFReportBuilder, summary: dict[str, Any]):
        """Add geopolitical analysis section"""
        pdf.add_section("Geopolitical Analysis")

        war_risks = summary.get("war_risks", [])
        rel_summary = summary.get("relationship_summary", {})

        pdf.add_paragraph(
            f"The simulation tracked inter-country relationships including alliances, rivalries, "
            f"and conflict probabilities. Among the simulated countries, there are "
            f"{rel_summary.get('allies', 0)} allied pairs, {rel_summary.get('rivals', 0)} rival pairs, "
            f"and {rel_summary.get('enemies', 0)} enemy pairs."
        )

        if war_risks:
            pdf.add_subsection("War Risk Assessment")
            pdf.add_paragraph(
                "The following country pairs show elevated conflict probability based on "
                "historical tensions, sanctions, and current relationship dynamics."
            )

            headers = ["Countries", "Probability", "Risk Level"]
            rows = []
            for risk in war_risks[:5]:  # Top 5
                countries = f"{risk['countries'][0]}-{risk['countries'][1]}"
                prob = format_percent(risk["probability"])
                level = (
                    "HIGH"
                    if risk["probability"] >= 0.10
                    else ("MODERATE" if risk["probability"] >= 0.05 else "LOW")
                )
                rows.append([countries, prob, level])

            pdf.add_comparison_table(
                headers, rows, "Highest war risk pairs", self._next_table()
            )

            pdf.add_note(
                "Austrian economics predicts that free trade reduces conflict probability. "
                "When nations trade freely, war becomes economically costly. Protectionism "
                "and sanctions increase conflict risk by reducing economic interdependence."
            )
        else:
            pdf.add_paragraph(
                "No significant war risks were detected among the simulated countries."
            )

    def _add_global_metrics_section(
        self, pdf: PDFReportBuilder, summary: dict[str, Any]
    ):
        """Add global metrics section"""
        pdf.add_section("Global Economic Metrics")

        global_metrics = summary.get("global_metrics", {})

        pdf.add_paragraph(
            "Aggregate metrics across all simulated economies provide insight into global "
            "economic conditions and total intervention damage."
        )

        formatted_metrics = {
            "Total GDP": global_metrics.get("total_gdp", 0),
            "Average Inflation": global_metrics.get("avg_inflation", 0),
            "Average Freedom Index": f"{global_metrics.get('avg_freedom_index', 0):.1f}",
            "Total CB Damage": global_metrics.get("total_cb_damage", 0),
            "Total Gov Damage": global_metrics.get("total_gov_damage", 0),
            "High War Risk Pairs": global_metrics.get("high_war_risk_pairs", 0),
        }

        pdf.add_metrics_table(
            "", formatted_metrics, "Aggregate global metrics", self._next_table()
        )

    def _add_evolution_charts_section(
        self, pdf: PDFReportBuilder, history: list[Any], summary: dict[str, Any]
    ):
        """Add comprehensive visualization section for multi-country report"""
        countries = summary.get("config", {}).get("countries", [])
        country_data = summary.get("countries", {})
        war_risks = summary.get("war_risks", [])
        global_metrics = summary.get("global_metrics", {})

        # ===========================================
        # SECTION: ECONOMIC PROFILE RADAR
        # ===========================================
        pdf.add_section("Economic Profiles")
        pdf.add_paragraph(
            "The radar chart below provides a multi-dimensional view of each country's "
            "economic profile, allowing comparison across multiple metrics simultaneously."
        )

        # Build radar data
        radar_metrics = ["freedom_index", "gdp", "unemployment", "inflation"]
        radar_data = {}
        for country in countries[:6]:  # Limit to 6 for readability
            if country in country_data:
                data = country_data[country]
                radar_data[country] = {
                    "freedom_index": data.get("freedom_index", 50),
                    "gdp": float(data.get("gdp", 0)) / 1e9,  # Normalize to billions
                    "unemployment": data.get("unemployment", 0) * 100
                    if data.get("unemployment", 0) < 1
                    else data.get("unemployment", 0),
                    "inflation": data.get("inflation", 0) * 100
                    if data.get("inflation", 0) < 1
                    else data.get("inflation", 0),
                }

        if radar_data:
            radar_chart = self.chart_generator.create_radar_chart(
                radar_data, radar_metrics, "Multi-Dimensional Economic Profile"
            )
            pdf.add_chart(
                radar_chart,
                "Radar chart comparing country profiles across key metrics",
                width=14,
                height=12,
                figure_num=self._next_figure(),
            )

        # ===========================================
        # SECTION: BUBBLE CHART - FREEDOM VS INFLATION
        # ===========================================
        pdf.add_section("Freedom-Inflation Analysis")
        pdf.add_paragraph(
            "The bubble chart below plots each country by economic freedom (x-axis) and inflation "
            "rate (y-axis), with bubble size representing GDP. Austrian theory predicts that "
            "countries with higher freedom will cluster in the lower-right quadrant (free & stable)."
        )

        bubble_data = {}
        for country in countries:
            if country in country_data:
                data = country_data[country]
                bubble_data[country] = {
                    "freedom_index": data.get("freedom_index", 50),
                    "inflation": data.get("inflation", 0),
                    "gdp": float(data.get("gdp", 1e9)),
                }

        if bubble_data:
            bubble_chart = self.chart_generator.create_bubble_chart(
                bubble_data,
                "freedom_index",
                "inflation",
                "gdp",
                "Freedom vs Inflation (Bubble Size = GDP)",
            )
            pdf.add_chart(
                bubble_chart,
                "Bubble chart: Freedom (x) vs Inflation (y), size = GDP",
                width=14,
                height=10,
                figure_num=self._next_figure(),
            )

        # ===========================================
        # SECTION: COUNTRY RANKINGS
        # ===========================================
        pdf.add_section("Country Rankings")

        # Freedom Index Ranking Bar Chart
        pdf.add_subsection("Freedom Index Ranking")
        freedom_rankings = {
            c: country_data[c].get("freedom_index", 0)
            for c in countries
            if c in country_data
        }
        if freedom_rankings:
            freedom_bar = self.chart_generator.create_ranking_horizontal_bar(
                freedom_rankings,
                "Economic Freedom Index by Country",
                "Freedom Index (0-100)",
            )
            pdf.add_chart(
                freedom_bar,
                "Countries ranked by economic freedom index",
                figure_num=self._next_figure(),
            )

        # Inflation Ranking (inverted - lower is better)
        pdf.add_subsection("Inflation Ranking")
        inflation_rankings = {
            c: country_data[c].get("inflation", 0) * 100
            if country_data[c].get("inflation", 0) < 1
            else country_data[c].get("inflation", 0)
            for c in countries
            if c in country_data
        }
        if inflation_rankings:
            inflation_bar = self.chart_generator.create_ranking_horizontal_bar(
                inflation_rankings,
                "Inflation Rate by Country",
                "Inflation (%)",
                highlight_best=False,
            )
            pdf.add_chart(
                inflation_bar,
                "Countries ranked by inflation rate (lower is better)",
                figure_num=self._next_figure(),
            )

        # ===========================================
        # SECTION: INTERVENTION DAMAGE ANALYSIS
        # ===========================================
        pdf.add_section("Intervention Damage Breakdown")
        pdf.add_paragraph(
            "The following visualizations show how intervention damage is distributed "
            "between central banks and governments globally."
        )

        # Global damage donut
        total_cb = float(global_metrics.get("total_cb_damage", 0))
        total_gov = float(global_metrics.get("total_gov_damage", 0))

        if total_cb > 0 or total_gov > 0:
            damage_donut = self.chart_generator.create_damage_breakdown_donut(
                total_cb, total_gov, "Global Intervention Damage Distribution"
            )
            pdf.add_chart(
                damage_donut,
                "Donut chart showing CB vs Government damage breakdown",
                width=10,
                height=10,
                figure_num=self._next_figure(),
            )

        # Damage by country
        pdf.add_subsection("Damage by Country")
        damage_by_country = {}
        for country in countries:
            if country in country_data:
                data = country_data[country]
                cb_dmg = float(data.get("cb_damage", 0))
                gov_dmg = float(data.get("gov_damage", 0))
                damage_by_country[country] = cb_dmg + gov_dmg

        if damage_by_country and any(v > 0 for v in damage_by_country.values()):
            damage_rankings = self.chart_generator.create_ranking_horizontal_bar(
                damage_by_country,
                "Total Intervention Damage by Country",
                "Damage (USD)",
                highlight_best=False,
            )
            pdf.add_chart(
                damage_rankings,
                "Countries ranked by total intervention damage",
                figure_num=self._next_figure(),
            )

        # ===========================================
        # SECTION: WAR PROBABILITY MATRIX
        # ===========================================
        if war_risks and len(countries) >= 2:
            pdf.add_section("War Probability Analysis")
            pdf.add_paragraph(
                "The heatmap below shows bilateral war probabilities between country pairs. "
                "Darker colors indicate higher conflict risk. Austrian economics predicts that "
                "countries with strong trade ties have lower war probabilities."
            )

            war_heatmap = self.chart_generator.create_war_probability_heatmap(
                war_risks, countries, "Bilateral War Probability Matrix"
            )
            pdf.add_chart(
                war_heatmap,
                "Heatmap of war probabilities between country pairs",
                width=12,
                height=10,
                figure_num=self._next_figure(),
            )

        # ===========================================
        # SECTION: REGIME COMPARISON
        # ===========================================
        pdf.add_section("Regime Type Analysis")
        pdf.add_paragraph(
            "The following chart compares economic outcomes grouped by political regime type, "
            "testing Hoppe's thesis that less interventionist regimes produce better outcomes."
        )

        regime_data = {}
        for country in countries:
            if country in country_data:
                data = country_data[country]
                regime_data[country] = {
                    "regime": data.get("regime", "unknown"),
                    "freedom_index": data.get("freedom_index", 50),
                    "inflation": data.get("inflation", 0),
                    "unemployment": data.get("unemployment", 0),
                }

        if regime_data:
            regime_chart = self.chart_generator.create_regime_comparison_chart(
                regime_data, "Economic Outcomes by Regime Type"
            )
            pdf.add_chart(
                regime_chart,
                "Comparison of key metrics by political regime type",
                width=16,
                height=8,
                figure_num=self._next_figure(),
            )

        # ===========================================
        # SECTION: GROUPED METRICS COMPARISON
        # ===========================================
        pdf.add_section("Multi-Metric Comparison")
        pdf.add_paragraph(
            "The grouped bar chart below allows direct comparison of multiple metrics "
            "across all countries simultaneously."
        )

        grouped_data = {}
        for country in countries[:8]:  # Limit for readability
            if country in country_data:
                data = country_data[country]
                grouped_data[country] = {
                    "freedom_index": data.get("freedom_index", 50),
                    "inflation": data.get("inflation", 0),
                    "unemployment": data.get("unemployment", 0),
                }

        if grouped_data:
            grouped_chart = self.chart_generator.create_grouped_bar_chart(
                grouped_data,
                ["freedom_index", "inflation", "unemployment"],
                "Key Metrics by Country",
            )
            pdf.add_chart(
                grouped_chart,
                "Grouped bar chart comparing Freedom, Inflation, and Unemployment",
                figure_num=self._next_figure(),
            )

        # ===========================================
        # SECTION: SCATTER CORRELATION
        # ===========================================
        pdf.add_section("Correlation Analysis")
        pdf.add_paragraph(
            "The scatter plot below shows the relationship between economic freedom and "
            "inflation rate. The trend line indicates the correlation direction. Austrian "
            "theory predicts a negative correlation: more freedom leads to lower inflation."
        )

        scatter_data = {}
        for country in countries:
            if country in country_data:
                data = country_data[country]
                scatter_data[country] = {
                    "freedom_index": data.get("freedom_index", 50),
                    "inflation": data.get("inflation", 0),
                }

        if scatter_data:
            scatter_chart = self.chart_generator.create_scatter_correlation(
                scatter_data,
                "freedom_index",
                "inflation",
                title="Freedom Index vs Inflation Rate Correlation",
            )
            pdf.add_chart(
                scatter_chart,
                "Scatter plot with trend line: Freedom vs Inflation",
                figure_num=self._next_figure(),
            )

        # ===========================================
        # SECTION: TIME SERIES EVOLUTION
        # ===========================================
        if history:
            pdf.add_section("Time Series Evolution")
            pdf.add_paragraph(
                "The following charts show how key indicators evolved over the simulation period "
                "for each country."
            )

            # Extract data by country over time
            freedom_data = {}
            inflation_data = {}
            damage_data = {}

            for metric in history:
                for country in countries:
                    country_metric = metric.country_metrics.get(country)
                    if country_metric:
                        if country not in freedom_data:
                            freedom_data[country] = []
                            inflation_data[country] = []
                            damage_data[country] = []
                        freedom_data[country].append(country_metric.freedom_index)
                        inflation_data[country].append(
                            country_metric.inflation_rate * 100
                            if country_metric.inflation_rate < 1
                            else country_metric.inflation_rate
                        )
                        damage_data[country].append(
                            float(country_metric.central_bank_damage)
                            + float(country_metric.government_damage)
                        )

            if freedom_data:
                freedom_chart = (
                    self.chart_generator.create_multi_country_evolution_chart(
                        freedom_data,
                        "Freedom Index",
                        "Economic Freedom Evolution by Country",
                    )
                )
                pdf.add_chart(
                    freedom_chart,
                    "Freedom index evolution over simulation period",
                    figure_num=self._next_figure(),
                )

            if inflation_data:
                inflation_chart = (
                    self.chart_generator.create_multi_country_evolution_chart(
                        inflation_data,
                        "Inflation Rate (%)",
                        "Inflation Rate Evolution by Country",
                    )
                )
                pdf.add_chart(
                    inflation_chart,
                    "Inflation rate evolution over simulation period",
                    figure_num=self._next_figure(),
                )

            # Stacked area for cumulative damage
            if damage_data and any(d[-1] > 0 for d in damage_data.values() if d):
                stacked_chart = self.chart_generator.create_stacked_area_evolution(
                    damage_data,
                    "Cumulative Damage",
                    "Cumulative Intervention Damage by Country",
                )
                pdf.add_chart(
                    stacked_chart,
                    "Stacked area showing cumulative damage over time",
                    figure_num=self._next_figure(),
                )


# ===========================================
# CONVENIENCE FUNCTIONS
# ===========================================


def generate_simulation_report(engine, output_dir: Path | None = None) -> Path:
    """Convenience function to generate a report from a simulation engine."""
    config = ReportConfig()
    if output_dir:
        config.output_dir = output_dir

    generator = ReportGenerator(config)
    summary = engine.get_summary()
    history = engine.metrics_history

    if hasattr(engine, "country_engines"):
        return generator.generate_multi_country_report(summary, history)
    else:
        return generator.generate_single_country_report(summary, history)
