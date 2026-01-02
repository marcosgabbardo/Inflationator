"""
Report Generator - Academic Whitepaper Style

Generates comprehensive PDF reports for economic simulations.
Output format inspired by academic papers and whitepapers.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from decimal import Decimal

from .charts import ChartGenerator
from .pdf_generator import PDFReportBuilder, format_currency, format_percent, format_number


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

    def __init__(self, config: Optional[ReportConfig] = None):
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
        simulation_summary: Dict[str, Any],
        metrics_history: List[Any],
        output_filename: Optional[str] = None
    ) -> Path:
        """Generate academic-style report for single-country simulation."""

        # Reset counters
        self.table_counter = 0
        self.figure_counter = 0

        # Setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        country = simulation_summary.get('config', {}).get('country', 'USA')
        regime = simulation_summary.get('config', {}).get('regime', 'democracy_liberal')

        if output_filename is None:
            output_filename = f"simulation_report_{country}_{timestamp}.pdf"

        output_path = self.config.output_dir / output_filename

        # Initialize PDF
        pdf = PDFReportBuilder(output_path)

        # Title section
        pdf.add_title_page(
            title="Austrian Economics Simulation Analysis",
            subtitle=f"Economic Analysis of {country} under {regime.replace('_', ' ').title()} Regime"
        )

        # Abstract
        abstract = self._generate_single_country_abstract(simulation_summary, metrics_history, country)
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

        # Methodology
        if self.config.include_methodology:
            pdf.add_methodology_section()

        # Simulation Parameters
        self._add_parameters_section(pdf, simulation_summary)

        # Results: Price Evolution
        self._add_price_results_section(pdf, simulation_summary, metrics_history, country)

        # Results: Monetary Analysis
        self._add_monetary_results_section(pdf, simulation_summary, metrics_history)

        # Results: Labor Market
        self._add_labor_results_section(pdf, simulation_summary, metrics_history)

        # Intervention Damage Analysis
        self._add_damage_analysis_section(pdf, simulation_summary)

        # Business Cycle
        self._add_business_cycle_section(pdf, simulation_summary)

        # Dashboard (if charts enabled)
        if self.config.include_charts and metrics_history:
            self._add_dashboard_section(pdf, metrics_history, country)

        # Conclusions
        final_metrics = metrics_history[-1] if metrics_history else None
        freedom_index = final_metrics.freedom_index if final_metrics else 50
        total_damage = (
            (final_metrics.central_bank_damage if final_metrics else Decimal("0")) +
            (final_metrics.government_damage if final_metrics else Decimal("0"))
        )
        pdf.add_conclusions_section(freedom_index, total_damage)

        # References
        if self.config.include_references:
            pdf.add_references_section()

        # Build
        pdf.build()
        return output_path

    def generate_multi_country_report(
        self,
        simulation_summary: Dict[str, Any],
        metrics_history: List[Any],
        output_filename: Optional[str] = None
    ) -> Path:
        """Generate academic-style report for multi-country simulation."""

        # Reset counters
        self.table_counter = 0
        self.figure_counter = 0

        # Setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        countries = simulation_summary.get('config', {}).get('countries', [])
        num_countries = len(countries)

        if output_filename is None:
            output_filename = f"multi_country_report_{num_countries}countries_{timestamp}.pdf"

        output_path = self.config.output_dir / output_filename

        # Initialize PDF
        pdf = PDFReportBuilder(output_path)

        # Title
        pdf.add_title_page(
            title="Comparative Economic Analysis",
            subtitle=f"Multi-Country Simulation: {num_countries} Economies"
        )

        # Abstract
        abstract = self._generate_multi_country_abstract(simulation_summary, metrics_history)
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
        country_summary = simulation_summary.get('countries', {})
        sorted_countries = sorted(
            country_summary.items(),
            key=lambda x: x[1].get('freedom_index', 0),
            reverse=True
        )

        best = sorted_countries[0][0] if sorted_countries else None
        worst = sorted_countries[-1][0] if sorted_countries else None
        best_freedom = sorted_countries[0][1].get('freedom_index', 50) if sorted_countries else 50

        global_metrics = simulation_summary.get('global_metrics', {})
        total_damage = (
            float(global_metrics.get('total_cb_damage', 0)) +
            float(global_metrics.get('total_gov_damage', 0))
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
        self,
        summary: Dict[str, Any],
        history: List[Any],
        country: str
    ) -> str:
        config = summary.get('config', {})
        metrics = summary.get('metrics', {})

        duration = config.get('ticks', 12)
        inflation = metrics.get('inflation_rate', 0)
        freedom = metrics.get('freedom_index', 50)
        cb_damage = metrics.get('central_bank_damage', 0)
        gov_damage = metrics.get('government_damage', 0)

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
        self,
        summary: Dict[str, Any],
        history: List[Any]
    ) -> str:
        countries = summary.get('config', {}).get('countries', [])
        global_metrics = summary.get('global_metrics', {})

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

    def _add_parameters_section(self, pdf: PDFReportBuilder, summary: Dict[str, Any]):
        """Add simulation parameters section"""
        pdf.add_section("Simulation Parameters")

        config = summary.get('config', {})
        params = {
            'Country': config.get('country', 'USA'),
            'Regime Type': config.get('regime', 'unknown').replace('_', ' ').title(),
            'Duration': f"{config.get('ticks', 12)} months",
            'Agent Population': f"{config.get('num_persons', 100000):,} persons",
            'Companies': f"{config.get('num_companies', 10000):,}",
            'Banks': f"{config.get('num_banks', 500):,}",
            'CB Intervention Level': format_percent(config.get('central_bank_intervention', 0.5)),
        }

        pdf.add_metrics_table("", params, "Simulation configuration parameters", self._next_table())

    def _add_price_results_section(
        self,
        pdf: PDFReportBuilder,
        summary: Dict[str, Any],
        history: List[Any],
        country: str
    ):
        """Add price evolution results section"""
        pdf.add_section("Results: Price Evolution")

        if not history:
            pdf.add_paragraph("No price history data available for this simulation.")
            return

        # Initial and final prices
        initial = history[0]
        final = history[-1]

        btc_change = (float(final.bitcoin_price) - float(initial.bitcoin_price)) / float(initial.bitcoin_price)
        gold_change = (float(final.gold_price) - float(initial.gold_price)) / float(initial.gold_price)

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
            'Initial BTC Price': initial.bitcoin_price,
            'Final BTC Price': final.bitcoin_price,
            'BTC Change': f"{btc_change*100:+.1f}%",
            'Initial Gold Price': initial.gold_price,
            'Final Gold Price': final.gold_price,
            'Gold Change': f"{gold_change*100:+.1f}%",
        }
        pdf.add_metrics_table("", price_data, "Price evolution during simulation", self._next_table())

        # Charts
        if self.config.include_charts:
            btc_prices = [m.bitcoin_price for m in history]
            btc_chart = self.chart_generator.create_bitcoin_price_chart(
                btc_prices, f"Bitcoin Price Evolution"
            )
            pdf.add_chart(btc_chart, "Bitcoin price trajectory showing monetary conditions impact", figure_num=self._next_figure())

    def _add_monetary_results_section(
        self,
        pdf: PDFReportBuilder,
        summary: Dict[str, Any],
        history: List[Any]
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
            'Initial Money Supply': initial.money_supply,
            'Final Money Supply': final.money_supply,
            'Credit Created': final.credit_expansion,
            'Inflation Rate (Annualized)': final.inflation_rate,
            'Central Bank Damage': final.central_bank_damage,
        }
        pdf.add_metrics_table("", monetary_data, "Monetary indicators", self._next_table())

    def _add_labor_results_section(
        self,
        pdf: PDFReportBuilder,
        summary: Dict[str, Any],
        history: List[Any]
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
            'Unemployment Rate': final.unemployment_rate,
            'GDP': final.gdp,
            'Freedom Index': f"{final.freedom_index:.1f}/100",
        }
        pdf.add_metrics_table("", labor_data, "Labor market indicators", self._next_table())

    def _add_damage_analysis_section(self, pdf: PDFReportBuilder, summary: Dict[str, Any]):
        """Add intervention damage analysis section"""
        pdf.add_section("Intervention Damage Analysis")

        damage = summary.get('damage_summary', {})
        metrics = summary.get('metrics', {})

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

        total_damage = float(metrics.get('central_bank_damage', 0)) + float(metrics.get('government_damage', 0))
        damage_data = {
            'Central Bank Damage': metrics.get('central_bank_damage', 0),
            'Government Damage': metrics.get('government_damage', 0),
            'Total Malinvestment': metrics.get('total_malinvestment', 0),
            'Total Intervention Damage': Decimal(str(total_damage)),
        }
        pdf.add_metrics_table("", damage_data, "Intervention damage summary", self._next_table())

    def _add_business_cycle_section(self, pdf: PDFReportBuilder, summary: Dict[str, Any]):
        """Add business cycle analysis section"""
        pdf.add_section("Business Cycle Analysis")

        cycle = summary.get('business_cycle', {})

        if not cycle:
            pdf.add_paragraph("No business cycle data available.")
            return

        phase = cycle.get('phase', 'unknown')
        pdf.add_paragraph(
            f"The economy is currently in the <b>{phase}</b> phase of the business cycle. "
            f"Boom intensity is {cycle.get('boom_intensity', 0):.2f}, indicating "
            f"{'strong expansion' if cycle.get('boom_intensity', 0) > 0.7 else 'moderate activity' if cycle.get('boom_intensity', 0) > 0.4 else 'weak conditions'}. "
            f"Interest rate distortion is {format_percent(cycle.get('rate_distortion', 0))}, "
            f"measuring how far current rates are from the natural rate."
        )

        cycle_data = {
            'Current Phase': phase.title(),
            'Boom Intensity': f"{cycle.get('boom_intensity', 0):.2f}",
            'Rate Distortion': format_percent(cycle.get('rate_distortion', 0)),
            'Credit Signal': cycle.get('credit_signal', 'unknown').title(),
            'Investment Signal': cycle.get('investment_signal', 'unknown').title(),
        }
        pdf.add_metrics_table("", cycle_data, "Business cycle indicators", self._next_table())

    def _add_dashboard_section(
        self,
        pdf: PDFReportBuilder,
        history: List[Any],
        country: str
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
        pdf.add_chart(dashboard_bytes, "Comprehensive economic dashboard", width=16, height=10, figure_num=self._next_figure())

    # ===========================================
    # HELPER METHODS - MULTI-COUNTRY SECTIONS
    # ===========================================

    def _add_country_overview_section(self, pdf: PDFReportBuilder, summary: Dict[str, Any]):
        """Add country overview section for multi-country report"""
        pdf.add_section("Country Overview")

        country_data = summary.get('countries', {})

        pdf.add_paragraph(
            f"The simulation includes {len(country_data)} countries with varying regime types "
            "and intervention levels. Each country operates its own central bank and government, "
            "with agents responding to local conditions while also engaging in international trade."
        )

        # Build comparison table
        headers = ["Country", "Regime", "Freedom", "Inflation", "Unemployment"]
        rows = []

        for code, data in country_data.items():
            regime = data.get('regime', 'N/A')
            if regime == 'N/A':
                # Try to get from config
                regime = 'Mixed'
            rows.append([
                code,
                str(regime).replace('_', ' ').title() if regime else 'N/A',
                f"{data.get('freedom_index', 0):.0f}",
                format_percent(data.get('inflation', 0)),
                format_percent(data.get('unemployment', 0)),
            ])

        pdf.add_comparison_table(headers, rows, "Overview of simulated countries", self._next_table())

    def _add_rankings_section(self, pdf: PDFReportBuilder, summary: Dict[str, Any]):
        """Add country rankings section"""
        pdf.add_section("Comparative Rankings")

        country_data = summary.get('countries', {})

        # Freedom ranking
        sorted_by_freedom = sorted(
            country_data.items(),
            key=lambda x: x[1].get('freedom_index', 0),
            reverse=True
        )

        pdf.add_subsection("Economic Freedom Ranking")
        pdf.add_paragraph(
            "Countries ranked by economic freedom index, where higher values indicate less "
            "government intervention and freer markets."
        )

        freedom_rankings = [(c, f"{data.get('freedom_index', 0):.1f}") for c, data in sorted_by_freedom]
        pdf.add_ranking_table("", freedom_rankings, "Countries by freedom index", self._next_table())

        # Inflation ranking
        sorted_by_inflation = sorted(
            country_data.items(),
            key=lambda x: x[1].get('inflation', 0)
        )

        pdf.add_subsection("Inflation Ranking")
        pdf.add_paragraph("Countries ranked by inflation rate (lower is better).")

        inflation_rankings = [(c, format_percent(data.get('inflation', 0))) for c, data in sorted_by_inflation]
        pdf.add_ranking_table("", inflation_rankings, "Countries by inflation rate", self._next_table())

    def _add_geopolitical_section(self, pdf: PDFReportBuilder, summary: Dict[str, Any]):
        """Add geopolitical analysis section"""
        pdf.add_section("Geopolitical Analysis")

        war_risks = summary.get('war_risks', [])
        rel_summary = summary.get('relationship_summary', {})

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
                prob = format_percent(risk['probability'])
                level = "HIGH" if risk['probability'] >= 0.10 else ("MODERATE" if risk['probability'] >= 0.05 else "LOW")
                rows.append([countries, prob, level])

            pdf.add_comparison_table(headers, rows, "Highest war risk pairs", self._next_table())

            pdf.add_note(
                "Austrian economics predicts that free trade reduces conflict probability. "
                "When nations trade freely, war becomes economically costly. Protectionism "
                "and sanctions increase conflict risk by reducing economic interdependence."
            )
        else:
            pdf.add_paragraph("No significant war risks were detected among the simulated countries.")

    def _add_global_metrics_section(self, pdf: PDFReportBuilder, summary: Dict[str, Any]):
        """Add global metrics section"""
        pdf.add_section("Global Economic Metrics")

        global_metrics = summary.get('global_metrics', {})

        pdf.add_paragraph(
            "Aggregate metrics across all simulated economies provide insight into global "
            "economic conditions and total intervention damage."
        )

        formatted_metrics = {
            'Total GDP': global_metrics.get('total_gdp', 0),
            'Average Inflation': global_metrics.get('avg_inflation', 0),
            'Average Freedom Index': f"{global_metrics.get('avg_freedom_index', 0):.1f}",
            'Total CB Damage': global_metrics.get('total_cb_damage', 0),
            'Total Gov Damage': global_metrics.get('total_gov_damage', 0),
            'High War Risk Pairs': global_metrics.get('high_war_risk_pairs', 0),
        }

        pdf.add_metrics_table("", formatted_metrics, "Aggregate global metrics", self._next_table())

    def _add_evolution_charts_section(
        self,
        pdf: PDFReportBuilder,
        history: List[Any],
        summary: Dict[str, Any]
    ):
        """Add evolution charts for multi-country report"""
        pdf.add_section("Comparative Evolution")

        pdf.add_paragraph(
            "The following charts show how key indicators evolved over time for each country, "
            "allowing visual comparison of economic trajectories."
        )

        countries = summary.get('config', {}).get('countries', [])

        # Extract data by country
        freedom_data = {}
        inflation_data = {}

        for metric in history:
            for country in countries:
                country_metric = metric.country_metrics.get(country)
                if country_metric:
                    if country not in freedom_data:
                        freedom_data[country] = []
                        inflation_data[country] = []
                    freedom_data[country].append(country_metric.freedom_index)
                    inflation_data[country].append(country_metric.inflation_rate)

        if freedom_data:
            freedom_chart = self.chart_generator.create_multi_country_evolution_chart(
                freedom_data, "Freedom Index", "Economic Freedom by Country"
            )
            pdf.add_chart(freedom_chart, "Freedom index evolution by country", figure_num=self._next_figure())

        if inflation_data:
            inflation_chart = self.chart_generator.create_multi_country_evolution_chart(
                inflation_data, "Inflation Rate", "Inflation Rate by Country"
            )
            pdf.add_chart(inflation_chart, "Inflation rate evolution by country", figure_num=self._next_figure())


# ===========================================
# CONVENIENCE FUNCTIONS
# ===========================================

def generate_simulation_report(engine, output_dir: Optional[Path] = None) -> Path:
    """Convenience function to generate a report from a simulation engine."""
    config = ReportConfig()
    if output_dir:
        config.output_dir = output_dir

    generator = ReportGenerator(config)
    summary = engine.get_summary()
    history = engine.metrics_history

    if hasattr(engine, 'country_engines'):
        return generator.generate_multi_country_report(summary, history)
    else:
        return generator.generate_single_country_report(summary, history)
