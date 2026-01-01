"""
Report Generator

Main module for generating comprehensive simulation reports.
Combines chart generation, PDF building, and data analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from decimal import Decimal
import json

from .charts import ChartGenerator
from .pdf_generator import PDFReportBuilder


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    output_dir: Path = field(default_factory=lambda: Path("reports"))
    include_charts: bool = True
    include_austrian_notes: bool = True
    include_theory_section: bool = True
    include_references: bool = True
    chart_width: float = 6.0


class ReportGenerator:
    """
    Generates comprehensive PDF reports for economic simulations.

    Features:
    - Scientific article formatting
    - Multiple chart visualizations
    - Austrian Economics explanations
    - Agent interaction summaries
    - Country comparison (multi-country)
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.chart_generator = ChartGenerator(self.config.output_dir / "charts")

    def generate_single_country_report(
        self,
        simulation_summary: Dict[str, Any],
        metrics_history: List[Any],
        output_filename: Optional[str] = None
    ) -> Path:
        """
        Generate a comprehensive report for a single-country simulation.

        Args:
            simulation_summary: Summary from engine.get_summary()
            metrics_history: List of SimulationMetrics from simulation
            output_filename: Optional custom filename

        Returns:
            Path to the generated PDF
        """
        # Setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        country = simulation_summary.get('config', {}).get('country', 'Unknown')
        regime = simulation_summary.get('config', {}).get('regime', 'Unknown')

        if output_filename is None:
            output_filename = f"simulation_report_{country}_{timestamp}.pdf"

        output_path = self.config.output_dir / output_filename

        # Initialize PDF builder
        pdf = PDFReportBuilder(
            output_path=output_path,
            title=f"Economic Simulation Report: {country}",
            author="Inflationator"
        )

        # =======================
        # TITLE PAGE
        # =======================
        pdf.add_title_page(
            title=f"Economic Simulation Report",
            subtitle=f"{country} under {regime} Regime",
            country=country,
            regime=regime
        )

        # =======================
        # ABSTRACT
        # =======================
        abstract = self._generate_abstract(simulation_summary, metrics_history, country)
        pdf.add_abstract(abstract)

        # =======================
        # TABLE OF CONTENTS (manual for now)
        # =======================
        pdf.add_section("Table of Contents")
        toc_items = [
            "1. Executive Summary",
            "2. Simulation Configuration",
            "3. Initial Economic Conditions",
            "4. Economic Dashboard",
            "5. Price Evolution Analysis",
            "6. Monetary Analysis",
            "7. Labor Market Analysis",
            "8. Intervention Damage Assessment",
            "9. Business Cycle Analysis",
            "10. Conclusions and Recommendations",
        ]
        if self.config.include_theory_section:
            toc_items.append("11. Theoretical Framework")
        if self.config.include_references:
            toc_items.append("12. References")

        pdf.add_numbered_list(toc_items)
        pdf.add_page_break()

        # =======================
        # 1. EXECUTIVE SUMMARY
        # =======================
        pdf.add_section("1. Executive Summary")
        self._add_executive_summary(pdf, simulation_summary, metrics_history, country)

        # =======================
        # 2. SIMULATION CONFIGURATION
        # =======================
        pdf.add_simulation_config_section(simulation_summary.get('config', {}))

        # =======================
        # 3. INITIAL CONDITIONS
        # =======================
        if 'initial_conditions' in simulation_summary:
            pdf.add_initial_conditions_section(
                simulation_summary['initial_conditions'],
                country
            )

        pdf.add_page_break()

        # =======================
        # 4. ECONOMIC DASHBOARD
        # =======================
        if self.config.include_charts and metrics_history:
            dashboard_bytes = self.chart_generator.create_simulation_dashboard(
                metrics_history,
                title=f"{country} Economic Simulation Dashboard"
            )
            pdf.add_dashboard(
                dashboard_bytes,
                f"Figure 1: Comprehensive dashboard showing key economic indicators for {country} simulation."
            )
            pdf.add_page_break()

        # =======================
        # 5. PRICE EVOLUTION ANALYSIS
        # =======================
        pdf.add_section("5. Price Evolution Analysis")
        self._add_price_analysis(pdf, metrics_history, country)
        pdf.add_page_break()

        # =======================
        # 6. MONETARY ANALYSIS
        # =======================
        pdf.add_section("6. Monetary Analysis")
        self._add_monetary_analysis(pdf, metrics_history, simulation_summary)
        pdf.add_page_break()

        # =======================
        # 7. LABOR MARKET ANALYSIS
        # =======================
        pdf.add_section("7. Labor Market Analysis")
        self._add_labor_analysis(pdf, metrics_history, simulation_summary)

        # =======================
        # 8. INTERVENTION DAMAGE
        # =======================
        damage_summary = simulation_summary.get('damage_summary', {})
        pdf.add_damage_summary_table(
            damage_summary.get('central_bank', {}),
            damage_summary.get('government', {})
        )
        pdf.add_page_break()

        # =======================
        # 9. BUSINESS CYCLE ANALYSIS
        # =======================
        pdf.add_section("9. Business Cycle Analysis")
        self._add_business_cycle_analysis(pdf, simulation_summary)

        # =======================
        # 10. CONCLUSIONS
        # =======================
        final_metrics = metrics_history[-1] if metrics_history else None
        freedom_index = final_metrics.freedom_index if final_metrics else 50
        total_damage = (
            (final_metrics.central_bank_damage if final_metrics else Decimal("0")) +
            (final_metrics.government_damage if final_metrics else Decimal("0"))
        )
        recommendation = damage_summary.get('recommendation', 'Reduce intervention for better outcomes.')

        pdf.add_conclusions_section(freedom_index, total_damage, recommendation)

        # =======================
        # 11. THEORETICAL FRAMEWORK (optional)
        # =======================
        if self.config.include_theory_section:
            pdf.add_page_break()
            pdf.add_austrian_theory_section()

        # =======================
        # 12. REFERENCES (optional)
        # =======================
        if self.config.include_references:
            pdf.add_references_section()

        # Build PDF
        pdf.build()
        print(f"Report generated: {output_path}")

        return output_path

    def generate_multi_country_report(
        self,
        simulation_summary: Dict[str, Any],
        metrics_history: List[Any],
        output_filename: Optional[str] = None
    ) -> Path:
        """
        Generate a comprehensive report for a multi-country simulation.

        Args:
            simulation_summary: Summary from MultiCountryEngine.get_summary()
            metrics_history: List of MultiCountryMetrics from simulation
            output_filename: Optional custom filename

        Returns:
            Path to the generated PDF
        """
        # Setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        countries = simulation_summary.get('config', {}).get('countries', [])
        num_countries = len(countries)

        if output_filename is None:
            output_filename = f"multi_country_report_{num_countries}countries_{timestamp}.pdf"

        output_path = self.config.output_dir / output_filename

        # Initialize PDF builder
        pdf = PDFReportBuilder(
            output_path=output_path,
            title="Multi-Country Economic Simulation Report",
            author="Inflationator"
        )

        # =======================
        # TITLE PAGE
        # =======================
        pdf.add_title_page(
            title="Multi-Country Economic Simulation",
            subtitle=f"Comparative Analysis of {num_countries} Countries",
        )

        # =======================
        # ABSTRACT
        # =======================
        abstract = self._generate_multi_country_abstract(simulation_summary, metrics_history)
        pdf.add_abstract(abstract)

        # =======================
        # PARTICIPATING COUNTRIES
        # =======================
        pdf.add_section("Participating Countries")
        country_summary = simulation_summary.get('countries', {})

        headers = ["Country", "Regime", "Freedom Index", "Inflation", "Unemployment"]
        rows = []
        for country, data in country_summary.items():
            rows.append([
                country,
                data.get('regime', 'N/A'),
                f"{data.get('freedom_index', 0):.1f}",
                f"{data.get('inflation', 0):.1%}",
                f"{data.get('unemployment', 0):.1%}",
            ])

        pdf.add_comparison_table(
            "Country Overview",
            headers,
            rows,
            "Table 1: Overview of all simulated countries and their final metrics."
        )
        pdf.add_page_break()

        # =======================
        # WAR RISK ANALYSIS
        # =======================
        pdf.add_section("Geopolitical Risk Analysis")
        war_risks = simulation_summary.get('war_risks', [])

        if war_risks:
            pdf.add_paragraph(
                "The following country pairs show elevated probability of military conflict based on "
                "their relationship dynamics, trade tensions, historical grievances, and current sanctions:"
            )

            # War risk chart
            if self.config.include_charts:
                war_chart = self.chart_generator.create_war_risk_chart(
                    war_risks,
                    "War Risk Assessment Between Country Pairs"
                )
                pdf.add_chart(war_chart, "Figure: War probability between country pairs", width=5.5, height=3)

            pdf.add_austrian_note(
                "Austrian/Classical Liberal View: Free trade creates peace. "
                "When nations trade freely, the cost of war increases dramatically (loss of trade partners). "
                "Protectionism and sanctions increase conflict probability."
            )
        else:
            pdf.add_paragraph("No significant war risks were detected during the simulation period.")

        pdf.add_page_break()

        # =======================
        # COUNTRY RANKINGS
        # =======================
        pdf.add_section("Country Rankings and Comparisons")
        self._add_country_rankings(pdf, simulation_summary)

        # =======================
        # COMPARATIVE EVOLUTION CHARTS
        # =======================
        if self.config.include_charts and metrics_history:
            pdf.add_section("Comparative Economic Evolution")
            self._add_multi_country_charts(pdf, metrics_history, simulation_summary)

        pdf.add_page_break()

        # =======================
        # GLOBAL METRICS
        # =======================
        pdf.add_section("Global Economic Metrics")
        global_metrics = simulation_summary.get('global_metrics', {})
        pdf.add_metrics_table("Aggregate Global Metrics", global_metrics)

        # =======================
        # CONCLUSIONS
        # =======================
        pdf.add_section("Conclusions")

        # Find best and worst countries
        sorted_by_freedom = sorted(
            country_summary.items(),
            key=lambda x: x[1].get('freedom_index', 0),
            reverse=True
        )

        if sorted_by_freedom:
            best = sorted_by_freedom[0]
            worst = sorted_by_freedom[-1]

            pdf.add_paragraph(
                f"<b>Best Performing Country:</b> {best[0]} with Freedom Index {best[1].get('freedom_index', 0):.1f}"
            )
            pdf.add_paragraph(
                f"<b>Worst Performing Country:</b> {worst[0]} with Freedom Index {worst[1].get('freedom_index', 0):.1f}"
            )

        pdf.add_austrian_note(
            "The data consistently shows that countries with higher economic freedom "
            "(lower intervention, lower taxes, sound money) outperform those with heavy intervention. "
            "This aligns with Austrian Economics predictions: free markets create prosperity; "
            "intervention destroys it."
        )

        # =======================
        # REFERENCES
        # =======================
        if self.config.include_references:
            pdf.add_references_section()

        # Build PDF
        pdf.build()
        print(f"Multi-country report generated: {output_path}")

        return output_path

    # ===========================================
    # HELPER METHODS
    # ===========================================

    def _generate_abstract(
        self,
        summary: Dict[str, Any],
        history: List[Any],
        country: str
    ) -> str:
        """Generate abstract text for single-country report"""
        config = summary.get('config', {})
        metrics = summary.get('metrics', {})
        damage = summary.get('damage_summary', {})

        duration = config.get('ticks', 12)
        regime = config.get('regime', 'unknown')
        inflation = metrics.get('inflation_rate', 0)
        freedom = metrics.get('freedom_index', 50)

        total_damage = damage.get('total_intervention_damage', '0')

        return (
            f"This report presents a comprehensive analysis of an economic simulation for {country} "
            f"operating under a {regime} regime over a period of {duration} months. "
            f"The simulation employs Austrian Economics principles to model agent behavior, "
            f"market dynamics, and the effects of government and central bank intervention. "
            f"Key findings include an annualized inflation rate of {inflation:.1%}, "
            f"a freedom index of {freedom:.1f}/100, and total intervention damage of ${total_damage}. "
            f"The analysis quantifies the economic distortions caused by monetary and fiscal policy, "
            f"providing evidence for the Austrian claim that intervention destroys value."
        )

    def _generate_multi_country_abstract(
        self,
        summary: Dict[str, Any],
        history: List[Any]
    ) -> str:
        """Generate abstract for multi-country report"""
        countries = summary.get('config', {}).get('countries', [])
        num_countries = len(countries)
        global_metrics = summary.get('global_metrics', {})

        return (
            f"This report presents a comparative analysis of economic simulations across {num_countries} countries. "
            f"Each country is modeled with its own government regime, central bank, and agent population. "
            f"The simulation tracks inter-country relationships including trade flows, tariffs, sanctions, "
            f"and war probabilities. Results demonstrate the Austrian Economics thesis that economic freedom "
            f"correlates strongly with prosperity, while intervention correlates with economic destruction. "
            f"Global metrics show average inflation of {global_metrics.get('avg_inflation', 0):.1%} and "
            f"average freedom index of {global_metrics.get('avg_freedom_index', 50):.1f}."
        )

    def _add_executive_summary(
        self,
        pdf: PDFReportBuilder,
        summary: Dict[str, Any],
        history: List[Any],
        country: str
    ):
        """Add executive summary content"""
        metrics = summary.get('metrics', {})
        damage = summary.get('damage_summary', {})

        pdf.add_paragraph(
            f"This simulation of {country}'s economy reveals the true cost of government and "
            f"central bank intervention. Unlike mainstream economic models that ignore or justify intervention, "
            f"this Austrian-framework simulation tracks every distortion and its economic impact."
        )

        # Key findings
        findings = []

        # Inflation finding
        inflation = metrics.get('inflation_rate', 0)
        if inflation > 0.10:
            findings.append(f"High inflation ({inflation:.1%}) driven primarily by monetary expansion")
        elif inflation > 0.05:
            findings.append(f"Moderate inflation ({inflation:.1%}) reflecting ongoing currency debasement")
        else:
            findings.append(f"Low inflation ({inflation:.1%}) suggesting relatively restrained monetary policy")

        # Freedom finding
        freedom = metrics.get('freedom_index', 50)
        if freedom >= 70:
            findings.append(f"High economic freedom ({freedom:.1f}) enabling market coordination")
        elif freedom >= 50:
            findings.append(f"Moderate economic freedom ({freedom:.1f}) with notable intervention costs")
        else:
            findings.append(f"Low economic freedom ({freedom:.1f}) causing significant market distortions")

        # Bitcoin finding
        btc_price = metrics.get('bitcoin_price', '0')
        findings.append(f"Bitcoin price of ${float(btc_price):,.0f} reflects dollar purchasing power")

        # Damage finding
        total_damage = damage.get('total_intervention_damage', '0')
        findings.append(f"Total intervention damage: ${total_damage}")

        pdf.add_bullet_list(findings, "Key Findings")

        pdf.add_austrian_note(
            "Every dollar of intervention damage represents real value that could have been "
            "created through voluntary exchange. This is not theoretical; it is measured "
            "through market distortions, malinvestment, and deadweight loss."
        )

    def _add_price_analysis(
        self,
        pdf: PDFReportBuilder,
        history: List[Any],
        country: str
    ):
        """Add price evolution analysis section"""
        if not history:
            pdf.add_paragraph("No price history data available.")
            return

        # Extract price data
        btc_prices = [m.bitcoin_price for m in history]
        gold_prices = [m.gold_price for m in history]

        pdf.add_subsection("5.1 Bitcoin Price Evolution")
        pdf.add_paragraph(
            "Bitcoin serves as a key indicator of fiat currency debasement. "
            "As central banks expand the money supply, rational actors move wealth into hard assets. "
            "Bitcoin's fixed supply (21 million) makes it an ideal measure of dollar purchasing power loss."
        )

        if self.config.include_charts:
            btc_chart = self.chart_generator.create_bitcoin_price_chart(
                btc_prices,
                f"Bitcoin Price During {country} Simulation"
            )
            pdf.add_chart(btc_chart, "Figure: Bitcoin price evolution reflecting monetary conditions", width=5.5, height=2.8)

        pdf.add_subsection("5.2 Gold Price Evolution")
        pdf.add_paragraph(
            "Gold remains the traditional inflation hedge with 5,000 years of monetary history. "
            "Its price movements often precede and predict inflation in consumer goods."
        )

        if self.config.include_charts:
            gold_chart = self.chart_generator.create_gold_price_chart(
                gold_prices,
                f"Gold Price During {country} Simulation"
            )
            pdf.add_chart(gold_chart, "Figure: Gold price evolution as inflation indicator", width=5.5, height=2.8)

        # Price comparison
        pdf.add_subsection("5.3 Hard Assets vs Fiat Comparison")
        if self.config.include_charts:
            comparison_chart = self.chart_generator.create_price_comparison_chart(
                btc_prices, gold_prices,
                "Hard Assets Performance (Normalized)"
            )
            pdf.add_chart(
                comparison_chart,
                "Figure: Comparative performance of Bitcoin and Gold normalized to starting value",
                width=5.5, height=2.8
            )

        pdf.add_austrian_note(
            "The appreciation of hard assets (Bitcoin, Gold) relative to fiat currency is not speculation; "
            "it is the market correctly pricing the ongoing debasement of government money. "
            "Holders of fiat currency are being taxed through inflation."
        )

    def _add_monetary_analysis(
        self,
        pdf: PDFReportBuilder,
        history: List[Any],
        summary: Dict[str, Any]
    ):
        """Add monetary analysis section"""
        if not history:
            pdf.add_paragraph("No monetary data available.")
            return

        pdf.add_paragraph(
            "Austrian Business Cycle Theory (ABCT) explains how central bank manipulation of interest rates "
            "and money supply causes boom-bust cycles. Credit expansion lowers rates below the natural rate, "
            "leading to malinvestment in longer-term projects that cannot be completed."
        )

        # Extract monetary data
        money_supply = [m.money_supply for m in history]
        credit = [m.credit_expansion for m in history]
        inflation = [m.inflation_rate for m in history]

        pdf.add_subsection("6.1 Money Supply and Credit")
        if self.config.include_charts:
            money_chart = self.chart_generator.create_money_supply_chart(
                money_supply, credit,
                "Money Supply and Credit Expansion"
            )
            pdf.add_chart(money_chart, "Figure: Base money and credit expansion showing monetary inflation", width=5.5, height=2.8)

        pdf.add_subsection("6.2 Inflation Rate")
        if self.config.include_charts:
            inflation_chart = self.chart_generator.create_inflation_chart(
                inflation,
                "Annualized Inflation Rate"
            )
            pdf.add_chart(inflation_chart, "Figure: Inflation rate evolution (Austrian calculation)", width=5.5, height=2.8)

        pdf.add_austrian_note(
            "Inflation is ALWAYS a monetary phenomenon. The central bank creates new money, "
            "which dilutes the value of existing money. CPI understates true inflation because "
            "it excludes asset prices and uses substitution bias. Our calculation uses market prices."
        )

    def _add_labor_analysis(
        self,
        pdf: PDFReportBuilder,
        history: List[Any],
        summary: Dict[str, Any]
    ):
        """Add labor market analysis section"""
        if not history:
            pdf.add_paragraph("No labor market data available.")
            return

        unemployment = [m.unemployment_rate for m in history]

        pdf.add_paragraph(
            "Labor markets clear at the market wage when unimpeded. Government interventions "
            "(minimum wage, labor regulations, payroll taxes) create unemployment by preventing "
            "mutually beneficial transactions between employers and workers."
        )

        if self.config.include_charts:
            unemployment_chart = self.chart_generator.create_unemployment_chart(
                unemployment,
                "Unemployment Rate Evolution"
            )
            pdf.add_chart(unemployment_chart, "Figure: Unemployment rate over simulation period", width=5.5, height=2.8)

        # Labor market statistics from summary
        labor_stats = summary.get('labor_market', {})
        if labor_stats:
            pdf.add_metrics_table("Labor Market Statistics", labor_stats)

        pdf.add_austrian_note(
            "True unemployment includes underemployment and discouraged workers, "
            "which government statistics systematically undercount. A free labor market "
            "with no minimum wage or regulations would clear at the market-determined wage, "
            "resulting in voluntary unemployment only."
        )

    def _add_business_cycle_analysis(
        self,
        pdf: PDFReportBuilder,
        summary: Dict[str, Any]
    ):
        """Add business cycle analysis section"""
        cycle = summary.get('business_cycle', {})

        pdf.add_paragraph(
            "The Austrian Business Cycle Theory explains economic fluctuations as the result of "
            "central bank credit expansion. When banks create credit not backed by real savings, "
            "interest rates fall below the natural rate, causing entrepreneurs to undertake projects "
            "that cannot be completed with available resources."
        )

        if cycle:
            cycle_metrics = {
                'Current Phase': cycle.get('phase', 'unknown').title(),
                'Boom Intensity': f"{cycle.get('boom_intensity', 0):.2f}",
                'Rate Distortion': f"{cycle.get('rate_distortion', 0):.2f}",
                'Credit Signal': cycle.get('credit_signal', 'neutral'),
                'Investment Signal': cycle.get('investment_signal', 'neutral'),
            }
            pdf.add_metrics_table("Business Cycle Indicators", cycle_metrics)

            if cycle.get('recommendation'):
                pdf.add_paragraph(f"<b>Analysis:</b> {cycle.get('recommendation')}")

        pdf.add_austrian_note(
            "The bust phase is not a problem to be solved; it is the solution. "
            "Liquidation of malinvestment frees resources for productive use. "
            "Government attempts to prevent or delay the bust only prolong the misallocation."
        )

    def _add_country_rankings(
        self,
        pdf: PDFReportBuilder,
        summary: Dict[str, Any]
    ):
        """Add country ranking tables for multi-country report"""
        countries = summary.get('countries', {})

        # Freedom ranking
        sorted_freedom = sorted(
            countries.items(),
            key=lambda x: x[1].get('freedom_index', 0),
            reverse=True
        )

        pdf.add_subsection("Freedom Index Ranking")
        freedom_rows = [
            [str(i+1), c, f"{data.get('freedom_index', 0):.1f}", data.get('regime', 'N/A')]
            for i, (c, data) in enumerate(sorted_freedom)
        ]
        pdf.add_comparison_table(
            "",
            ["Rank", "Country", "Freedom Index", "Regime"],
            freedom_rows,
            "Table: Countries ranked by economic freedom (higher = freer)"
        )

        # Inflation ranking (lower is better)
        sorted_inflation = sorted(
            countries.items(),
            key=lambda x: x[1].get('inflation', float('inf'))
        )

        pdf.add_subsection("Inflation Ranking")
        inflation_rows = [
            [str(i+1), c, f"{data.get('inflation', 0):.1%}"]
            for i, (c, data) in enumerate(sorted_inflation)
        ]
        pdf.add_comparison_table(
            "",
            ["Rank", "Country", "Inflation Rate"],
            inflation_rows,
            "Table: Countries ranked by inflation rate (lower = better)"
        )

    def _add_multi_country_charts(
        self,
        pdf: PDFReportBuilder,
        history: List[Any],
        summary: Dict[str, Any]
    ):
        """Add comparative evolution charts for multi-country report"""
        if not history:
            return

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

        # Freedom evolution chart
        if freedom_data:
            freedom_chart = self.chart_generator.create_multi_country_evolution_chart(
                freedom_data,
                "Freedom Index",
                "Economic Freedom Evolution by Country"
            )
            pdf.add_chart(
                freedom_chart,
                "Figure: Evolution of economic freedom across countries",
                width=6, height=3
            )

        # Inflation evolution chart
        if inflation_data:
            inflation_chart = self.chart_generator.create_multi_country_evolution_chart(
                inflation_data,
                "Inflation Rate",
                "Inflation Rate Evolution by Country"
            )
            pdf.add_chart(
                inflation_chart,
                "Figure: Evolution of inflation rates across countries",
                width=6, height=3
            )


# ===========================================
# CONVENIENCE FUNCTIONS
# ===========================================

def generate_simulation_report(
    engine,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Convenience function to generate a report from a simulation engine.

    Args:
        engine: SimulationEngine or MultiCountrySimulationEngine
        output_dir: Optional output directory

    Returns:
        Path to generated PDF
    """
    config = ReportConfig()
    if output_dir:
        config.output_dir = output_dir

    generator = ReportGenerator(config)

    summary = engine.get_summary()
    history = engine.metrics_history

    # Check if multi-country
    if hasattr(engine, 'country_engines'):
        return generator.generate_multi_country_report(summary, history)
    else:
        return generator.generate_single_country_report(summary, history)
