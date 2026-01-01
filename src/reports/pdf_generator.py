"""
PDF Report Generator

Creates professional PDF reports using ReportLab.
Designed for scientific/academic publication quality.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, ListFlowable, ListItem
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from decimal import Decimal
import io


class PDFReportBuilder:
    """
    Builds professional PDF reports for economic simulations.

    Features:
    - Scientific article formatting
    - Tables for data presentation
    - Embedded charts
    - Austrian Economics annotations
    """

    def __init__(
        self,
        output_path: Path,
        title: str = "Economic Simulation Report",
        author: str = "Inflationator Simulation Engine"
    ):
        self.output_path = output_path
        self.title = title
        self.author = author

        # Document setup
        self.doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2.5*cm,
            bottomMargin=2.5*cm,
        )

        # Styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

        # Content elements
        self.elements: List[Any] = []

    def _setup_custom_styles(self):
        """Setup custom paragraph styles for scientific formatting"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1a1a2e'),
        ))

        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#4a4a4a'),
            fontName='Helvetica-Oblique',
        ))

        # Section heading
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor('#1a1a2e'),
            borderWidth=1,
            borderColor=colors.HexColor('#1f77b4'),
            borderPadding=5,
        ))

        # Subsection heading
        self.styles.add(ParagraphStyle(
            name='SubsectionHeading',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#2a2a3e'),
        ))

        # Body text (justified)
        self.styles.add(ParagraphStyle(
            name='BodyJustified',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
            leading=14,
        ))

        # Austrian theory note (italics, gray)
        self.styles.add(ParagraphStyle(
            name='AustrianNote',
            parent=self.styles['Normal'],
            fontSize=9,
            fontName='Helvetica-Oblique',
            textColor=colors.HexColor('#666666'),
            leftIndent=20,
            spaceAfter=10,
            spaceBefore=5,
            backColor=colors.HexColor('#f5f5f5'),
            borderWidth=0,
            borderPadding=8,
        ))

        # Metric value (bold)
        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=11,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER,
        ))

        # Caption style
        self.styles.add(ParagraphStyle(
            name='Caption',
            parent=self.styles['Normal'],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#4a4a4a'),
            fontName='Helvetica-Oblique',
            spaceBefore=5,
            spaceAfter=15,
        ))

        # Abstract style
        self.styles.add(ParagraphStyle(
            name='Abstract',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            leftIndent=30,
            rightIndent=30,
            spaceAfter=20,
            leading=14,
        ))

    # ===========================================
    # DOCUMENT STRUCTURE
    # ===========================================

    def add_title_page(
        self,
        title: str,
        subtitle: str,
        date: Optional[datetime] = None,
        country: Optional[str] = None,
        regime: Optional[str] = None
    ):
        """Add a title page with simulation info"""
        date = date or datetime.now()

        # Main title
        self.elements.append(Spacer(1, 2*inch))
        self.elements.append(Paragraph(title, self.styles['ReportTitle']))

        # Subtitle
        self.elements.append(Paragraph(subtitle, self.styles['ReportSubtitle']))

        # Spacer
        self.elements.append(Spacer(1, 0.5*inch))

        # Simulation metadata
        meta_data = [
            ["Simulation Date:", date.strftime("%B %d, %Y")],
            ["Generated by:", "Inflationator - Austrian Economics Simulator"],
        ]
        if country:
            meta_data.append(["Country:", country])
        if regime:
            meta_data.append(["Regime Type:", regime])

        meta_table = Table(meta_data, colWidths=[2*inch, 3*inch])
        meta_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        self.elements.append(meta_table)

        # Austrian economics note at bottom
        self.elements.append(Spacer(1, 2*inch))
        self.elements.append(Paragraph(
            "This simulation is based on Austrian Economics principles (Mises, Hayek, Rothbard). "
            "All interventions are tracked for their true economic cost.",
            self.styles['AustrianNote']
        ))

        self.elements.append(PageBreak())

    def add_abstract(self, text: str):
        """Add an abstract section"""
        self.elements.append(Paragraph("Abstract", self.styles['SectionHeading']))
        self.elements.append(Paragraph(text, self.styles['Abstract']))
        self.elements.append(Spacer(1, 0.3*inch))

    def add_section(self, title: str, content: Optional[str] = None):
        """Add a new section with heading"""
        self.elements.append(Paragraph(title, self.styles['SectionHeading']))
        if content:
            self.elements.append(Paragraph(content, self.styles['BodyJustified']))

    def add_subsection(self, title: str, content: Optional[str] = None):
        """Add a subsection"""
        self.elements.append(Paragraph(title, self.styles['SubsectionHeading']))
        if content:
            self.elements.append(Paragraph(content, self.styles['BodyJustified']))

    def add_paragraph(self, text: str):
        """Add a body paragraph"""
        self.elements.append(Paragraph(text, self.styles['BodyJustified']))

    def add_austrian_note(self, text: str):
        """Add an Austrian Economics explanation note"""
        self.elements.append(Paragraph(
            f"<b>Austrian Economics Note:</b> {text}",
            self.styles['AustrianNote']
        ))

    def add_spacer(self, height: float = 0.2):
        """Add vertical space (in inches)"""
        self.elements.append(Spacer(1, height*inch))

    def add_page_break(self):
        """Add a page break"""
        self.elements.append(PageBreak())

    def add_horizontal_line(self):
        """Add a horizontal divider line"""
        self.elements.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor('#cccccc'),
            spaceBefore=10,
            spaceAfter=10
        ))

    # ===========================================
    # TABLES
    # ===========================================

    def add_metrics_table(
        self,
        title: str,
        metrics: Dict[str, Any],
        caption: Optional[str] = None
    ):
        """Add a metrics summary table"""
        self.elements.append(Paragraph(title, self.styles['SubsectionHeading']))

        # Convert metrics to table data
        data = [["Metric", "Value"]]
        for key, value in metrics.items():
            # Format key
            key_formatted = key.replace('_', ' ').title()

            # Format value
            if isinstance(value, float):
                if 'rate' in key.lower() or 'index' in key.lower():
                    value_formatted = f"{value:.2%}" if value < 1 else f"{value:.1f}"
                else:
                    value_formatted = f"{value:,.2f}"
            elif isinstance(value, Decimal):
                value_formatted = f"${float(value):,.0f}"
            else:
                value_formatted = str(value)

            data.append([key_formatted, value_formatted])

        table = Table(data, colWidths=[3*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f8f8')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dddddd')),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ]))

        self.elements.append(table)

        if caption:
            self.elements.append(Paragraph(caption, self.styles['Caption']))

        self.elements.append(Spacer(1, 0.2*inch))

    def add_comparison_table(
        self,
        title: str,
        headers: List[str],
        rows: List[List[str]],
        caption: Optional[str] = None
    ):
        """Add a comparison table (e.g., for multi-country)"""
        self.elements.append(Paragraph(title, self.styles['SubsectionHeading']))

        data = [headers] + rows

        # Calculate column widths based on number of columns
        num_cols = len(headers)
        col_width = (6.5*inch) / num_cols

        table = Table(data, colWidths=[col_width] * num_cols)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dddddd')),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ]))

        self.elements.append(table)

        if caption:
            self.elements.append(Paragraph(caption, self.styles['Caption']))

        self.elements.append(Spacer(1, 0.2*inch))

    def add_damage_summary_table(
        self,
        cb_damage: Dict[str, Any],
        gov_damage: Dict[str, Any]
    ):
        """Add intervention damage summary table"""
        self.add_section("Intervention Damage Assessment")

        self.add_austrian_note(
            "Every government and central bank intervention creates deadweight loss, "
            "distorts price signals, and causes malinvestment. This table quantifies the damage."
        )

        # Central Bank damage
        self.add_subsection("Central Bank Damage")
        data = [["Damage Type", "Amount (USD)", "Description"]]
        for key, value in cb_damage.items():
            if key not in ['total', 'recommendation']:
                desc = self._get_damage_description(key, "cb")
                amount = f"${float(value):,.0f}" if isinstance(value, (Decimal, float, int)) else str(value)
                data.append([key.replace('_', ' ').title(), amount, desc])

        if len(data) > 1:
            table = Table(data, colWidths=[1.8*inch, 1.5*inch, 3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc143c')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (1, -1), 'CENTER'),
                ('ALIGN', (2, 0), (2, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dddddd')),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fff0f0')]),
            ]))
            self.elements.append(table)

        # Government damage
        self.add_subsection("Government Damage")
        data = [["Damage Type", "Amount (USD)", "Description"]]
        for key, value in gov_damage.items():
            if key not in ['total', 'recommendation']:
                desc = self._get_damage_description(key, "gov")
                amount = f"${float(value):,.0f}" if isinstance(value, (Decimal, float, int)) else str(value)
                data.append([key.replace('_', ' ').title(), amount, desc])

        if len(data) > 1:
            table = Table(data, colWidths=[1.8*inch, 1.5*inch, 3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff7f0e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (1, -1), 'CENTER'),
                ('ALIGN', (2, 0), (2, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dddddd')),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fff8f0')]),
            ]))
            self.elements.append(table)

    def _get_damage_description(self, damage_type: str, source: str) -> str:
        """Get explanation for damage type"""
        descriptions = {
            "cb": {
                "malinvestment": "Capital allocated to unsustainable projects due to artificial rates",
                "inflation_tax": "Hidden tax on savers through currency debasement",
                "purchasing_power_loss": "Real value destroyed by money printing",
                "asset_bubble": "Price distortions in financial assets",
                "boom_bust": "Economic instability from credit cycles",
            },
            "gov": {
                "deadweight_loss": "Economic value destroyed by taxation",
                "compliance_costs": "Resources wasted on bureaucracy",
                "trade_disruption": "Value destroyed by tariffs/sanctions",
                "capital_destroyed": "Physical capital destroyed by regulations",
                "spending_waste": "Inefficient government spending",
            }
        }
        return descriptions.get(source, {}).get(damage_type.lower(), "Economic distortion")

    # ===========================================
    # CHARTS AND IMAGES
    # ===========================================

    def add_chart(
        self,
        chart_bytes: bytes,
        caption: str,
        width: float = 6,
        height: Optional[float] = None
    ):
        """Add a chart image to the report"""
        # Create image from bytes
        img_buffer = io.BytesIO(chart_bytes)

        # Default height based on typical chart aspect ratio (16:9 or 4:3)
        if height is None:
            height = width * 0.5  # 2:1 aspect ratio for most charts

        img = Image(img_buffer, width=width*inch, height=height*inch)

        # Keep chart and caption together
        elements = [
            img,
            Paragraph(caption, self.styles['Caption'])
        ]
        self.elements.append(KeepTogether(elements))
        self.elements.append(Spacer(1, 0.1*inch))

    def add_dashboard(
        self,
        dashboard_bytes: bytes,
        caption: str = "Simulation Dashboard: Key Economic Indicators"
    ):
        """Add the full simulation dashboard"""
        self.add_section("Economic Dashboard")
        self.add_chart(dashboard_bytes, caption, width=6.5, height=4.5)

    # ===========================================
    # LISTS
    # ===========================================

    def add_bullet_list(self, items: List[str], title: Optional[str] = None):
        """Add a bulleted list"""
        if title:
            self.elements.append(Paragraph(title, self.styles['SubsectionHeading']))

        bullet_items = [
            ListItem(Paragraph(item, self.styles['BodyJustified']))
            for item in items
        ]
        self.elements.append(ListFlowable(
            bullet_items,
            bulletType='bullet',
            start='bulletchar',
        ))
        self.elements.append(Spacer(1, 0.1*inch))

    def add_numbered_list(self, items: List[str], title: Optional[str] = None):
        """Add a numbered list"""
        if title:
            self.elements.append(Paragraph(title, self.styles['SubsectionHeading']))

        numbered_items = [
            ListItem(Paragraph(item, self.styles['BodyJustified']))
            for item in items
        ]
        self.elements.append(ListFlowable(
            numbered_items,
            bulletType='1',
        ))
        self.elements.append(Spacer(1, 0.1*inch))

    # ===========================================
    # SPECIALIZED SECTIONS
    # ===========================================

    def add_initial_conditions_section(
        self,
        conditions: Dict[str, Any],
        country: str
    ):
        """Add section describing initial economic conditions"""
        self.add_section("Initial Economic Conditions")

        self.add_paragraph(
            f"The simulation for {country} began with the following real-world economic conditions, "
            "fetched from private market data sources (no government statistics):"
        )

        # Key metrics table
        initial_metrics = {
            'Bitcoin Price': conditions.get('btc_price', 'N/A'),
            'Gold Price': conditions.get('gold_price', 'N/A'),
            'S&P 500': conditions.get('sp500', 'N/A'),
            'VIX (Fear Index)': conditions.get('vix', 'N/A'),
            'Dollar Index (DXY)': conditions.get('dxy', 'N/A'),
            '10Y Treasury Yield': f"{conditions.get('treasury_10y', 'N/A')}%",
            'Market Sentiment': conditions.get('sentiment', 'N/A'),
            'Inflation Estimate': f"{conditions.get('inflation_estimate', 'N/A')}%",
            'Recession Probability': f"{conditions.get('recession_prob', 'N/A')}%",
        }

        self.add_metrics_table("Initial Market Conditions", initial_metrics)

        self.add_austrian_note(
            "Initial prices are derived from market data, not government statistics. "
            "Austrian economists recognize that prices contain distributed knowledge (Hayek) "
            "and government statistics systematically understate true inflation."
        )

    def add_simulation_config_section(
        self,
        config: Dict[str, Any]
    ):
        """Add section describing simulation configuration"""
        self.add_section("Simulation Configuration")

        config_metrics = {
            'Country': config.get('country', 'USA'),
            'Number of Persons': f"{config.get('num_persons', 0):,}",
            'Number of Companies': f"{config.get('num_companies', 0):,}",
            'Number of Banks': f"{config.get('num_banks', 0):,}",
            'Regime Type': config.get('regime', 'Unknown'),
            'Simulation Duration': f"{config.get('ticks', 12)} months",
            'Central Bank Intervention': f"{config.get('cb_intervention', 0.5):.0%}",
        }

        self.add_metrics_table("Simulation Parameters", config_metrics)

    def add_austrian_theory_section(self):
        """Add a section explaining the Austrian Economics framework"""
        self.add_section("Theoretical Framework: Austrian Economics")

        self.add_paragraph(
            "This simulation is built upon the principles of the Austrian School of Economics, "
            "as developed by Ludwig von Mises, Friedrich Hayek, Murray Rothbard, and Hans-Hermann Hoppe."
        )

        principles = [
            "<b>Subjective Value Theory:</b> Value is determined by individual preferences, not labor or intrinsic properties.",
            "<b>Praxeology:</b> Economic laws are derived from the logic of human action, not empirical statistics.",
            "<b>Business Cycle Theory:</b> Credit expansion by central banks causes artificial booms followed by inevitable busts.",
            "<b>Price Signals:</b> Free market prices coordinate economic activity; intervention distorts these signals.",
            "<b>Time Preference:</b> Individuals prefer present goods to future goods; savings rate reflects societal time preference.",
            "<b>Sound Money:</b> Money should emerge from the market (gold, bitcoin); fiat money enables inflation.",
        ]

        self.add_bullet_list(principles, "Core Principles")

        self.add_austrian_note(
            "Unlike mainstream (Keynesian) economics, Austrian economics recognizes that all government "
            "and central bank intervention creates economic distortions and destroys value. "
            "This simulation quantifies that damage."
        )

    def add_conclusions_section(
        self,
        freedom_index: float,
        total_damage: Decimal,
        recommendation: str
    ):
        """Add conclusions section"""
        self.add_section("Conclusions and Recommendations")

        # Freedom assessment
        if freedom_index >= 90:
            assessment = "Excellent - Near-optimal free market conditions"
            color = "green"
        elif freedom_index >= 70:
            assessment = "Good - Some intervention, but manageable"
            color = "blue"
        elif freedom_index >= 50:
            assessment = "Moderate - Significant intervention affecting economy"
            color = "orange"
        elif freedom_index >= 30:
            assessment = "Poor - Heavy intervention distorting markets"
            color = "red"
        else:
            assessment = "Critical - Extreme intervention destroying economy"
            color = "darkred"

        self.add_paragraph(
            f"<b>Freedom Index: {freedom_index:.1f}/100</b> - {assessment}"
        )

        self.add_paragraph(
            f"<b>Total Intervention Damage:</b> ${float(total_damage):,.0f}"
        )

        self.add_paragraph(f"<b>Recommendation:</b> {recommendation}")

        self.add_austrian_note(
            "The path to prosperity is through reduced intervention: lower taxes, "
            "sound money, free trade, and respect for private property. "
            "Every step toward the free market creates value; every step toward intervention destroys it."
        )

    def add_references_section(self):
        """Add references to Austrian Economics literature"""
        self.add_section("References")

        references = [
            "Mises, Ludwig von. <i>Human Action: A Treatise on Economics</i> (1949)",
            "Hayek, Friedrich A. <i>Prices and Production</i> (1931)",
            "Rothbard, Murray N. <i>Man, Economy, and State</i> (1962)",
            "Rothbard, Murray N. <i>America's Great Depression</i> (1963)",
            "Hoppe, Hans-Hermann. <i>Democracy: The God That Failed</i> (2001)",
            "Hazlitt, Henry. <i>Economics in One Lesson</i> (1946)",
            "Menger, Carl. <i>Principles of Economics</i> (1871)",
        ]

        self.add_numbered_list(references)

    # ===========================================
    # BUILD AND SAVE
    # ===========================================

    def build(self):
        """Build and save the PDF document"""
        self.doc.build(self.elements)
        return self.output_path
