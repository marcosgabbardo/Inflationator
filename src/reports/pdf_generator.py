"""
PDF Report Generator - Academic Whitepaper Style

Creates professional PDF reports in academic whitepaper format,
similar to Satoshi Nakamoto's Bitcoin whitepaper.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm, mm
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, ListFlowable, ListItem
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.pdfbase import pdfmetrics

from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from decimal import Decimal
import io


def format_currency(value: Any, decimals: int = 0) -> str:
    """Format a value as currency"""
    try:
        if isinstance(value, Decimal):
            num = float(value)
        elif isinstance(value, str):
            num = float(value.replace(',', ''))
        else:
            num = float(value)

        if abs(num) >= 1e12:
            return f"${num/1e12:,.2f}T"
        elif abs(num) >= 1e9:
            return f"${num/1e9:,.2f}B"
        elif abs(num) >= 1e6:
            return f"${num/1e6:,.2f}M"
        elif abs(num) >= 1e3:
            return f"${num/1e3:,.1f}K"
        else:
            return f"${num:,.{decimals}f}"
    except:
        return str(value)


def format_percent(value: Any, decimals: int = 1) -> str:
    """Format a value as percentage"""
    try:
        num = float(value)
        if num < 1:  # Already decimal
            return f"{num*100:.{decimals}f}%"
        return f"{num:.{decimals}f}%"
    except:
        return str(value)


def format_number(value: Any, decimals: int = 2) -> str:
    """Format a number with thousands separator"""
    try:
        num = float(value) if not isinstance(value, (int, float)) else value
        return f"{num:,.{decimals}f}"
    except:
        return str(value)


class PDFReportBuilder:
    """
    Builds academic-style PDF reports (whitepaper format).

    Design principles:
    - Clean, minimal formatting
    - Compact tables
    - Clear section numbering
    - Academic references style
    """

    def __init__(
        self,
        output_path: Path,
        title: str = "Economic Simulation Report",
        author: str = "Inflationator"
    ):
        self.output_path = output_path
        self.title = title
        self.author = author
        self.section_counter = 0

        # Document setup - tighter margins for academic style
        self.doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=2.5*cm,
            leftMargin=2.5*cm,
            topMargin=2*cm,
            bottomMargin=2*cm,
        )

        # Styles
        self.styles = getSampleStyleSheet()
        self._setup_academic_styles()

        # Content elements
        self.elements: List[Any] = []

    def _setup_academic_styles(self):
        """Setup academic/whitepaper paragraph styles"""

        # Title - centered, bold
        self.styles.add(ParagraphStyle(
            name='PaperTitle',
            parent=self.styles['Title'],
            fontSize=18,
            spaceAfter=6,
            alignment=TA_CENTER,
            textColor=colors.black,
            fontName='Times-Bold',
        ))

        # Author/subtitle
        self.styles.add(ParagraphStyle(
            name='PaperAuthor',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=4,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#333333'),
            fontName='Times-Roman',
        ))

        # Abstract title
        self.styles.add(ParagraphStyle(
            name='AbstractTitle',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceBefore=12,
            spaceAfter=4,
            alignment=TA_CENTER,
            fontName='Times-Bold',
        ))

        # Abstract text - indented, italic
        self.styles.add(ParagraphStyle(
            name='Abstract',
            parent=self.styles['Normal'],
            fontSize=9,
            alignment=TA_JUSTIFY,
            leftIndent=1.5*cm,
            rightIndent=1.5*cm,
            spaceAfter=12,
            leading=12,
            fontName='Times-Italic',
        ))

        # Section heading (numbered)
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=11,
            spaceBefore=14,
            spaceAfter=6,
            textColor=colors.black,
            fontName='Times-Bold',
        ))

        # Subsection heading
        self.styles.add(ParagraphStyle(
            name='SubsectionHeading',
            parent=self.styles['Heading3'],
            fontSize=10,
            spaceBefore=10,
            spaceAfter=4,
            textColor=colors.black,
            fontName='Times-Bold',
        ))

        # Body text - justified, Times font
        self.styles.add(ParagraphStyle(
            name='AcademicBody',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            leading=12,
            fontName='Times-Roman',
        ))

        # Note/annotation style
        self.styles.add(ParagraphStyle(
            name='Note',
            parent=self.styles['Normal'],
            fontSize=9,
            fontName='Times-Italic',
            textColor=colors.HexColor('#444444'),
            leftIndent=0.5*cm,
            spaceAfter=8,
            spaceBefore=4,
        ))

        # Table caption
        self.styles.add(ParagraphStyle(
            name='TableCaption',
            parent=self.styles['Normal'],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#333333'),
            fontName='Times-Italic',
            spaceBefore=4,
            spaceAfter=10,
        ))

        # Figure caption
        self.styles.add(ParagraphStyle(
            name='FigureCaption',
            parent=self.styles['Normal'],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#333333'),
            fontName='Times-Italic',
            spaceBefore=4,
            spaceAfter=8,
        ))

        # Reference style
        self.styles.add(ParagraphStyle(
            name='Reference',
            parent=self.styles['Normal'],
            fontSize=9,
            fontName='Times-Roman',
            leftIndent=0.5*cm,
            firstLineIndent=-0.5*cm,
            spaceAfter=3,
        ))

    # ===========================================
    # DOCUMENT STRUCTURE
    # ===========================================

    def add_title_page(
        self,
        title: str,
        subtitle: str,
        date: Optional[datetime] = None,
        author: str = "Inflationator Economic Simulator",
        email: str = "simulation@inflationator.io"
    ):
        """Add academic-style title section (not full page)"""
        date = date or datetime.now()

        # Title
        self.elements.append(Paragraph(title, self.styles['PaperTitle']))

        # Subtitle
        if subtitle:
            self.elements.append(Paragraph(subtitle, self.styles['PaperAuthor']))

        # Author and date
        self.elements.append(Spacer(1, 0.3*cm))
        self.elements.append(Paragraph(author, self.styles['PaperAuthor']))
        self.elements.append(Paragraph(email, self.styles['PaperAuthor']))
        self.elements.append(Paragraph(date.strftime("%B %d, %Y"), self.styles['PaperAuthor']))

        self.elements.append(Spacer(1, 0.5*cm))

    def add_abstract(self, text: str):
        """Add abstract section"""
        self.elements.append(Paragraph("<b>Abstract.</b>", self.styles['AbstractTitle']))
        self.elements.append(Paragraph(text, self.styles['Abstract']))
        self.elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.gray, spaceAfter=10))

    def add_section(self, title: str, content: Optional[str] = None):
        """Add a numbered section"""
        self.section_counter += 1
        section_title = f"{self.section_counter}. {title}"
        self.elements.append(Paragraph(section_title, self.styles['SectionHeading']))
        if content:
            self.elements.append(Paragraph(content, self.styles['AcademicBody']))

    def add_subsection(self, title: str, content: Optional[str] = None):
        """Add a subsection (not numbered)"""
        self.elements.append(Paragraph(title, self.styles['SubsectionHeading']))
        if content:
            self.elements.append(Paragraph(content, self.styles['AcademicBody']))

    def add_paragraph(self, text: str):
        """Add a body paragraph"""
        self.elements.append(Paragraph(text, self.styles['AcademicBody']))

    def add_note(self, text: str):
        """Add an annotation/note"""
        self.elements.append(Paragraph(text, self.styles['Note']))

    def add_spacer(self, height: float = 0.3):
        """Add vertical space (in cm)"""
        self.elements.append(Spacer(1, height*cm))

    def add_page_break(self):
        """Add a page break"""
        self.elements.append(PageBreak())

    def add_horizontal_line(self):
        """Add a thin horizontal line"""
        self.elements.append(HRFlowable(
            width="100%",
            thickness=0.5,
            color=colors.gray,
            spaceBefore=6,
            spaceAfter=6
        ))

    # ===========================================
    # TABLES - COMPACT ACADEMIC STYLE
    # ===========================================

    def add_metrics_table(
        self,
        title: str,
        metrics: Dict[str, Any],
        caption: Optional[str] = None,
        table_num: Optional[int] = None
    ):
        """Add a compact metrics table"""

        # Convert metrics to formatted table data
        data = []
        for key, value in metrics.items():
            key_formatted = key.replace('_', ' ').title()

            # Smart formatting based on key name
            if any(x in key.lower() for x in ['damage', 'gdp', 'price', 'supply', 'money', 'cost']):
                value_formatted = format_currency(value)
            elif any(x in key.lower() for x in ['rate', 'inflation', 'unemployment', 'index']):
                if isinstance(value, (int, float)) and value < 1:
                    value_formatted = format_percent(value)
                else:
                    value_formatted = format_number(value, 1)
            else:
                value_formatted = str(value) if not isinstance(value, float) else format_number(value, 2)

            data.append([key_formatted, value_formatted])

        if not data:
            return

        # Create compact table
        table = Table(data, colWidths=[7*cm, 5*cm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LINEBELOW', (0, 0), (-1, -2), 0.5, colors.HexColor('#cccccc')),
            ('LINEBELOW', (0, -1), (-1, -1), 1, colors.black),
            ('LINEABOVE', (0, 0), (-1, 0), 1, colors.black),
        ]))

        self.elements.append(table)

        if caption:
            cap_text = f"Table {table_num}: {caption}" if table_num else caption
            self.elements.append(Paragraph(cap_text, self.styles['TableCaption']))

    def add_comparison_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        caption: Optional[str] = None,
        table_num: Optional[int] = None
    ):
        """Add a compact comparison table with headers"""

        data = [headers] + rows

        # Calculate column widths
        num_cols = len(headers)
        col_width = (16*cm) / num_cols

        table = Table(data, colWidths=[col_width] * num_cols)
        table.setStyle(TableStyle([
            # Header styling
            ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f0f0')),
            ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),
            ('LINEABOVE', (0, 0), (-1, 0), 1, colors.black),

            # Body styling
            ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LINEBELOW', (0, -1), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fafafa')]),
        ]))

        self.elements.append(table)

        if caption:
            cap_text = f"Table {table_num}: {caption}" if table_num else caption
            self.elements.append(Paragraph(cap_text, self.styles['TableCaption']))

    def add_ranking_table(
        self,
        title: str,
        rankings: List[tuple],
        caption: Optional[str] = None,
        table_num: Optional[int] = None
    ):
        """Add a compact ranking table (rank, name, value)"""

        data = [["#", "Country", "Value"]]
        for i, (name, value) in enumerate(rankings, 1):
            data.append([str(i), name, value])

        table = Table(data, colWidths=[1*cm, 4*cm, 4*cm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),
            ('LINEABOVE', (0, 0), (-1, 0), 1, colors.black),
            ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.black),
        ]))

        if title:
            self.elements.append(Paragraph(f"<b>{title}</b>", self.styles['SubsectionHeading']))

        self.elements.append(table)

        if caption:
            cap_text = f"Table {table_num}: {caption}" if table_num else caption
            self.elements.append(Paragraph(cap_text, self.styles['TableCaption']))

    # ===========================================
    # CHARTS AND FIGURES
    # ===========================================

    def add_chart(
        self,
        chart_bytes: bytes,
        caption: str,
        width: float = 14,
        height: float = 7,
        figure_num: Optional[int] = None
    ):
        """Add a chart/figure with caption"""
        img_buffer = io.BytesIO(chart_bytes)

        # Convert cm to points
        img = Image(img_buffer, width=width*cm, height=height*cm)

        # Center the image
        self.elements.append(Spacer(1, 0.3*cm))
        self.elements.append(img)

        cap_text = f"Figure {figure_num}: {caption}" if figure_num else f"Figure: {caption}"
        self.elements.append(Paragraph(cap_text, self.styles['FigureCaption']))

    def add_dashboard(
        self,
        dashboard_bytes: bytes,
        caption: str = "Economic simulation dashboard showing key indicators"
    ):
        """Add the full simulation dashboard"""
        self.add_chart(dashboard_bytes, caption, width=16, height=10)

    # ===========================================
    # LISTS
    # ===========================================

    def add_bullet_list(self, items: List[str]):
        """Add a simple bullet list"""
        for item in items:
            self.elements.append(Paragraph(f"• {item}", self.styles['AcademicBody']))

    def add_numbered_list(self, items: List[str]):
        """Add a numbered list"""
        for i, item in enumerate(items, 1):
            self.elements.append(Paragraph(f"[{i}] {item}", self.styles['Reference']))

    # ===========================================
    # SPECIALIZED SECTIONS
    # ===========================================

    def add_methodology_section(self):
        """Add methodology explanation"""
        self.add_section("Methodology")

        self.add_paragraph(
            "This simulation employs agent-based modeling within an Austrian Economics framework. "
            "Individual agents (persons, companies, banks) make decisions based on subjective value theory, "
            "time preference, and price signals. The central bank and government act as exogenous "
            "intervention sources whose damage is tracked throughout the simulation."
        )

        self.add_subsection("Agent Behavior")
        self.add_paragraph(
            "Agents follow praxeological principles: they act purposefully to achieve subjective goals, "
            "demonstrate time preference (preferring present goods to future goods), and respond to "
            "price signals as information carriers. Credit expansion by banks follows fractional reserve "
            "mechanics, creating the conditions for business cycles as described by Mises and Hayek."
        )

    def add_austrian_framework_section(self):
        """Add theoretical framework section"""
        self.add_section("Theoretical Framework")

        self.add_paragraph(
            "The simulation is built on Austrian School principles, particularly the contributions of "
            "Ludwig von Mises, Friedrich Hayek, Murray Rothbard, and Hans-Hermann Hoppe. Key theoretical foundations include:"
        )

        points = [
            "<b>Subjective Value Theory:</b> All value is determined by individual preferences, not intrinsic properties or labor inputs.",
            "<b>Austrian Business Cycle Theory:</b> Credit expansion artificially lowers interest rates below the natural rate, leading to malinvestment in higher-order capital goods.",
            "<b>Economic Calculation:</b> Prices serve as essential signals for resource allocation; intervention distorts these signals.",
            "<b>Time Preference:</b> The ratio at which individuals value present goods over future goods determines the natural interest rate.",
            "<b>Spontaneous Order:</b> Complex economic coordination emerges from individual actions without central planning."
        ]

        for point in points:
            self.add_paragraph(point)

    def add_hayek_section(self, inflation_rate: float, price_distortion: float = 0):
        """Add section on Hayek's knowledge problem and price signals"""
        self.add_section("Knowledge and Price Signals (Hayek)")

        self.add_paragraph(
            "Friedrich Hayek's seminal contribution to economics was demonstrating that prices serve as "
            "information carriers in a complex economy. No central planner can possess the dispersed, "
            "tacit knowledge held by millions of individuals. Prices aggregate this knowledge, allowing "
            "coordination without central direction."
        )

        self.add_paragraph(
            f"In this simulation, inflation of {format_percent(inflation_rate)} represents price signal "
            "distortion. When central banks expand the money supply, relative prices no longer accurately "
            "reflect true scarcity. Entrepreneurs receive false signals, leading to systematic errors in "
            "resource allocation that Hayek called 'malinvestment.'"
        )

        self.add_note(
            '"The curious task of economics is to demonstrate to men how little they really know about '
            'what they imagine they can design." - F.A. Hayek, The Fatal Conceit (1988)'
        )

    def add_rothbard_section(self, cb_damage: float, money_supply_growth: float = 0):
        """Add section on Rothbard's critique of central banking"""
        self.add_section("Central Bank Damage (Rothbard)")

        self.add_paragraph(
            "Murray Rothbard argued that central banking is inherently inflationary and destructive. "
            "Unlike market-based money (gold, Bitcoin), fiat currency allows unlimited expansion. "
            "The central bank acts as a legalized counterfeiter, transferring wealth from savers to "
            "debtors and first-receivers of new money (Cantillon Effect)."
        )

        self.add_paragraph(
            f"The simulation tracked {format_currency(cb_damage)} in central bank damage. This represents "
            "the wealth destruction caused by: (1) purchasing power loss for holders of the currency, "
            "(2) malinvestment triggered by artificial credit expansion, (3) business cycle boom-bust "
            "volatility, and (4) wealth redistribution to politically-connected entities."
        )

        self.add_note(
            '"The government can, for a time, simply print money to finance its deficits. But the process '
            'of money printing leads to rising prices and distorts production." - Murray Rothbard, '
            'What Has Government Done to Our Money? (1963)'
        )

    def add_hoppe_section(self, regime_type: str, freedom_index: float, gov_damage: float):
        """Add section on Hoppe's analysis of democracy and time preference"""
        self.add_section("Regime Analysis (Hoppe)")

        self.add_paragraph(
            "Hans-Hermann Hoppe's work 'Democracy: The God That Failed' provides a framework for "
            "understanding how different political regimes affect economic outcomes. Hoppe argues that "
            "democracy systematically promotes high time preference behavior and economic destruction "
            "compared to private property order or even monarchy."
        )

        self.add_subsection("Time Preference and Regime Type")
        regime_analysis = {
            'totalitarian': "Totalitarian regimes exhibit extremely high time preference - rulers extract "
                           "maximum value in the short term with no regard for long-term capital preservation. "
                           "This explains the economic devastation in communist and fascist states.",
            'democracy_socialist': "Democratic socialist regimes combine democratic short-termism with "
                                  "socialist economic intervention. Politicians maximize vote-buying today "
                                  "at the expense of future generations through debt and monetary expansion.",
            'democracy_liberal': "Liberal democracies temper intervention with some respect for property "
                                "rights, but still suffer from electoral incentives that favor present "
                                "consumption over capital accumulation.",
            'monarchy': "Monarchies, as Hoppe notes, tend toward lower time preference because the king "
                       "'owns' the country and has incentive to preserve its capital value for heirs. "
                       "This doesn't justify monarchy, but explains its relative economic stability.",
            'minarchy': "Minimal states restrict intervention to essential services, allowing most "
                       "economic activity to be governed by voluntary exchange. Time preference is "
                       "determined by market rather than political factors.",
            'ancap': "Anarcho-capitalism represents the theoretical optimum - pure private property order "
                    "with no systematic intervention. Time preference is entirely market-determined."
        }

        regime_key = regime_type.lower().replace(' ', '_')
        if regime_key in regime_analysis:
            self.add_paragraph(regime_analysis[regime_key])
        else:
            self.add_paragraph(
                f"The {regime_type} regime shows a freedom index of {freedom_index:.1f}/100 and government "
                f"damage of {format_currency(gov_damage)}. Higher freedom correlates with better economic outcomes."
            )

        self.add_note(
            '"Under democratic conditions, the government apparatus tends to grow relative to the economy... '
            'The democratic system inexorably tends toward fiscal irresponsibility and exploitation." '
            '- Hans-Hermann Hoppe, Democracy: The God That Failed (2001)'
        )

    def add_mises_section(self, business_cycle_phase: str, rate_distortion: float = 0):
        """Add section on Mises' theory of money and credit"""
        self.add_section("Business Cycle Theory (Mises)")

        self.add_paragraph(
            "Ludwig von Mises developed the Austrian Business Cycle Theory, explaining how credit "
            "expansion without prior savings creates unsustainable booms that must end in busts. "
            "When banks create credit 'out of thin air,' interest rates fall below the natural rate, "
            "signaling to entrepreneurs that more savings exist than actually do."
        )

        phase_analysis = {
            'boom': "The economy is currently in the BOOM phase. Credit expansion has artificially lowered "
                   "interest rates, encouraging investment in long-term capital projects. These projects "
                   "appear profitable at current rates but will prove unprofitable when rates normalize.",
            'crisis': "The economy is experiencing CRISIS. The boom's malinvestments are being revealed as "
                     "unprofitable. Resources must be reallocated from failed projects to viable ones. "
                     "This painful process is necessary to correct previous distortions.",
            'recession': "The RECESSION phase represents the economy purging malinvestments. Unemployment "
                        "rises as resources shift from failed boom-era projects. Government attempts to "
                        "'stimulate' the economy will only delay necessary adjustments.",
            'recovery': "The economy is in RECOVERY. Resources are being reallocated toward sustainable "
                       "production. If the central bank resists the temptation to re-inflate, genuine "
                       "prosperity can emerge based on real savings rather than credit illusion."
        }

        phase_key = business_cycle_phase.lower()
        if phase_key in phase_analysis:
            self.add_paragraph(phase_analysis[phase_key])
        else:
            self.add_paragraph(
                f"The current business cycle phase is '{business_cycle_phase}' with interest rate "
                f"distortion of {format_percent(rate_distortion)}. This distortion measures how far "
                "central bank policy has pushed rates from their natural market level."
            )

        self.add_note(
            '"There is no means of avoiding the final collapse of a boom brought about by credit expansion. '
            'The alternative is only whether the crisis should come sooner as the result of voluntary abandonment '
            'of further credit expansion, or later as a final and total catastrophe of the currency system involved." '
            '- Ludwig von Mises, Human Action (1949)'
        )

    def add_bitcoin_analysis_section(self, btc_price: float, btc_change: float, dollar_debasement: float = 0):
        """Add section analyzing Bitcoin as sound money"""
        self.add_section("Bitcoin as Sound Money")

        self.add_paragraph(
            "Bitcoin represents the first successful implementation of a decentralized, "
            "algorithmically-scarce digital money. With a fixed supply of 21 million coins, "
            "Bitcoin cannot be inflated by any central authority. This makes it the closest "
            "approximation to Rothbard's ideal of market-chosen money since the gold standard."
        )

        self.add_paragraph(
            f"In this simulation, Bitcoin price moved to {format_currency(btc_price)} "
            f"({format_percent(btc_change)} change). Bitcoin's price in fiat currency is not merely "
            "speculation - it represents the market's assessment of fiat purchasing power loss. "
            "When central banks expand money supply, Bitcoin's price in that currency tends to rise."
        )

        self.add_subsection("Austrian Interpretation")
        self.add_paragraph(
            "From an Austrian perspective, Bitcoin adoption accelerates when: (1) inflation erodes "
            "fiat purchasing power, (2) capital controls restrict financial freedom, (3) monetary "
            "policy becomes unpredictable, and (4) trust in central banking institutions declines. "
            "The simulation tracks these conditions and their effect on Bitcoin demand."
        )

        self.add_note(
            '"Bitcoin is money without a state, without a central bank, without a trusted third party. '
            'It is the spontaneous emergence of a new monetary order." - Saifedean Ammous, The Bitcoin Standard (2018)'
        )

    def add_conclusions_section(
        self,
        freedom_index: float,
        total_damage: Any,
        best_country: Optional[str] = None,
        worst_country: Optional[str] = None
    ):
        """Add conclusions section"""
        self.add_section("Conclusions")

        if best_country and worst_country:
            self.add_paragraph(
                f"The simulation results demonstrate significant variance in economic outcomes based on "
                f"regime type and intervention levels. {best_country} achieved the highest economic freedom "
                f"index ({freedom_index:.1f}), while {worst_country} showed the lowest freedom levels. "
                f"Total intervention damage across all countries reached {format_currency(total_damage)}."
            )
        else:
            self.add_paragraph(
                f"The simulation achieved a freedom index of {freedom_index:.1f}/100, with total "
                f"intervention damage of {format_currency(total_damage)}. These results align with "
                f"Austrian predictions: intervention creates deadweight loss and distorts price signals."
            )

        self.add_paragraph(
            "The data consistently supports the Austrian thesis that economic freedom correlates "
            "positively with prosperity indicators, while government and central bank intervention "
            "correlates with economic destruction and malinvestment."
        )

    def add_references_section(self):
        """Add academic references"""
        self.add_section("References")

        references = [
            "Mises, L. v. (1949). <i>Human Action: A Treatise on Economics</i>. Yale University Press.",
            "Hayek, F. A. (1931). <i>Prices and Production</i>. Routledge.",
            "Rothbard, M. N. (1962). <i>Man, Economy, and State</i>. Van Nostrand.",
            "Rothbard, M. N. (1963). <i>America's Great Depression</i>. Van Nostrand.",
            "Hoppe, H. H. (2001). <i>Democracy: The God That Failed</i>. Transaction Publishers.",
            "Hazlitt, H. (1946). <i>Economics in One Lesson</i>. Harper & Brothers.",
            "Menger, C. (1871). <i>Grundsätze der Volkswirtschaftslehre</i>. Wilhelm Braumüller.",
        ]

        self.add_numbered_list(references)

    # ===========================================
    # BUILD AND SAVE
    # ===========================================

    def build(self):
        """Build and save the PDF document"""
        self.doc.build(self.elements)
        return self.output_path
