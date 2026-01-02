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
            "Ludwig von Mises, Friedrich Hayek, and Murray Rothbard. Key theoretical foundations include:"
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
