"""
Report Generation Module

Generates scientific PDF reports for economic simulations.
Includes charts, analysis, and Austrian Economics explanations.
"""

from .generator import ReportGenerator, ReportConfig
from .charts import ChartGenerator
from .pdf_generator import PDFReportBuilder

__all__ = [
    "ReportGenerator",
    "ReportConfig",
    "ChartGenerator",
    "PDFReportBuilder",
]
