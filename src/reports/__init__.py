"""
Report Generation Module

Generates scientific PDF reports for economic simulations.
Includes charts, analysis, and Austrian Economics explanations.
"""

from .charts import ChartGenerator
from .generator import ReportConfig, ReportGenerator
from .pdf_generator import PDFReportBuilder

__all__ = [
    "ChartGenerator",
    "PDFReportBuilder",
    "ReportConfig",
    "ReportGenerator",
]
