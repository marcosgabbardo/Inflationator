"""Inflationator - Austrian Economics World Simulator"""

from setuptools import setup, find_packages

setup(
    name="inflationator",
    version="0.1.0",
    description="Agent-based economic simulator based on Austrian Economics",
    author="Inflationator Team",
    python_requires=">=3.11",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "mesa>=2.1.0",
        "sqlalchemy>=2.0.0",
        "pymysql>=1.1.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "httpx>=0.25.0",
        "yfinance>=0.2.30",
        "pycoingecko>=3.1.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "plotly>=5.18.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
    ],
    entry_points={
        "console_scripts": [
            "inflationator=cli.main:app",
        ],
    },
)
