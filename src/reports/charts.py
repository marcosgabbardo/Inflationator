"""
Chart Generation Module - Publication Quality

Creates high-quality matplotlib charts suitable for academic publications.
Style inspired by The Economist, Financial Times, and academic journals.
"""

import io
from decimal import Decimal
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def setup_publication_style():
    """Configure matplotlib for publication-quality output"""
    plt.rcParams.update(
        {
            # Figure
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "figure.dpi": 150,
            "figure.figsize": (10, 5),
            # Fonts - use serif for academic look
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 11,
            # Axes
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.labelweight": "normal",
            "axes.labelpad": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(
                color=[
                    "#1a476f",  # Dark blue
                    "#e37222",  # Orange
                    "#2f7e44",  # Green
                    "#c42c2c",  # Red
                    "#6e4c9e",  # Purple
                    "#d4a100",  # Gold
                    "#17becf",  # Cyan
                    "#e377c2",  # Pink
                ]
            ),
            # Grid
            "axes.grid": True,
            "axes.grid.which": "major",
            "grid.color": "#e0e0e0",
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            # Ticks
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            # Legend
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.loc": "best",
            # Lines
            "lines.linewidth": 2,
            "lines.markersize": 6,
            # Savefig
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


# Apply style on import
setup_publication_style()


class ChartGenerator:
    """
    Generates publication-quality charts for economic reports.

    Design principles:
    - Clean, minimal design (no chartjunk)
    - Clear labels and titles
    - Consistent color palette
    - Proper typography
    """

    # Professional color palette
    COLORS = {
        "primary": "#1a476f",  # Dark blue (main data)
        "secondary": "#e37222",  # Orange (comparison)
        "positive": "#2f7e44",  # Green (gains)
        "negative": "#c42c2c",  # Red (losses)
        "highlight": "#d4a100",  # Gold (highlights)
        "neutral": "#666666",  # Gray
        "light": "#e0e0e0",  # Light gray
        "bitcoin": "#f7931a",  # Bitcoin orange
        "gold": "#cfb53b",  # Metallic gold
    }

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path("reports/charts")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chart_counter = 0

    def _to_float_list(self, data: list[Any]) -> list[float]:
        """Convert list of Decimals/Any to floats"""
        result = []
        for v in data:
            if isinstance(v, (Decimal, int, float)):
                result.append(float(v))
            else:
                result.append(0.0)
        return result

    def _format_currency(self, x, pos):
        """Format axis values as currency"""
        if abs(x) >= 1e9:
            return f"${x / 1e9:.0f}B"
        elif abs(x) >= 1e6:
            return f"${x / 1e6:.0f}M"
        elif abs(x) >= 1e3:
            return f"${x / 1e3:.0f}K"
        else:
            return f"${x:,.0f}"

    def _format_currency_full(self, x, pos):
        """Format axis values as currency with more precision"""
        if abs(x) >= 1e12:
            return f"${x / 1e12:.1f}T"
        elif abs(x) >= 1e9:
            return f"${x / 1e9:.1f}B"
        elif abs(x) >= 1e6:
            return f"${x / 1e6:.1f}M"
        elif abs(x) >= 1e3:
            return f"${x / 1e3:.1f}K"
        else:
            return f"${x:,.0f}"

    def _save_chart(self, fig: Figure, filename: str) -> bytes:
        """Save chart and return bytes"""
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        buf.seek(0)
        chart_bytes = buf.getvalue()

        # Also save to file
        filepath = self.output_dir / filename
        fig.savefig(
            filepath,
            format="png",
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )

        plt.close(fig)
        self.chart_counter += 1

        return chart_bytes

    def _add_source_annotation(
        self, ax: Axes, source: str = "Inflationator Simulation"
    ):
        """Add source annotation to chart"""
        ax.annotate(
            f"Source: {source}",
            xy=(0, -0.12),
            xycoords="axes fraction",
            fontsize=8,
            color=self.COLORS["neutral"],
            style="italic",
        )

    # ===========================================
    # PRICE CHARTS
    # ===========================================

    def create_bitcoin_price_chart(
        self, prices: list[Decimal], title: str = "Bitcoin Price"
    ) -> bytes:
        """Create professional Bitcoin price chart"""
        fig, ax = plt.subplots(figsize=(10, 5))

        prices_float = self._to_float_list(prices)
        months = range(len(prices_float))

        # Main line
        ax.plot(
            months,
            prices_float,
            color=self.COLORS["bitcoin"],
            linewidth=2.5,
            label="BTC/USD",
            zorder=3,
        )

        # Fill under line (gradient effect)
        ax.fill_between(
            months, prices_float, alpha=0.15, color=self.COLORS["bitcoin"], zorder=2
        )

        # Add start and end annotations
        if len(prices_float) > 1:
            ax.annotate(
                f"${prices_float[0]:,.0f}",
                xy=(0, prices_float[0]),
                xytext=(-30, 10),
                textcoords="offset points",
                fontsize=9,
                color=self.COLORS["neutral"],
            )
            ax.annotate(
                f"${prices_float[-1]:,.0f}",
                xy=(len(prices_float) - 1, prices_float[-1]),
                xytext=(5, 10),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color=self.COLORS["bitcoin"],
            )

        ax.set_xlabel("Month")
        ax.set_ylabel("Price (USD)")
        ax.set_title(title)

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(self._format_currency))

        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
        self._add_source_annotation(ax)

        plt.tight_layout()
        return self._save_chart(fig, f"bitcoin_price_{self.chart_counter}.png")

    def create_gold_price_chart(
        self, prices: list[Decimal], title: str = "Gold Price"
    ) -> bytes:
        """Create professional Gold price chart"""
        fig, ax = plt.subplots(figsize=(10, 5))

        prices_float = self._to_float_list(prices)
        months = range(len(prices_float))

        ax.plot(
            months,
            prices_float,
            color=self.COLORS["gold"],
            linewidth=2.5,
            label="Gold/USD (oz)",
            zorder=3,
        )
        ax.fill_between(
            months, prices_float, alpha=0.15, color=self.COLORS["gold"], zorder=2
        )

        ax.set_xlabel("Month")
        ax.set_ylabel("Price (USD per oz)")
        ax.set_title(title)

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(self._format_currency))
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
        self._add_source_annotation(ax)

        plt.tight_layout()
        return self._save_chart(fig, f"gold_price_{self.chart_counter}.png")

    def create_price_comparison_chart(
        self,
        btc_prices: list[Decimal],
        gold_prices: list[Decimal],
        title: str = "Asset Performance (Indexed to 100)",
    ) -> bytes:
        """Create comparison chart with normalized values"""
        fig, ax = plt.subplots(figsize=(10, 5))

        btc = self._to_float_list(btc_prices)
        gold = self._to_float_list(gold_prices)

        # Normalize to 100
        btc_norm = [100 * p / btc[0] if btc[0] > 0 else 100 for p in btc]
        gold_norm = [100 * p / gold[0] if gold[0] > 0 else 100 for p in gold]

        months = range(len(btc_norm))

        ax.plot(
            months,
            btc_norm,
            color=self.COLORS["bitcoin"],
            linewidth=2.5,
            label="Bitcoin",
            zorder=3,
        )
        ax.plot(
            months,
            gold_norm,
            color=self.COLORS["gold"],
            linewidth=2.5,
            label="Gold",
            zorder=3,
        )
        ax.axhline(
            y=100,
            color=self.COLORS["neutral"],
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Starting Value",
            zorder=1,
        )

        ax.set_xlabel("Month")
        ax.set_ylabel("Index (Start = 100)")
        ax.set_title(title)

        ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="none")
        self._add_source_annotation(ax)

        plt.tight_layout()
        return self._save_chart(fig, f"price_comparison_{self.chart_counter}.png")

    # ===========================================
    # MONETARY CHARTS
    # ===========================================

    def create_inflation_chart(
        self, inflation_rates: list[float], title: str = "Inflation Rate (Annualized)"
    ) -> bytes:
        """Create professional inflation bar chart"""
        fig, ax = plt.subplots(figsize=(10, 5))

        months = range(len(inflation_rates))
        rates_pct = [r * 100 if r < 1 else r for r in inflation_rates]

        # Color bars by value
        colors = []
        for r in rates_pct:
            if r > 10:
                colors.append(self.COLORS["negative"])
            elif r > 5:
                colors.append(self.COLORS["secondary"])
            elif r > 2:
                colors.append(self.COLORS["highlight"])
            else:
                colors.append(self.COLORS["positive"])

        ax.bar(
            months,
            rates_pct,
            color=colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

        # Target line
        ax.axhline(
            y=2,
            color=self.COLORS["neutral"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label="2% Target",
        )

        ax.set_xlabel("Month")
        ax.set_ylabel("Inflation Rate (%)")
        ax.set_title(title)

        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
        self._add_source_annotation(ax)

        plt.tight_layout()
        return self._save_chart(fig, f"inflation_{self.chart_counter}.png")

    def create_money_supply_chart(
        self,
        money_supply: list[Decimal],
        credit_expansion: list[Decimal],
        title: str = "Money Supply & Credit",
    ) -> bytes:
        """Create dual-axis money supply chart"""
        fig, ax1 = plt.subplots(figsize=(10, 5))

        money = self._to_float_list(money_supply)
        credit = self._to_float_list(credit_expansion)
        months = range(len(money))

        # Money supply on left axis
        color1 = self.COLORS["primary"]
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Base Money", color=color1)
        line1 = ax1.plot(
            months, money, color=color1, linewidth=2.5, label="Base Money", zorder=3
        )
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(self._format_currency_full))
        ax1.fill_between(months, money, alpha=0.1, color=color1)

        # Credit on right axis
        ax2 = ax1.twinx()
        color2 = self.COLORS["secondary"]
        ax2.set_ylabel("Credit Expansion", color=color2)
        line2 = ax2.plot(
            months,
            credit,
            color=color2,
            linewidth=2.5,
            linestyle="--",
            label="Credit",
            zorder=3,
        )
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(self._format_currency_full))

        ax1.set_title(title)

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(
            lines,
            labels,
            loc="upper left",
            frameon=True,
            facecolor="white",
            edgecolor="none",
        )

        self._add_source_annotation(ax1)
        plt.tight_layout()
        return self._save_chart(fig, f"money_supply_{self.chart_counter}.png")

    # ===========================================
    # LABOR CHARTS
    # ===========================================

    def create_unemployment_chart(
        self, unemployment_rates: list[float], title: str = "Unemployment Rate"
    ) -> bytes:
        """Create unemployment rate area chart"""
        fig, ax = plt.subplots(figsize=(10, 5))

        months = range(len(unemployment_rates))
        rates_pct = [r * 100 if r < 1 else r for r in unemployment_rates]

        ax.fill_between(
            months, rates_pct, alpha=0.3, color=self.COLORS["negative"], zorder=2
        )
        ax.plot(
            months, rates_pct, color=self.COLORS["negative"], linewidth=2.5, zorder=3
        )

        # Add markers at start and end
        if rates_pct:
            ax.scatter(
                [0], [rates_pct[0]], color=self.COLORS["negative"], s=50, zorder=4
            )
            ax.scatter(
                [len(rates_pct) - 1],
                [rates_pct[-1]],
                color=self.COLORS["negative"],
                s=50,
                zorder=4,
            )

        ax.set_xlabel("Month")
        ax.set_ylabel("Unemployment Rate (%)")
        ax.set_title(title)
        ax.set_ylim(bottom=0)

        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
        self._add_source_annotation(ax)

        plt.tight_layout()
        return self._save_chart(fig, f"unemployment_{self.chart_counter}.png")

    # ===========================================
    # DAMAGE CHARTS
    # ===========================================

    def create_damage_chart(
        self,
        cb_damage: list[Decimal],
        gov_damage: list[Decimal],
        title: str = "Cumulative Intervention Damage",
    ) -> bytes:
        """Create stacked area chart of intervention damage"""
        fig, ax = plt.subplots(figsize=(10, 5))

        cb = self._to_float_list(cb_damage)
        gov = self._to_float_list(gov_damage)
        months = range(len(cb))

        ax.stackplot(
            months,
            cb,
            gov,
            labels=["Central Bank", "Government"],
            colors=[self.COLORS["negative"], self.COLORS["secondary"]],
            alpha=0.8,
        )

        ax.set_xlabel("Month")
        ax.set_ylabel("Cumulative Damage")
        ax.set_title(title)

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(self._format_currency_full))
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
        self._add_source_annotation(ax)

        plt.tight_layout()
        return self._save_chart(fig, f"damage_{self.chart_counter}.png")

    def create_freedom_index_chart(
        self, freedom_indices: list[float], title: str = "Economic Freedom Index"
    ) -> bytes:
        """Create freedom index horizontal bar chart"""
        fig, ax = plt.subplots(figsize=(10, 5))

        months = range(len(freedom_indices))

        # Color by value
        colors = []
        for f in freedom_indices:
            if f >= 80:
                colors.append(self.COLORS["positive"])
            elif f >= 60:
                colors.append("#7cb342")  # Light green
            elif f >= 40:
                colors.append(self.COLORS["highlight"])
            elif f >= 20:
                colors.append(self.COLORS["secondary"])
            else:
                colors.append(self.COLORS["negative"])

        ax.bar(
            months,
            freedom_indices,
            color=colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

        # Reference lines
        ax.axhline(
            y=80,
            color=self.COLORS["positive"],
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Free (80+)",
        )
        ax.axhline(
            y=50,
            color=self.COLORS["neutral"],
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Moderate (50)",
        )

        ax.set_xlabel("Month")
        ax.set_ylabel("Freedom Index")
        ax.set_title(title)
        ax.set_ylim(0, 100)

        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
        self._add_source_annotation(ax)

        plt.tight_layout()
        return self._save_chart(fig, f"freedom_{self.chart_counter}.png")

    # ===========================================
    # MULTI-COUNTRY CHARTS
    # ===========================================

    # Extended color palette for many countries
    COUNTRY_COLORS = [
        "#1a476f",
        "#e37222",
        "#2f7e44",
        "#c42c2c",
        "#6e4c9e",
        "#d4a100",
        "#17becf",
        "#e377c2",
        "#8c564b",
        "#7f7f7f",
        "#bcbd22",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#1f77b4",
        "#98df8a",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
    ]

    def create_country_comparison_chart(
        self, country_data: dict[str, dict[str, Any]], metric: str, title: str
    ) -> bytes:
        """Create horizontal bar chart comparing countries"""
        fig, ax = plt.subplots(figsize=(10, 6))

        countries = list(country_data.keys())
        values = [country_data[c].get(metric, 0) for c in countries]

        # Sort by value
        sorted_data = sorted(
            zip(countries, values, strict=False), key=lambda x: x[1], reverse=True
        )
        countries, values = zip(*sorted_data, strict=False) if sorted_data else ([], [])

        # Color gradient
        colors = [
            self.COLORS["primary"] if i % 2 == 0 else self.COLORS["secondary"]
            for i in range(len(values))
        ]

        y_pos = np.arange(len(countries))
        bars = ax.barh(
            y_pos, values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(countries)
        ax.invert_yaxis()
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_title(title)

        # Add value labels
        for bar, val in zip(bars, values, strict=False):
            ax.text(
                bar.get_width() + max(values) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}",
                va="center",
                fontsize=9,
            )

        self._add_source_annotation(ax)
        plt.tight_layout()
        return self._save_chart(fig, f"country_comparison_{self.chart_counter}.png")

    def create_grouped_bar_chart(
        self,
        country_data: dict[str, dict[str, Any]],
        metrics: list[str],
        title: str = "Multi-Metric Country Comparison",
    ) -> bytes:
        """Create grouped bar chart comparing multiple metrics across countries"""
        fig, ax = plt.subplots(figsize=(12, 6))

        countries = list(country_data.keys())
        n_countries = len(countries)
        n_metrics = len(metrics)

        x = np.arange(n_countries)
        width = 0.8 / n_metrics

        for i, metric in enumerate(metrics):
            values = [country_data[c].get(metric, 0) for c in countries]
            # Normalize percentages
            if any(v < 1 for v in values if isinstance(v, (int, float))):
                values = [v * 100 if v < 1 else v for v in values]

            offset = (i - n_metrics / 2 + 0.5) * width
            ax.bar(
                x + offset,
                values,
                width,
                label=metric.replace("_", " ").title(),
                color=self.COUNTRY_COLORS[i % len(self.COUNTRY_COLORS)],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_xlabel("Country")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(countries, rotation=45, ha="right")
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

        self._add_source_annotation(ax)
        plt.tight_layout()
        return self._save_chart(fig, f"grouped_bar_{self.chart_counter}.png")

    def create_radar_chart(
        self,
        country_data: dict[str, dict[str, float]],
        metrics: list[str],
        title: str = "Country Economic Profile",
    ) -> bytes:
        """Create radar/spider chart for country comparison"""
        from math import pi

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})

        # Number of metrics
        N = len(metrics)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Complete the loop

        # Format metric labels
        metric_labels = [m.replace("_", " ").title() for m in metrics]

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, size=10)

        # Plot each country
        for i, (country, data) in enumerate(country_data.items()):
            values = [data.get(m, 0) for m in metrics]
            # Normalize values to 0-100 scale
            max_vals = {
                m: max(country_data[c].get(m, 1) for c in country_data) for m in metrics
            }
            values_norm = [
                v / max_vals[m] * 100 if max_vals[m] > 0 else 0
                for v, m in zip(values, metrics, strict=False)
            ]
            values_norm += values_norm[:1]

            color = self.COUNTRY_COLORS[i % len(self.COUNTRY_COLORS)]
            ax.plot(
                angles,
                values_norm,
                linewidth=2,
                linestyle="solid",
                label=country,
                color=color,
            )
            ax.fill(angles, values_norm, alpha=0.15, color=color)

        ax.set_ylim(0, 100)
        ax.set_title(title, size=14, fontweight="bold", y=1.08)
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.3, 1.0),
            frameon=True,
            facecolor="white",
            edgecolor="none",
        )

        plt.tight_layout()
        return self._save_chart(fig, f"radar_{self.chart_counter}.png")

    def create_heatmap(
        self,
        matrix: dict[str, dict[str, float]],
        title: str = "Relationship Matrix",
        cmap: str = "RdYlGn_r",
        value_format: str = ".1f",
        show_values: bool = True,
    ) -> bytes:
        """Create heatmap for relationship/correlation matrix"""
        fig, ax = plt.subplots(figsize=(10, 8))

        countries = list(matrix.keys())
        n = len(countries)

        # Build matrix
        data = np.zeros((n, n))
        for i, c1 in enumerate(countries):
            for j, c2 in enumerate(countries):
                data[i, j] = matrix.get(c1, {}).get(c2, 0)

        # Create heatmap
        im = ax.imshow(
            data, cmap=cmap, aspect="auto", vmin=0, vmax=max(0.2, data.max())
        )

        # Ticks
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(countries, rotation=45, ha="right")
        ax.set_yticklabels(countries)

        # Add values in cells
        if show_values:
            for i in range(n):
                for j in range(n):
                    val = data[i, j]
                    if val > 0:
                        text_color = "white" if val > data.max() * 0.5 else "black"
                        ax.text(
                            j,
                            i,
                            f"{val * 100:{value_format}}%",
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=9,
                            fontweight="bold",
                        )

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Probability", rotation=270, labelpad=15)

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        self._add_source_annotation(ax)

        plt.tight_layout()
        return self._save_chart(fig, f"heatmap_{self.chart_counter}.png")

    def create_war_probability_heatmap(
        self,
        war_risks: list[dict[str, Any]],
        countries: list[str],
        title: str = "War Probability Matrix",
    ) -> bytes:
        """Create heatmap specifically for war probabilities between countries"""
        # Build matrix from war risks
        matrix = {c: dict.fromkeys(countries, 0.0) for c in countries}

        for risk in war_risks:
            c1, c2 = risk["countries"]
            prob = risk["probability"]
            if c1 in matrix and c2 in matrix[c1]:
                matrix[c1][c2] = prob
                matrix[c2][c1] = prob  # Symmetric

        return self.create_heatmap(matrix, title=title, cmap="YlOrRd")

    def create_scatter_correlation(
        self,
        country_data: dict[str, dict[str, float]],
        x_metric: str,
        y_metric: str,
        size_metric: str | None = None,
        title: str = "Economic Correlation",
    ) -> bytes:
        """Create scatter plot showing correlation between two metrics"""
        fig, ax = plt.subplots(figsize=(10, 8))

        countries = list(country_data.keys())
        x_values = [country_data[c].get(x_metric, 0) for c in countries]
        y_values = [country_data[c].get(y_metric, 0) for c in countries]

        # Normalize percentages
        x_values = [v * 100 if v < 1 else v for v in x_values]
        y_values = [v * 100 if v < 1 else v for v in y_values]

        # Size based on third metric or uniform
        if size_metric:
            sizes = [country_data[c].get(size_metric, 50) for c in countries]
            # Normalize sizes
            max_size = max(sizes) if sizes else 1
            sizes = [max(100, s / max_size * 1000) for s in sizes]
        else:
            sizes = [200] * len(countries)

        # Color by regime or index
        colors = [
            self.COUNTRY_COLORS[i % len(self.COUNTRY_COLORS)]
            for i in range(len(countries))
        ]

        ax.scatter(
            x_values,
            y_values,
            s=sizes,
            c=colors,
            alpha=0.7,
            edgecolors="white",
            linewidth=1.5,
        )

        # Add country labels
        for i, country in enumerate(countries):
            ax.annotate(
                country,
                (x_values[i], y_values[i]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=10,
                fontweight="bold",
            )

        # Add trend line
        if len(x_values) > 2:
            z = np.polyfit(x_values, y_values, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(x_values), max(x_values), 100)
            ax.plot(
                x_line,
                p(x_line),
                "--",
                color=self.COLORS["neutral"],
                alpha=0.5,
                linewidth=1.5,
                label="Trend",
            )

        ax.set_xlabel(x_metric.replace("_", " ").title())
        ax.set_ylabel(y_metric.replace("_", " ").title())
        ax.set_title(title, fontsize=14, fontweight="bold")

        self._add_source_annotation(ax)
        plt.tight_layout()
        return self._save_chart(fig, f"scatter_{self.chart_counter}.png")

    def create_bubble_chart(
        self,
        country_data: dict[str, dict[str, float]],
        x_metric: str = "freedom_index",
        y_metric: str = "inflation",
        size_metric: str = "gdp",
        title: str = "Economic Bubble Chart",
    ) -> bytes:
        """Create bubble chart: x=freedom, y=inflation, size=GDP"""
        fig, ax = plt.subplots(figsize=(12, 8))

        countries = list(country_data.keys())

        x_values = [country_data[c].get(x_metric, 50) for c in countries]
        y_values = [country_data[c].get(y_metric, 0) for c in countries]
        sizes = [float(country_data[c].get(size_metric, 1)) for c in countries]

        # Normalize y if percentage
        y_values = [v * 100 if v < 1 else v for v in y_values]

        # Normalize sizes for display (100-2000 range)
        max_size = max(sizes) if sizes else 1
        sizes_norm = [max(100, s / max_size * 2000) for s in sizes]

        # Color by freedom index (green=high, red=low)
        colors = []
        for x in x_values:
            if x >= 70:
                colors.append(self.COLORS["positive"])
            elif x >= 50:
                colors.append(self.COLORS["highlight"])
            elif x >= 30:
                colors.append(self.COLORS["secondary"])
            else:
                colors.append(self.COLORS["negative"])

        ax.scatter(
            x_values,
            y_values,
            s=sizes_norm,
            c=colors,
            alpha=0.6,
            edgecolors="white",
            linewidth=2,
        )

        # Add country labels
        for i, country in enumerate(countries):
            ax.annotate(
                country,
                (x_values[i], y_values[i]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=11,
                fontweight="bold",
            )

        # Add quadrant lines
        ax.axvline(
            x=50, color=self.COLORS["light"], linestyle="--", linewidth=1, alpha=0.7
        )
        ax.axhline(
            y=10, color=self.COLORS["light"], linestyle="--", linewidth=1, alpha=0.7
        )

        # Quadrant labels
        ax.text(
            75,
            2,
            "Free & Stable\n(Ideal)",
            ha="center",
            va="center",
            fontsize=10,
            color=self.COLORS["positive"],
            alpha=0.7,
        )
        ax.text(
            25,
            2,
            "Unfree & Stable",
            ha="center",
            va="center",
            fontsize=10,
            color=self.COLORS["neutral"],
            alpha=0.7,
        )
        ax.text(
            75,
            25,
            "Free & Inflationary",
            ha="center",
            va="center",
            fontsize=10,
            color=self.COLORS["highlight"],
            alpha=0.7,
        )
        ax.text(
            25,
            25,
            "Unfree & Inflationary\n(Worst)",
            ha="center",
            va="center",
            fontsize=10,
            color=self.COLORS["negative"],
            alpha=0.7,
        )

        ax.set_xlabel(f"{x_metric.replace('_', ' ').title()}", fontsize=12)
        ax.set_ylabel(f"{y_metric.replace('_', ' ').title()} (%)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(0, 100)

        # Size legend
        legend_sizes = [100, 500, 1000]
        legend_labels = ["Small", "Medium", "Large"]
        for size, label in zip(legend_sizes, legend_labels, strict=False):
            ax.scatter([], [], s=size, c="gray", alpha=0.5, label=f"{label} GDP")
        ax.legend(
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            title="GDP Size",
        )

        self._add_source_annotation(ax)
        plt.tight_layout()
        return self._save_chart(fig, f"bubble_{self.chart_counter}.png")

    def create_donut_chart(
        self,
        values: dict[str, float],
        title: str = "Distribution",
        center_text: str = "",
    ) -> bytes:
        """Create donut/pie chart"""
        fig, ax = plt.subplots(figsize=(8, 8))

        labels = list(values.keys())
        sizes = [abs(float(v)) for v in values.values()]

        # Sort by size
        sorted_data = sorted(
            zip(labels, sizes, strict=False), key=lambda x: x[1], reverse=True
        )
        labels, sizes = zip(*sorted_data, strict=False) if sorted_data else ([], [])

        colors = [
            self.COUNTRY_COLORS[i % len(self.COUNTRY_COLORS)]
            for i in range(len(labels))
        ]

        # Create donut
        _wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
            startangle=90,
            pctdistance=0.75,
            wedgeprops={"width": 0.5, "edgecolor": "white"},
        )

        # Style the text
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        # Center text
        if center_text:
            ax.text(
                0,
                0,
                center_text,
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        plt.tight_layout()
        return self._save_chart(fig, f"donut_{self.chart_counter}.png")

    def create_damage_breakdown_donut(
        self,
        cb_damage: float,
        gov_damage: float,
        title: str = "Intervention Damage Breakdown",
    ) -> bytes:
        """Create donut chart for CB vs Gov damage"""
        total = cb_damage + gov_damage
        values = {"Central Bank": cb_damage, "Government": gov_damage}

        fig, ax = plt.subplots(figsize=(8, 8))

        labels = list(values.keys())
        sizes = list(values.values())
        colors = [self.COLORS["negative"], self.COLORS["secondary"]]

        _wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct=lambda p: f"{p:.1f}%",
            startangle=90,
            pctdistance=0.75,
            wedgeprops={"width": 0.5, "edgecolor": "white"},
            explode=(0.02, 0.02),
        )

        for text in texts:
            text.set_fontsize(11)
            text.set_fontweight("bold")
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        # Center text with total
        center_text = f"Total\n${total / 1e9:.1f}B"
        ax.text(
            0, 0, center_text, ha="center", va="center", fontsize=12, fontweight="bold"
        )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        plt.tight_layout()
        return self._save_chart(fig, f"damage_donut_{self.chart_counter}.png")

    def create_stacked_area_evolution(
        self,
        country_metrics: dict[str, list[float]],
        metric_name: str,
        title: str = "Cumulative Evolution by Country",
    ) -> bytes:
        """Create stacked area chart showing evolution over time"""
        fig, ax = plt.subplots(figsize=(12, 6))

        countries = list(country_metrics.keys())
        n_months = len(next(iter(country_metrics.values()))) if country_metrics else 0
        months = range(n_months)

        # Prepare data for stacking
        data = [country_metrics[c] for c in countries]
        colors = [
            self.COUNTRY_COLORS[i % len(self.COUNTRY_COLORS)]
            for i in range(len(countries))
        ]

        ax.stackplot(months, *data, labels=countries, colors=colors, alpha=0.8)

        ax.set_xlabel("Month")
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_title(title, fontsize=14, fontweight="bold")

        ax.legend(
            loc="upper left",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            ncol=min(5, len(countries)),
        )
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(self._format_currency_full))

        self._add_source_annotation(ax)
        plt.tight_layout()
        return self._save_chart(fig, f"stacked_area_{self.chart_counter}.png")

    def create_histogram(
        self,
        values: list[float],
        title: str = "Distribution",
        xlabel: str = "Value",
        bins: int = 20,
    ) -> bytes:
        """Create histogram showing distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))

        _n, _bins_edges, _patches = ax.hist(
            values,
            bins=bins,
            color=self.COLORS["primary"],
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )

        # Add mean and median lines
        mean_val = np.mean(values)
        median_val = np.median(values)

        ax.axvline(
            mean_val,
            color=self.COLORS["negative"],
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.2f}",
        )
        ax.axvline(
            median_val,
            color=self.COLORS["positive"],
            linestyle="-",
            linewidth=2,
            label=f"Median: {median_val:.2f}",
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

        self._add_source_annotation(ax)
        plt.tight_layout()
        return self._save_chart(fig, f"histogram_{self.chart_counter}.png")

    def create_box_plot(
        self,
        data: dict[str, list[float]],
        title: str = "Distribution Comparison",
        ylabel: str = "Value",
    ) -> bytes:
        """Create box plot comparing distributions across categories"""
        fig, ax = plt.subplots(figsize=(10, 6))

        labels = list(data.keys())
        values = [data[k] for k in labels]

        bp = ax.boxplot(values, labels=labels, patch_artist=True)

        # Color boxes
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(self.COUNTRY_COLORS[i % len(self.COUNTRY_COLORS)])
            box.set_alpha(0.7)

        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Rotate labels if many
        if len(labels) > 5:
            plt.xticks(rotation=45, ha="right")

        self._add_source_annotation(ax)
        plt.tight_layout()
        return self._save_chart(fig, f"boxplot_{self.chart_counter}.png")

    def create_ranking_horizontal_bar(
        self,
        rankings: dict[str, float],
        title: str = "Country Rankings",
        xlabel: str = "Score",
        highlight_best: bool = True,
    ) -> bytes:
        """Create horizontal bar chart for rankings"""
        fig, ax = plt.subplots(figsize=(10, max(6, len(rankings) * 0.4)))

        # Sort by value
        sorted_items = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        # Color by rank
        colors = []
        for i, _v in enumerate(values):
            if i == 0 and highlight_best:
                colors.append(self.COLORS["positive"])  # Best = green
            elif i == len(values) - 1:
                colors.append(self.COLORS["negative"])  # Worst = red
            else:
                colors.append(self.COLORS["primary"])

        y_pos = np.arange(len(labels))
        bars = ax.barh(
            y_pos, values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Add value labels
        max_val = max(values) if values else 1
        for bar, val in zip(bars, values, strict=False):
            ax.text(
                val + max_val * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        # Add rank numbers
        for i, (bar, _label) in enumerate(zip(bars, labels, strict=False)):
            ax.text(
                -max_val * 0.08,
                bar.get_y() + bar.get_height() / 2,
                f"#{i + 1}",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=self.COLORS["neutral"],
            )

        self._add_source_annotation(ax)
        plt.tight_layout()
        return self._save_chart(fig, f"ranking_{self.chart_counter}.png")

    def create_regime_comparison_chart(
        self,
        country_data: dict[str, dict[str, Any]],
        title: str = "Economic Outcomes by Regime Type",
    ) -> bytes:
        """Create chart comparing outcomes grouped by regime type"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))

        # Group countries by regime
        regime_groups = {}
        for country, data in country_data.items():
            regime = data.get("regime", "unknown")
            if regime not in regime_groups:
                regime_groups[regime] = []
            regime_groups[regime].append((country, data))

        # Metrics to compare
        metrics = ["freedom_index", "inflation", "unemployment"]
        metric_labels = ["Freedom Index", "Inflation (%)", "Unemployment (%)"]

        for idx, (metric, label) in enumerate(
            zip(metrics, metric_labels, strict=False)
        ):
            ax = axes[idx]

            regimes = list(regime_groups.keys())
            avg_values = []

            for regime in regimes:
                countries_in_regime = regime_groups[regime]
                values = [cd[1].get(metric, 0) for cd in countries_in_regime]
                values = [
                    v * 100 if v < 1 and metric != "freedom_index" else v
                    for v in values
                ]
                avg_values.append(np.mean(values) if values else 0)

            # Sort by value
            sorted_data = sorted(
                zip(regimes, avg_values, strict=False),
                key=lambda x: x[1],
                reverse=(metric == "freedom_index"),
            )
            regimes, avg_values = (
                zip(*sorted_data, strict=False) if sorted_data else ([], [])
            )

            # Color by regime type
            colors = []
            for r in regimes:
                r_lower = r.lower() if isinstance(r, str) else str(r)
                if "ancap" in r_lower or "minarch" in r_lower:
                    colors.append(self.COLORS["positive"])
                elif "monarch" in r_lower:
                    colors.append(self.COLORS["highlight"])
                elif "liberal" in r_lower:
                    colors.append(self.COLORS["primary"])
                elif "socialist" in r_lower:
                    colors.append(self.COLORS["secondary"])
                else:  # totalitarian
                    colors.append(self.COLORS["negative"])

            y_pos = np.arange(len(regimes))
            bars = ax.barh(
                y_pos,
                avg_values,
                color=colors,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )

            ax.set_yticks(y_pos)
            ax.set_yticklabels([r.replace("_", " ").title() for r in regimes])
            ax.invert_yaxis()
            ax.set_xlabel(label)
            ax.set_title(label, fontsize=11, fontweight="bold")

            # Add value labels
            max_val = max(avg_values) if avg_values else 1
            for bar, val in zip(bars, avg_values, strict=False):
                ax.text(
                    val + max_val * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}",
                    va="center",
                    fontsize=9,
                )

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        return self._save_chart(fig, f"regime_comparison_{self.chart_counter}.png")

    def create_multi_country_evolution_chart(
        self, country_metrics: dict[str, list[float]], metric_name: str, title: str
    ) -> bytes:
        """Create line chart showing metric evolution for multiple countries"""
        fig, ax = plt.subplots(figsize=(10, 5))

        for _i, (country, values) in enumerate(country_metrics.items()):
            months = range(len(values))
            ax.plot(
                months,
                values,
                linewidth=2,
                label=country,
                marker="o",
                markersize=4,
                markevery=max(1, len(values) // 6),
            )

        ax.set_xlabel("Month")
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_title(title)

        ax.legend(
            loc="best",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            ncol=min(4, len(country_metrics)),
        )
        self._add_source_annotation(ax)

        plt.tight_layout()
        return self._save_chart(fig, f"multi_country_{self.chart_counter}.png")

    def create_war_risk_chart(
        self, war_risks: list[dict[str, Any]], title: str = "War Risk Assessment"
    ) -> bytes:
        """Create horizontal bar chart of war risk"""
        fig, ax = plt.subplots(figsize=(10, 5))

        if not war_risks:
            ax.text(
                0.5,
                0.5,
                "No significant war risks detected",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                color=self.COLORS["neutral"],
            )
            ax.axis("off")
            return self._save_chart(fig, f"war_risk_{self.chart_counter}.png")

        pairs = [f"{r['countries'][0]}-{r['countries'][1]}" for r in war_risks[:8]]
        probs = [r["probability"] * 100 for r in war_risks[:8]]

        colors = []
        for p in probs:
            if p >= 10:
                colors.append(self.COLORS["negative"])
            elif p >= 5:
                colors.append(self.COLORS["secondary"])
            else:
                colors.append(self.COLORS["highlight"])

        y_pos = np.arange(len(pairs))
        bars = ax.barh(
            y_pos, probs, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(pairs)
        ax.invert_yaxis()
        ax.set_xlabel("War Probability (%)")
        ax.set_title(title)

        # Add value labels
        for bar, prob in zip(bars, probs, strict=False):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{prob:.1f}%",
                va="center",
                fontsize=9,
            )

        self._add_source_annotation(ax)
        plt.tight_layout()
        return self._save_chart(fig, f"war_risk_{self.chart_counter}.png")

    # ===========================================
    # DASHBOARD
    # ===========================================

    def create_simulation_dashboard(
        self, metrics_history: list[Any], title: str = "Economic Dashboard"
    ) -> bytes:
        """Create comprehensive 6-panel dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

        months = list(range(len(metrics_history)))
        btc_prices = [float(m.bitcoin_price) for m in metrics_history]
        gold_prices = [float(m.gold_price) for m in metrics_history]
        inflation_rates = [
            m.inflation_rate * 100 if m.inflation_rate < 1 else m.inflation_rate
            for m in metrics_history
        ]
        unemployment_rates = [
            m.unemployment_rate * 100
            if m.unemployment_rate < 1
            else m.unemployment_rate
            for m in metrics_history
        ]
        freedom_indices = [m.freedom_index for m in metrics_history]
        cb_damage = [float(m.central_bank_damage) for m in metrics_history]
        gov_damage = [float(m.government_damage) for m in metrics_history]

        # Panel 1: Bitcoin Price
        ax = axes[0, 0]
        ax.plot(months, btc_prices, color=self.COLORS["bitcoin"], linewidth=2)
        ax.fill_between(months, btc_prices, alpha=0.15, color=self.COLORS["bitcoin"])
        ax.set_title("Bitcoin Price", fontsize=11, fontweight="bold")
        ax.set_ylabel("USD")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(self._format_currency))
        ax.set_xlabel("Month")

        # Panel 2: Gold Price
        ax = axes[0, 1]
        ax.plot(months, gold_prices, color=self.COLORS["gold"], linewidth=2)
        ax.fill_between(months, gold_prices, alpha=0.15, color=self.COLORS["gold"])
        ax.set_title("Gold Price", fontsize=11, fontweight="bold")
        ax.set_ylabel("USD/oz")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(self._format_currency))
        ax.set_xlabel("Month")

        # Panel 3: Inflation Rate
        ax = axes[0, 2]
        colors = [
            self.COLORS["negative"]
            if r > 5
            else self.COLORS["secondary"]
            if r > 2
            else self.COLORS["positive"]
            for r in inflation_rates
        ]
        ax.bar(months, inflation_rates, color=colors, alpha=0.85)
        ax.axhline(
            y=2, color=self.COLORS["neutral"], linestyle="--", linewidth=1, alpha=0.7
        )
        ax.set_title("Inflation Rate", fontsize=11, fontweight="bold")
        ax.set_ylabel("%")
        ax.set_xlabel("Month")

        # Panel 4: Unemployment
        ax = axes[1, 0]
        ax.fill_between(
            months, unemployment_rates, alpha=0.3, color=self.COLORS["negative"]
        )
        ax.plot(months, unemployment_rates, color=self.COLORS["negative"], linewidth=2)
        ax.set_title("Unemployment Rate", fontsize=11, fontweight="bold")
        ax.set_ylabel("%")
        ax.set_xlabel("Month")
        ax.set_ylim(bottom=0)

        # Panel 5: Freedom Index
        ax = axes[1, 1]
        colors = [
            self.COLORS["positive"]
            if f >= 70
            else self.COLORS["highlight"]
            if f >= 50
            else self.COLORS["negative"]
            for f in freedom_indices
        ]
        ax.bar(months, freedom_indices, color=colors, alpha=0.85)
        ax.axhline(
            y=50, color=self.COLORS["neutral"], linestyle="--", linewidth=1, alpha=0.7
        )
        ax.set_title("Freedom Index", fontsize=11, fontweight="bold")
        ax.set_ylabel("Index")
        ax.set_xlabel("Month")
        ax.set_ylim(0, 100)

        # Panel 6: Cumulative Damage
        ax = axes[1, 2]
        ax.stackplot(
            months,
            cb_damage,
            gov_damage,
            labels=["Central Bank", "Government"],
            colors=[self.COLORS["negative"], self.COLORS["secondary"]],
            alpha=0.8,
        )
        ax.set_title("Intervention Damage", fontsize=11, fontweight="bold")
        ax.set_ylabel("USD")
        ax.set_xlabel("Month")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(self._format_currency_full))
        ax.legend(
            loc="upper left",
            fontsize=8,
            frameon=True,
            facecolor="white",
            edgecolor="none",
        )

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        return self._save_chart(fig, f"dashboard_{self.chart_counter}.png")
