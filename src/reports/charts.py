"""
Chart Generation Module - Publication Quality

Creates high-quality Plotly charts suitable for academic publications.
Style inspired by The Economist, Financial Times, and academic journals.
"""

from decimal import Decimal
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

    def _format_currency(self, x: float) -> str:
        """Format values as currency"""
        if abs(x) >= 1e9:
            return f"${x / 1e9:.0f}B"
        elif abs(x) >= 1e6:
            return f"${x / 1e6:.0f}M"
        elif abs(x) >= 1e3:
            return f"${x / 1e3:.0f}K"
        else:
            return f"${x:,.0f}"

    def _format_currency_full(self, x: float) -> str:
        """Format values as currency with more precision"""
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

    def _get_layout(
        self,
        title: str,
        xaxis_title: str = "",
        yaxis_title: str = "",
        showlegend: bool = True,
    ) -> dict[str, Any]:
        """Get standard layout configuration"""
        return {
            "title": {
                "text": title,
                "font": {"size": 16, "family": "Times New Roman, serif", "color": "#333"},
                "x": 0.5,
                "xanchor": "center",
            },
            "xaxis": {
                "title": {"text": xaxis_title, "font": {"size": 12}},
                "showgrid": True,
                "gridcolor": "#e0e0e0",
                "gridwidth": 0.5,
                "linecolor": "#333333",
                "linewidth": 0.8,
                "tickfont": {"size": 10},
            },
            "yaxis": {
                "title": {"text": yaxis_title, "font": {"size": 12}},
                "showgrid": True,
                "gridcolor": "#e0e0e0",
                "gridwidth": 0.5,
                "linecolor": "#333333",
                "linewidth": 0.8,
                "tickfont": {"size": 10},
            },
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
            "font": {"family": "Times New Roman, serif", "size": 11},
            "showlegend": showlegend,
            "legend": {
                "bgcolor": "rgba(255,255,255,0.9)",
                "borderwidth": 0,
                "font": {"size": 10},
            },
            "margin": {"l": 60, "r": 40, "t": 60, "b": 60},
            "annotations": [
                {
                    "text": "Source: Inflationator Simulation",
                    "showarrow": False,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0,
                    "y": -0.12,
                    "font": {"size": 9, "color": self.COLORS["neutral"], "style": "italic"},
                }
            ],
        }

    def _save_chart(self, fig: go.Figure, filename: str) -> bytes:
        """Save chart and return bytes"""
        # Save to bytes
        img_bytes = fig.to_image(format="png", width=1000, height=500, scale=2)

        # Also save to file
        filepath = self.output_dir / filename
        with open(filepath, "wb") as f:
            f.write(img_bytes)

        self.chart_counter += 1
        return img_bytes

    # ===========================================
    # PRICE CHARTS
    # ===========================================

    def create_bitcoin_price_chart(
        self, prices: list[Decimal], title: str = "Bitcoin Price"
    ) -> bytes:
        """Create professional Bitcoin price chart"""
        prices_float = self._to_float_list(prices)
        months = list(range(len(prices_float)))

        fig = go.Figure()

        # Fill area under line
        fig.add_trace(
            go.Scatter(
                x=months,
                y=prices_float,
                fill="tozeroy",
                fillcolor=f"rgba(247, 147, 26, 0.15)",
                line={"color": self.COLORS["bitcoin"], "width": 2.5},
                name="BTC/USD",
                mode="lines",
            )
        )

        # Add start and end annotations
        if len(prices_float) > 1:
            fig.add_annotation(
                x=0,
                y=prices_float[0],
                text=f"${prices_float[0]:,.0f}",
                showarrow=False,
                xshift=-30,
                yshift=10,
                font={"size": 10, "color": self.COLORS["neutral"]},
            )
            fig.add_annotation(
                x=len(prices_float) - 1,
                y=prices_float[-1],
                text=f"${prices_float[-1]:,.0f}",
                showarrow=False,
                xshift=5,
                yshift=10,
                font={"size": 10, "color": self.COLORS["bitcoin"], "weight": "bold"},
            )

        layout = self._get_layout(title, "Month", "Price (USD)")
        fig.update_layout(**layout)

        return self._save_chart(fig, f"bitcoin_price_{self.chart_counter}.png")

    def create_gold_price_chart(
        self, prices: list[Decimal], title: str = "Gold Price"
    ) -> bytes:
        """Create professional Gold price chart"""
        prices_float = self._to_float_list(prices)
        months = list(range(len(prices_float)))

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=months,
                y=prices_float,
                fill="tozeroy",
                fillcolor="rgba(207, 181, 59, 0.15)",
                line={"color": self.COLORS["gold"], "width": 2.5},
                name="Gold/USD (oz)",
                mode="lines",
            )
        )

        layout = self._get_layout(title, "Month", "Price (USD per oz)")
        fig.update_layout(**layout)

        return self._save_chart(fig, f"gold_price_{self.chart_counter}.png")

    def create_price_comparison_chart(
        self,
        btc_prices: list[Decimal],
        gold_prices: list[Decimal],
        title: str = "Asset Performance (Indexed to 100)",
    ) -> bytes:
        """Create comparison chart with normalized values"""
        btc = self._to_float_list(btc_prices)
        gold = self._to_float_list(gold_prices)

        # Normalize to 100
        btc_norm = [100 * p / btc[0] if btc[0] > 0 else 100 for p in btc]
        gold_norm = [100 * p / gold[0] if gold[0] > 0 else 100 for p in gold]
        months = list(range(len(btc_norm)))

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=months,
                y=btc_norm,
                line={"color": self.COLORS["bitcoin"], "width": 2.5},
                name="Bitcoin",
                mode="lines",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=months,
                y=gold_norm,
                line={"color": self.COLORS["gold"], "width": 2.5},
                name="Gold",
                mode="lines",
            )
        )

        # Starting value line
        fig.add_hline(
            y=100,
            line_dash="dash",
            line_color=self.COLORS["neutral"],
            line_width=1,
            opacity=0.7,
            annotation_text="Starting Value",
            annotation_position="right",
        )

        layout = self._get_layout(title, "Month", "Index (Start = 100)")
        fig.update_layout(**layout)

        return self._save_chart(fig, f"price_comparison_{self.chart_counter}.png")

    # ===========================================
    # MONETARY CHARTS
    # ===========================================

    def create_inflation_chart(
        self, inflation_rates: list[float], title: str = "Inflation Rate (Annualized)"
    ) -> bytes:
        """Create professional inflation bar chart"""
        months = list(range(len(inflation_rates)))
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

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=months,
                y=rates_pct,
                marker_color=colors,
                opacity=0.85,
                name="Inflation",
            )
        )

        # Target line
        fig.add_hline(
            y=2,
            line_dash="dash",
            line_color=self.COLORS["neutral"],
            line_width=1.5,
            opacity=0.8,
            annotation_text="2% Target",
            annotation_position="right",
        )

        layout = self._get_layout(title, "Month", "Inflation Rate (%)")
        fig.update_layout(**layout)

        return self._save_chart(fig, f"inflation_{self.chart_counter}.png")

    def create_money_supply_chart(
        self,
        money_supply: list[Decimal],
        credit_expansion: list[Decimal],
        title: str = "Money Supply & Credit",
    ) -> bytes:
        """Create dual-axis money supply chart"""
        money = self._to_float_list(money_supply)
        credit = self._to_float_list(credit_expansion)
        months = list(range(len(money)))

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Base money on primary axis
        fig.add_trace(
            go.Scatter(
                x=months,
                y=money,
                fill="tozeroy",
                fillcolor="rgba(26, 71, 111, 0.1)",
                line={"color": self.COLORS["primary"], "width": 2.5},
                name="Base Money",
                mode="lines",
            ),
            secondary_y=False,
        )

        # Credit on secondary axis
        fig.add_trace(
            go.Scatter(
                x=months,
                y=credit,
                line={"color": self.COLORS["secondary"], "width": 2.5, "dash": "dash"},
                name="Credit",
                mode="lines",
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title={
                "text": title,
                "font": {"size": 16, "family": "Times New Roman, serif"},
                "x": 0.5,
            },
            xaxis_title="Month",
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend={"bgcolor": "rgba(255,255,255,0.9)", "borderwidth": 0},
        )

        fig.update_yaxes(
            title_text="Base Money",
            secondary_y=False,
            tickcolor=self.COLORS["primary"],
            title_font={"color": self.COLORS["primary"]},
        )
        fig.update_yaxes(
            title_text="Credit Expansion",
            secondary_y=True,
            tickcolor=self.COLORS["secondary"],
            title_font={"color": self.COLORS["secondary"]},
        )

        return self._save_chart(fig, f"money_supply_{self.chart_counter}.png")

    # ===========================================
    # LABOR CHARTS
    # ===========================================

    def create_unemployment_chart(
        self, unemployment_rates: list[float], title: str = "Unemployment Rate"
    ) -> bytes:
        """Create unemployment rate area chart"""
        months = list(range(len(unemployment_rates)))
        rates_pct = [r * 100 if r < 1 else r for r in unemployment_rates]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=months,
                y=rates_pct,
                fill="tozeroy",
                fillcolor="rgba(196, 44, 44, 0.3)",
                line={"color": self.COLORS["negative"], "width": 2.5},
                mode="lines+markers",
                marker={"size": 8, "color": self.COLORS["negative"]},
                name="Unemployment",
            )
        )

        layout = self._get_layout(title, "Month", "Unemployment Rate (%)")
        layout["yaxis"]["rangemode"] = "tozero"
        fig.update_layout(**layout)

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
        cb = self._to_float_list(cb_damage)
        gov = self._to_float_list(gov_damage)
        months = list(range(len(cb)))

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=months,
                y=cb,
                fill="tozeroy",
                fillcolor="rgba(196, 44, 44, 0.8)",
                line={"color": self.COLORS["negative"], "width": 0},
                name="Central Bank",
                stackgroup="one",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=months,
                y=gov,
                fill="tonexty",
                fillcolor="rgba(227, 114, 34, 0.8)",
                line={"color": self.COLORS["secondary"], "width": 0},
                name="Government",
                stackgroup="one",
            )
        )

        layout = self._get_layout(title, "Month", "Cumulative Damage")
        fig.update_layout(**layout)

        return self._save_chart(fig, f"damage_{self.chart_counter}.png")

    def create_freedom_index_chart(
        self, freedom_indices: list[float], title: str = "Economic Freedom Index"
    ) -> bytes:
        """Create freedom index bar chart"""
        months = list(range(len(freedom_indices)))

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

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=months,
                y=freedom_indices,
                marker_color=colors,
                opacity=0.85,
                name="Freedom Index",
            )
        )

        # Reference lines
        fig.add_hline(
            y=80,
            line_dash="dash",
            line_color=self.COLORS["positive"],
            line_width=1,
            opacity=0.5,
            annotation_text="Free (80+)",
        )
        fig.add_hline(
            y=50,
            line_dash="dash",
            line_color=self.COLORS["neutral"],
            line_width=1,
            opacity=0.5,
            annotation_text="Moderate (50)",
        )

        layout = self._get_layout(title, "Month", "Freedom Index")
        layout["yaxis"]["range"] = [0, 100]
        fig.update_layout(**layout)

        return self._save_chart(fig, f"freedom_{self.chart_counter}.png")

    # ===========================================
    # MULTI-COUNTRY CHARTS
    # ===========================================

    def create_country_comparison_chart(
        self, country_data: dict[str, dict[str, Any]], metric: str, title: str
    ) -> bytes:
        """Create horizontal bar chart comparing countries"""
        countries = list(country_data.keys())
        values = [float(country_data[c].get(metric, 0)) for c in countries]

        # Sort by value
        sorted_data = sorted(
            zip(countries, values, strict=False), key=lambda x: x[1], reverse=True
        )
        countries, values = zip(*sorted_data, strict=False) if sorted_data else ([], [])

        # Alternate colors
        colors = [
            self.COLORS["primary"] if i % 2 == 0 else self.COLORS["secondary"]
            for i in range(len(values))
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=list(countries),
                x=list(values),
                orientation="h",
                marker_color=colors,
                opacity=0.85,
                text=[f"{v:.1f}" for v in values],
                textposition="outside",
            )
        )

        layout = self._get_layout(title, metric.replace("_", " ").title(), "")
        layout["yaxis"]["autorange"] = "reversed"
        layout["height"] = max(400, len(countries) * 30)
        fig.update_layout(**layout)

        return self._save_chart(fig, f"country_comparison_{self.chart_counter}.png")

    def create_grouped_bar_chart(
        self,
        country_data: dict[str, dict[str, Any]],
        metrics: list[str],
        title: str = "Multi-Metric Country Comparison",
    ) -> bytes:
        """Create grouped bar chart comparing multiple metrics across countries"""
        countries = list(country_data.keys())

        fig = go.Figure()

        for i, metric in enumerate(metrics):
            values = [float(country_data[c].get(metric, 0)) for c in countries]
            # Normalize percentages
            if any(v < 1 for v in values if isinstance(v, (int, float))):
                values = [v * 100 if v < 1 else v for v in values]

            fig.add_trace(
                go.Bar(
                    name=metric.replace("_", " ").title(),
                    x=countries,
                    y=values,
                    marker_color=self.COUNTRY_COLORS[i % len(self.COUNTRY_COLORS)],
                    opacity=0.85,
                )
            )

        layout = self._get_layout(title, "Country", "Value")
        layout["barmode"] = "group"
        fig.update_layout(**layout)
        fig.update_xaxes(tickangle=45)

        return self._save_chart(fig, f"grouped_bar_{self.chart_counter}.png")

    def create_radar_chart(
        self,
        country_data: dict[str, dict[str, float]],
        metrics: list[str],
        title: str = "Country Economic Profile",
    ) -> bytes:
        """Create radar/spider chart for country comparison"""
        metric_labels = [m.replace("_", " ").title() for m in metrics]

        fig = go.Figure()

        for i, (country, data) in enumerate(country_data.items()):
            values = [float(data.get(m, 0)) for m in metrics]
            # Normalize values to 0-100 scale
            max_vals = {
                m: max(float(country_data[c].get(m, 1)) for c in country_data)
                for m in metrics
            }
            values_norm = [
                v / max_vals[m] * 100 if max_vals[m] > 0 else 0
                for v, m in zip(values, metrics, strict=False)
            ]
            values_norm += values_norm[:1]  # Close the polygon

            color = self.COUNTRY_COLORS[i % len(self.COUNTRY_COLORS)]

            fig.add_trace(
                go.Scatterpolar(
                    r=values_norm,
                    theta=metric_labels + metric_labels[:1],
                    fill="toself",
                    fillcolor=f"rgba{tuple(int(color[j:j+2], 16) for j in (1, 3, 5)) + (0.15,)}",
                    line={"color": color, "width": 2},
                    name=country,
                )
            )

        fig.update_layout(
            polar={
                "radialaxis": {"visible": True, "range": [0, 100]},
                "bgcolor": "white",
            },
            title={
                "text": title,
                "font": {"size": 16, "family": "Times New Roman, serif"},
                "x": 0.5,
            },
            showlegend=True,
            legend={"x": 1.1, "y": 1},
            paper_bgcolor="white",
        )

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
        countries = list(matrix.keys())
        n = len(countries)

        # Build matrix data
        data = []
        for c1 in countries:
            row = []
            for c2 in countries:
                row.append(matrix.get(c1, {}).get(c2, 0))
            data.append(row)

        # Create text annotations
        text = None
        if show_values:
            text = [[f"{v * 100:{value_format}}%" if v > 0 else "" for v in row] for row in data]

        colorscale = "YlOrRd" if "YlOrRd" in cmap else "RdYlGn_r"

        fig = go.Figure(
            data=go.Heatmap(
                z=data,
                x=countries,
                y=countries,
                colorscale=colorscale,
                text=text,
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate="From: %{y}<br>To: %{x}<br>Value: %{z:.2%}<extra></extra>",
                colorbar={"title": "Probability"},
            )
        )

        fig.update_layout(
            title={
                "text": title,
                "font": {"size": 16, "family": "Times New Roman, serif"},
                "x": 0.5,
            },
            xaxis={"side": "bottom", "tickangle": 45},
            yaxis={"autorange": "reversed"},
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        return self._save_chart(fig, f"heatmap_{self.chart_counter}.png")

    def create_war_probability_heatmap(
        self,
        war_risks: list[dict[str, Any]],
        countries: list[str],
        title: str = "War Probability Matrix",
    ) -> bytes:
        """Create heatmap specifically for war probabilities between countries"""
        # Build matrix from war risks
        matrix: dict[str, dict[str, float]] = {c: dict.fromkeys(countries, 0.0) for c in countries}

        for risk in war_risks:
            c1, c2 = risk["countries"]
            prob = float(risk["probability"])
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
        countries = list(country_data.keys())
        x_values = [float(country_data[c].get(x_metric, 0)) for c in countries]
        y_values = [float(country_data[c].get(y_metric, 0)) for c in countries]

        # Normalize percentages
        x_values = [v * 100 if v < 1 else v for v in x_values]
        y_values = [v * 100 if v < 1 else v for v in y_values]

        # Size based on third metric or uniform
        if size_metric:
            sizes = [float(country_data[c].get(size_metric, 50)) for c in countries]
            max_size = max(sizes) if sizes else 1
            sizes = [max(20, s / max_size * 60) for s in sizes]
        else:
            sizes = [30] * len(countries)

        colors = [
            self.COUNTRY_COLORS[i % len(self.COUNTRY_COLORS)]
            for i in range(len(countries))
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers+text",
                marker={"size": sizes, "color": colors, "opacity": 0.7, "line": {"width": 1.5, "color": "white"}},
                text=countries,
                textposition="top center",
                textfont={"size": 10, "weight": "bold"},
                hovertemplate="%{text}<br>%{xaxis.title.text}: %{x:.1f}<br>%{yaxis.title.text}: %{y:.1f}<extra></extra>",
            )
        )

        # Add trend line if enough data
        if len(x_values) > 2:
            import numpy as np
            z = np.polyfit(x_values, y_values, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(x_values), max(x_values), 100)
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=p(x_line),
                    mode="lines",
                    line={"color": self.COLORS["neutral"], "width": 1.5, "dash": "dash"},
                    name="Trend",
                    opacity=0.5,
                )
            )

        layout = self._get_layout(
            title, x_metric.replace("_", " ").title(), y_metric.replace("_", " ").title()
        )
        fig.update_layout(**layout)

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
        countries = list(country_data.keys())

        x_values = [float(country_data[c].get(x_metric, 50)) for c in countries]
        y_values = [float(country_data[c].get(y_metric, 0)) for c in countries]
        sizes = [float(country_data[c].get(size_metric, 1)) for c in countries]

        # Normalize y if percentage
        y_values = [v * 100 if v < 1 else v for v in y_values]

        # Normalize sizes for display
        max_size = max(sizes) if sizes else 1
        sizes_norm = [max(20, s / max_size * 80) for s in sizes]

        # Color by freedom index
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

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers+text",
                marker={"size": sizes_norm, "color": colors, "opacity": 0.6, "line": {"width": 2, "color": "white"}},
                text=countries,
                textposition="top center",
                textfont={"size": 11, "weight": "bold"},
            )
        )

        # Add quadrant lines
        fig.add_vline(x=50, line_dash="dash", line_color=self.COLORS["light"], line_width=1, opacity=0.7)
        fig.add_hline(y=10, line_dash="dash", line_color=self.COLORS["light"], line_width=1, opacity=0.7)

        # Quadrant labels
        quadrant_annotations = [
            {"x": 75, "y": 2, "text": "Free & Stable<br>(Ideal)", "color": self.COLORS["positive"]},
            {"x": 25, "y": 2, "text": "Unfree & Stable", "color": self.COLORS["neutral"]},
            {"x": 75, "y": 25, "text": "Free & Inflationary", "color": self.COLORS["highlight"]},
            {"x": 25, "y": 25, "text": "Unfree & Inflationary<br>(Worst)", "color": self.COLORS["negative"]},
        ]

        for ann in quadrant_annotations:
            fig.add_annotation(
                x=ann["x"],
                y=ann["y"],
                text=ann["text"],
                showarrow=False,
                font={"size": 10, "color": ann["color"]},
                opacity=0.7,
            )

        layout = self._get_layout(
            title,
            f"{x_metric.replace('_', ' ').title()}",
            f"{y_metric.replace('_', ' ').title()} (%)",
        )
        layout["xaxis"]["range"] = [0, 100]
        fig.update_layout(**layout)

        return self._save_chart(fig, f"bubble_{self.chart_counter}.png")

    def create_donut_chart(
        self,
        values: dict[str, float],
        title: str = "Distribution",
        center_text: str = "",
    ) -> bytes:
        """Create donut/pie chart"""
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

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(labels),
                    values=list(sizes),
                    hole=0.5,
                    marker={"colors": colors, "line": {"color": "white", "width": 2}},
                    textinfo="label+percent",
                    textposition="outside",
                    textfont={"size": 11},
                )
            ]
        )

        if center_text:
            fig.add_annotation(
                x=0.5,
                y=0.5,
                text=center_text,
                showarrow=False,
                font={"size": 14, "weight": "bold"},
                xref="paper",
                yref="paper",
            )

        fig.update_layout(
            title={
                "text": title,
                "font": {"size": 16, "family": "Times New Roman, serif"},
                "x": 0.5,
            },
            showlegend=True,
            paper_bgcolor="white",
        )

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
        colors = [self.COLORS["negative"], self.COLORS["secondary"]]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(values.keys()),
                    values=list(values.values()),
                    hole=0.5,
                    marker={"colors": colors, "line": {"color": "white", "width": 2}},
                    textinfo="label+percent",
                    textposition="outside",
                    textfont={"size": 11, "weight": "bold"},
                    pull=[0.02, 0.02],
                )
            ]
        )

        # Center text with total
        center_text = f"Total<br>${total / 1e9:.1f}B"
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=center_text,
            showarrow=False,
            font={"size": 12, "weight": "bold"},
            xref="paper",
            yref="paper",
        )

        fig.update_layout(
            title={
                "text": title,
                "font": {"size": 16, "family": "Times New Roman, serif"},
                "x": 0.5,
            },
            showlegend=True,
            paper_bgcolor="white",
        )

        return self._save_chart(fig, f"damage_donut_{self.chart_counter}.png")

    def create_stacked_area_evolution(
        self,
        country_metrics: dict[str, list[float]],
        metric_name: str,
        title: str = "Cumulative Evolution by Country",
    ) -> bytes:
        """Create stacked area chart showing evolution over time"""
        countries = list(country_metrics.keys())
        n_months = len(next(iter(country_metrics.values()))) if country_metrics else 0
        months = list(range(n_months))

        fig = go.Figure()

        for i, country in enumerate(countries):
            values = country_metrics[country]
            color = self.COUNTRY_COLORS[i % len(self.COUNTRY_COLORS)]

            fig.add_trace(
                go.Scatter(
                    x=months,
                    y=values,
                    mode="lines",
                    stackgroup="one",
                    name=country,
                    line={"color": color, "width": 0},
                    fillcolor=color,
                )
            )

        layout = self._get_layout(title, "Month", metric_name.replace("_", " ").title())
        fig.update_layout(**layout)

        return self._save_chart(fig, f"stacked_area_{self.chart_counter}.png")

    def create_histogram(
        self,
        values: list[float],
        title: str = "Distribution",
        xlabel: str = "Value",
        bins: int = 20,
    ) -> bytes:
        """Create histogram showing distribution"""
        import numpy as np

        mean_val = np.mean(values)
        median_val = np.median(values)

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=bins,
                marker_color=self.COLORS["primary"],
                opacity=0.7,
                name="Distribution",
            )
        )

        # Mean line
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color=self.COLORS["negative"],
            line_width=2,
            annotation_text=f"Mean: {mean_val:.2f}",
            annotation_position="top",
        )

        # Median line
        fig.add_vline(
            x=median_val,
            line_color=self.COLORS["positive"],
            line_width=2,
            annotation_text=f"Median: {median_val:.2f}",
            annotation_position="top right",
        )

        layout = self._get_layout(title, xlabel, "Frequency")
        fig.update_layout(**layout)

        return self._save_chart(fig, f"histogram_{self.chart_counter}.png")

    def create_box_plot(
        self,
        data: dict[str, list[float]],
        title: str = "Distribution Comparison",
        ylabel: str = "Value",
    ) -> bytes:
        """Create box plot comparing distributions across categories"""
        fig = go.Figure()

        for i, (label, values) in enumerate(data.items()):
            color = self.COUNTRY_COLORS[i % len(self.COUNTRY_COLORS)]
            fig.add_trace(
                go.Box(
                    y=values,
                    name=label,
                    marker_color=color,
                    boxmean=True,
                    opacity=0.7,
                )
            )

        layout = self._get_layout(title, "", ylabel)
        fig.update_layout(**layout)

        return self._save_chart(fig, f"boxplot_{self.chart_counter}.png")

    def create_ranking_horizontal_bar(
        self,
        rankings: dict[str, float],
        title: str = "Country Rankings",
        xlabel: str = "Score",
        highlight_best: bool = True,
    ) -> bytes:
        """Create horizontal bar chart for rankings"""
        # Sort by value
        sorted_items = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        # Color by rank
        colors = []
        for i, _v in enumerate(values):
            if i == 0 and highlight_best:
                colors.append(self.COLORS["positive"])
            elif i == len(values) - 1:
                colors.append(self.COLORS["negative"])
            else:
                colors.append(self.COLORS["primary"])

        # Add rank numbers to labels
        labels_with_rank = [f"#{i+1} {label}" for i, label in enumerate(labels)]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=labels_with_rank,
                x=values,
                orientation="h",
                marker_color=colors,
                opacity=0.85,
                text=[f"{v:.1f}" for v in values],
                textposition="outside",
                textfont={"weight": "bold"},
            )
        )

        layout = self._get_layout(title, xlabel, "")
        layout["yaxis"]["autorange"] = "reversed"
        layout["height"] = max(400, len(rankings) * 30)
        fig.update_layout(**layout)

        return self._save_chart(fig, f"ranking_{self.chart_counter}.png")

    def create_regime_comparison_chart(
        self,
        country_data: dict[str, dict[str, Any]],
        title: str = "Economic Outcomes by Regime Type",
    ) -> bytes:
        """Create chart comparing outcomes grouped by regime type"""
        import numpy as np

        # Group countries by regime
        regime_groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for country, data in country_data.items():
            regime = data.get("regime", "unknown")
            if regime not in regime_groups:
                regime_groups[regime] = []
            regime_groups[regime].append((country, data))

        # Metrics to compare
        metrics = ["freedom_index", "inflation", "unemployment"]
        metric_labels = ["Freedom Index", "Inflation (%)", "Unemployment (%)"]

        fig = make_subplots(rows=1, cols=3, subplot_titles=metric_labels)

        for idx, (metric, _label) in enumerate(zip(metrics, metric_labels, strict=False)):
            regimes = list(regime_groups.keys())
            avg_values = []

            for regime in regimes:
                countries_in_regime = regime_groups[regime]
                values = [float(cd[1].get(metric, 0)) for cd in countries_in_regime]
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
            regimes_sorted, avg_sorted = (
                zip(*sorted_data, strict=False) if sorted_data else ([], [])
            )

            # Color by regime type
            colors = []
            for r in regimes_sorted:
                r_lower = r.lower() if isinstance(r, str) else str(r)
                if "ancap" in r_lower or "minarch" in r_lower:
                    colors.append(self.COLORS["positive"])
                elif "monarch" in r_lower:
                    colors.append(self.COLORS["highlight"])
                elif "liberal" in r_lower:
                    colors.append(self.COLORS["primary"])
                elif "socialist" in r_lower:
                    colors.append(self.COLORS["secondary"])
                else:
                    colors.append(self.COLORS["negative"])

            fig.add_trace(
                go.Bar(
                    y=[r.replace("_", " ").title() for r in regimes_sorted],
                    x=list(avg_sorted),
                    orientation="h",
                    marker_color=colors,
                    opacity=0.85,
                    text=[f"{v:.1f}" for v in avg_sorted],
                    textposition="outside",
                    showlegend=False,
                ),
                row=1,
                col=idx + 1,
            )

        fig.update_layout(
            title={
                "text": title,
                "font": {"size": 16, "family": "Times New Roman, serif"},
                "x": 0.5,
            },
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=500,
        )

        for i in range(1, 4):
            fig.update_yaxes(autorange="reversed", row=1, col=i)

        return self._save_chart(fig, f"regime_comparison_{self.chart_counter}.png")

    def create_multi_country_evolution_chart(
        self, country_metrics: dict[str, list[float]], metric_name: str, title: str
    ) -> bytes:
        """Create line chart showing metric evolution for multiple countries"""
        fig = go.Figure()

        for i, (country, values) in enumerate(country_metrics.items()):
            months = list(range(len(values)))
            color = self.COUNTRY_COLORS[i % len(self.COUNTRY_COLORS)]

            fig.add_trace(
                go.Scatter(
                    x=months,
                    y=values,
                    mode="lines+markers",
                    line={"color": color, "width": 2},
                    marker={"size": 6, "color": color},
                    name=country,
                )
            )

        layout = self._get_layout(title, "Month", metric_name.replace("_", " ").title())
        fig.update_layout(**layout)

        return self._save_chart(fig, f"multi_country_{self.chart_counter}.png")

    def create_war_risk_chart(
        self, war_risks: list[dict[str, Any]], title: str = "War Risk Assessment"
    ) -> bytes:
        """Create horizontal bar chart of war risk"""
        if not war_risks:
            fig = go.Figure()
            fig.add_annotation(
                x=0.5,
                y=0.5,
                text="No significant war risks detected",
                showarrow=False,
                font={"size": 14, "color": self.COLORS["neutral"]},
                xref="paper",
                yref="paper",
            )
            fig.update_layout(
                xaxis={"visible": False},
                yaxis={"visible": False},
                paper_bgcolor="white",
                plot_bgcolor="white",
            )
            return self._save_chart(fig, f"war_risk_{self.chart_counter}.png")

        pairs = [f"{r['countries'][0]}-{r['countries'][1]}" for r in war_risks[:8]]
        probs = [float(r["probability"]) * 100 for r in war_risks[:8]]

        colors = []
        for p in probs:
            if p >= 10:
                colors.append(self.COLORS["negative"])
            elif p >= 5:
                colors.append(self.COLORS["secondary"])
            else:
                colors.append(self.COLORS["highlight"])

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=pairs,
                x=probs,
                orientation="h",
                marker_color=colors,
                opacity=0.85,
                text=[f"{p:.1f}%" for p in probs],
                textposition="outside",
            )
        )

        layout = self._get_layout(title, "War Probability (%)", "")
        layout["yaxis"]["autorange"] = "reversed"
        fig.update_layout(**layout)

        return self._save_chart(fig, f"war_risk_{self.chart_counter}.png")

    # ===========================================
    # DASHBOARD
    # ===========================================

    def create_simulation_dashboard(
        self, metrics_history: list[Any], title: str = "Economic Dashboard"
    ) -> bytes:
        """Create comprehensive 6-panel dashboard"""
        months = list(range(len(metrics_history)))
        btc_prices = [float(m.bitcoin_price) for m in metrics_history]
        gold_prices = [float(m.gold_price) for m in metrics_history]
        inflation_rates = [
            m.inflation_rate * 100 if m.inflation_rate < 1 else m.inflation_rate
            for m in metrics_history
        ]
        unemployment_rates = [
            m.unemployment_rate * 100 if m.unemployment_rate < 1 else m.unemployment_rate
            for m in metrics_history
        ]
        freedom_indices = [m.freedom_index for m in metrics_history]
        cb_damage = [float(m.central_bank_damage) for m in metrics_history]
        gov_damage = [float(m.government_damage) for m in metrics_history]

        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=[
                "Bitcoin Price",
                "Gold Price",
                "Inflation Rate",
                "Unemployment Rate",
                "Freedom Index",
                "Intervention Damage",
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        # Panel 1: Bitcoin Price
        fig.add_trace(
            go.Scatter(
                x=months,
                y=btc_prices,
                fill="tozeroy",
                fillcolor="rgba(247, 147, 26, 0.15)",
                line={"color": self.COLORS["bitcoin"], "width": 2},
                mode="lines",
                name="Bitcoin",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Panel 2: Gold Price
        fig.add_trace(
            go.Scatter(
                x=months,
                y=gold_prices,
                fill="tozeroy",
                fillcolor="rgba(207, 181, 59, 0.15)",
                line={"color": self.COLORS["gold"], "width": 2},
                mode="lines",
                name="Gold",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Panel 3: Inflation Rate
        colors_infl = [
            self.COLORS["negative"] if r > 5 else self.COLORS["secondary"] if r > 2 else self.COLORS["positive"]
            for r in inflation_rates
        ]
        fig.add_trace(
            go.Bar(
                x=months,
                y=inflation_rates,
                marker_color=colors_infl,
                opacity=0.85,
                name="Inflation",
                showlegend=False,
            ),
            row=1,
            col=3,
        )
        fig.add_hline(y=2, line_dash="dash", line_color=self.COLORS["neutral"], line_width=1, opacity=0.7, row=1, col=3)

        # Panel 4: Unemployment
        fig.add_trace(
            go.Scatter(
                x=months,
                y=unemployment_rates,
                fill="tozeroy",
                fillcolor="rgba(196, 44, 44, 0.3)",
                line={"color": self.COLORS["negative"], "width": 2},
                mode="lines",
                name="Unemployment",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Panel 5: Freedom Index
        colors_free = [
            self.COLORS["positive"] if f >= 70 else self.COLORS["highlight"] if f >= 50 else self.COLORS["negative"]
            for f in freedom_indices
        ]
        fig.add_trace(
            go.Bar(
                x=months,
                y=freedom_indices,
                marker_color=colors_free,
                opacity=0.85,
                name="Freedom",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        fig.add_hline(y=50, line_dash="dash", line_color=self.COLORS["neutral"], line_width=1, opacity=0.7, row=2, col=2)

        # Panel 6: Cumulative Damage (stacked)
        fig.add_trace(
            go.Scatter(
                x=months,
                y=cb_damage,
                fill="tozeroy",
                fillcolor="rgba(196, 44, 44, 0.8)",
                line={"color": self.COLORS["negative"], "width": 0},
                mode="lines",
                name="Central Bank",
                stackgroup="damage",
            ),
            row=2,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=months,
                y=gov_damage,
                fill="tonexty",
                fillcolor="rgba(227, 114, 34, 0.8)",
                line={"color": self.COLORS["secondary"], "width": 0},
                mode="lines",
                name="Government",
                stackgroup="damage",
            ),
            row=2,
            col=3,
        )

        fig.update_layout(
            title={
                "text": title,
                "font": {"size": 18, "family": "Times New Roman, serif"},
                "x": 0.5,
            },
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=700,
            width=1200,
            showlegend=True,
            legend={"x": 0.85, "y": 0.15, "bgcolor": "rgba(255,255,255,0.9)"},
        )

        # Update all axes
        for i in range(1, 3):
            for j in range(1, 4):
                fig.update_xaxes(
                    title_text="Month",
                    showgrid=True,
                    gridcolor="#e0e0e0",
                    row=i,
                    col=j,
                )
                fig.update_yaxes(
                    showgrid=True,
                    gridcolor="#e0e0e0",
                    row=i,
                    col=j,
                )

        # Set specific y-axis labels
        fig.update_yaxes(title_text="USD", row=1, col=1)
        fig.update_yaxes(title_text="USD/oz", row=1, col=2)
        fig.update_yaxes(title_text="%", row=1, col=3)
        fig.update_yaxes(title_text="%", rangemode="tozero", row=2, col=1)
        fig.update_yaxes(title_text="Index", range=[0, 100], row=2, col=2)
        fig.update_yaxes(title_text="USD", row=2, col=3)

        return self._save_chart(fig, f"dashboard_{self.chart_counter}.png")
