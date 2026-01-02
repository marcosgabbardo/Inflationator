"""
Chart Generation Module - Publication Quality

Creates high-quality matplotlib charts suitable for academic publications.
Style inspired by The Economist, Financial Times, and academic journals.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import io


def setup_publication_style():
    """Configure matplotlib for publication-quality output"""
    plt.rcParams.update({
        # Figure
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'figure.dpi': 150,
        'figure.figsize': (10, 5),

        # Fonts - use serif for academic look
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 11,

        # Axes
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.8,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.titlepad': 12,
        'axes.labelsize': 10,
        'axes.labelweight': 'normal',
        'axes.labelpad': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.prop_cycle': plt.cycler(color=[
            '#1a476f',  # Dark blue
            '#e37222',  # Orange
            '#2f7e44',  # Green
            '#c42c2c',  # Red
            '#6e4c9e',  # Purple
            '#d4a100',  # Gold
            '#17becf',  # Cyan
            '#e377c2',  # Pink
        ]),

        # Grid
        'axes.grid': True,
        'axes.grid.which': 'major',
        'grid.color': '#e0e0e0',
        'grid.linewidth': 0.5,
        'grid.linestyle': '-',

        # Ticks
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.color': '#333333',
        'ytick.color': '#333333',

        # Legend
        'legend.frameon': False,
        'legend.fontsize': 9,
        'legend.loc': 'best',

        # Lines
        'lines.linewidth': 2,
        'lines.markersize': 6,

        # Savefig
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


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
        'primary': '#1a476f',      # Dark blue (main data)
        'secondary': '#e37222',    # Orange (comparison)
        'positive': '#2f7e44',     # Green (gains)
        'negative': '#c42c2c',     # Red (losses)
        'highlight': '#d4a100',    # Gold (highlights)
        'neutral': '#666666',      # Gray
        'light': '#e0e0e0',        # Light gray
        'bitcoin': '#f7931a',      # Bitcoin orange
        'gold': '#cfb53b',         # Metallic gold
    }

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("reports/charts")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chart_counter = 0

    def _to_float_list(self, data: List[Any]) -> List[float]:
        """Convert list of Decimals/Any to floats"""
        result = []
        for v in data:
            if isinstance(v, Decimal):
                result.append(float(v))
            elif isinstance(v, (int, float)):
                result.append(float(v))
            else:
                result.append(0.0)
        return result

    def _format_currency(self, x, pos):
        """Format axis values as currency"""
        if abs(x) >= 1e9:
            return f'${x/1e9:.0f}B'
        elif abs(x) >= 1e6:
            return f'${x/1e6:.0f}M'
        elif abs(x) >= 1e3:
            return f'${x/1e3:.0f}K'
        else:
            return f'${x:,.0f}'

    def _format_currency_full(self, x, pos):
        """Format axis values as currency with more precision"""
        if abs(x) >= 1e12:
            return f'${x/1e12:.1f}T'
        elif abs(x) >= 1e9:
            return f'${x/1e9:.1f}B'
        elif abs(x) >= 1e6:
            return f'${x/1e6:.1f}M'
        elif abs(x) >= 1e3:
            return f'${x/1e3:.1f}K'
        else:
            return f'${x:,.0f}'

    def _save_chart(self, fig: Figure, filename: str) -> bytes:
        """Save chart and return bytes"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        chart_bytes = buf.getvalue()

        # Also save to file
        filepath = self.output_dir / filename
        fig.savefig(filepath, format='png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')

        plt.close(fig)
        self.chart_counter += 1

        return chart_bytes

    def _add_source_annotation(self, ax: Axes, source: str = "Inflationator Simulation"):
        """Add source annotation to chart"""
        ax.annotate(
            f'Source: {source}',
            xy=(0, -0.12), xycoords='axes fraction',
            fontsize=8, color=self.COLORS['neutral'],
            style='italic'
        )

    # ===========================================
    # PRICE CHARTS
    # ===========================================

    def create_bitcoin_price_chart(
        self,
        prices: List[Decimal],
        title: str = "Bitcoin Price"
    ) -> bytes:
        """Create professional Bitcoin price chart"""
        fig, ax = plt.subplots(figsize=(10, 5))

        prices_float = self._to_float_list(prices)
        months = range(len(prices_float))

        # Main line
        ax.plot(months, prices_float, color=self.COLORS['bitcoin'],
               linewidth=2.5, label='BTC/USD', zorder=3)

        # Fill under line (gradient effect)
        ax.fill_between(months, prices_float, alpha=0.15,
                       color=self.COLORS['bitcoin'], zorder=2)

        # Add start and end annotations
        if len(prices_float) > 1:
            ax.annotate(f'${prices_float[0]:,.0f}',
                       xy=(0, prices_float[0]),
                       xytext=(-30, 10), textcoords='offset points',
                       fontsize=9, color=self.COLORS['neutral'])
            ax.annotate(f'${prices_float[-1]:,.0f}',
                       xy=(len(prices_float)-1, prices_float[-1]),
                       xytext=(5, 10), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       color=self.COLORS['bitcoin'])

        ax.set_xlabel('Month')
        ax.set_ylabel('Price (USD)')
        ax.set_title(title)

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(self._format_currency))

        ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
        self._add_source_annotation(ax)

        plt.tight_layout()
        return self._save_chart(fig, f'bitcoin_price_{self.chart_counter}.png')

    def create_gold_price_chart(
        self,
        prices: List[Decimal],
        title: str = "Gold Price"
    ) -> bytes:
        """Create professional Gold price chart"""
        fig, ax = plt.subplots(figsize=(10, 5))

        prices_float = self._to_float_list(prices)
        months = range(len(prices_float))

        ax.plot(months, prices_float, color=self.COLORS['gold'],
               linewidth=2.5, label='Gold/USD (oz)', zorder=3)
        ax.fill_between(months, prices_float, alpha=0.15,
                       color=self.COLORS['gold'], zorder=2)

        ax.set_xlabel('Month')
        ax.set_ylabel('Price (USD per oz)')
        ax.set_title(title)

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(self._format_currency))
        ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
        self._add_source_annotation(ax)

        plt.tight_layout()
        return self._save_chart(fig, f'gold_price_{self.chart_counter}.png')

    def create_price_comparison_chart(
        self,
        btc_prices: List[Decimal],
        gold_prices: List[Decimal],
        title: str = "Asset Performance (Indexed to 100)"
    ) -> bytes:
        """Create comparison chart with normalized values"""
        fig, ax = plt.subplots(figsize=(10, 5))

        btc = self._to_float_list(btc_prices)
        gold = self._to_float_list(gold_prices)

        # Normalize to 100
        btc_norm = [100 * p / btc[0] if btc[0] > 0 else 100 for p in btc]
        gold_norm = [100 * p / gold[0] if gold[0] > 0 else 100 for p in gold]

        months = range(len(btc_norm))

        ax.plot(months, btc_norm, color=self.COLORS['bitcoin'],
               linewidth=2.5, label='Bitcoin', zorder=3)
        ax.plot(months, gold_norm, color=self.COLORS['gold'],
               linewidth=2.5, label='Gold', zorder=3)
        ax.axhline(y=100, color=self.COLORS['neutral'], linestyle='--',
                  linewidth=1, alpha=0.7, label='Starting Value', zorder=1)

        ax.set_xlabel('Month')
        ax.set_ylabel('Index (Start = 100)')
        ax.set_title(title)

        ax.legend(loc='best', frameon=True, facecolor='white', edgecolor='none')
        self._add_source_annotation(ax)

        plt.tight_layout()
        return self._save_chart(fig, f'price_comparison_{self.chart_counter}.png')

    # ===========================================
    # MONETARY CHARTS
    # ===========================================

    def create_inflation_chart(
        self,
        inflation_rates: List[float],
        title: str = "Inflation Rate (Annualized)"
    ) -> bytes:
        """Create professional inflation bar chart"""
        fig, ax = plt.subplots(figsize=(10, 5))

        months = range(len(inflation_rates))
        rates_pct = [r * 100 if r < 1 else r for r in inflation_rates]

        # Color bars by value
        colors = []
        for r in rates_pct:
            if r > 10:
                colors.append(self.COLORS['negative'])
            elif r > 5:
                colors.append(self.COLORS['secondary'])
            elif r > 2:
                colors.append(self.COLORS['highlight'])
            else:
                colors.append(self.COLORS['positive'])

        bars = ax.bar(months, rates_pct, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)

        # Target line
        ax.axhline(y=2, color=self.COLORS['neutral'], linestyle='--',
                  linewidth=1.5, alpha=0.8, label='2% Target')

        ax.set_xlabel('Month')
        ax.set_ylabel('Inflation Rate (%)')
        ax.set_title(title)

        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
        self._add_source_annotation(ax)

        plt.tight_layout()
        return self._save_chart(fig, f'inflation_{self.chart_counter}.png')

    def create_money_supply_chart(
        self,
        money_supply: List[Decimal],
        credit_expansion: List[Decimal],
        title: str = "Money Supply & Credit"
    ) -> bytes:
        """Create dual-axis money supply chart"""
        fig, ax1 = plt.subplots(figsize=(10, 5))

        money = self._to_float_list(money_supply)
        credit = self._to_float_list(credit_expansion)
        months = range(len(money))

        # Money supply on left axis
        color1 = self.COLORS['primary']
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Base Money', color=color1)
        line1 = ax1.plot(months, money, color=color1, linewidth=2.5,
                        label='Base Money', zorder=3)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(self._format_currency_full))
        ax1.fill_between(months, money, alpha=0.1, color=color1)

        # Credit on right axis
        ax2 = ax1.twinx()
        color2 = self.COLORS['secondary']
        ax2.set_ylabel('Credit Expansion', color=color2)
        line2 = ax2.plot(months, credit, color=color2, linewidth=2.5,
                        linestyle='--', label='Credit', zorder=3)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(self._format_currency_full))

        ax1.set_title(title)

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', frameon=True,
                  facecolor='white', edgecolor='none')

        self._add_source_annotation(ax1)
        plt.tight_layout()
        return self._save_chart(fig, f'money_supply_{self.chart_counter}.png')

    # ===========================================
    # LABOR CHARTS
    # ===========================================

    def create_unemployment_chart(
        self,
        unemployment_rates: List[float],
        title: str = "Unemployment Rate"
    ) -> bytes:
        """Create unemployment rate area chart"""
        fig, ax = plt.subplots(figsize=(10, 5))

        months = range(len(unemployment_rates))
        rates_pct = [r * 100 if r < 1 else r for r in unemployment_rates]

        ax.fill_between(months, rates_pct, alpha=0.3,
                       color=self.COLORS['negative'], zorder=2)
        ax.plot(months, rates_pct, color=self.COLORS['negative'],
               linewidth=2.5, zorder=3)

        # Add markers at start and end
        if rates_pct:
            ax.scatter([0], [rates_pct[0]], color=self.COLORS['negative'],
                      s=50, zorder=4)
            ax.scatter([len(rates_pct)-1], [rates_pct[-1]], color=self.COLORS['negative'],
                      s=50, zorder=4)

        ax.set_xlabel('Month')
        ax.set_ylabel('Unemployment Rate (%)')
        ax.set_title(title)
        ax.set_ylim(bottom=0)

        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
        self._add_source_annotation(ax)

        plt.tight_layout()
        return self._save_chart(fig, f'unemployment_{self.chart_counter}.png')

    # ===========================================
    # DAMAGE CHARTS
    # ===========================================

    def create_damage_chart(
        self,
        cb_damage: List[Decimal],
        gov_damage: List[Decimal],
        title: str = "Cumulative Intervention Damage"
    ) -> bytes:
        """Create stacked area chart of intervention damage"""
        fig, ax = plt.subplots(figsize=(10, 5))

        cb = self._to_float_list(cb_damage)
        gov = self._to_float_list(gov_damage)
        months = range(len(cb))

        ax.stackplot(months, cb, gov,
                    labels=['Central Bank', 'Government'],
                    colors=[self.COLORS['negative'], self.COLORS['secondary']],
                    alpha=0.8)

        ax.set_xlabel('Month')
        ax.set_ylabel('Cumulative Damage')
        ax.set_title(title)

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(self._format_currency_full))
        ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
        self._add_source_annotation(ax)

        plt.tight_layout()
        return self._save_chart(fig, f'damage_{self.chart_counter}.png')

    def create_freedom_index_chart(
        self,
        freedom_indices: List[float],
        title: str = "Economic Freedom Index"
    ) -> bytes:
        """Create freedom index horizontal bar chart"""
        fig, ax = plt.subplots(figsize=(10, 5))

        months = range(len(freedom_indices))

        # Color by value
        colors = []
        for f in freedom_indices:
            if f >= 80:
                colors.append(self.COLORS['positive'])
            elif f >= 60:
                colors.append('#7cb342')  # Light green
            elif f >= 40:
                colors.append(self.COLORS['highlight'])
            elif f >= 20:
                colors.append(self.COLORS['secondary'])
            else:
                colors.append(self.COLORS['negative'])

        ax.bar(months, freedom_indices, color=colors, alpha=0.85,
              edgecolor='white', linewidth=0.5)

        # Reference lines
        ax.axhline(y=80, color=self.COLORS['positive'], linestyle='--',
                  linewidth=1, alpha=0.5, label='Free (80+)')
        ax.axhline(y=50, color=self.COLORS['neutral'], linestyle='--',
                  linewidth=1, alpha=0.5, label='Moderate (50)')

        ax.set_xlabel('Month')
        ax.set_ylabel('Freedom Index')
        ax.set_title(title)
        ax.set_ylim(0, 100)

        ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
        self._add_source_annotation(ax)

        plt.tight_layout()
        return self._save_chart(fig, f'freedom_{self.chart_counter}.png')

    # ===========================================
    # MULTI-COUNTRY CHARTS
    # ===========================================

    def create_country_comparison_chart(
        self,
        country_data: Dict[str, Dict[str, Any]],
        metric: str,
        title: str
    ) -> bytes:
        """Create horizontal bar chart comparing countries"""
        fig, ax = plt.subplots(figsize=(10, 6))

        countries = list(country_data.keys())
        values = [country_data[c].get(metric, 0) for c in countries]

        # Sort by value
        sorted_data = sorted(zip(countries, values), key=lambda x: x[1], reverse=True)
        countries, values = zip(*sorted_data) if sorted_data else ([], [])

        # Color gradient
        colors = [self.COLORS['primary'] if i % 2 == 0 else self.COLORS['secondary']
                 for i in range(len(values))]

        y_pos = np.arange(len(countries))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.85,
                      edgecolor='white', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(countries)
        ax.invert_yaxis()
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_title(title)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values)*0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}', va='center', fontsize=9)

        self._add_source_annotation(ax)
        plt.tight_layout()
        return self._save_chart(fig, f'country_comparison_{self.chart_counter}.png')

    def create_multi_country_evolution_chart(
        self,
        country_metrics: Dict[str, List[float]],
        metric_name: str,
        title: str
    ) -> bytes:
        """Create line chart showing metric evolution for multiple countries"""
        fig, ax = plt.subplots(figsize=(10, 5))

        for i, (country, values) in enumerate(country_metrics.items()):
            months = range(len(values))
            ax.plot(months, values, linewidth=2, label=country,
                   marker='o', markersize=4, markevery=max(1, len(values)//6))

        ax.set_xlabel('Month')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(title)

        ax.legend(loc='best', frameon=True, facecolor='white', edgecolor='none',
                 ncol=min(4, len(country_metrics)))
        self._add_source_annotation(ax)

        plt.tight_layout()
        return self._save_chart(fig, f'multi_country_{self.chart_counter}.png')

    def create_war_risk_chart(
        self,
        war_risks: List[Dict[str, Any]],
        title: str = "War Risk Assessment"
    ) -> bytes:
        """Create horizontal bar chart of war risk"""
        fig, ax = plt.subplots(figsize=(10, 5))

        if not war_risks:
            ax.text(0.5, 0.5, "No significant war risks detected",
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, color=self.COLORS['neutral'])
            ax.axis('off')
            return self._save_chart(fig, f'war_risk_{self.chart_counter}.png')

        pairs = [f"{r['countries'][0]}-{r['countries'][1]}" for r in war_risks[:8]]
        probs = [r['probability'] * 100 for r in war_risks[:8]]

        colors = []
        for p in probs:
            if p >= 10:
                colors.append(self.COLORS['negative'])
            elif p >= 5:
                colors.append(self.COLORS['secondary'])
            else:
                colors.append(self.COLORS['highlight'])

        y_pos = np.arange(len(pairs))
        bars = ax.barh(y_pos, probs, color=colors, alpha=0.85,
                      edgecolor='white', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(pairs)
        ax.invert_yaxis()
        ax.set_xlabel('War Probability (%)')
        ax.set_title(title)

        # Add value labels
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{prob:.1f}%', va='center', fontsize=9)

        self._add_source_annotation(ax)
        plt.tight_layout()
        return self._save_chart(fig, f'war_risk_{self.chart_counter}.png')

    # ===========================================
    # DASHBOARD
    # ===========================================

    def create_simulation_dashboard(
        self,
        metrics_history: List[Any],
        title: str = "Economic Dashboard"
    ) -> bytes:
        """Create comprehensive 6-panel dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        months = list(range(len(metrics_history)))
        btc_prices = [float(m.bitcoin_price) for m in metrics_history]
        gold_prices = [float(m.gold_price) for m in metrics_history]
        inflation_rates = [m.inflation_rate * 100 if m.inflation_rate < 1 else m.inflation_rate
                         for m in metrics_history]
        unemployment_rates = [m.unemployment_rate * 100 if m.unemployment_rate < 1 else m.unemployment_rate
                             for m in metrics_history]
        freedom_indices = [m.freedom_index for m in metrics_history]
        cb_damage = [float(m.central_bank_damage) for m in metrics_history]
        gov_damage = [float(m.government_damage) for m in metrics_history]

        # Panel 1: Bitcoin Price
        ax = axes[0, 0]
        ax.plot(months, btc_prices, color=self.COLORS['bitcoin'], linewidth=2)
        ax.fill_between(months, btc_prices, alpha=0.15, color=self.COLORS['bitcoin'])
        ax.set_title('Bitcoin Price', fontsize=11, fontweight='bold')
        ax.set_ylabel('USD')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(self._format_currency))
        ax.set_xlabel('Month')

        # Panel 2: Gold Price
        ax = axes[0, 1]
        ax.plot(months, gold_prices, color=self.COLORS['gold'], linewidth=2)
        ax.fill_between(months, gold_prices, alpha=0.15, color=self.COLORS['gold'])
        ax.set_title('Gold Price', fontsize=11, fontweight='bold')
        ax.set_ylabel('USD/oz')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(self._format_currency))
        ax.set_xlabel('Month')

        # Panel 3: Inflation Rate
        ax = axes[0, 2]
        colors = [self.COLORS['negative'] if r > 5 else self.COLORS['secondary'] if r > 2
                 else self.COLORS['positive'] for r in inflation_rates]
        ax.bar(months, inflation_rates, color=colors, alpha=0.85)
        ax.axhline(y=2, color=self.COLORS['neutral'], linestyle='--', linewidth=1, alpha=0.7)
        ax.set_title('Inflation Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel('%')
        ax.set_xlabel('Month')

        # Panel 4: Unemployment
        ax = axes[1, 0]
        ax.fill_between(months, unemployment_rates, alpha=0.3, color=self.COLORS['negative'])
        ax.plot(months, unemployment_rates, color=self.COLORS['negative'], linewidth=2)
        ax.set_title('Unemployment Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel('%')
        ax.set_xlabel('Month')
        ax.set_ylim(bottom=0)

        # Panel 5: Freedom Index
        ax = axes[1, 1]
        colors = [self.COLORS['positive'] if f >= 70 else self.COLORS['highlight'] if f >= 50
                 else self.COLORS['negative'] for f in freedom_indices]
        ax.bar(months, freedom_indices, color=colors, alpha=0.85)
        ax.axhline(y=50, color=self.COLORS['neutral'], linestyle='--', linewidth=1, alpha=0.7)
        ax.set_title('Freedom Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Index')
        ax.set_xlabel('Month')
        ax.set_ylim(0, 100)

        # Panel 6: Cumulative Damage
        ax = axes[1, 2]
        ax.stackplot(months, cb_damage, gov_damage,
                    labels=['Central Bank', 'Government'],
                    colors=[self.COLORS['negative'], self.COLORS['secondary']],
                    alpha=0.8)
        ax.set_title('Intervention Damage', fontsize=11, fontweight='bold')
        ax.set_ylabel('USD')
        ax.set_xlabel('Month')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(self._format_currency_full))
        ax.legend(loc='upper left', fontsize=8, frameon=True, facecolor='white', edgecolor='none')

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        return self._save_chart(fig, f'dashboard_{self.chart_counter}.png')
