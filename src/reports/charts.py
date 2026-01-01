"""
Chart Generation Module

Creates matplotlib charts for economic simulation reports.
All charts follow a consistent scientific style.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import io


# Configure matplotlib for consistent styling
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    # Fallback for older matplotlib versions
    plt.style.use('ggplot')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 100,  # Lower DPI for PDF compatibility
})


class ChartGenerator:
    """
    Generates charts for economic simulation reports.

    All charts are designed for scientific publication:
    - Clear labels and titles
    - Consistent color scheme
    - Austrian Economics annotations
    """

    # Color scheme for consistent charts
    COLORS = {
        'primary': '#1f77b4',      # Blue
        'secondary': '#ff7f0e',    # Orange
        'tertiary': '#2ca02c',     # Green
        'quaternary': '#d62728',   # Red
        'bitcoin': '#f7931a',      # Bitcoin orange
        'gold': '#ffd700',         # Gold
        'inflation': '#dc143c',    # Crimson
        'freedom': '#228b22',      # Forest green
        'damage': '#8b0000',       # Dark red
        'neutral': '#7f7f7f',      # Gray
    }

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("reports/charts")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chart_counter = 0

    def _get_months_axis(self, num_months: int, start_date: Optional[datetime] = None) -> List[str]:
        """Generate month labels for x-axis"""
        if start_date is None:
            start_date = datetime.now()

        labels = []
        for i in range(num_months):
            date = start_date + timedelta(days=30*i)
            labels.append(date.strftime('%b %Y'))
        return labels

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

    def _save_or_return(self, fig: Figure, filename: str) -> bytes:
        """Save chart to file and return bytes"""
        # Save to bytes buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        chart_bytes = buf.getvalue()

        # Also save to file
        filepath = self.output_dir / filename
        fig.savefig(filepath, format='png', dpi=150, bbox_inches='tight')

        plt.close(fig)
        self.chart_counter += 1

        return chart_bytes

    # ===========================================
    # PRICE EVOLUTION CHARTS
    # ===========================================

    def create_bitcoin_price_chart(
        self,
        prices: List[Decimal],
        title: str = "Bitcoin Price Evolution",
        annotations: Optional[Dict[int, str]] = None
    ) -> bytes:
        """
        Create Bitcoin price evolution chart.

        Austrian Theory Note:
        - Bitcoin as sound money (fixed supply)
        - Price reflects dollar debasement
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        prices_float = self._to_float_list(prices)
        months = range(len(prices_float))

        ax.plot(months, prices_float, color=self.COLORS['bitcoin'], linewidth=2, label='BTC/USD')
        ax.fill_between(months, prices_float, alpha=0.3, color=self.COLORS['bitcoin'])

        ax.set_xlabel('Month')
        ax.set_ylabel('Price (USD)')
        ax.set_title(title)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Add annotations if provided
        if annotations:
            for month, text in annotations.items():
                if month < len(prices_float):
                    ax.annotate(text, xy=(month, prices_float[month]),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=8, arrowprops=dict(arrowstyle='->', color='gray'))

        # Add Austrian note
        ax.text(0.02, 0.98, "Austrian View: BTC price = USD debasement measure",
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                style='italic', color='gray')

        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        return self._save_or_return(fig, f'bitcoin_price_{self.chart_counter}.png')

    def create_gold_price_chart(
        self,
        prices: List[Decimal],
        title: str = "Gold Price Evolution"
    ) -> bytes:
        """Create Gold price evolution chart"""
        fig, ax = plt.subplots(figsize=(10, 5))

        prices_float = self._to_float_list(prices)
        months = range(len(prices_float))

        ax.plot(months, prices_float, color=self.COLORS['gold'], linewidth=2, label='Gold/USD')
        ax.fill_between(months, prices_float, alpha=0.3, color=self.COLORS['gold'])

        ax.set_xlabel('Month')
        ax.set_ylabel('Price (USD per oz)')
        ax.set_title(title)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        ax.text(0.02, 0.98, "Austrian View: Gold as traditional inflation hedge",
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                style='italic', color='gray')

        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        return self._save_or_return(fig, f'gold_price_{self.chart_counter}.png')

    def create_price_comparison_chart(
        self,
        btc_prices: List[Decimal],
        gold_prices: List[Decimal],
        title: str = "Hard Assets vs Fiat"
    ) -> bytes:
        """Create comparative chart of BTC and Gold normalized to starting value"""
        fig, ax = plt.subplots(figsize=(10, 5))

        btc = self._to_float_list(btc_prices)
        gold = self._to_float_list(gold_prices)

        # Normalize to 100 at start
        btc_norm = [100 * p / btc[0] if btc[0] > 0 else 100 for p in btc]
        gold_norm = [100 * p / gold[0] if gold[0] > 0 else 100 for p in gold]

        months = range(len(btc_norm))

        ax.plot(months, btc_norm, color=self.COLORS['bitcoin'], linewidth=2, label='Bitcoin')
        ax.plot(months, gold_norm, color=self.COLORS['gold'], linewidth=2, label='Gold')
        ax.axhline(y=100, color=self.COLORS['neutral'], linestyle='--', alpha=0.5, label='Starting Value')

        ax.set_xlabel('Month')
        ax.set_ylabel('Normalized Value (Start = 100)')
        ax.set_title(title)

        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        return self._save_or_return(fig, f'price_comparison_{self.chart_counter}.png')

    # ===========================================
    # INFLATION CHARTS
    # ===========================================

    def create_inflation_chart(
        self,
        inflation_rates: List[float],
        title: str = "Inflation Rate Evolution (Annualized)"
    ) -> bytes:
        """
        Create inflation rate evolution chart.

        Austrian Theory:
        - Inflation is ALWAYS monetary
        - CPI understates true inflation
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        months = range(len(inflation_rates))

        # Color bars by intensity
        colors = [self.COLORS['inflation'] if r > 0.05 else
                  self.COLORS['secondary'] if r > 0.02 else
                  self.COLORS['primary'] for r in inflation_rates]

        ax.bar(months, [r * 100 for r in inflation_rates], color=colors, alpha=0.7)
        ax.axhline(y=2, color=self.COLORS['neutral'], linestyle='--', alpha=0.7, label='CB Target (2%)')

        ax.set_xlabel('Month')
        ax.set_ylabel('Inflation Rate (%)')
        ax.set_title(title)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))

        ax.text(0.02, 0.98, "Austrian View: Real inflation >> CPI (monetary expansion)",
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                style='italic', color='gray')

        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        return self._save_or_return(fig, f'inflation_{self.chart_counter}.png')

    def create_money_supply_chart(
        self,
        money_supply: List[Decimal],
        credit_expansion: List[Decimal],
        title: str = "Money Supply and Credit Expansion"
    ) -> bytes:
        """
        Create money supply and credit expansion chart.

        Austrian Theory:
        - Credit expansion causes boom-bust cycles
        - Bank money creation multiplies base money
        """
        fig, ax1 = plt.subplots(figsize=(10, 5))

        money = self._to_float_list(money_supply)
        credit = self._to_float_list(credit_expansion)
        months = range(len(money))

        # Money supply on left axis
        color1 = self.COLORS['primary']
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Base Money (USD)', color=color1)
        ax1.plot(months, money, color=color1, linewidth=2, label='Base Money')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e12:.1f}T'))

        # Credit on right axis
        ax2 = ax1.twinx()
        color2 = self.COLORS['secondary']
        ax2.set_ylabel('Credit Created (USD)', color=color2)
        ax2.plot(months, credit, color=color2, linewidth=2, linestyle='--', label='Credit')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e9:.0f}B'))

        ax1.set_title(title)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax1.text(0.02, 0.02, "Austrian View: Credit expansion → Malinvestment → Bust",
                transform=ax1.transAxes, fontsize=8, verticalalignment='bottom',
                style='italic', color='gray')

        return self._save_or_return(fig, f'money_supply_{self.chart_counter}.png')

    # ===========================================
    # EMPLOYMENT CHARTS
    # ===========================================

    def create_unemployment_chart(
        self,
        unemployment_rates: List[float],
        title: str = "Unemployment Rate Evolution"
    ) -> bytes:
        """Create unemployment rate evolution chart"""
        fig, ax = plt.subplots(figsize=(10, 5))

        months = range(len(unemployment_rates))
        rates_pct = [r * 100 for r in unemployment_rates]

        ax.plot(months, rates_pct, color=self.COLORS['quaternary'], linewidth=2)
        ax.fill_between(months, rates_pct, alpha=0.3, color=self.COLORS['quaternary'])

        ax.set_xlabel('Month')
        ax.set_ylabel('Unemployment Rate (%)')
        ax.set_title(title)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))

        ax.text(0.02, 0.98, "Austrian View: Natural rate reflects market clearing; \ngovernment intervention raises it",
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                style='italic', color='gray')

        ax.grid(True, alpha=0.3)

        return self._save_or_return(fig, f'unemployment_{self.chart_counter}.png')

    # ===========================================
    # DAMAGE AND FREEDOM CHARTS
    # ===========================================

    def create_damage_chart(
        self,
        cb_damage: List[Decimal],
        gov_damage: List[Decimal],
        title: str = "Cumulative Intervention Damage"
    ) -> bytes:
        """
        Create stacked area chart of intervention damage.

        Austrian Theory:
        - All intervention causes deadweight loss
        - Damage compounds over time
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        cb = self._to_float_list(cb_damage)
        gov = self._to_float_list(gov_damage)
        months = range(len(cb))

        ax.stackplot(months, cb, gov,
                    labels=['Central Bank Damage', 'Government Damage'],
                    colors=[self.COLORS['damage'], self.COLORS['secondary']],
                    alpha=0.7)

        ax.set_xlabel('Month')
        ax.set_ylabel('Cumulative Damage (USD)')
        ax.set_title(title)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e9:.1f}B'))

        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        ax.text(0.98, 0.02, "Austrian View: Every intervention destroys value",
                transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
                horizontalalignment='right', style='italic', color='gray')

        return self._save_or_return(fig, f'damage_{self.chart_counter}.png')

    def create_freedom_index_chart(
        self,
        freedom_indices: List[float],
        title: str = "Economic Freedom Index Evolution"
    ) -> bytes:
        """Create freedom index evolution chart"""
        fig, ax = plt.subplots(figsize=(10, 5))

        months = range(len(freedom_indices))

        # Color gradient based on freedom level
        colors = [self.COLORS['freedom'] if f >= 70 else
                  self.COLORS['primary'] if f >= 50 else
                  self.COLORS['secondary'] if f >= 30 else
                  self.COLORS['damage'] for f in freedom_indices]

        ax.bar(months, freedom_indices, color=colors, alpha=0.8)

        # Reference lines
        ax.axhline(y=90, color=self.COLORS['freedom'], linestyle='--', alpha=0.5, label='Excellent (90+)')
        ax.axhline(y=50, color=self.COLORS['neutral'], linestyle='--', alpha=0.5, label='Moderate (50)')

        ax.set_xlabel('Month')
        ax.set_ylabel('Freedom Index')
        ax.set_title(title)
        ax.set_ylim(0, 100)

        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        return self._save_or_return(fig, f'freedom_{self.chart_counter}.png')

    # ===========================================
    # MULTI-COUNTRY CHARTS
    # ===========================================

    def create_country_comparison_chart(
        self,
        country_data: Dict[str, Dict[str, Any]],
        metric: str,
        title: str
    ) -> bytes:
        """Create bar chart comparing countries on a metric"""
        fig, ax = plt.subplots(figsize=(12, 6))

        countries = list(country_data.keys())
        values = [country_data[c].get(metric, 0) for c in countries]

        # Sort by value
        sorted_pairs = sorted(zip(countries, values), key=lambda x: x[1], reverse=True)
        countries, values = zip(*sorted_pairs)

        colors = [self.COLORS['freedom'] if v >= 70 else
                  self.COLORS['primary'] if v >= 50 else
                  self.COLORS['secondary'] if v >= 30 else
                  self.COLORS['damage'] for v in values]

        bars = ax.barh(countries, values, color=colors, alpha=0.8)

        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_title(title)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}', va='center', fontsize=8)

        ax.grid(True, alpha=0.3, axis='x')

        return self._save_or_return(fig, f'country_comparison_{self.chart_counter}.png')

    def create_multi_country_evolution_chart(
        self,
        country_metrics: Dict[str, List[float]],
        metric_name: str,
        title: str
    ) -> bytes:
        """Create line chart showing metric evolution for multiple countries"""
        fig, ax = plt.subplots(figsize=(12, 6))

        color_cycle = list(self.COLORS.values())

        for i, (country, values) in enumerate(country_metrics.items()):
            color = color_cycle[i % len(color_cycle)]
            months = range(len(values))
            ax.plot(months, values, color=color, linewidth=1.5, label=country, alpha=0.8)

        ax.set_xlabel('Month')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(title)

        ax.legend(loc='upper left', ncol=4, fontsize=8)
        ax.grid(True, alpha=0.3)

        return self._save_or_return(fig, f'multi_country_{self.chart_counter}.png')

    def create_war_risk_chart(
        self,
        war_risks: List[Dict[str, Any]],
        title: str = "War Risk Assessment"
    ) -> bytes:
        """Create horizontal bar chart of war risk between country pairs"""
        fig, ax = plt.subplots(figsize=(10, 6))

        if not war_risks:
            ax.text(0.5, 0.5, "No significant war risks detected",
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            return self._save_or_return(fig, f'war_risk_{self.chart_counter}.png')

        pairs = [f"{r['countries'][0]}-{r['countries'][1]}" for r in war_risks[:10]]
        probs = [r['probability'] * 100 for r in war_risks[:10]]

        colors = [self.COLORS['damage'] if p >= 10 else
                  self.COLORS['secondary'] if p >= 5 else
                  self.COLORS['primary'] for p in probs]

        bars = ax.barh(pairs, probs, color=colors, alpha=0.8)

        ax.set_xlabel('War Probability (%)')
        ax.set_title(title)

        for bar, prob in zip(bars, probs):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                   f'{prob:.1f}%', va='center', fontsize=8)

        ax.grid(True, alpha=0.3, axis='x')

        return self._save_or_return(fig, f'war_risk_{self.chart_counter}.png')

    # ===========================================
    # BUSINESS CYCLE CHART
    # ===========================================

    def create_business_cycle_chart(
        self,
        phases: List[str],
        boom_intensity: List[float],
        title: str = "Austrian Business Cycle Phases"
    ) -> bytes:
        """
        Create business cycle visualization.

        Austrian Theory:
        - Boom caused by credit expansion
        - Bust is healthy correction of malinvestment
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        months = range(len(phases))

        # Color by phase
        phase_colors = {
            'boom': self.COLORS['quaternary'],
            'bust': self.COLORS['damage'],
            'recovery': self.COLORS['tertiary'],
            'trough': self.COLORS['neutral'],
        }

        colors = [phase_colors.get(p.lower(), self.COLORS['neutral']) for p in phases]

        ax.bar(months, boom_intensity, color=colors, alpha=0.7)
        ax.axhline(y=0.5, color=self.COLORS['neutral'], linestyle='--', alpha=0.5)

        ax.set_xlabel('Month')
        ax.set_ylabel('Cycle Intensity')
        ax.set_title(title)
        ax.set_ylim(0, 1)

        # Add phase labels
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=p.title())
                          for p, c in phase_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.text(0.02, 0.98, "Austrian View: Credit expansion → Boom → Malinvestment → Bust (correction)",
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                style='italic', color='gray')

        ax.grid(True, alpha=0.3)

        return self._save_or_return(fig, f'business_cycle_{self.chart_counter}.png')

    # ===========================================
    # GDP AND ECONOMIC GROWTH
    # ===========================================

    def create_gdp_chart(
        self,
        gdp_values: List[Decimal],
        title: str = "GDP Evolution"
    ) -> bytes:
        """Create GDP evolution chart"""
        fig, ax = plt.subplots(figsize=(10, 5))

        gdp = self._to_float_list(gdp_values)
        months = range(len(gdp))

        ax.plot(months, gdp, color=self.COLORS['tertiary'], linewidth=2)
        ax.fill_between(months, gdp, alpha=0.3, color=self.COLORS['tertiary'])

        ax.set_xlabel('Month')
        ax.set_ylabel('GDP (USD)')
        ax.set_title(title)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e12:.2f}T'))

        ax.grid(True, alpha=0.3)

        return self._save_or_return(fig, f'gdp_{self.chart_counter}.png')

    # ===========================================
    # COMPREHENSIVE DASHBOARD
    # ===========================================

    def create_simulation_dashboard(
        self,
        metrics_history: List[Any],
        title: str = "Simulation Dashboard"
    ) -> bytes:
        """Create a comprehensive 6-panel dashboard of key metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # Extract data from metrics history
        months = list(range(len(metrics_history)))
        btc_prices = [float(m.bitcoin_price) for m in metrics_history]
        gold_prices = [float(m.gold_price) for m in metrics_history]
        inflation_rates = [m.inflation_rate * 100 for m in metrics_history]
        unemployment_rates = [m.unemployment_rate * 100 for m in metrics_history]
        freedom_indices = [m.freedom_index for m in metrics_history]
        cb_damage = [float(m.central_bank_damage) for m in metrics_history]
        gov_damage = [float(m.government_damage) for m in metrics_history]

        # Panel 1: Bitcoin Price
        ax = axes[0, 0]
        ax.plot(months, btc_prices, color=self.COLORS['bitcoin'], linewidth=2)
        ax.set_title('Bitcoin Price')
        ax.set_ylabel('USD')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax.grid(True, alpha=0.3)

        # Panel 2: Gold Price
        ax = axes[0, 1]
        ax.plot(months, gold_prices, color=self.COLORS['gold'], linewidth=2)
        ax.set_title('Gold Price')
        ax.set_ylabel('USD/oz')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax.grid(True, alpha=0.3)

        # Panel 3: Inflation Rate
        ax = axes[0, 2]
        colors = [self.COLORS['damage'] if r > 5 else self.COLORS['secondary'] if r > 2 else self.COLORS['primary']
                  for r in inflation_rates]
        ax.bar(months, inflation_rates, color=colors, alpha=0.7)
        ax.axhline(y=2, color='gray', linestyle='--', alpha=0.7)
        ax.set_title('Inflation Rate (Annualized)')
        ax.set_ylabel('%')
        ax.grid(True, alpha=0.3, axis='y')

        # Panel 4: Unemployment
        ax = axes[1, 0]
        ax.plot(months, unemployment_rates, color=self.COLORS['quaternary'], linewidth=2)
        ax.fill_between(months, unemployment_rates, alpha=0.3, color=self.COLORS['quaternary'])
        ax.set_title('Unemployment Rate')
        ax.set_ylabel('%')
        ax.set_xlabel('Month')
        ax.grid(True, alpha=0.3)

        # Panel 5: Freedom Index
        ax = axes[1, 1]
        colors = [self.COLORS['freedom'] if f >= 70 else self.COLORS['primary'] if f >= 50 else self.COLORS['damage']
                  for f in freedom_indices]
        ax.bar(months, freedom_indices, color=colors, alpha=0.8)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
        ax.set_title('Freedom Index')
        ax.set_ylabel('Index (0-100)')
        ax.set_xlabel('Month')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')

        # Panel 6: Cumulative Damage
        ax = axes[1, 2]
        total_damage = [c + g for c, g in zip(cb_damage, gov_damage)]
        ax.stackplot(months, cb_damage, gov_damage,
                    labels=['Central Bank', 'Government'],
                    colors=[self.COLORS['damage'], self.COLORS['secondary']],
                    alpha=0.7)
        ax.set_title('Cumulative Intervention Damage')
        ax.set_ylabel('USD')
        ax.set_xlabel('Month')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e9:.0f}B'))
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        return self._save_or_return(fig, f'dashboard_{self.chart_counter}.png')
