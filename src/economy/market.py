"""
Market System

Implements price discovery through supply and demand.
Based on Austrian Economics price theory:
- Prices emerge from individual valuations
- No "just price" set by authority
- Prices convey information about scarcity
- Market clearing through voluntary exchange
"""

import uuid
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any


class MarketType(str, Enum):
    """Types of markets"""

    LABOR = "labor"
    CONSUMER_GOODS = "consumer_goods"
    CAPITAL_GOODS = "capital_goods"
    COMMODITIES = "commodities"
    CRYPTO = "crypto"


@dataclass
class Order:
    """A buy or sell order"""

    id: str
    agent_id: str
    is_buy: bool  # True = buy order, False = sell order
    price: Decimal  # Limit price
    quantity: Decimal
    timestamp: int  # Simulation tick

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class Trade:
    """A completed trade"""

    id: str
    buyer_id: str
    seller_id: str
    price: Decimal
    quantity: Decimal
    total_value: Decimal
    timestamp: int

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


class OrderBook:
    """
    Order book for a market.

    Austrian Theory:
    - Prices discovered through bidding process
    - Buyers and sellers reveal preferences
    - Market clears at equilibrium
    """

    def __init__(self):
        self.buy_orders: list[Order] = []  # Sorted by price DESC
        self.sell_orders: list[Order] = []  # Sorted by price ASC

    def add_buy_order(self, order: Order):
        """Add a buy order (sorted by price descending)"""
        self.buy_orders.append(order)
        self.buy_orders.sort(key=lambda o: o.price, reverse=True)

    def add_sell_order(self, order: Order):
        """Add a sell order (sorted by price ascending)"""
        self.sell_orders.append(order)
        self.sell_orders.sort(key=lambda o: o.price)

    def get_best_bid(self) -> Decimal | None:
        """Highest buy price"""
        if self.buy_orders:
            return self.buy_orders[0].price
        return None

    def get_best_ask(self) -> Decimal | None:
        """Lowest sell price"""
        if self.sell_orders:
            return self.sell_orders[0].price
        return None

    def get_spread(self) -> Decimal | None:
        """Bid-ask spread"""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid and ask:
            return ask - bid
        return None

    def match_orders(self, tick: int) -> list[Trade]:
        """
        Match buy and sell orders.

        Price discovery happens here:
        - Buyers willing to pay >= sellers' ask get matched
        - Trade happens at midpoint or ask price
        """
        trades = []

        while self.buy_orders and self.sell_orders:
            best_buy = self.buy_orders[0]
            best_sell = self.sell_orders[0]

            # Can we match?
            if best_buy.price >= best_sell.price:
                # Trade at midpoint
                trade_price = (best_buy.price + best_sell.price) / 2
                trade_quantity = min(best_buy.quantity, best_sell.quantity)

                trade = Trade(
                    id="",
                    buyer_id=best_buy.agent_id,
                    seller_id=best_sell.agent_id,
                    price=trade_price,
                    quantity=trade_quantity,
                    total_value=trade_price * trade_quantity,
                    timestamp=tick,
                )
                trades.append(trade)

                # Update order quantities
                best_buy.quantity -= trade_quantity
                best_sell.quantity -= trade_quantity

                # Remove filled orders
                if best_buy.quantity <= 0:
                    self.buy_orders.pop(0)
                if best_sell.quantity <= 0:
                    self.sell_orders.pop(0)
            else:
                # No more matches possible
                break

        return trades

    def clear(self):
        """Clear all orders"""
        self.buy_orders = []
        self.sell_orders = []


class Market:
    """
    A market for a specific good or service.

    Austrian Price Theory:
    - Prices are not set, they are discovered
    - Prices reflect subjective valuations
    - Prices signal scarcity to producers
    - Prices coordinate economic activity
    """

    def __init__(
        self,
        market_id: str | None = None,
        name: str = "Unnamed Market",
        market_type: MarketType = MarketType.CONSUMER_GOODS,
        country: str = "USA",
        initial_price: Decimal = Decimal("100"),
    ):
        self.id = market_id or str(uuid.uuid4())
        self.name = name
        self.market_type = market_type
        self.country = country

        # Price
        self.current_price = initial_price
        self.previous_price = initial_price
        self.price_history: list[Decimal] = [initial_price]

        # Volume
        self.volume = Decimal("0")
        self.volume_history: list[Decimal] = []

        # Order book
        self.order_book = OrderBook()

        # Supply and demand tracking
        self.total_supply = Decimal("0")
        self.total_demand = Decimal("0")

        # Market metrics
        self.volatility = 0.1
        self.liquidity = 1.0

        # Trade history
        self.trades: list[Trade] = []

    @property
    def price_change(self) -> Decimal:
        """Absolute price change"""
        return self.current_price - self.previous_price

    @property
    def price_change_pct(self) -> float:
        """Percentage price change"""
        if self.previous_price > 0:
            return float(self.price_change / self.previous_price * 100)
        return 0.0

    # ===========================================
    # ORDER SUBMISSION
    # ===========================================

    def submit_buy_order(
        self, agent_id: str, quantity: Decimal, max_price: Decimal, tick: int
    ) -> Order:
        """Submit a buy order"""
        order = Order(
            id="",
            agent_id=agent_id,
            is_buy=True,
            price=max_price,
            quantity=quantity,
            timestamp=tick,
        )
        self.order_book.add_buy_order(order)
        self.total_demand += quantity
        return order

    def submit_sell_order(
        self, agent_id: str, quantity: Decimal, min_price: Decimal, tick: int
    ) -> Order:
        """Submit a sell order"""
        order = Order(
            id="",
            agent_id=agent_id,
            is_buy=False,
            price=min_price,
            quantity=quantity,
            timestamp=tick,
        )
        self.order_book.add_sell_order(order)
        self.total_supply += quantity
        return order

    # ===========================================
    # PRICE DISCOVERY
    # ===========================================

    def clear_market(self, tick: int) -> list[Trade]:
        """
        Clear the market - match orders and discover price.

        Austrian Theory:
        - This is where the "catallactic" process happens
        - Individual valuations aggregate into prices
        - Prices emerge, not designed

        Note: For CRYPTO and COMMODITIES, prices are exogenously determined
        by global markets (presidential cycle, etc.), so we skip price discovery
        for these markets. They can still process trades but price is set elsewhere.
        """
        # Match orders
        trades = self.order_book.match_orders(tick)

        # For exogenously priced markets (BTC, Gold), skip price discovery
        # Price is set by _apply_monetary_effects in the engine
        exogenous_markets = {MarketType.CRYPTO, MarketType.COMMODITIES}

        # Update price based on trades (only for endogenous markets)
        if trades and self.market_type not in exogenous_markets:
            # Price is last trade price
            self.previous_price = self.current_price
            self.current_price = trades[-1].price

            # Calculate volume
            self.volume = sum(t.quantity for t in trades)
            self.volume_history.append(self.volume)

            # Store trades
            self.trades.extend(trades)

        elif trades and self.market_type in exogenous_markets:
            # Still track volume and trades, but don't update price
            self.volume = sum(t.quantity for t in trades)
            self.volume_history.append(self.volume)
            self.trades.extend(trades)

        # Price adjustment if no trades but order imbalance (only for endogenous markets)
        elif self.total_demand > 0 or self.total_supply > 0:
            if self.market_type not in exogenous_markets:
                self._adjust_price_on_imbalance()

        # Update metrics
        self._update_metrics()

        # Record price
        self.price_history.append(self.current_price)
        if len(self.price_history) > 1000:  # Keep last 1000
            self.price_history.pop(0)

        # Reset for next period
        self.total_supply = Decimal("0")
        self.total_demand = Decimal("0")

        return trades

    def _adjust_price_on_imbalance(self):
        """
        Adjust price based on supply/demand imbalance.

        Walrasian tatonnement approximation:
        - Excess demand → price up
        - Excess supply → price down
        """
        if self.total_supply > 0:
            imbalance = float(self.total_demand / self.total_supply)
        else:
            imbalance = 2.0 if self.total_demand > 0 else 1.0

        # Price adjustment factor
        if imbalance > 1.1:  # Excess demand
            adjustment = Decimal(str(min(1.1, 1 + (imbalance - 1) * 0.1)))
        elif imbalance < 0.9:  # Excess supply
            adjustment = Decimal(str(max(0.9, 1 - (1 - imbalance) * 0.1)))
        else:
            adjustment = Decimal("1")

        self.previous_price = self.current_price
        self.current_price *= adjustment

    def _update_metrics(self):
        """Update market metrics"""
        # Volatility (standard deviation of recent price changes)
        if len(self.price_history) >= 10:
            recent_prices = self.price_history[-10:]
            returns = [
                float((recent_prices[i] - recent_prices[i - 1]) / recent_prices[i - 1])
                for i in range(1, len(recent_prices))
                if recent_prices[i - 1] > 0
            ]
            if returns:
                mean_return = sum(returns) / len(returns)
                variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
                self.volatility = variance**0.5

        # Liquidity (based on order book depth)
        bid = self.order_book.get_best_bid()
        ask = self.order_book.get_best_ask()
        if bid and ask and ask > 0:
            spread_pct = float((ask - bid) / ask)
            self.liquidity = max(0.1, 1 - spread_pct * 10)
        else:
            self.liquidity = 0.5

    # ===========================================
    # MARKET INFORMATION
    # ===========================================

    def get_market_data(self) -> dict[str, Any]:
        """Get current market data"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.market_type.value,
            "country": self.country,
            "price": str(self.current_price),
            "price_change_pct": self.price_change_pct,
            "volume": str(self.volume),
            "volatility": self.volatility,
            "liquidity": self.liquidity,
            "best_bid": str(self.order_book.get_best_bid() or 0),
            "best_ask": str(self.order_book.get_best_ask() or 0),
            "spread": str(self.order_book.get_spread() or 0),
        }

    def get_price_signal(self) -> dict[str, Any]:
        """
        Get price signal information.

        Austrian Theory:
        - Prices are signals about scarcity
        - Rising prices → scarcity, attract production
        - Falling prices → abundance, reduce production
        """
        trend = "neutral"
        if len(self.price_history) >= 5:
            recent_avg = sum(self.price_history[-5:]) / 5
            older_avg = (
                sum(self.price_history[-10:-5]) / 5
                if len(self.price_history) >= 10
                else recent_avg
            )

            if recent_avg > older_avg * Decimal("1.05"):
                trend = "bullish"
            elif recent_avg < older_avg * Decimal("0.95"):
                trend = "bearish"

        return {
            "current_price": str(self.current_price),
            "trend": trend,
            "volatility": self.volatility,
            "signal": "increase_production"
            if trend == "bullish"
            else "decrease_production"
            if trend == "bearish"
            else "maintain",
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize market"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.market_type.value,
            "country": self.country,
            "current_price": str(self.current_price),
            "volume": str(self.volume),
            "volatility": self.volatility,
            "liquidity": self.liquidity,
        }


class MarketManager:
    """
    Manages all markets in the simulation.

    Coordinates price discovery across markets.
    """

    def __init__(self, country: str = "USA"):
        self.country = country
        self.markets: dict[str, Market] = {}

    def create_market(
        self,
        name: str,
        market_type: MarketType,
        initial_price: Decimal = Decimal("100"),
    ) -> Market:
        """Create a new market"""
        market = Market(
            name=name,
            market_type=market_type,
            country=self.country,
            initial_price=initial_price,
        )
        self.markets[market.id] = market
        return market

    def get_market(self, market_id: str) -> Market | None:
        """Get a market by ID"""
        return self.markets.get(market_id)

    def get_markets_by_type(self, market_type: MarketType) -> list[Market]:
        """Get all markets of a type"""
        return [m for m in self.markets.values() if m.market_type == market_type]

    def get_market_by_name(self, name: str) -> Market | None:
        """Get a market by name"""
        for market in self.markets.values():
            if market.name == name:
                return market
        return None

    def clear_all_markets(self, tick: int) -> dict[str, list[Trade]]:
        """Clear all markets and return trades"""
        all_trades = {}
        for market_id, market in self.markets.items():
            trades = market.clear_market(tick)
            if trades:
                all_trades[market_id] = trades
        return all_trades

    def get_all_prices(self) -> dict[str, Decimal]:
        """Get prices for all markets"""
        return {m.name: m.current_price for m in self.markets.values()}

    def get_inflation_index(self, lookback_months: int = 3) -> float:
        """
        Calculate a CPI-like inflation index using rolling price changes.

        This is a "real" inflation measure that:
        - Only includes consumer goods and labor (like CPI)
        - EXCLUDES crypto and commodities (too volatile)
        - Uses longer lookback for stability
        - Dampens short-term noise

        Austrian Theory Reminder:
        - This measures PRICE changes, not true inflation (which is money supply growth)
        - Real inflation happens in money printing, prices just reflect it
        - But this metric is useful for simulation feedback

        Args:
            lookback_months: Number of months to look back for rolling inflation (default 3 for quarterly)
        """
        if not self.markets:
            return 0.0

        weighted_changes = []
        total_weight = 0.0

        # CPI-like weights: only consumer goods and labor
        # Exclude crypto and commodities which are investment assets, not consumption
        weights = {
            MarketType.CONSUMER_GOODS: 0.7,  # Main component like CPI
            MarketType.LABOR: 0.3,  # Wage inflation component
            MarketType.CAPITAL_GOODS: 0.0,  # Not in CPI
            MarketType.COMMODITIES: 0.0,  # Exclude - too volatile, investment asset
            MarketType.CRYPTO: 0.0,  # Exclude - investment asset, not consumption
        }

        for market in self.markets.values():
            weight = weights.get(market.market_type, 0.0)

            # Skip markets with zero weight
            if weight <= 0:
                continue

            # Calculate rolling price change from price history
            if len(market.price_history) >= lookback_months + 1:
                # Compare current price to price N months ago
                old_price = market.price_history[-(lookback_months + 1)]
                current_price = market.price_history[-1]

                if old_price > 0:
                    # Calculate period change
                    period_change = float((current_price - old_price) / old_price)
                    # Scale to approximate annual rate (12 months = 1 year)
                    annualized_change = period_change * (12 / lookback_months)
                    # Soft cap using tanh to smooth extreme values
                    # This maps any value to (-0.3, 0.3) range smoothly
                    import math

                    annualized_change = 0.3 * math.tanh(annualized_change / 0.3)
                    weighted_changes.append(annualized_change * weight)
                    total_weight += weight

            elif len(market.price_history) >= 2:
                # Fallback to simple month-over-month change
                old_price = market.price_history[-2]
                current_price = market.price_history[-1]

                if old_price > 0:
                    import math

                    change = float((current_price - old_price) / old_price)
                    # Annualize and soft cap using tanh (12 months = 1 year)
                    annualized_change = change * 12
                    annualized_change = 0.3 * math.tanh(annualized_change / 0.3)
                    weighted_changes.append(annualized_change * weight)
                    total_weight += weight

        if weighted_changes and total_weight > 0:
            return sum(weighted_changes) / total_weight
        return 0.0

    def setup_default_markets(self) -> None:
        """Create default markets for MVP"""
        # Consumer goods markets
        self.create_market(
            "Food & Groceries", MarketType.CONSUMER_GOODS, Decimal("100")
        )
        self.create_market("Housing & Rent", MarketType.CONSUMER_GOODS, Decimal("2000"))
        self.create_market("General Consumer", MarketType.CONSUMER_GOODS, Decimal("50"))

        # Capital goods markets
        self.create_market("Machinery", MarketType.CAPITAL_GOODS, Decimal("10000"))
        self.create_market("Equipment", MarketType.CAPITAL_GOODS, Decimal("5000"))

        # Commodities
        self.create_market("Gold (oz)", MarketType.COMMODITIES, Decimal("2000"))
        self.create_market("Silver (oz)", MarketType.COMMODITIES, Decimal("25"))
        self.create_market("Oil (barrel)", MarketType.COMMODITIES, Decimal("80"))

        # Crypto
        self.create_market("Bitcoin", MarketType.CRYPTO, Decimal("50000"))

        # Labor market
        self.create_market("Labor", MarketType.LABOR, Decimal("1000"))  # Weekly wage
