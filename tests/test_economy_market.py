"""
Unit tests for the Market system.

Tests Austrian Economics price theory including:
- Order book mechanics
- Price discovery
- Supply/demand dynamics
"""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock

from src.economy.market import (
    Market,
    MarketManager,
    MarketType,
    Order,
    OrderBook,
    Trade,
)


class TestOrderBook:
    """Test OrderBook functionality."""

    def test_buy_order_sorting(self) -> None:
        """Buy orders should be sorted by price descending."""
        book = OrderBook()

        book.add_buy_order(
            Order(
                id="1",
                agent_id="a1",
                is_buy=True,
                price=Decimal("100"),
                quantity=Decimal("10"),
                timestamp=1,
            )
        )
        book.add_buy_order(
            Order(
                id="2",
                agent_id="a2",
                is_buy=True,
                price=Decimal("110"),
                quantity=Decimal("10"),
                timestamp=1,
            )
        )
        book.add_buy_order(
            Order(
                id="3",
                agent_id="a3",
                is_buy=True,
                price=Decimal("105"),
                quantity=Decimal("10"),
                timestamp=1,
            )
        )

        assert book.buy_orders[0].price == Decimal("110")
        assert book.buy_orders[1].price == Decimal("105")
        assert book.buy_orders[2].price == Decimal("100")

    def test_sell_order_sorting(self) -> None:
        """Sell orders should be sorted by price ascending."""
        book = OrderBook()

        book.add_sell_order(
            Order(
                id="1",
                agent_id="a1",
                is_buy=False,
                price=Decimal("100"),
                quantity=Decimal("10"),
                timestamp=1,
            )
        )
        book.add_sell_order(
            Order(
                id="2",
                agent_id="a2",
                is_buy=False,
                price=Decimal("90"),
                quantity=Decimal("10"),
                timestamp=1,
            )
        )
        book.add_sell_order(
            Order(
                id="3",
                agent_id="a3",
                is_buy=False,
                price=Decimal("95"),
                quantity=Decimal("10"),
                timestamp=1,
            )
        )

        assert book.sell_orders[0].price == Decimal("90")
        assert book.sell_orders[1].price == Decimal("95")
        assert book.sell_orders[2].price == Decimal("100")

    def test_best_bid(self) -> None:
        """Best bid should be highest buy price."""
        book = OrderBook()

        book.add_buy_order(
            Order(
                id="1",
                agent_id="a1",
                is_buy=True,
                price=Decimal("100"),
                quantity=Decimal("10"),
                timestamp=1,
            )
        )
        book.add_buy_order(
            Order(
                id="2",
                agent_id="a2",
                is_buy=True,
                price=Decimal("110"),
                quantity=Decimal("10"),
                timestamp=1,
            )
        )

        assert book.get_best_bid() == Decimal("110")

    def test_best_ask(self) -> None:
        """Best ask should be lowest sell price."""
        book = OrderBook()

        book.add_sell_order(
            Order(
                id="1",
                agent_id="a1",
                is_buy=False,
                price=Decimal("100"),
                quantity=Decimal("10"),
                timestamp=1,
            )
        )
        book.add_sell_order(
            Order(
                id="2",
                agent_id="a2",
                is_buy=False,
                price=Decimal("90"),
                quantity=Decimal("10"),
                timestamp=1,
            )
        )

        assert book.get_best_ask() == Decimal("90")

    def test_spread_calculation(self) -> None:
        """Spread should be difference between ask and bid."""
        book = OrderBook()

        book.add_buy_order(
            Order(
                id="1",
                agent_id="a1",
                is_buy=True,
                price=Decimal("95"),
                quantity=Decimal("10"),
                timestamp=1,
            )
        )
        book.add_sell_order(
            Order(
                id="2",
                agent_id="a2",
                is_buy=False,
                price=Decimal("100"),
                quantity=Decimal("10"),
                timestamp=1,
            )
        )

        assert book.get_spread() == Decimal("5")

    def test_order_matching(self) -> None:
        """Orders should match when buy price >= sell price."""
        book = OrderBook()

        book.add_buy_order(
            Order(
                id="1",
                agent_id="buyer",
                is_buy=True,
                price=Decimal("100"),
                quantity=Decimal("10"),
                timestamp=1,
            )
        )
        book.add_sell_order(
            Order(
                id="2",
                agent_id="seller",
                is_buy=False,
                price=Decimal("95"),
                quantity=Decimal("10"),
                timestamp=1,
            )
        )

        trades = book.match_orders(tick=1)

        assert len(trades) == 1
        assert trades[0].buyer_id == "buyer"
        assert trades[0].seller_id == "seller"
        assert trades[0].quantity == Decimal("10")

    def test_no_match_when_spread_exists(self) -> None:
        """No trade when buy price < sell price."""
        book = OrderBook()

        book.add_buy_order(
            Order(
                id="1",
                agent_id="buyer",
                is_buy=True,
                price=Decimal("90"),
                quantity=Decimal("10"),
                timestamp=1,
            )
        )
        book.add_sell_order(
            Order(
                id="2",
                agent_id="seller",
                is_buy=False,
                price=Decimal("100"),
                quantity=Decimal("10"),
                timestamp=1,
            )
        )

        trades = book.match_orders(tick=1)

        assert len(trades) == 0

    def test_partial_fill(self) -> None:
        """Partial fills should leave remaining quantity."""
        book = OrderBook()

        book.add_buy_order(
            Order(
                id="1",
                agent_id="buyer",
                is_buy=True,
                price=Decimal("100"),
                quantity=Decimal("10"),
                timestamp=1,
            )
        )
        book.add_sell_order(
            Order(
                id="2",
                agent_id="seller",
                is_buy=False,
                price=Decimal("95"),
                quantity=Decimal("5"),
                timestamp=1,
            )
        )

        trades = book.match_orders(tick=1)

        assert len(trades) == 1
        assert trades[0].quantity == Decimal("5")
        assert len(book.buy_orders) == 1
        assert book.buy_orders[0].quantity == Decimal("5")
        assert len(book.sell_orders) == 0


class TestMarket:
    """Test Market functionality."""

    def test_market_initialization(self) -> None:
        """Market should initialize with correct defaults."""
        market = Market(
            name="Test Market",
            market_type=MarketType.CONSUMER_GOODS,
            initial_price=Decimal("100"),
        )

        assert market.name == "Test Market"
        assert market.market_type == MarketType.CONSUMER_GOODS
        assert market.current_price == Decimal("100")
        assert market.previous_price == Decimal("100")
        assert market.volume == Decimal("0")

    def test_submit_buy_order(self) -> None:
        """Submitting buy order should update demand."""
        market = Market(
            name="Test Market",
            market_type=MarketType.CONSUMER_GOODS,
            initial_price=Decimal("100"),
        )

        order = market.submit_buy_order(
            agent_id="buyer",
            quantity=Decimal("10"),
            max_price=Decimal("105"),
            tick=1,
        )

        assert order.is_buy is True
        assert market.total_demand == Decimal("10")

    def test_submit_sell_order(self) -> None:
        """Submitting sell order should update supply."""
        market = Market(
            name="Test Market",
            market_type=MarketType.CONSUMER_GOODS,
            initial_price=Decimal("100"),
        )

        order = market.submit_sell_order(
            agent_id="seller",
            quantity=Decimal("10"),
            min_price=Decimal("95"),
            tick=1,
        )

        assert order.is_buy is False
        assert market.total_supply == Decimal("10")

    def test_price_change_calculation(self) -> None:
        """Price change should be calculated correctly."""
        market = Market(
            name="Test Market",
            market_type=MarketType.CONSUMER_GOODS,
            initial_price=Decimal("100"),
        )

        market.previous_price = Decimal("100")
        market.current_price = Decimal("110")

        assert market.price_change == Decimal("10")
        assert market.price_change_pct == 10.0

    def test_clear_market_updates_price(self) -> None:
        """Clearing market with trades should update price."""
        market = Market(
            name="Test Market",
            market_type=MarketType.CONSUMER_GOODS,
            initial_price=Decimal("100"),
        )

        market.submit_buy_order(
            agent_id="buyer",
            quantity=Decimal("10"),
            max_price=Decimal("110"),
            tick=1,
        )
        market.submit_sell_order(
            agent_id="seller",
            quantity=Decimal("10"),
            min_price=Decimal("100"),
            tick=1,
        )

        trades = market.clear_market(tick=1)

        assert len(trades) == 1
        # Trade price should be midpoint
        assert trades[0].price == Decimal("105")


class TestMarketManager:
    """Test MarketManager functionality."""

    def test_create_market(self) -> None:
        """Creating a market should add it to the manager."""
        manager = MarketManager(country="USA")

        market = manager.create_market(
            name="Test Market",
            market_type=MarketType.CONSUMER_GOODS,
            initial_price=Decimal("100"),
        )

        assert market.name == "Test Market"
        assert market.id in manager.markets

    def test_get_market_by_name(self) -> None:
        """Should be able to retrieve market by name."""
        manager = MarketManager(country="USA")

        manager.create_market(
            name="Bitcoin",
            market_type=MarketType.CRYPTO,
            initial_price=Decimal("50000"),
        )

        market = manager.get_market_by_name("Bitcoin")

        assert market is not None
        assert market.name == "Bitcoin"

    def test_get_markets_by_type(self) -> None:
        """Should filter markets by type."""
        manager = MarketManager(country="USA")

        manager.create_market("Food", MarketType.CONSUMER_GOODS, Decimal("100"))
        manager.create_market("Housing", MarketType.CONSUMER_GOODS, Decimal("2000"))
        manager.create_market("Gold", MarketType.COMMODITIES, Decimal("2000"))

        consumer_markets = manager.get_markets_by_type(MarketType.CONSUMER_GOODS)

        assert len(consumer_markets) == 2

    def test_get_all_prices(self) -> None:
        """Should return all market prices."""
        manager = MarketManager(country="USA")

        manager.create_market("Food", MarketType.CONSUMER_GOODS, Decimal("100"))
        manager.create_market("Gold", MarketType.COMMODITIES, Decimal("2000"))

        prices = manager.get_all_prices()

        assert prices["Food"] == Decimal("100")
        assert prices["Gold"] == Decimal("2000")

    def test_setup_default_markets(self) -> None:
        """Setup should create expected default markets."""
        manager = MarketManager(country="USA")

        manager.setup_default_markets()

        assert manager.get_market_by_name("Bitcoin") is not None
        assert manager.get_market_by_name("Gold (oz)") is not None
        assert manager.get_market_by_name("Food & Groceries") is not None
