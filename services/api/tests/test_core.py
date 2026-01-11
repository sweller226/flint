import pytest
import pandas as pd
from services.api.trading.paper_engine import PaperTradingEngine
from services.api.data_manager import DataManager

# Mock Order/Position classes if they are simple dataclasses, or import them
from services.api.trading.paper_engine import Order, Position

def test_paper_engine_initialization():
    engine = PaperTradingEngine(initial_balance=10000.0)
    state = engine.get_state()
    assert state.balance == 10000.0
    assert state.equity == 10000.0
    assert len(state.positions) == 0
    assert len(state.orders) == 0

def test_paper_engine_order_placement():
    engine = PaperTradingEngine()
    engine.place_order("AAPL", "LONG", 10, "LIMIT", 150.0)
    
    state = engine.get_state()
    assert len(state.orders) == 1
    order = state.orders[0]
    assert order.symbol == "AAPL"
    assert order.side == "LONG"
    assert order.size == 10
    assert order.price == 150.0
    assert order.status == "OPEN"

def test_paper_engine_execution():
    engine = PaperTradingEngine()
    # Place Limit Buy at 150
    engine.place_order("AAPL", "LONG", 10, "LIMIT", 150.0)
    
    # Process tick above limit (should not fill)
    engine.process_tick("AAPL", 151.0)
    assert len(engine.positions) == 0
    
    # Process tick at limit (should fill)
    engine.process_tick("AAPL", 150.0)
    assert "AAPL" in engine.positions
    pos = engine.positions["AAPL"]
    assert pos.size == 10
    assert pos.average_price == 150.0
    
    # Check balance (should remain same until realized)
    # Check equity (should move with price)
    
    # Price moves to 160
    engine.process_tick("AAPL", 160.0)
    state = engine.get_state()
    # Unrealized PnL = (160 - 150) * 10 = 100
    assert state.equity == 100000.0 + 100.0

def test_data_manager_chunking():
    dm = DataManager()
    # Create dummy data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1min")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": range(100),
        "high": range(100),
        "low": range(100),
        "close": range(100),
        "volume": [1000]*100
    })
    dm.history = df
    
    # Test valid chunk
    chunk = dm.get_candles(start_index=0, limit=10)
    assert len(chunk) == 10
    assert chunk[0]["open"] == 0
    assert chunk[9]["open"] == 9
    
    # Test offset chunk
    chunk = dm.get_candles(start_index=10, limit=10)
    assert len(chunk) == 10
    assert chunk[0]["open"] == 10
    
    # Test out of bounds
    chunk = dm.get_candles(start_index=95, limit=10)
    assert len(chunk) == 5
