import asyncio
import sys
import os
# Add current directory to path so imports work
sys.path.append(os.getcwd())

from trading.paper_engine import PaperTradingEngine
from data_manager import DataManager
import pandas as pd

def test_paper_engine():
    print("[TEST] PaperTradingEngine")
    engine = PaperTradingEngine(initial_balance=10000.0)
    state = engine.get_state()
    
    if state.balance != 10000.0:
        print("FAIL: Initial balance incorrect")
        return False
        
    engine.place_order("AAPL", "LONG", 10, "LIMIT", 150.0)
    if len(engine.get_state().orders) != 1:
        print("FAIL: Order not placed")
        return False
        
    # Process tick to fill order
    engine.process_tick("AAPL", 150.0)
    if "AAPL" not in engine.positions:
        print("FAIL: Order not filled")
        return False
        
    # Process tick to move price
    engine.process_tick("AAPL", 160.0)
    pnl = engine.get_state().equity - 10000.0
    if pnl != 100.0:
        print(f"FAIL: PnL incorrect. Expected 100.0, got {pnl}")
        return False
        
    print("PASS: PaperTradingEngine logic verified.")
    return True

async def test_data_manager():
    print("[TEST] DataManager")
    dm = DataManager()
    
    # Mock some data
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
    
    chunk = dm.get_candles(0, 10)
    if len(chunk) != 10:
        print("FAIL: Chunk size incorrect")
        return False
        
    # Replay logic check
    dm.set_replay_state(True)
    tick = dm.get_replay_tick()
    if tick is None or tick["open"] != 0:
        print("FAIL: Replay tick incorrect")
        return False
        
    print("PASS: DataManager logic verified.")
    return True

async def main():
    p_pass = test_paper_engine()
    d_pass = await test_data_manager()
    
    if p_pass and d_pass:
        print("\nALL SYSTEMS VERIFIED.")
    else:
        print("\nVERIFICATION FAILED.")

if __name__ == "__main__":
    asyncio.run(main())
