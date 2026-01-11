import asyncio
import sys
import os
# Add current directory to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "services", "api"))

from trading.paper_engine import PaperTradingEngine
from trading.bot_manager import BotManager
# Mock inference service
class MockInferenceService:
    def analyze_tick(self, history):
        return {"action": "BUY", "confidence": 0.88}
        
    def mock_signal(self):
        return self.analyze_tick(None)

async def test_bot_flow():
    print("[TEST] Starting Bot Flow Check...")
    
    # 1. Setup
    pe = PaperTradingEngine()
    ins = MockInferenceService()
    bot = BotManager(pe, ins)
    
    # 2. Check initial state
    if bot.is_enabled:
        print("FAIL: Bot should start disabled.")
        return
        
    # 3. Enable Bot
    bot.set_enabled(True)
    if not bot.is_enabled:
        print("FAIL: Bot failed to enable.")
        return
        
    # 4. Simulate Tick & Signal
    tick = {"close": 4150.0}
    signal = ins.mock_signal() # BUY
    
    print(f"[TEST] Processing tick with Signal: {signal}")
    
    # 5. Execute
    result = await bot.execute_strategy(tick, signal)
    
    # 6. Verify Trade
    if result:
        print(f"[TEST] Bot returned trade log: {result['type']}")
        print(result)
        
        # Verify Paper Engine State
        state = pe.get_state()
        if len(state.positions) > 0:
            print(f"[PASS] Position created: {state.positions}")
        else:
            print("[FAIL] No position created in engine.")
            
        if "signature" in result["data"]:
             print(f"[PASS] Solana Signature found: {result['data']['signature']}")
        else:
             print("[FAIL] Missing Solana signature.")
             
    else:
        print("[FAIL] Bot did not execute trade on BUY signal.")

if __name__ == "__main__":
    asyncio.run(test_bot_flow())
