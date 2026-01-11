import asyncio
from typing import Dict, Optional
from datetime import datetime
from flint_solana.log import log_trade_to_solana

class BotManager:
    def __init__(self, paper_engine, inference_service):
        self.paper_engine = paper_engine
        self.inference_service = inference_service
        self.is_enabled = False
        
    def set_enabled(self, enabled: bool):
        self.is_enabled = enabled
        print(f"[BotManager] Auto-Trading {'ENABLED' if enabled else 'DISABLED'}")
        
    async def process_tick(self, candle: Dict) -> Optional[Dict]:
        """
        Process a new candle tick. 
        Returns a trade log dictionary if a trade was executed, else None.
        """
        # 1. Get Signal
        # We assume inference_service can handle a single candle update or we pass the whole history?
        # For this demo, let's assume we pass a dataframe-like structure or just let it mock.
        # Ideally, main.py passes the full history to inference_service, but here we just ask for a signal.
        # Since InferenceService mock logic uses history, we rely on Main passing it.
        # Wait, Main calls process_tick. Let's assume Main handles data updates and we simply ask for "latest signal".
        
        # Actually, let's pass the latest price info to mock simple logic inside here if needed
        # or call a method on inference_service that uses its internal state (if it had one).
        # Given InferenceService design in main:
        # signal = inference_service.analyze_tick(...)
        
        # So we will change the signature to accept the signal directly from Main
        pass

    async def execute_strategy(self, candle: Dict, signal: Dict) -> Optional[Dict]:
        if not self.is_enabled:
            return None
            
        action = signal.get("action", "HOLD")
        confidence = signal.get("confidence", 0.0)
        
        if action == "HOLD":
            return None
            
        # Execute Trade
        symbol = "ES=F" # Hardcoded for demo
        price = candle["close"]
        size = 1.0 # Standard lot for demo
        
        # Check if we already have a position to avoid spamming (simplest logic: 1 open pos max)
        state = self.paper_engine.get_state()
        current_pos = state.positions.get(symbol)
        
        # Logic: 
        # If BUY -> Enter Long if Flat (or flip if Short?)
        # For simplicity: Only enter if Flat.
        if current_pos:
            # If we have a position, we only exit if signal opposes?
            # Or simplified: Bot only ENTRIES. Exits handled by TP/SL (which PaperEngine supports?)
            # Or Bot flips.
            # Let's do: Bot flips.
            if (action == "BUY" and current_pos.side == "SHORT") or \
               (action == "SELL" and current_pos.side == "LONG"):
                # Close existing
                self.paper_engine.place_order(symbol, "MARKET", current_pos.size, side=("BUY" if current_pos.side == "SHORT" else "SELL"))
                
        # Place NEW Entry if we are now flat (or if we just closed)
        # Re-check state? No, just place order.
        
        trade_side = "BUY" if action == "BUY" else "SELL"
        
        # Place Order
        self.paper_engine.place_order(symbol, trade_side, size, "MARKET", price)
        
        print(f"[BotManager] Executing {trade_side} {size} @ {price}")
        
        # Log to Solana
        trade_data = {
            "symbol": symbol,
            "side": trade_side,
            "price": price,
            "size": size,
            "reason": f"Auto-Bot Signal {confidence:.2f}"
        }
        
        # Fire and forget / await
        try:
            tx_sig = await log_trade_to_solana(trade_data)
        except Exception as e:
            print(f"[BotManager] Solana Log Failed: {e}")
            tx_sig = "failed_log"
            
        return {
            "type": "trade_log",
            "data": {
                **trade_data,
                "timestamp": datetime.now().isoformat(),
                "signature": tx_sig
            }
        }
