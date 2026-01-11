from datetime import datetime
from pydantic import BaseModel

class AccountState(BaseModel):
    balance: float
    equity: float
    pnl_unrealized: float
    pnl_realized: float
    position_size: float # +ve for Long, -ve for Short
    entry_price: float
    last_updated: str

class DemoAccount:
    def __init__(self, initial_balance=50000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance # Cash
        self.pnl_realized = 0.0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.last_price = 0.0
        
    def get_state(self) -> dict:
        pnl_unrealized = self._calc_unrealized_pnl(self.last_price)
        equity = self.balance + pnl_unrealized
        
        return {
            "balance": round(self.balance, 2),
            "equity": round(equity, 2),
            "pnl_unrealized": round(pnl_unrealized, 2),
            "pnl_realized": round(self.pnl_realized, 2),
            "position_size": self.position_size,
            "entry_price": round(self.entry_price, 2),
            "last_updated": datetime.utcnow().isoformat()
        }

    def _calc_unrealized_pnl(self, current_price: float) -> float:
        if self.position_size == 0:
            return 0.0
        
        # PnL = (Current - Entry) * Size
        # If Short (Size < 0): (Entry - Current) * Abs(Size) = (Current - Entry) * Size
        return (current_price - self.entry_price) * self.position_size

    def update_price(self, price: float):
        self.last_price = price

    def execute_signal(self, signal: dict, current_price: float):
        """
        signal: { "data": { "side": "LONG"|"SHORT" } }
        Simple Logic:
        - If LONG signal and Flat/Short -> Buy to Open Long (Flip if Short)
        - If SHORT signal and Flat/Long -> Sell to Open Short (Flip if Long)
        Fix size: 1 Contract
        """
        side = signal.get("data", {}).get("side")
        if not side:
            return

        self.last_price = current_price
        
        # Determine target position
        target_size = 1.0 if side == "LONG" else -1.0
        
        if self.position_size == target_size:
            return # Already positioned

        # Close existing if any
        if self.position_size != 0:
            pnl = self._calc_unrealized_pnl(current_price)
            self.pnl_realized += pnl
            self.balance += pnl # Realize PnL into cash
            print(f"Closed {self.position_size} @ {current_price}. PnL: {pnl}")
            self.position_size = 0
            self.entry_price = 0

        # Open new position
        self.position_size = target_size
        self.entry_price = current_price
        print(f"Opened {side} @ {current_price}")
