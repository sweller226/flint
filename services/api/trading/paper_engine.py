from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from datetime import datetime
import uuid

@dataclass
class Position:
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    size: float
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    
    def update(self, current_price: float):
        self.current_price = current_price
        if self.side == 'LONG':
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.size

@dataclass
class Order:
    id: str
    symbol: str
    side: str
    size: float
    price: float  # Limit price, or execution price for market
    type: str  # 'MARKET', 'LIMIT', 'STOP'
    status: str  # 'OPEN', 'FILLED', 'CANCELLED'
    timestamp: datetime
    filled_price: Optional[float] = None

@dataclass
class PortfolioState:
    balance: float
    equity: float
    positions: Dict[str, Position] = field(default_factory=dict)
    orders: List[Order] = field(default_factory=list)

class PaperTradingEngine:
    def __init__(self, initial_balance: float = 100000.0):
        self.balance = initial_balance
        self.positions: Dict[str, Position] = {} # symbol -> Position
        self.orders: List[Order] = []
        self.trade_history: List[dict] = []

    def place_order(self, symbol: str, side: str, size: float, order_type: str = 'MARKET', price: float = 0.0) -> Order:
        order_id = str(uuid.uuid4())
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side.upper(),
            size=size,
            price=price,
            type=order_type.upper(),
            status='OPEN',
            timestamp=datetime.now()
        )
        self.orders.append(order)
        # For simplicity in this iteration, immediately fill MARKET orders if we have a price
        # In a real loop, we'd check against current price tick
        return order

    def process_tick(self, symbol: str, current_price: float):
        """Called whenever a new price comes in."""
        
        # 1. Update Position PnL
        if symbol in self.positions:
            self.positions[symbol].update(current_price)

        # 2. Check pending orders
        for order in self.orders:
            if order.status == 'OPEN' and order.symbol == symbol:
                if order.type == 'MARKET':
                    self._fill_order(order, current_price)
                elif order.type == 'LIMIT':
                    if (order.side == 'LONG' and current_price <= order.price) or \
                       (order.side == 'SHORT' and current_price >= order.price):
                        self._fill_order(order, order.price) # Assume fill at limit
                elif order.type == 'STOP':
                    if (order.side == 'LONG' and current_price >= order.price) or \
                       (order.side == 'SHORT' and current_price <= order.price):
                        self._fill_order(order, current_price)

    def _fill_order(self, order: Order, price: float):
        order.status = 'FILLED'
        order.filled_price = price
        
        # Calculate Cost/Proceeds
        cost = price * order.size
        
        # Update Balance?? 
        # For futures, balance usually only changes on Realized PnL.
        # But for simple spot-like logic: 
        # If Buying, we might reduce buying power. 
        # Let's verify if we are netting positions.
        
        # Netting Logic
        if order.symbol in self.positions:
            pos = self.positions[order.symbol]
            if pos.side == order.side:
                # Add to position
                total_cost = (pos.entry_price * pos.size) + (price * order.size)
                new_size = pos.size + order.size
                pos.entry_price = total_cost / new_size
                pos.size = new_size
            else:
                # Reduce/Flip position
                if order.size <= pos.size:
                    # Partial close or full close
                    closed_size = order.size
                    remaining_size = pos.size - order.size
                    
                    # Realized PnL
                    if pos.side == 'LONG': # Selling to close Long
                        pnl = (price - pos.entry_price) * closed_size
                    else: # Buying to close Short
                        pnl = (pos.entry_price - price) * closed_size
                    
                    self.balance += pnl
                    self.trade_history.append({
                        "symbol": order.symbol,
                        "pnl": pnl,
                        "timestamp": datetime.now()
                    })
                    
                    if remaining_size == 0:
                        del self.positions[order.symbol]
                    else:
                        pos.size = remaining_size
                else:
                    # Flip position
                    # 1. Close existing
                    closed_size = pos.size
                    
                    if pos.side == 'LONG':
                        pnl = (price - pos.entry_price) * closed_size
                    else:
                        pnl = (pos.entry_price - price) * closed_size
                        
                    self.balance += pnl
                    
                    # 2. Open new remainder
                    new_size = order.size - pos.size
                    self.positions[order.symbol] = Position(
                        symbol=order.symbol,
                        side=order.side,
                        size=new_size,
                        entry_price=price,
                        current_price=price
                    )
        else:
            # New Position
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                side=order.side,
                size=order.size,
                entry_price=price,
                current_price=price
            )

    def get_state(self) -> PortfolioState:
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return PortfolioState(
            balance=self.balance,
            equity=self.balance + total_unrealized,
            positions=self.positions,
            orders=self.orders
        )
