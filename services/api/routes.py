from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
from flint_solana.log import log_trade_to_solana

# We will inject these dependencies from main.py or use a singleton pattern.
# For simplicity, we'll import instances from main if circular import allows, 
# OR (better) we move initialization to a shared module or use FastAPI dependency injection.
# Since main.py imports routes, we cannot import main.py here.
# So we will define the router but expect the "manager" objects to be attached to `app.state` or similar,
# OR we rely on a shared `services` module.
# Plan: I will use a simple global reference set by main.py for this "Audit" speed.
# In production, use Dependency Injection.

router = APIRouter()

# Global references (will be set by main.py)
paper_engine = None
data_manager = None
inference_service = None
replay_manager = None # merged into data_manager for now

def set_services(pe, dm, ins, bm=None):
    global paper_engine, data_manager, inference_service
    paper_engine = pe
    data_manager = dm
    inference_service = ins
    global bot_manager
    bot_manager = bm


class Trade(BaseModel):
    symbol: str
    side: str
    price: float
    size: float
    reason: str

class PaperOrder(BaseModel):
    symbol: str
    side: str
    size: float
    type: str = "MARKET"
    price: float = 0.0

@router.post("/log-trade")
async def log_trade(trade: Trade):
    result = await log_trade_to_solana(trade.dict())
    return result

@router.get("/candles")
def get_history(symbol: str = "ES=F", start: int = 0, limit: int = 1000):
    if not data_manager:
        raise HTTPException(503, "Services not initialized")
    return data_manager.get_candles(start, limit)

@router.post("/order")
def place_order(order: PaperOrder):
    if not paper_engine:
        raise HTTPException(503, "Services not initialized")
    
    # Place order
    placed = paper_engine.place_order(
        symbol=order.symbol,
        side=order.side,
        size=order.size,
        order_type=order.type,
        price=order.price
    )
    return placed

@router.get("/portfolio")
def get_portfolio():
    if not paper_engine:
        raise HTTPException(503, "Services not initialized")
    return paper_engine.get_state()

@router.post("/replay/{action}")
def control_replay(action: str, speed: float = 1.0, index: int = 0):
    if not data_manager:
        raise HTTPException(503, "Services not initialized")
    
    if action == "start":
        data_manager.set_replay_state(True, speed)
    elif action == "stop" or action == "pause":
        data_manager.set_replay_state(False)
    elif action == "seek":
        data_manager.set_replay_index(index)
    elif action == "speed":
        data_manager.set_replay_state(data_manager.is_playing, speed)
    
    return {"status": "ok", "playing": data_manager.is_playing, "index": data_manager.replay_index}

@router.get("/signals")
def get_latest_signal():
    if not inference_service:
        raise HTTPException(503, "Services not initialized")
    
    # In a real app, we'd pass the latest history from data_manager
    # For now, simplistic
    return inference_service.analyze_tick(data_manager.history)

@router.post("/bot/toggle")
def toggle_bot(enabled: bool):
    if not bot_manager:
        raise HTTPException(503, "Services not initialized")
    bot_manager.set_enabled(enabled)
    return {"status": "ok", "enabled": bot_manager.is_enabled}

@router.get("/bot/status")
def get_bot_status():
    if not bot_manager:
        raise HTTPException(503, "Services not initialized")
    return {"enabled": bot_manager.is_enabled}

