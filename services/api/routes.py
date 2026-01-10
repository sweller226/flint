from fastapi import APIRouter
from pydantic import BaseModel
from solana.log import log_trade_to_solana

router = APIRouter()

class Trade(BaseModel):
    symbol: str
    side: str
    price: float
    size: float
    reason: str

@router.post("/log-trade")
async def log_trade(trade: Trade):
    result = await log_trade_to_solana(trade.dict())
    return result
