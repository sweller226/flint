from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Optional
import asyncio
from api.dependencies import get_market_state, parse_timestamp
from services.market_state import MarketState, Candle

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/candles")
async def websocket_candles(
    websocket: WebSocket,
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    delay_ms: int = Query(100, ge=10, le=5000)
):
    """WebSocket endpoint for streaming candles."""
    await websocket.accept()
    
    try:
        if get_market_state() is None:
            await websocket.send_json({"type": "error", "message": "Market data not loaded"})
            return
        
        market_state = get_market_state()
        
        # Parse time window
        start_ts = parse_timestamp(start_time) if start_time else None
        end_ts = parse_timestamp(end_time) if end_time else None
        
        # Load window
        if start_ts and end_ts:
            df = market_state.load_window_by_time(start_ts, end_ts)
        elif start_ts:
            df = market_state.load_window_by_time(start_ts, market_state.df['timestamp'].max())
        elif end_ts:
            df = market_state.load_window_by_time(market_state.df['timestamp'].min(), end_ts)
        else:
            df = market_state.df.copy()
        
        # Create temporary MarketState for iteration
        temp_state = MarketState(df)
        
        # Stream candles
        for candle in temp_state.iter_bars():
            await websocket.send_json({
                "type": "candle",
                "data": {
                    "timestamp": candle.timestamp,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume
                }
            })
            await asyncio.sleep(delay_ms / 1000.0)
        
        await websocket.send_json({"type": "complete"})
        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
