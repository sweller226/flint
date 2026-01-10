from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional
from api.models import CandleResponse, CandlesListResponse
from api.dependencies import get_market_state, parse_timestamp
from services.market_state import MarketState

router = APIRouter(tags=["candles"])

@router.get("/candles", response_model=CandlesListResponse)
async def get_candles(
    start_time: Optional[str] = Query(None, description="Start timestamp (ISO8601)"),
    end_time: Optional[str] = Query(None, description="End timestamp (ISO8601)"),
    limit: int = Query(100, ge=1, le=10000),
    market_state: MarketState = Depends(get_market_state)
):
    """Get historical candles within a time range."""
    try:
        start_ts = parse_timestamp(start_time)
        end_ts = parse_timestamp(end_time)
        
        # Get data window
        if start_ts and end_ts:
            df = market_state.load_window_by_time(start_ts, end_ts)
        elif start_ts:
            df = market_state.load_window_by_time(start_ts, market_state.df['timestamp'].max())
        elif end_ts:
            df = market_state.load_window_by_time(market_state.df['timestamp'].min(), end_ts)
        else:
            df = market_state.df
        
        # Apply limit
        if len(df) > limit:
            df = df.tail(limit)
        
        # Convert to response
        candles = [
            CandleResponse(
                timestamp=row['timestamp'].isoformat(),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume'])
            )
            for _, row in df.iterrows()
        ]
        
        return CandlesListResponse(candles=candles, count=len(candles))
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))