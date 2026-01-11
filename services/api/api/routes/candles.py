import pandas as pd
from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from api.models import CandleResponse, CandlesListResponse
from api.dependencies import get_market_state, parse_timestamp

router = APIRouter(tags=["candles"])

@router.get("/candles", response_model=CandlesListResponse)
async def get_candles(
    contract: str = Query("H", description="Contract quarter code: H (March), M (June), U (Sep), Z (Dec)"),
    width_seconds: Optional[int] = Query(None, ge=1, description="Candle width in seconds. If provided, resamples data."),
    limit: int = Query(100, ge=1, le=10000),
):
    """Get historical candles for a specific ES futures contract."""
    try:
        market_state = get_market_state(contract)
        
        # Default to loading the most recent 'limit' candles
        # Unless width_seconds is set, then we need to estimate time window to fetch efficiently
        
        start_ts = None
        end_ts = market_state.df['timestamp'].max()

        if width_seconds:
             # Estimate needed history: limit * width * safety_factor
            fetch_seconds = limit * width_seconds * 5
            
            # Cap lookback to avoid overflow
            if fetch_seconds > (50 * 365 * 24 * 3600):
                start_ts = market_state.df['timestamp'].min()
            else:
                start_ts = end_ts - pd.Timedelta(seconds=fetch_seconds)
        
        # Load window
        if start_ts:
             df = market_state.load_window_by_time(start_ts, end_ts)
        else:
             # Just get everything or use simple logic. 
             # To keep behavior consistent (tail limit), we can just load the whole DF or huge chunk?
             # But 'load_window_by_time' is the standard way.
             # If no start_ts estimated, and no explicit start/end params...
             # We should probably just load the whole DF and tail it?
             df = market_state.df

        # Resample if requested
        if width_seconds:
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.set_index('timestamp')
            
            df = df.resample(f"{width_seconds}s").agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            df = df.reset_index()

        # Apply limit
        if len(df) > limit:
            df = df.tail(limit)
        
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