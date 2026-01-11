import pandas as pd
from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from api.models import CandleResponse, CandlesListResponse
from api.dependencies import get_market_state, parse_timestamp

router = APIRouter(tags=["candles"])

@router.get("/candles", response_model=CandlesListResponse)
async def get_candles(
    contract: str = Query("H", description="Contract quarter code: H (March), M (June), U (Sep), Z (Dec)"),
    start_time: Optional[str] = Query(None, description="Start timestamp (ISO8601)"),
    end_time: Optional[str] = Query(None, description="End timestamp (ISO8601)"),
    width_seconds: Optional[int] = Query(None, ge=1, description="Candle width in seconds. If provided, resamples data."),
    limit: int = Query(100, ge=1, le=100000),
):
    """Get historical candles for a specific ES futures contract."""
    try:
        market_state = get_market_state(contract)
        
        start_ts = parse_timestamp(start_time)
        end_ts = parse_timestamp(end_time)

        # Optimization: If resampling is requested but no start time, 
        # calculate a lookback to avoid resampling the entire history.
        if width_seconds and not start_ts:
            latest_ts = end_ts if end_ts else market_state.df['timestamp'].max()
            # Estimate needed history: limit * width * safety_factor (for weekends/gaps)
            fetch_seconds = limit * width_seconds * 5 
            
            # Cap lookback to avoid overflow (Pandas limit is ~584 years). Use 50 years as safe max.
            if fetch_seconds > (50 * 365 * 24 * 3600):
                start_ts = market_state.df['timestamp'].min()
            else:
                start_ts = latest_ts - pd.Timedelta(seconds=fetch_seconds)
        
        # Get data window
        if start_ts and end_ts:
            df = market_state.load_window_by_time(start_ts, end_ts)
        elif start_ts:
            df = market_state.load_window_by_time(start_ts, market_state.df['timestamp'].max())
        elif end_ts:
            df = market_state.load_window_by_time(market_state.df['timestamp'].min(), end_ts)
        else:
            df = market_state.df
        
        # Resample if requested
        if width_seconds:
            # Ensure index is datetime for resampling
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.set_index('timestamp')
            
            df = df.resample(f"{width_seconds}s").agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Reset index to make timestamp a column again
            df = df.reset_index()

        # Apply limit to the final candle count
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