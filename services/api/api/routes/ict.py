from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional
import pandas as pd
from api.models import ICTLevelsResponse
from api.dependencies import get_market_state, parse_timestamp
from services.market_state import MarketState, compute_ict_metrics

router = APIRouter(prefix="/ict", tags=["ict"])


@router.get("/levels", response_model=ICTLevelsResponse)
async def get_ict_levels(
    ref_time: Optional[str] = Query(None, description="Reference timestamp"),
    lookback_days: int = Query(2, ge=1, le=30),
    market_state: MarketState = Depends(get_market_state)
):
    """Get ICT levels for a given time."""
    try:
        # Parse reference time or use latest
        if ref_time:
            ref_ts = parse_timestamp(ref_time)
        else:
            ref_ts = market_state.df['timestamp'].max()
        
        # Load window with lookback
        start_ts = ref_ts - pd.Timedelta(days=lookback_days)
        window_df = market_state.load_window_by_time(start_ts, ref_ts)
        
        if window_df.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        ict_metrics = compute_ict_metrics(window_df, ref_date=ref_ts)
        return ICTLevelsResponse(**ict_metrics)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))