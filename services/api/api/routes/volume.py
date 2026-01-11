from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional
import pandas as pd
from api.models import VolumeProfileResponse, VolumeBinResponse
from api.dependencies import get_market_state, parse_timestamp
from services.market_state import MarketState, compute_volume_metrics

router = APIRouter(prefix="/volume", tags=["volume"])

@router.get("/profile", response_model=VolumeProfileResponse)
async def get_volume_profile(
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    num_bins: int = Query(30, ge=5, le=100),
    market_state: MarketState = Depends(get_market_state)
):
    """Get volume profile for a time window."""
    try:
        start_ts = parse_timestamp(start_time) if start_time else None
        end_ts = parse_timestamp(end_time) if end_time else market_state.df['timestamp'].max()
        
        # Get window
        if start_ts and end_ts:
            window_df = market_state.load_window_by_time(start_ts, end_ts)
        elif start_ts:
            window_df = market_state.load_window_by_time(start_ts, market_state.df['timestamp'].max())
        else:
            # Default to last day
            end_ts = market_state.df['timestamp'].max()
            start_ts = end_ts - pd.Timedelta(days=1)
            window_df = market_state.load_window_by_time(start_ts, end_ts)
        
        if window_df.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        # Compute volume profile
        vol_metrics = compute_volume_metrics(window_df, num_bins=num_bins)
        
        bins = [VolumeBinResponse(**b) for b in vol_metrics['bins']]
        
        return VolumeProfileResponse(
            bins=bins,
            poc_price=vol_metrics['poc_price']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
