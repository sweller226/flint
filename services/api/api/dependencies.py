from fastapi import HTTPException
from typing import Optional
import pandas as pd
from services.market_state import MarketState, load_data

_market_state: Optional[MarketState] = None


def get_market_state() -> MarketState:
    """Get the global MarketState instance."""
    global _market_state
    if _market_state is None:
        raise HTTPException(
            status_code=503,
            detail="Market data not loaded. Initialize the application first."
        )
    return _market_state


def initialize_market_state(data_path: str) -> None:
    """Initialize the global market state from a data file"""
    global _market_state
    try:
        _market_state = load_data(data_path)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize market state: {e}")


def parse_timestamp(ts_str: Optional[str]) -> Optional[pd.Timestamp]:
    """Parse timestamp string to pandas Timestamp"""
    if ts_str is None:
        return None
    try:
        return pd.Timestamp(ts_str, tz="UTC")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timestamp: {ts_str}. Use ISO8601 format."
        )