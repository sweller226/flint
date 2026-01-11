from pydantic import BaseModel, Field
from typing import List, Optional


class CandleResponse(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class CandlesListResponse(BaseModel):
    candles: List[CandleResponse]
    count: int


class ICTLevelsResponse(BaseModel):
    session_high: Optional[float] = None
    session_low: Optional[float] = None
    session_eq: Optional[float] = None
    prev_day_high: Optional[float] = None
    prev_day_low: Optional[float] = None
    prev_day_close: Optional[float] = None


class VolumeBinResponse(BaseModel):
    price_min: float
    price_max: float
    volume: float


class VolumeProfileResponse(BaseModel):
    bins: List[VolumeBinResponse]
    poc_price: Optional[float] = None