# services/market_state.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Iterator, Any, Tuple

import pandas as pd
import numpy as np


@dataclass(frozen=True)
class Candle:
    timestamp: str  # ISO8601 string
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketState:
    """
    Wraps a pandas DataFrame of Databento-style OHLCV bars and exposes
    iterator + analytics that return pure Python types.
    """

    def __init__(self, df: pd.DataFrame, instrument: str | None = None):
        """
        Parameters
        ----------
        df : DataFrame
            Must contain OHLCV columns:
            ts_event, open, high, low, close, volume
            Optional: symbol or contract_month
        instrument : str, optional
            If provided, filter rows to only this symbol (e.g., "ESH6")
        """

        # Core required columns (symbol is optional - contracts may be in separate files)
        required_cols = {
            "ts_event", "open", "high", "low", "close", "volume"
        }
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Handle symbol/contract_month column aliasing
        if "symbol" not in df.columns and "contract_month" in df.columns:
            df = df.copy()
            df["symbol"] = df["contract_month"]

        df = df.copy()

        # Optional: filter by instrument
        if instrument is not None:
            df = df[df["symbol"] == instrument].copy()

        # Normalize timestamp
        df["timestamp"] = pd.to_datetime(df["ts_event"], utc=True)

        # Convert numeric strings (Databento uses 9-decimal strings)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort + dedupe
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

        self._df = df
        self._cursor = 0

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def reset(self) -> None:
        """Reset the iterator cursor to the beginning."""
        self._cursor = 0

    def load_window_by_time(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
        """Load candles within a specific time window."""
        m = (self._df["timestamp"] >= start_ts) & (self._df["timestamp"] <= end_ts)
        return self._df.loc[m].copy()

    def get_next_bar(self) -> Optional[Candle]:
        """Return the next candle as a pure-Python Candle object."""
        if self._cursor >= len(self._df):
            return None

        row = self._df.iloc[self._cursor]
        self._cursor += 1

        return Candle(
            timestamp=row["timestamp"].isoformat(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )

    def iter_bars(self) -> Iterator[Candle]:
        """Iterate through all candles from the current cursor position."""
        while True:
            bar = self.get_next_bar()
            if bar is None:
                break
            yield bar


def load_data(path: str) -> MarketState:
    """
    Load CSV or Parquet of candles into MarketState.
    
    Args:
        path: Path to CSV or Parquet file
        
    Returns:
        MarketState instance with loaded and validated data
    """
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file format; use CSV or Parquet.")

    return MarketState(df)


def _compute_prev_day_levels(
    df_window: pd.DataFrame,
    ref_date: Optional[pd.Timestamp] = None
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute previous trading day's H/L/C relative to a reference date.
    """
    if df_window.empty:
        return None, None, None

    dts = df_window["timestamp"].dt.date
    
    if ref_date is None:
        current_day = dts.iloc[-1]
    else:
        current_day = ref_date.date() if isinstance(ref_date, pd.Timestamp) else ref_date
    
    # Find previous day with data
    prev_mask = dts < current_day
    if not prev_mask.any():
        return None, None, None

    # Get the most recent previous day
    unique_prev_days = sorted(dts[prev_mask].unique())
    prev_day = unique_prev_days[-1]
    
    prev_day_mask = dts == prev_day
    prev_day_df = df_window.loc[prev_day_mask]

    prev_high = float(prev_day_df["high"].max())
    prev_low = float(prev_day_df["low"].min())
    prev_close = float(prev_day_df["close"].iloc[-1])
    
    return prev_high, prev_low, prev_close


def compute_ict_metrics(
    df_window: pd.DataFrame,
    ref_date: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """
    Compute ICT-style levels on a window. This is ai bullshit Alex I'll leave this to you.
    
    Returns:
        Dict with session_high, session_low, session_eq, prev_day_high/low/close
    """
    if df_window.empty:
        return {
            "session_high": None,
            "session_low": None,
            "session_eq": None,
            "prev_day_high": None,
            "prev_day_low": None,
            "prev_day_close": None,
        }

    # Current session high/low/equilibrium
    session_high = float(df_window["high"].max())
    session_low = float(df_window["low"].min())
    session_eq = (session_high + session_low) / 2.0

    # Compute previous day levels
    prev_high, prev_low, prev_close = _compute_prev_day_levels(df_window, ref_date)

    return {
        "session_high": session_high,
        "session_low": session_low,
        "session_eq": session_eq,
        "prev_day_high": prev_high,
        "prev_day_low": prev_low,
        "prev_day_close": prev_close,
    }


def compute_volume_metrics(
    df_window: pd.DataFrame,
    num_bins: int = 20,
) -> Dict[str, Any]:
    """
    Compute volume profile. This is AI bullshit Alex I'll leave this to you.
    
    Args:
        df_window: DataFrame of candles
        num_bins: Number of price bins for the profile
    
    Returns:
        Dict with bins and poc_price (Point of Control)
    """
    if df_window.empty:
        return {
            "bins": [],
            "poc_price": None,
        }

    price_min = float(df_window["low"].min())
    price_max = float(df_window["high"].max())
    
    if price_min == price_max:
        # Degenerate case: single price
        total_volume = float(df_window["volume"].sum())
        return {
            "bins": [
                {"price_min": price_min, "price_max": price_max, "volume": total_volume}
            ],
            "poc_price": price_min,
        }

    # Construct price bins
    edges = np.linspace(price_min, price_max, num_bins + 1)
    vol_per_bin = np.zeros(num_bins, dtype=float)
    
    for _, row in df_window.iterrows():
        price = row["close"]
        vol = row["volume"]

        # Find the bin for this price
        bin_idx = np.searchsorted(edges, price, side="right") - 1
        bin_idx = max(0, min(bin_idx, num_bins - 1))

        vol_per_bin[bin_idx] += vol


    # Build bins list
    bins: List[Dict[str, float]] = []
    for i in range(num_bins):
        bins.append({
            "price_min": float(edges[i]),
            "price_max": float(edges[i + 1]),
            "volume": float(vol_per_bin[i]),
        })

    # Find Point of Control (highest volume bin)
    poc_idx = np.argmax(vol_per_bin)
    poc_price = float((edges[poc_idx] + edges[poc_idx + 1]) / 2)

    return {
        "bins": bins,
        "poc_price": poc_price,
    }