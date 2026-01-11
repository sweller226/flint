"""Build timestamp-aligned forecasting samples from the Flint futures CSVs.

Goal
- Given a requested start timestamp `t0` (must exist in the CSV), build:
  - model input window `x` ending at `t0 - 1 minute`
  - optional ground-truth window `y` covering `[t0, t0 + 59 minutes]`

Notes
- This uses *row counts* (trading minutes) rather than assuming 24/7 minute continuity.
- It matches the notebook's notion of "trading minutes per day" by estimating the median
  rows/day and using it to set the context length.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DF_COLUMNS = ["ts_event", "open", "high", "low", "close", "volume", "contract_month"]
FEATURE_COLS_DEFAULT = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class SampleSpec:
    start_ts: pd.Timestamp
    horizon: int = 60
    days_per_sample: int = 7
    day_len: int = 390  # will be inferred from data unless explicitly set

    @property
    def total_len(self) -> int:
        return int(self.days_per_sample * self.day_len)

    @property
    def input_len(self) -> int:
        total = self.total_len
        if total <= self.horizon:
            raise ValueError(f"Invalid total_len={total} <= horizon={self.horizon}")
        return int(total - self.horizon)


def parse_utc_timestamp(ts: str) -> pd.Timestamp:
    """Parse an ISO8601 timestamp string into a tz-aware UTC Timestamp."""
    t = pd.to_datetime(ts, utc=True, errors="raise")
    if not isinstance(t, pd.Timestamp):
        t = pd.Timestamp(t)
    if t.tz is None:
        # Force UTC if user passed naive.
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def validate_request_timestamp_range(ts: pd.Timestamp) -> None:
    """Validate the user's allowed timestamp range."""
    lo = pd.Timestamp("2016-01-10 23:00:00+00:00")
    hi = pd.Timestamp("2026-01-08 23:59:00+00:00")
    if ts < lo or ts > hi:
        raise ValueError(f"start_ts={ts.isoformat()} out of allowed range [{lo.isoformat()}, {hi.isoformat()}]")


def load_contract_csv(path: Path, *, log_volume: bool = True) -> pd.DataFrame:
    """Load a Flint contract CSV (df_h.csv etc.) into a clean DataFrame.

    Returns a DataFrame with at least DF_COLUMNS. `ts_event` is tz-aware UTC.

    IMPORTANT:
    - If `log_volume=True`, volume is transformed via log1p(max(volume, 0)).
      This must match the transformation used during training.
    """

    df = pd.read_csv(path)
    missing = [c for c in DF_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    df = df.copy()
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_event"]).sort_values("ts_event").reset_index(drop=True)

    for col in FEATURE_COLS_DEFAULT:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS_DEFAULT)

    if log_volume:
        v = df["volume"].to_numpy(dtype=np.float64)
        v = np.log1p(np.maximum(v, 0.0))
        df["volume"] = v.astype(np.float32)

    for col in FEATURE_COLS_DEFAULT:
        df[col] = df[col].astype(np.float32)

    df["contract_month"] = df["contract_month"].astype(str)
    return df


def compute_median_trading_minutes_per_day(df: pd.DataFrame) -> int:
    """Estimate 'minutes per trading day' as the median row count per date."""
    day_counts = df.groupby(df["ts_event"].dt.date).size().astype(int)
    if day_counts.empty:
        raise ValueError("No daily rows found")
    m = int(day_counts.median())
    if m <= 0:
        raise ValueError("Median daily count is <= 0")
    return m


def find_start_index(
    df: pd.DataFrame,
    start_ts: pd.Timestamp,
    *,
    align: str = "exact",
) -> int:
    """Return the row index for a requested start timestamp.

    align:
      - "exact": require an exact row timestamp match
      - "next":  first row with ts_event >= start_ts
      - "prev":  last row with ts_event <= start_ts
      - "nearest": whichever of prev/next is closer
    """

    align = str(align).strip().lower()
    if align not in {"exact", "next", "prev", "nearest"}:
        raise ValueError(f"align must be one of exact|next|prev|nearest, got {align!r}")

    # Convert tz-aware timestamps to int64 nanoseconds since epoch (UTC)
    # tz-aware datetime64[ns, UTC] -> int64 nanoseconds
    ts_ns = df["ts_event"].astype("int64").to_numpy()
    target_ns = int(start_ts.value)

    if align == "exact":
        matches = np.where(ts_ns == target_ns)[0]
        if matches.size == 0:
            raise KeyError(
                f"start_ts={start_ts.isoformat()} not found in data. "
                "Use --align next/prev/nearest to snap to an available timestamp."
            )
        return int(matches[0])

    # searchsorted requires sorted ts (load_contract_csv sorts)
    insert = int(np.searchsorted(ts_ns, target_ns, side="left"))

    if align == "next":
        if insert >= len(ts_ns):
            raise KeyError(f"No row timestamp >= {start_ts.isoformat()} (past end of data).")
        return insert

    if align == "prev":
        idx = insert - 1 if insert > 0 else 0
        if ts_ns[idx] > target_ns:
            raise KeyError(f"No row timestamp <= {start_ts.isoformat()} (before start of data).")
        return int(idx)

    # nearest
    prev_idx = max(insert - 1, 0)
    next_idx = min(insert, len(ts_ns) - 1)
    prev_dt = abs(ts_ns[prev_idx] - target_ns)
    next_dt = abs(ts_ns[next_idx] - target_ns)
    return int(prev_idx if prev_dt <= next_dt else next_idx)


def build_sample_windows(
    df: pd.DataFrame,
    *,
    start_ts: pd.Timestamp,
    horizon: int = 60,
    days_per_sample: int = 7,
    day_len: Optional[int] = None,
    feature_cols: Sequence[str] = FEATURE_COLS_DEFAULT,

    align: str = "exact",
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, str, int, pd.Timestamp]:
    """Build (x_abs, y_abs, y_timestamps, contract_month, start_idx, aligned_start_ts).

    - x_abs: (input_len, F) float32
    - y_abs: (horizon, F) float32 (ground truth for that hour)
    - y_timestamps: DatetimeIndex of length horizon (exact from df)

    This uses the previous `input_len` rows as context and the next `horizon` rows as target.
    """

    if day_len is None:
        day_len = compute_median_trading_minutes_per_day(df)

    spec = SampleSpec(start_ts=start_ts, horizon=horizon, days_per_sample=days_per_sample, day_len=int(day_len))
    start_idx = find_start_index(df, start_ts, align=align)
    aligned_start_ts = pd.to_datetime(df["ts_event"].iloc[start_idx], utc=True)

    if start_idx < spec.input_len:
        earliest_ok_idx = spec.input_len
        if earliest_ok_idx < len(df):
            earliest_ok_ts = pd.to_datetime(df["ts_event"].iloc[earliest_ok_idx], utc=True)
            hint = f" Earliest usable start_ts is about {earliest_ok_ts.isoformat()} (needs {spec.input_len} prior rows)."
        else:
            hint = ""
        raise ValueError(
            f"Not enough history before {start_ts.isoformat()} to build input window: "
            f"need {spec.input_len} rows but start_idx={start_idx}." + hint
        )

    end_idx = start_idx + spec.horizon
    if end_idx > len(df):
        latest_ok_idx = len(df) - spec.horizon
        if latest_ok_idx >= 0 and latest_ok_idx < len(df):
            latest_ok_ts = pd.to_datetime(df["ts_event"].iloc[latest_ok_idx], utc=True)
            hint = f" Latest usable start_ts is about {latest_ok_ts.isoformat()} (needs {spec.horizon} future rows)."
        else:
            hint = ""
        raise ValueError(
            f"Not enough future rows after {start_ts.isoformat()} to build a {spec.horizon}-step forecast: "
            f"need end_idx={end_idx} but len(df)={len(df)}." + hint
        )

    x_df = df.iloc[start_idx - spec.input_len : start_idx]
    y_df = df.iloc[start_idx : end_idx]

    x_abs = x_df[list(feature_cols)].to_numpy(dtype=np.float32, copy=True)
    y_abs = y_df[list(feature_cols)].to_numpy(dtype=np.float32, copy=True)

    y_ts = pd.DatetimeIndex(y_df["ts_event"].astype("datetime64[ns, UTC]").to_numpy())
    contract_month = str(y_df["contract_month"].iloc[0])

    return x_abs, y_abs, y_ts, contract_month, start_idx, aligned_start_ts


def build_forecast_timestamps(start_ts: pd.Timestamp, horizon: int = 60) -> pd.DatetimeIndex:
    """Build an exact 1-minute grid from start_ts for `horizon` minutes (UTC)."""
    if start_ts.tz is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    return pd.date_range(start=start_ts, periods=int(horizon), freq="1min", tz="UTC")
