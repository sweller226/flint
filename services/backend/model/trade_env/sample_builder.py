from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd

# Reuse the same feature ordering + data cleaning as the forecasting pipeline.
from ..forecaster.sample_builder import (
	FEATURE_COLS,
	build_day_segments,
	build_disjoint_week_starts,
	compute_median_trading_minutes_per_day,
	load_contract_csv,
)


@dataclass(frozen=True)
class WeekSample:
	"""One 7-trading-day sample.

	- `history`: first part of the week window (all but last hour)
	- `actual_hour`: last 60 minutes of the window
	- `timestamps_hour`: timestamps aligned with `actual_hour`
	"""

	history: np.ndarray
	actual_hour: np.ndarray
	timestamps_hour: List[pd.Timestamp]
	contract_month: str
	forecast_hour: Optional[np.ndarray] = None


def build_week_samples(
	*,
	csv_path: Path,
	days_per_sample: int = 7,
	horizon: int = 60,
	bar_minutes: int = 1,
	stride_days: int = 7,
	min_day_len_ratio: float = 0.9,
	max_samples: Optional[int] = None,
) -> Iterator[WeekSample]:
	"""Yield week samples from a contract CSV.

	This matches the forecasting setup: a sample is 7 day-segments, and the
	last `horizon` minutes are the target hour.
	"""
	# RL env expects raw OHLCV scale; do not log-transform volume.
	df = load_contract_csv(csv_path, log_volume=False)
	bar_minutes = int(bar_minutes)
	if bar_minutes <= 0:
		raise ValueError("bar_minutes must be > 0")
	if int(horizon) % bar_minutes != 0:
		raise ValueError(f"horizon={horizon} must be divisible by bar_minutes={bar_minutes}")
	# Build day segments using the data-driven median minutes/day.
	day_len = compute_median_trading_minutes_per_day(df)
	if day_len % bar_minutes != 0:
		raise ValueError(f"day_len={day_len} must be divisible by bar_minutes={bar_minutes}")
	data, day_segments, _day_starts = build_day_segments(
		df,
		day_len=day_len,
		min_day_len_ratio=min_day_len_ratio,
		feature_cols=FEATURE_COLS,
	)

	n_days = len(day_segments)
	if stride_days <= 0:
		raise ValueError("stride_days must be > 0")

	# Disjoint by default (stride 7). If you want overlapping weeks, set stride_days=1.
	if stride_days == days_per_sample:
		week_starts = build_disjoint_week_starts(n_days, days_per_sample=days_per_sample)
	else:
		week_starts = list(range(0, max(0, n_days - days_per_sample + 1), stride_days))

	total_len = days_per_sample * day_len
	if total_len <= horizon:
		raise ValueError("total_len must be > horizon")
	input_len = total_len - horizon
	if input_len % bar_minutes != 0:
		raise ValueError("input_len must be divisible by bar_minutes")

	def _agg_chunks(x: np.ndarray) -> np.ndarray:
		if bar_minutes == 1:
			return x.astype(np.float32, copy=False)
		if x.shape[0] % bar_minutes != 0:
			raise ValueError("array length not divisible by bar_minutes")
		out = np.zeros((x.shape[0] // bar_minutes, x.shape[1]), dtype=np.float32)
		for i in range(out.shape[0]):
			chunk = x[i * bar_minutes : (i + 1) * bar_minutes]
			out[i, 0] = float(chunk[0, 0])
			out[i, 1] = float(np.max(chunk[:, 1]))
			out[i, 2] = float(np.min(chunk[:, 2]))
			out[i, 3] = float(chunk[-1, 3])
			out[i, 4] = float(np.sum(chunk[:, 4]))
		return out

	yielded = 0
	for start_day_idx in week_starts:
		segs = day_segments[start_day_idx : start_day_idx + days_per_sample]
		parts = [data[s:e] for (s, e) in segs]
		week_arr = np.concatenate(parts, axis=0)

		history = week_arr[:input_len]
		actual_hour = week_arr[input_len:]
		history = _agg_chunks(history)
		actual_hour = _agg_chunks(actual_hour)

		# Build timestamps for the hour from the original df index range.
		# We reconstruct the same indices by concatenating segment ranges.
		seg_indices: List[int] = []
		for (s, e) in segs:
			seg_indices.extend(list(range(s, e)))
		idx_hour = seg_indices[input_len:]
		ts = df.loc[idx_hour, "ts_event"].tolist()
		if bar_minutes != 1:
			# Use the bar-close timestamp (last timestamp in each chunk).
			ts = [ts[(i + 1) * bar_minutes - 1] for i in range(len(ts) // bar_minutes)]

		contract_month = str(df["contract_month"].iloc[idx_hour[0]])
		yield WeekSample(
			history=history.astype(np.float32, copy=False),
			actual_hour=actual_hour.astype(np.float32, copy=False),
			timestamps_hour=list(ts),
			contract_month=contract_month,
		)
		yielded += 1
		if max_samples is not None and yielded >= max_samples:
			break
