from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd

# Reuse the same feature ordering + data cleaning as the forecasting pipeline.
from ..forecaster.forecasting_engine import (
	FEATURE_COLS,
	_compute_median_trading_minutes_per_day,
	_build_day_segments,
	_build_disjoint_week_starts,
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


def build_week_samples(
	*,
	csv_path: Path,
	days_per_sample: int = 7,
	horizon: int = 60,
	stride_days: int = 7,
	min_day_len_ratio: float = 0.9,
	max_samples: Optional[int] = None,
) -> Iterator[WeekSample]:
	"""Yield week samples from a contract CSV.

	This matches the forecasting setup: a sample is 7 day-segments, and the
	last `horizon` minutes are the target hour.
	"""
	df = load_contract_csv(csv_path)
	# Build day segments using the data-driven median minutes/day.
	day_len = _compute_median_trading_minutes_per_day(df)
	data, day_segments, _day_starts = _build_day_segments(
		df,
		day_len=day_len,
		min_day_len_ratio=min_day_len_ratio,
	)

	n_days = len(day_segments)
	if stride_days <= 0:
		raise ValueError("stride_days must be > 0")

	# Disjoint by default (stride 7). If you want overlapping weeks, set stride_days=1.
	if stride_days == days_per_sample:
		week_starts = _build_disjoint_week_starts(n_days, days_per_sample=days_per_sample)
	else:
		week_starts = list(range(0, max(0, n_days - days_per_sample + 1), stride_days))

	total_len = days_per_sample * day_len
	if total_len <= horizon:
		raise ValueError("total_len must be > horizon")
	input_len = total_len - horizon

	yielded = 0
	for start_day_idx in week_starts:
		segs = day_segments[start_day_idx : start_day_idx + days_per_sample]
		parts = [data[s:e] for (s, e) in segs]
		week_arr = np.concatenate(parts, axis=0)

		history = week_arr[:input_len]
		actual_hour = week_arr[input_len:]

		# Build timestamps for the hour from the original df index range.
		# We reconstruct the same indices by concatenating segment ranges.
		seg_indices: List[int] = []
		for (s, e) in segs:
			seg_indices.extend(list(range(s, e)))
		idx_hour = seg_indices[input_len:]
		ts = df.loc[idx_hour, "ts_event"].tolist()

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
