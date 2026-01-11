
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix


FEATURE_COLS = ["open", "high", "low", "close", "volume"]
CLOSE_IDX = FEATURE_COLS.index("close")


def _require_torch():
	try:
		import torch  # noqa: F401
	except Exception as exc:  # pragma: no cover
		raise RuntimeError(
			"PyTorch is required for LSTM forecasting. Install it with `pip install torch`."
		) from exc


def _data_dir() -> Path:
	# services/backend/model/forecaster/forecasting_engine.py -> services/backend/model/data
	return Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class RunConfig:
	# Data/splitting
	data_dir: Optional[Path] = None
	train_frac: float = 0.8
	val_frac: float = 0.1
	test_frac: float = 0.1
	min_day_len_ratio: float = 0.9
	horizon: int = 60
	days_per_sample: int = 7
	stride_days: int = 1  # 7 for fully disjoint windows

	# Training
	batch_size: int = 32
	final_epochs: int = 5

	# Hyperparameter tuning
	tune_trials: int = 8
	tune_epochs: int = 2
	max_train_batches_per_epoch: int = 200  # keeps tuning fast
	max_val_batches: int = 50
	rng_seed: int = 42

	# Model family
	model_family: str = "direct_lstm"  # "direct_lstm" or "seq2seq_lstm"

	# Output
	artifacts_dir: Path = Path(__file__).resolve().parent / "artifacts"
	plot_contract: str = "H"
	plot_idx: int = 0
	plot_split: str = "test"  # train/val/test
	plot_show: bool = True
	plot_path: Optional[Path] = None

	def resolved_data_dir(self) -> Path:
		return self.data_dir if self.data_dir is not None else _data_dir()


def load_contract_csv(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	expected = {"ts_event", "open", "high", "low", "close", "volume", "contract_month"}
	missing = expected - set(df.columns)
	if missing:
		raise ValueError(f"{path.name} missing columns: {sorted(missing)}")

	df = df.copy()
	df["ts_event"] = pd.to_datetime(df["ts_event"], errors="coerce")
	df = df.dropna(subset=["ts_event"])
	df = df.sort_values("ts_event").reset_index(drop=True)

	for col in FEATURE_COLS:
		df[col] = pd.to_numeric(df[col], errors="coerce")
	df = df.dropna(subset=FEATURE_COLS)

	# Ensure a consistent dtype for modeling
	for col in FEATURE_COLS:
		df[col] = df[col].astype(np.float32)

	df["contract_month"] = df["contract_month"].astype(str)
	return df


@dataclass(frozen=True)
class SplitSpec:
	train_frac: float = 0.8
	val_frac: float = 0.1
	test_frac: float = 0.1

	def __post_init__(self):
		total = self.train_frac + self.val_frac + self.test_frac
		if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
			raise ValueError(f"Split fractions must sum to 1.0, got {total}")


def _compute_median_trading_minutes_per_day(df: pd.DataFrame) -> int:
	# Estimate "trading minutes per day" from the data itself.
	day_counts = df.groupby(df["ts_event"].dt.date).size().astype(int)
	if day_counts.empty:
		raise ValueError("No daily rows found")
	median_count = int(day_counts.median())
	if median_count <= 0:
		raise ValueError("Median daily count is <= 0")
	return median_count


def _build_day_segments(
	df: pd.DataFrame,
	day_len: int,
	min_day_len_ratio: float = 0.9,
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[pd.Timestamp]]:
	"""Return:
	- data array of shape (n_rows, 5)
	- list of (seg_start, seg_end) indices (each segment has length day_len)
	- list of day start timestamps (aligned with segments)

	We select the *last* `day_len` rows of each sufficiently-complete day.
	"""
	if day_len <= 60:
		raise ValueError("day_len must be > 60 to hold a 1-hour target")

	df = df.sort_values("ts_event").reset_index(drop=True)
	data = df[FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)

	groups = df.groupby(df["ts_event"].dt.date, sort=True)
	day_segments: List[Tuple[int, int]] = []
	day_starts: List[pd.Timestamp] = []
	min_len = int(day_len * min_day_len_ratio)

	for _, g in groups:
		if len(g) < min_len:
			continue
		# Indices in the original df
		day_start_idx = int(g.index.min())
		day_end_idx = int(g.index.max()) + 1

		if (day_end_idx - day_start_idx) < day_len:
			continue
		seg_end = day_end_idx
		seg_start = seg_end - day_len
		day_segments.append((seg_start, seg_end))
		day_starts.append(pd.Timestamp(g["ts_event"].iloc[0]))

	if not day_segments:
		raise ValueError("No valid day segments were created; check day_len/min_day_len_ratio")

	return data, day_segments, day_starts


def _build_disjoint_week_starts(num_days: int, days_per_sample: int = 7) -> List[int]:
	starts: List[int] = []
	i = 0
	while i + days_per_sample <= num_days:
		starts.append(i)
		i += days_per_sample
	return starts


def _split_indices(n: int, split: SplitSpec) -> Tuple[range, range, range]:
	if n <= 0:
		raise ValueError("n must be > 0")
	train_end = int(n * split.train_frac)
	val_end = train_end + int(n * split.val_frac)
	train_end = max(1, train_end)
	val_end = max(train_end + 1, val_end) if n >= 3 else min(n, train_end)
	val_end = min(n, val_end)
	return range(0, train_end), range(train_end, val_end), range(val_end, n)


class WeekToHourDataset:
	"""7 trading days -> predict last hour (60 minutes), disjoint samples.

	Samples are built from 7 consecutive "day segments" each of length `day_len`.
	"""

	def __init__(
		self,
		*,
		contract_name: str,
		data: np.ndarray,
		day_segments: List[Tuple[int, int]],
		week_starts: List[int],
		day_len: int,
		days_per_sample: int = 7,
		horizon: int = 60,
		mean: Optional[np.ndarray] = None,
		std: Optional[np.ndarray] = None,
	):
		self.contract_name = contract_name
		self.data = data
		self.day_segments = day_segments
		self.week_starts = week_starts
		self.day_len = day_len
		self.days_per_sample = days_per_sample
		self.horizon = horizon
		self.mean = mean
		self.std = std

		self.total_len = self.days_per_sample * day_len
		if self.total_len <= horizon:
			raise ValueError("Total sample length must be > horizon")
		self.input_len = self.total_len - horizon

	def __len__(self) -> int:
		return len(self.week_starts)

	def set_scaler(self, mean: np.ndarray, std: np.ndarray) -> None:
		self.mean = mean.astype(np.float32)
		self.std = std.astype(np.float32)

	def _get_week_array(self, start_day_idx: int) -> np.ndarray:
		segs = self.day_segments[start_day_idx : start_day_idx + self.days_per_sample]
		parts = [self.data[s:e] for (s, e) in segs]
		return np.concatenate(parts, axis=0)

	def __getitem__(self, idx: int):
		_require_torch()
		import torch

		start_day = self.week_starts[idx]
		seq = self._get_week_array(start_day)
		x = seq[: self.input_len]
		y = seq[self.input_len :]

		if self.mean is not None and self.std is not None:
			x = (x - self.mean) / self.std
			y = (y - self.mean) / self.std

		return (
			torch.from_numpy(x).to(torch.float32),
			torch.from_numpy(y).to(torch.float32),
		)


def compute_global_scaler_from_train(
	train_datasets: Sequence[WeekToHourDataset],
	*,
	eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Compute mean/std over *inputs only* across all training samples."""
	sum_vec = np.zeros((len(FEATURE_COLS),), dtype=np.float64)
	sumsq_vec = np.zeros((len(FEATURE_COLS),), dtype=np.float64)
	count = 0

	for ds in train_datasets:
		for start_day in ds.week_starts:
			seq = ds._get_week_array(start_day)
			x = seq[: ds.input_len]
			sum_vec += x.sum(axis=0)
			sumsq_vec += (x * x).sum(axis=0)
			count += x.shape[0]

	if count == 0:
		raise ValueError("No training rows to compute scaler")
	mean = (sum_vec / count).astype(np.float32)
	var = (sumsq_vec / count) - (mean.astype(np.float64) ** 2)
	std = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
	std = np.where(std < eps, np.float32(1.0), std)
	return mean, std


def build_datasets_for_contract(
	contract_name: str,
	df: pd.DataFrame,
	*,
	split: SplitSpec,
	day_len_override: Optional[int] = None,
	days_per_sample: int = 7,
	stride_days: int = 1,
	horizon: int = 60,
	min_day_len_ratio: float = 0.9,
) -> Dict[str, WeekToHourDataset]:
	day_len = day_len_override or _compute_median_trading_minutes_per_day(df)
	data, day_segments, _day_starts = _build_day_segments(
		df,
		day_len=day_len,
		min_day_len_ratio=min_day_len_ratio,
	)

	if days_per_sample < 2:
		raise ValueError("days_per_sample must be >= 2")
	if stride_days < 1:
		raise ValueError("stride_days must be >= 1")
	if len(day_segments) < days_per_sample:
		raise ValueError(
			f"Not enough days to form a sample for {contract_name}: "
			f"days={len(day_segments)} days_per_sample={days_per_sample}"
		)

	# Leakage-safe split based on day indices (not sample indices), so that
	# overlapping windows never share days across train/val/test.
	n_days = len(day_segments)
	train_day_end = int(n_days * split.train_frac)
	val_day_end = int(n_days * (split.train_frac + split.val_frac))
	train_day_end = max(days_per_sample, train_day_end)
	val_day_end = max(train_day_end + days_per_sample, val_day_end)
	val_day_end = min(n_days, val_day_end)

	def _starts_in_range(day_start: int, day_end: int) -> List[int]:
		last_start = day_end - days_per_sample
		if last_start < day_start:
			return []
		return list(range(day_start, last_start + 1, stride_days))

	train_starts = _starts_in_range(0, train_day_end)
	val_starts = _starts_in_range(train_day_end, val_day_end)
	test_starts = _starts_in_range(val_day_end, n_days)

	if len(train_starts) < 5 or len(val_starts) < 1 or len(test_starts) < 1:
		raise ValueError(
			f"Not enough samples after split for {contract_name}: "
			f"train={len(train_starts)} val={len(val_starts)} test={len(test_starts)} "
			f"(days={n_days}, days_per_sample={days_per_sample}, stride={stride_days})"
		)

	datasets = {
		"train": WeekToHourDataset(
			contract_name=contract_name,
			data=data,
			day_segments=day_segments,
			week_starts=train_starts,
			day_len=day_len,
			days_per_sample=days_per_sample,
			horizon=horizon,
		),
		"val": WeekToHourDataset(
			contract_name=contract_name,
			data=data,
			day_segments=day_segments,
			week_starts=val_starts,
			day_len=day_len,
			days_per_sample=days_per_sample,
			horizon=horizon,
		),
		"test": WeekToHourDataset(
			contract_name=contract_name,
			data=data,
			day_segments=day_segments,
			week_starts=test_starts,
			day_len=day_len,
			days_per_sample=days_per_sample,
			horizon=horizon,
		),
	}
	return datasets


def build_seq2seq_lstm_forecaster(
	*,
	input_size: int = 5,
	hidden_size: int = 128,
	num_layers: int = 2,
	dropout: float = 0.1,
):
	"""Factory that returns an `nn.Module` without relying on unsafe dynamic base changes."""
	_require_torch()
	import torch
	from torch import nn

	class Seq2SeqLSTMForecaster(nn.Module):
		def __init__(self):
			super().__init__()
			self.encoder = nn.LSTM(
				input_size=input_size,
				hidden_size=hidden_size,
				num_layers=num_layers,
				batch_first=True,
				dropout=dropout if num_layers > 1 else 0.0,
			)
			self.decoder = nn.LSTM(
				input_size=input_size,
				hidden_size=hidden_size,
				num_layers=num_layers,
				batch_first=True,
				dropout=dropout if num_layers > 1 else 0.0,
			)
			self.proj = nn.Linear(hidden_size, input_size)

		def forward(
			self,
			x,
			y=None,
			*,
			horizon: int = 60,
			teacher_forcing: bool = True,
			target_is_delta: bool = False,
		):
			"""x: (B, Tin, 5)
			y: (B, Tout, 5) optional (for teacher forcing)
			returns: (B, horizon, 5)
			"""
			_, (h, c) = self.encoder(x)

			last_x = x[:, -1:, :]
			start_token = torch.zeros_like(last_x) if target_is_delta else last_x

			if teacher_forcing and y is not None:
				decoder_in = torch.cat([start_token, y[:, :-1, :]], dim=1)
				dec_out, _ = self.decoder(decoder_in, (h, c))
				return self.proj(dec_out)

			preds = []
			inp = start_token
			state = (h, c)
			for _ in range(horizon):
				dec_out, state = self.decoder(inp, state)
				step = self.proj(dec_out)  # (B, 1, 5)
				preds.append(step)
				inp = step
			return torch.cat(preds, dim=1)

	return Seq2SeqLSTMForecaster()


def build_direct_lstm_forecaster(
	*,
	input_size: int = 5,
	hidden_size: int = 128,
	num_layers: int = 2,
	dropout: float = 0.1,
	horizon: int = 60,
	output_size: int = 5,
):
	"""Direct multi-step forecaster.

	Encodes the full input sequence and projects the final hidden state to all horizon steps at once.
	This tends to be more stable than seq2seq early on (less exposure bias).
	"""
	_require_torch()
	import torch
	from torch import nn

	class DirectLSTMForecaster(nn.Module):
		def __init__(self):
			super().__init__()
			self.lstm = nn.LSTM(
				input_size=input_size,
				hidden_size=hidden_size,
				num_layers=num_layers,
				batch_first=True,
				dropout=dropout if num_layers > 1 else 0.0,
			)
			self.head = nn.Sequential(
				nn.Linear(hidden_size, hidden_size),
				nn.ReLU(),
				nn.Linear(hidden_size, horizon * output_size),
			)
			self.horizon = horizon
			self.output_size = output_size

		def forward(self, x, y=None, *, horizon: Optional[int] = None, teacher_forcing: bool = True):
			_, (h, _c) = self.lstm(x)
			last = h[-1]  # (B, hidden)
			h = horizon if horizon is not None else self.horizon
			out = self.head(last).view(x.shape[0], h, self.output_size)
			return out

	return DirectLSTMForecaster()


def evaluate_model(model, loader, device) -> float:
	_require_torch()
	import torch
	from torch import nn

	model.eval()
	loss_fn = nn.MSELoss()
	losses = []
	with torch.no_grad():
		for xb, yb in loader:
			xb = xb.to(device)
			yb = yb.to(device)
			pred = model(xb, yb, horizon=yb.shape[1], teacher_forcing=True)
			loss = loss_fn(pred, yb)
			losses.append(loss.item())
	return float(np.mean(losses)) if losses else float("nan")


def evaluate_model_limited(model, loader, device, *, max_batches: Optional[int] = None) -> float:
	"""Same as evaluate_model but caps the number of batches (useful for fast tuning)."""
	_require_torch()
	import torch
	from torch import nn

	model.eval()
	loss_fn = nn.MSELoss()
	losses = []
	with torch.no_grad():
		for i, (xb, yb) in enumerate(loader):
			if max_batches is not None and i >= max_batches:
				break
			xb = xb.to(device)
			yb = yb.to(device)
			pred = model(xb, yb, horizon=yb.shape[1], teacher_forcing=True)
			losses.append(loss_fn(pred, yb).item())
	return float(np.mean(losses)) if losses else float("nan")


def _predict_one(model, dataset: WeekToHourDataset, idx: int, device) -> Tuple[np.ndarray, np.ndarray]:
	"""Returns (pred, actual) in original units, arrays shaped (horizon, 5)."""
	_require_torch()
	import torch

	model.eval()
	xb, yb = dataset[idx]
	with torch.no_grad():
		pred = model(xb.unsqueeze(0).to(device), yb.unsqueeze(0).to(device), horizon=yb.shape[0])
		pred = pred.squeeze(0).cpu().numpy().astype(np.float32)
	actual = yb.cpu().numpy().astype(np.float32)

	if dataset.mean is not None and dataset.std is not None:
		pred = pred * dataset.std + dataset.mean
		actual = actual * dataset.std + dataset.mean

	return pred, actual


def plot_forecast_vs_actual(
	*,
	model,
	dataset: WeekToHourDataset,
	idx: int,
	device,
	save_path: Optional[Path] = None,
	show: bool = True,
):
	"""Plot predicted vs actual for the last-hour horizon for a single sample."""
	pred, actual = _predict_one(model, dataset, idx, device)
	t = np.arange(pred.shape[0])

	fig, axes = plt.subplots(nrows=len(FEATURE_COLS), ncols=1, figsize=(10, 10), sharex=True)
	if len(FEATURE_COLS) == 1:
		axes = [axes]
	for i, col in enumerate(FEATURE_COLS):
		ax = axes[i]
		ax.plot(t, actual[:, i], label="actual", linewidth=2)
		ax.plot(t, pred[:, i], label="forecast", linewidth=2)
		ax.set_ylabel(col)
		ax.grid(True, alpha=0.3)
		if i == 0:
			ax.legend(loc="best")
	axes[-1].set_xlabel("minute (last hour)")
	fig.suptitle(f"{dataset.contract_name} sample={idx} (last-hour forecast)")
	fig.tight_layout()

	if save_path is not None:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, dpi=160)
		print("plot_saved:", str(save_path))
	if show:
		plt.show()
	plt.close(fig)


def train(
	model,
	*,
	train_loader,
	val_loaders: Dict[str, object],
	epochs: int,
	lr: float,
	device,
) -> None:
	_require_torch()
	import torch
	from torch import nn

	opt = torch.optim.AdamW(model.parameters(), lr=lr)
	loss_fn = nn.MSELoss()

	for epoch in range(1, epochs + 1):
		model.train()
		train_losses = []
		for xb, yb in train_loader:
			xb = xb.to(device)
			yb = yb.to(device)
			opt.zero_grad(set_to_none=True)
			pred = model(xb, yb, horizon=yb.shape[1], teacher_forcing=True)
			loss = loss_fn(pred, yb)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			opt.step()
			train_losses.append(loss.item())

		train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
		val_report = {
			name: evaluate_model(model, loader, device) for name, loader in val_loaders.items()
		}
		val_str = " ".join([f"{k}={v:.6f}" for k, v in val_report.items()])
		print(f"epoch={epoch} train_mse={train_loss:.6f} {val_str}")


def train_limited(
	model,
	*,
	train_loader,
	val_loaders: Dict[str, object],
	epochs: int,
	lr: float,
	device,
	max_train_batches_per_epoch: Optional[int] = None,
	max_val_batches: Optional[int] = None,
) -> Dict[str, float]:
	"""Train for a few epochs and return a val report (used by hyperparameter tuning)."""
	_require_torch()
	import torch
	from torch import nn

	opt = torch.optim.AdamW(model.parameters(), lr=lr)
	loss_fn = nn.MSELoss()

	for _epoch in range(1, epochs + 1):
		model.train()
		for i, (xb, yb) in enumerate(train_loader):
			if max_train_batches_per_epoch is not None and i >= max_train_batches_per_epoch:
				break
			xb = xb.to(device)
			yb = yb.to(device)
			opt.zero_grad(set_to_none=True)
			pred = model(xb, yb, horizon=yb.shape[1], teacher_forcing=True)
			loss = loss_fn(pred, yb)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			opt.step()

	val_report = {
		name: evaluate_model_limited(model, loader, device, max_batches=max_val_batches)
		for name, loader in val_loaders.items()
	}
	return {k: float(v) for k, v in val_report.items()}


def _mean_metric(report: Dict[str, float]) -> float:
	vals = [v for v in report.values() if not (math.isnan(v) or math.isinf(v))]
	return float(np.mean(vals)) if vals else float("inf")


def tune_hyperparameters(
	*,
	config: RunConfig,
	contract_splits: Dict[str, Dict[str, WeekToHourDataset]],
	device,
):
	"""Lightweight random search using val MSE averaged across contracts."""
	_require_torch()
	import torch
	from torch.utils.data import ConcatDataset, DataLoader

	rng = random.Random(config.rng_seed)

	# Candidate spaces (kept small & sane for hackathon speed)
	hidden_sizes = [64, 128, 256]
	num_layers_list = [1, 2, 3]
	dropouts = [0.0, 0.1, 0.2]
	lrs = [1e-3, 3e-4, 1e-4]

	train_concat = ConcatDataset([contract_splits[c]["train"] for c in contract_splits])
	train_loader = DataLoader(
		train_concat,
		batch_size=config.batch_size,
		shuffle=True,
		num_workers=0,
		drop_last=True,
	)
	val_loaders = {
		f"val_{c}": DataLoader(
			contract_splits[c]["val"],
			batch_size=config.batch_size,
			shuffle=False,
			num_workers=0,
		)
		for c in contract_splits
	}

	best = {"score": float("inf"), "params": None, "report": None}

	for trial in range(1, config.tune_trials + 1):
		hidden = rng.choice(hidden_sizes)
		layers = rng.choice(num_layers_list)
		drop = rng.choice(dropouts)
		lr = rng.choice(lrs)

		if config.model_family == "seq2seq_lstm":
			model = build_seq2seq_lstm_forecaster(
				input_size=len(FEATURE_COLS),
				hidden_size=hidden,
				num_layers=layers,
				dropout=drop,
			).to(device)
		else:
			model = build_direct_lstm_forecaster(
				input_size=len(FEATURE_COLS),
				hidden_size=hidden,
				num_layers=layers,
				dropout=drop,
				horizon=config.horizon,
				output_size=len(FEATURE_COLS),
			).to(device)

		report = train_limited(
			model,
			train_loader=train_loader,
			val_loaders=val_loaders,
			epochs=config.tune_epochs,
			lr=lr,
			device=device,
			max_train_batches_per_epoch=config.max_train_batches_per_epoch,
			max_val_batches=config.max_val_batches,
		)
		score = _mean_metric(report)
		print(f"tune trial={trial}/{config.tune_trials} score={score:.6f} params={{'hidden':{hidden},'layers':{layers},'dropout':{drop},'lr':{lr}}} report={report}")
		if score < best["score"]:
			best = {
				"score": score,
				"params": {"hidden_size": hidden, "num_layers": layers, "dropout": drop, "lr": lr},
				"report": report,
			}

	print("best_tune:", best)
	return best


def _last_input_close_unscaled(dataset: WeekToHourDataset, idx: int) -> float:
	start_day = dataset.week_starts[idx]
	seq = dataset._get_week_array(start_day)
	last_input = seq[dataset.input_len - 1]
	return float(last_input[CLOSE_IDX])


def compute_direction_confusion_matrix(
	*,
	model,
	test_sets: Dict[str, WeekToHourDataset],
	device,
	max_samples_per_contract: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Directional confusion matrix for close moves.

	We binarize each forecast minute as up(1) if close_t - close_{t-1} > 0 else down/flat(0).
	"""
	y_true: List[int] = []
	y_pred: List[int] = []

	for c, ds in test_sets.items():
		n = min(len(ds), max_samples_per_contract)
		for i in range(n):
			pred, actual = _predict_one(model, ds, i, device)
			last_close = _last_input_close_unscaled(ds, i)
			pred_close = pred[:, CLOSE_IDX]
			act_close = actual[:, CLOSE_IDX]

			pred_prev = np.concatenate([[last_close], pred_close[:-1]])
			act_prev = np.concatenate([[last_close], act_close[:-1]])

			pred_dir = (pred_close - pred_prev > 0).astype(np.int32)
			act_dir = (act_close - act_prev > 0).astype(np.int32)

			y_true.extend(act_dir.tolist())
			y_pred.extend(pred_dir.tolist())

	cm = confusion_matrix(np.array(y_true), np.array(y_pred), labels=[0, 1])
	return cm, np.array([len(y_true)])



def main() -> None:
	config = RunConfig()

	_require_torch()
	import torch
	from torch.utils.data import ConcatDataset, DataLoader

	random.seed(config.rng_seed)
	np.random.seed(config.rng_seed)
	try:
		torch.manual_seed(config.rng_seed)
	except Exception:
		pass

	data_dir = config.resolved_data_dir()
	files = {
		"H": data_dir / "df_h.csv",
		"M": data_dir / "df_m.csv",
		"U": data_dir / "df_u.csv",
		"Z": data_dir / "df_z.csv",
	}

	split = SplitSpec(config.train_frac, config.val_frac, config.test_frac)

	# Load all contracts first so we can pick a single global `day_len`.
	contract_dfs: Dict[str, pd.DataFrame] = {}
	median_day_lens: Dict[str, int] = {}
	for contract_name, path in files.items():
		if not path.exists():
			raise FileNotFoundError(f"Missing {path}")
		df = load_contract_csv(path)
		contract_dfs[contract_name] = df
		median_day_lens[contract_name] = _compute_median_trading_minutes_per_day(df)

	global_day_len = int(min(median_day_lens.values()))
	if global_day_len <= config.horizon:
		raise ValueError(
			f"global_day_len ({global_day_len}) must be > horizon ({config.horizon})"
		)
	print("median_day_lens:", median_day_lens, "global_day_len:", global_day_len)

	contract_splits: Dict[str, Dict[str, WeekToHourDataset]] = {}
	for contract_name, df in contract_dfs.items():
		contract_splits[contract_name] = build_datasets_for_contract(
			contract_name,
			df,
			split=split,
			day_len_override=global_day_len,
			days_per_sample=config.days_per_sample,
			stride_days=config.stride_days,
			horizon=config.horizon,
			min_day_len_ratio=config.min_day_len_ratio,
		)
		print(
			f"{contract_name}: train={len(contract_splits[contract_name]['train'])} "
			f"val={len(contract_splits[contract_name]['val'])} "
			f"test={len(contract_splits[contract_name]['test'])} "
			f"day_len={contract_splits[contract_name]['train'].day_len} "
			f"input_len={contract_splits[contract_name]['train'].input_len}"
		)

	# Global scaler computed from all training inputs across all contract months
	train_datasets = [contract_splits[c]["train"] for c in contract_splits]
	mean, std = compute_global_scaler_from_train(train_datasets)
	for c in contract_splits:
		for split_name in ("train", "val", "test"):
			contract_splits[c][split_name].set_scaler(mean, std)

	print("global_scaler:", {k: float(v) for k, v in zip(FEATURE_COLS, mean)})

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("device:", device)

	# Hyperparameter tuning
	best = tune_hyperparameters(config=config, contract_splits=contract_splits, device=device)
	params = best["params"]
	assert params is not None

	# Final training
	if config.model_family == "seq2seq_lstm":
		model = build_seq2seq_lstm_forecaster(
			input_size=len(FEATURE_COLS),
			hidden_size=params["hidden_size"],
			num_layers=params["num_layers"],
			dropout=params["dropout"],
		).to(device)
	else:
		model = build_direct_lstm_forecaster(
			input_size=len(FEATURE_COLS),
			hidden_size=params["hidden_size"],
			num_layers=params["num_layers"],
			dropout=params["dropout"],
			horizon=config.horizon,
			output_size=len(FEATURE_COLS),
		).to(device)

	train_concat = ConcatDataset([contract_splits[c]["train"] for c in contract_splits])
	train_loader = DataLoader(
		train_concat,
		batch_size=config.batch_size,
		shuffle=True,
		num_workers=0,
		drop_last=True,
	)
	val_loaders = {
		f"val_{c}": DataLoader(
			contract_splits[c]["val"],
			batch_size=config.batch_size,
			shuffle=False,
			num_workers=0,
		)
		for c in contract_splits
	}

	train(
		model,
		train_loader=train_loader,
		val_loaders=val_loaders,
		epochs=config.final_epochs,
		lr=params["lr"],
		device=device,
	)

	test_loaders = {
		f"test_{c}": DataLoader(
			contract_splits[c]["test"],
			batch_size=config.batch_size,
			shuffle=False,
			num_workers=0,
		)
		for c in contract_splits
	}
	test_report = {name: evaluate_model(model, loader, device) for name, loader in test_loaders.items()}
	print("test_mse:", {k: float(v) for k, v in test_report.items()})

	# Directional confusion matrix over test sets
	test_sets = {c: contract_splits[c]["test"] for c in contract_splits}
	cm, n = compute_direction_confusion_matrix(model=model, test_sets=test_sets, device=device)
	print("test_direction_confusion_matrix (rows=true [down/flat, up], cols=pred [down/flat, up]):")
	print(cm)

	artifacts_dir = Path(config.artifacts_dir)
	artifacts_dir.mkdir(parents=True, exist_ok=True)
	ckpt_path = artifacts_dir / "lstm_forecaster.pt"
	meta_path = artifacts_dir / "scaler.json"

	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"feature_cols": FEATURE_COLS,
			"mean": mean,
			"std": std,
			"config": {
				"model_family": config.model_family,
				"hidden_size": params["hidden_size"],
				"num_layers": params["num_layers"],
				"dropout": params["dropout"],
				"lr": params["lr"],
				"horizon": config.horizon,
				"days_per_sample": config.days_per_sample,
				"stride_days": config.stride_days,
			},
		},
		ckpt_path,
	)
	meta_path.write_text(
		json.dumps({"feature_cols": FEATURE_COLS, "mean": mean.tolist(), "std": std.tolist()}, indent=2)
	)
	print("saved:", str(ckpt_path))

	# Always plot one example in main
	plot_contract = config.plot_contract
	plot_split = config.plot_split
	plot_idx = int(config.plot_idx)
	if plot_contract not in contract_splits:
		plot_contract = "H"
	ds = contract_splits[plot_contract][plot_split]
	if plot_idx < 0 or plot_idx >= len(ds):
		plot_idx = 0
	save_path = config.plot_path or (artifacts_dir / f"plot_{plot_contract}_{plot_split}_{plot_idx}.png")
	try:
		plot_forecast_vs_actual(
			model=model,
			dataset=ds,
			idx=plot_idx,
			device=device,
			save_path=save_path,
			show=config.plot_show,
		)
	except Exception as exc:
		# If running headless, still save the plot.
		print("plot_warning:", str(exc))
		plot_forecast_vs_actual(
			model=model,
			dataset=ds,
			idx=plot_idx,
			device=device,
			save_path=save_path,
			show=False,
		)


if __name__ == "__main__":
	main()
