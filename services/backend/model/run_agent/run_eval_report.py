from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..forecaster.sample_builder import FEATURE_COLS, find_start_index, load_contract_csv, parse_utc_timestamp
from ..trade_env.environment import EnvConfig
from ..trade_env.gym_env import ObservationConfig, TradingHourGymEnv
from ..trade_env.sample_builder import WeekSample
from .model_metadata import load_model_meta
from .ppo_pipeline import PPOEvalConfig, evaluate_ppo_multi


# No-args evaluation settings
EPISODES = 200
MAX_SAMPLES_PER_CONTRACT = 300
CONTRACTS = ("H", "M", "U", "Z")


def _latest_model(artifacts_dir: Path) -> Path:
	zips = list(artifacts_dir.glob("*.zip"))
	if not zips:
		raise RuntimeError(
			f"No PPO model found in {artifacts_dir}. Train one first, e.g. "
			"`python -m services.backend.model.run_agent.train_ppo --contract H`"
		)
	# Prefer mixed-contract models if present.
	mixed = [p for p in zips if p.stem.startswith("ppo_mixed_")]
	pool = mixed if mixed else zips
	return max(pool, key=lambda p: p.stat().st_mtime)


def _default_single_contract_model(artifacts_dir: Path, contract: str) -> Path:
	# Prefer a contract-specific model if it exists.
	p = artifacts_dir / f"ppo_{contract}.zip"
	return p if p.exists() else _latest_model(artifacts_dir)


def _run_agent_artifacts_dir() -> Path:
	# services/backend/model/run_agent/artifacts
	return Path(__file__).resolve().parent / "artifacts"


def _forecaster_forecasts_dir(model_root: Path) -> Path:
	# services/backend/model/forecaster/artifacts/forecasts
	return model_root / "forecaster" / "artifacts" / "forecasts"


def _default_forecast_csv(model_root: Path, contract: str, start_ts: pd.Timestamp) -> Path:
	safe = start_ts.strftime("%Y-%m-%dT%H-%M-%SZ")
	return _forecaster_forecasts_dir(model_root) / f"forecast_{contract}_{safe}.csv"


def _load_forecast_hour_csv(path: Path) -> np.ndarray:
	# Forecast CSVs are typically 1-minute for 60 rows; we may downsample later.
	df = pd.read_csv(path)
	missing = [c for c in FEATURE_COLS if c not in df.columns]
	if missing:
		raise ValueError(f"Forecast CSV missing columns: {missing}")
	if len(df) != 60:
		raise ValueError(f"Forecast CSV must have exactly 60 rows, got {len(df)}")
	for c in FEATURE_COLS:
		df[c] = pd.to_numeric(df[c], errors="coerce")
	if df[FEATURE_COLS].isna().any().any():
		raise ValueError("Forecast CSV contains NaNs in feature columns")
	arr = df[FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)
	if arr.shape != (60, len(FEATURE_COLS)):
		raise ValueError(f"Forecast hour array must be shape (60,{len(FEATURE_COLS)}), got {arr.shape}")
	return arr


def _agg_ohlcv_chunks(x: np.ndarray, chunk: int) -> np.ndarray:
	chunk = int(chunk)
	if chunk <= 1:
		return np.asarray(x, dtype=np.float32)
	if x.shape[0] % chunk != 0:
		raise ValueError(f"Array length {x.shape[0]} not divisible by chunk={chunk}")
	out = np.zeros((x.shape[0] // chunk, x.shape[1]), dtype=np.float32)
	for i in range(out.shape[0]):
		c = x[i * chunk : (i + 1) * chunk]
		out[i, 0] = float(c[0, 0])
		out[i, 1] = float(np.max(c[:, 1]))
		out[i, 2] = float(np.min(c[:, 2]))
		out[i, 3] = float(c[-1, 3])
		out[i, 4] = float(np.sum(c[:, 4]))
	return out


def _build_168h_episode_sample(
	*,
	df: pd.DataFrame,
	window_start_ts: pd.Timestamp,
	align: str = "next",
) -> Tuple[WeekSample, pd.Timestamp]:
	"""Build a single episode sample from a 168-hour window.

	We treat `window_start_ts` as the start of the 168-hour window.
	History is the first 167 hours, and the episode runs on the 168th hour.

	This is intentionally row-based (1 row = 1 minute bar). It requires:
	- 168*60 rows starting at (aligned) window_start_ts
	"""
	start_idx = find_start_index(df, window_start_ts, align=align)

	history_len = 167 * 60
	horizon = 60
	total_len = history_len + horizon
	end_idx = start_idx + total_len
	if end_idx > len(df):
		raise ValueError(
			f"Not enough rows from {window_start_ts.isoformat()} to build 168h window "
			f"(need {total_len} rows; start_idx={start_idx}, end_idx={end_idx}, len={len(df)})."
		)

	history_df = df.iloc[start_idx : start_idx + history_len]
	actual_df = df.iloc[start_idx + history_len : end_idx]
	episode_hour_start_ts = pd.Timestamp(actual_df["ts_event"].iloc[0])

	history = history_df[list(FEATURE_COLS)].to_numpy(dtype=np.float32, copy=True)
	actual_hour = actual_df[list(FEATURE_COLS)].to_numpy(dtype=np.float32, copy=True)
	timestamps = actual_df["ts_event"].tolist()
	contract_month = str(actual_df["contract_month"].iloc[0])

	# Note: forecast_hour will be attached by caller.
	sample = WeekSample(
		history=history,
		actual_hour=actual_hour,
		timestamps_hour=list(timestamps),
		contract_month=contract_month,
		forecast_hour=None,
	)
	return sample, episode_hour_start_ts


def _run_single_episode_and_export_trades(
	*,
	model_path: Path,
	sample: WeekSample,
	out_csv: Path,
	deterministic: bool = True,
) -> Dict[str, object]:
	try:
		with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
			from stable_baselines3 import PPO
	except Exception as exc:  # pragma: no cover
		raise RuntimeError(
			"stable-baselines3 is required. Install with `pip install stable-baselines3 gymnasium`."
		) from exc

	model = PPO.load(str(model_path))
	meta = load_model_meta(model_path)
	model_action_n = int(getattr(model.action_space, "n", 0) or 0)

	if meta is not None and meta.obs_config is not None:
		obs_cfg = ObservationConfig(**meta.obs_config)
	else:
		# Fallback: infer from the PPO model's observation dimension.
		# Our obs vector is: base(8) + last_closes + forecast_closes
		base_dim = 8
		obs_dim = int(getattr(model.observation_space, "shape", (0,))[0] or 0)
		if obs_dim <= base_dim:
			raise ValueError(f"Unexpected model observation dim: {obs_dim}")
		remaining = obs_dim - base_dim
		last_closes = min(60, remaining)
		forecast_closes = max(0, remaining - last_closes)
		obs_cfg = ObservationConfig(last_closes=int(last_closes), forecast_closes=int(forecast_closes))
	# Reward shaping (defaults): encourage at least one trade, but avoid pushing minute-0 entry.
	# - no_trade_penalty ensures "HOLD forever" is suboptimal.
	# - hold_action_penalty starts after a warmup window.
	if meta is not None and meta.env_config is not None:
		env_cfg = EnvConfig(**meta.env_config)
	else:
		env_cfg = EnvConfig(
			auto_close_on_done=True,
			invalid_action_penalty=0.0,
			no_trade_penalty=0.0,
			hold_action_penalty=0.0,
			hold_action_penalty_start_step=0,
			flat_penalty_per_step=0.0,
			flat_penalty_start_step=0,
			holding_penalty_per_step=0.0,
			holding_penalty_start_step=0,
			trade_cost=0.01,
		)

	# Use the same action encoding used in training.
	action_scheme = None
	if meta is not None and getattr(meta, "action_scheme", None):
		action_scheme = str(meta.action_scheme)
	elif model_action_n == 2:
		action_scheme = "toggle2"
	else:
		action_scheme = "legacy3"

	env = TradingHourGymEnv(
		samples=[sample],
		forecast_mode="provided",
		action_scheme=action_scheme,
		obs_config=obs_cfg,
		env_config=env_cfg,
	)

	obs, _info = env.reset()
	done = False
	step_idx = 0
	total_reward = 0.0
	last_info: Dict[str, object] = {}
	rows: List[Dict[str, object]] = []

	while not done:
		action, _state = model.predict(obs, deterministic=bool(deterministic))
		obs, reward, terminated, truncated, info = env.step(int(action))
		last_info = dict(info)
		total_reward += float(reward)
		done = bool(terminated or truncated)

		policy_a = int(info.get("policy_action", -1))
		act = int(info.get("action", -1))
		req = int(info.get("requested_action", act))
		act_name = "HOLD" if act == 0 else "BUY" if act == 1 else "SELL" if act == 2 else "UNKNOWN"
		req_name = "HOLD" if req == 0 else "BUY" if req == 1 else "SELL" if req == 2 else "UNKNOWN"
		rows.append(
			{
				"ts_event": pd.Timestamp(info.get("timestamp")).isoformat(),
				"step": int(info.get("t", step_idx)),
				"policy_action": int(policy_a),
				"action": act_name,
				"requested_action": req_name,
				"executed_trade": bool(info.get("executed_trade", False)),
				"invalid_action": bool(info.get("invalid_action", False)),
				"forced_liquidation": bool(info.get("forced_liquidation", False)),
				"auto_close": bool(info.get("auto_close", False)),
				"price": float(info.get("price", 0.0)),
				"reward": float(reward),
				"pnl_net": float(info.get("pnl_net", info.get("pnl", 0.0))),
				"pnl_gross": float(info.get("pnl_gross", 0.0)),
				"shaped_return": float(info.get("shaped_return", 0.0)),
				"roi_net": float(info.get("roi_net", 0.0)),
				"roi_gross": float(info.get("roi_gross", 0.0)),
				"position": int(info.get("position", 0)),
				"trade_count": int(info.get("trade_count", 0)),
				"buy_count": int(info.get("buy_count", 0)),
				"sell_count": int(info.get("sell_count", 0)),
			}
		)
		step_idx += 1

	out_csv.parent.mkdir(parents=True, exist_ok=True)
	pd.DataFrame(rows).to_csv(out_csv, index=False)
	return {
		"total_reward": float(total_reward),
		"realized_pnl_net": float(last_info.get("pnl_net", last_info.get("pnl", 0.0))) if last_info else 0.0,
		"realized_pnl_gross": float(last_info.get("pnl_gross", 0.0)) if last_info else 0.0,
		"roi_net": float(last_info.get("roi_net", 0.0)) if last_info else 0.0,
		"roi_gross": float(last_info.get("roi_gross", 0.0)) if last_info else 0.0,
		"trade_count": int(last_info.get("trade_count", 0)) if last_info else 0,
		"out_csv": str(out_csv),
	}


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Either run the standard multi-episode eval report (default), or run a single 1-hour episode "
			"defined by an episode start timestamp + a provided forecast CSV and export executed trades."
		)
	)
	parser.add_argument(
		"--start-ts",
		default=None,
		help=(
			"Episode hour start timestamp (UTC). If set, runs single-episode mode. "
			"(This should match the timestamp used to generate the forecast CSV.)"
		),
	)
	parser.add_argument(
		"--window-start-ts",
		default=None,
		help=(
			"Optional override for the 168h window start timestamp (UTC). "
			"If omitted, the window start is computed as (start-ts - 167 hours)."
		),
	)
	parser.add_argument("--contract", default="H", choices=["H", "M", "U", "Z"], help="Contract for single-episode mode")
	parser.add_argument("--forecast-csv", default=None, help="Path to forecast CSV for that episode hour")
	parser.add_argument(
		"--bar-minutes",
		type=int,
		default=None,
		help="Bar interval in minutes (1=1m bars, 5=5m bars). Must match the trained model.",
	)
	parser.add_argument("--align", default="next", choices=["exact", "next", "prev", "nearest"], help="Align start-ts to data")
	parser.add_argument("--model", default=None, help="Path to PPO .zip model (default: latest in model/artifacts)")
	parser.add_argument("--out", default=None, help="Output trades CSV path")
	parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
	args = parser.parse_args()

	model_root = Path(__file__).resolve().parents[1]
	artifacts_dir = model_root / "artifacts"

	if args.start_ts:
		requested_episode_start_ts = parse_utc_timestamp(str(args.start_ts))
		contract = str(args.contract)
		csv_path = model_root / "data" / f"df_{contract.lower()}.csv"
		# Use raw-volume scale to match forecast CSV.
		df = load_contract_csv(csv_path, log_volume=False)

		model_path = Path(args.model) if args.model else _default_single_contract_model(artifacts_dir, contract)
		meta = load_model_meta(model_path)
		bar_minutes = int(args.bar_minutes) if args.bar_minutes is not None else int(getattr(meta, "bar_minutes", 1) or 1)
		if bar_minutes not in {1, 5}:
			raise ValueError("bar_minutes must be 1 or 5")
		if meta is not None and args.bar_minutes is not None and int(getattr(meta, "bar_minutes", 1) or 1) != int(bar_minutes):
			raise ValueError(
				f"--bar-minutes {bar_minutes} does not match model metadata bar_minutes={getattr(meta, 'bar_minutes', 1)}"
			)
		# Single-episode mode: treat start-ts as the episode hour start (aligned to available rows).
		# Build history as the preceding 167*60 rows in the dataset (trading minutes), not wall-clock hours.
		if args.window_start_ts:
			window_start_ts = parse_utc_timestamp(str(args.window_start_ts))
			sample_min, episode_hour_start_ts = _build_168h_episode_sample(
				df=df, window_start_ts=window_start_ts, align=str(args.align)
			)
		else:
			history_len = 167 * 60
			horizon = 60
			episode_start_idx = find_start_index(df, requested_episode_start_ts, align=str(args.align))
			history_start_idx = int(episode_start_idx) - int(history_len)
			end_idx = int(episode_start_idx) + int(horizon)
			if history_start_idx < 0:
				raise ValueError(
					f"Not enough history before {requested_episode_start_ts.isoformat()} "
					f"(need {history_len} rows; episode_start_idx={episode_start_idx})."
				)
			if end_idx > len(df):
				raise ValueError(
					f"Not enough rows from {requested_episode_start_ts.isoformat()} to build 1h episode "
					f"(need {horizon} rows; episode_start_idx={episode_start_idx}, end_idx={end_idx}, len={len(df)})."
				)
			history_df = df.iloc[history_start_idx:episode_start_idx]
			actual_df = df.iloc[episode_start_idx:end_idx]
			window_start_ts = pd.Timestamp(history_df["ts_event"].iloc[0])
			episode_hour_start_ts = pd.Timestamp(actual_df["ts_event"].iloc[0])
			sample_min = WeekSample(
				history=history_df[list(FEATURE_COLS)].to_numpy(dtype=np.float32, copy=True),
				actual_hour=actual_df[list(FEATURE_COLS)].to_numpy(dtype=np.float32, copy=True),
				timestamps_hour=list(actual_df["ts_event"].tolist()),
				contract_month=str(actual_df["contract_month"].iloc[0]),
				forecast_hour=None,
			)
		# Downsample the episode windows if requested.
		if bar_minutes != 1:
			sample = WeekSample(
				history=_agg_ohlcv_chunks(sample_min.history, bar_minutes),
				actual_hour=_agg_ohlcv_chunks(sample_min.actual_hour, bar_minutes),
				timestamps_hour=[
					sample_min.timestamps_hour[(i + 1) * bar_minutes - 1]
					for i in range(len(sample_min.timestamps_hour) // bar_minutes)
				],
				contract_month=sample_min.contract_month,
				forecast_hour=None,
			)
		else:
			sample = sample_min

		if args.forecast_csv:
			forecast_csv = Path(args.forecast_csv)
		else:
			# First try a forecast named for the requested episode start (matches export_forecast naming).
			c1 = _default_forecast_csv(model_root, contract, requested_episode_start_ts)
			# Fallback: forecast named for the aligned episode hour start (in case start-ts had to be aligned).
			c2 = _default_forecast_csv(model_root, contract, episode_hour_start_ts)
			forecast_csv = c1 if c1.exists() else c2
		if not forecast_csv.exists():
			raise FileNotFoundError(
				"Forecast CSV not found. Generate it with export_forecast using the same episode start timestamp. "
				f"Tried: {forecast_csv} (requested={requested_episode_start_ts.isoformat()}, aligned={episode_hour_start_ts.isoformat()})"
			)
		forecast_hour = _load_forecast_hour_csv(forecast_csv)
		if bar_minutes != 1:
			forecast_hour = _agg_ohlcv_chunks(forecast_hour, bar_minutes)
		sample = WeekSample(
			history=sample.history,
			actual_hour=sample.actual_hour,
			timestamps_hour=sample.timestamps_hour,
			contract_month=sample.contract_month,
			forecast_hour=forecast_hour,
		)

		# model_path already resolved above.
		run_agent_artifacts = _run_agent_artifacts_dir()
		out_path = (
			Path(args.out)
			if args.out
			else (run_agent_artifacts / "agent_trades" / f"actions_{contract}_{episode_hour_start_ts.strftime('%Y-%m-%dT%H-%M-%SZ')}.csv")
		)

		result = _run_single_episode_and_export_trades(
			model_path=model_path,
			sample=sample,
			out_csv=out_path,
			deterministic=bool(args.deterministic),
		)

		print("Model:", model_path)
		print("Requested episode hour start:", requested_episode_start_ts.isoformat())
		print("Window start:", window_start_ts.isoformat())
		print("Episode hour start:", episode_hour_start_ts.isoformat())
		print("Forecast CSV:", forecast_csv)
		print("Actions CSV:", result["out_csv"])
		print("Trade count:", result["trade_count"])
		print("Total reward:", f"{result['total_reward']:.4f}")
		print("Realized PnL (net):", f"{result['realized_pnl_net']:.4f}")
		print("Realized PnL (gross):", f"{result['realized_pnl_gross']:.4f}")
		print("ROI (net):", f"{result['roi_net']:.6f}")
		print("ROI (gross):", f"{result['roi_gross']:.6f}")
		return

	# Default: legacy multi-episode report.
	model_path = _latest_model(artifacts_dir)
	data_dir = model_root / "data"

	eval_cfg = PPOEvalConfig(
		contracts=CONTRACTS,
		max_samples_per_contract=MAX_SAMPLES_PER_CONTRACT,
		episodes=EPISODES,
		deterministic=True,
		artifacts_dir=artifacts_dir,
	)

	out = evaluate_ppo_multi(model_path=model_path, config=eval_cfg, data_dir=data_dir)
	m = out["eval_metrics"]

	print("Data dir:", data_dir)
	print("Model:", model_path)
	print("Episodes:", m["episodes"], "Contracts:", ",".join(m["contracts"]))
	print("--- Performance ---")
	print(
		"Realized PnL mean/std:",
		f"{m['realized_pnl_mean']:.4f}",
		"/",
		f"{m['realized_pnl_std']:.4f}",
		"win_rate:",
		f"{m['realized_pnl_win_rate']:.3f}",
	)
	print(
		"Total reward mean/std:",
		f"{m['total_reward_mean']:.4f}",
		"/",
		f"{m['total_reward_std']:.4f}",
		"win_rate:",
		f"{m['total_reward_win_rate']:.3f}",
	)
	print("--- Behavior ---")
	print(
		"Actions (total):",
		m["action_counts"],
		"per_episode:",
		{k: round(v, 2) for k, v in m["actions_per_episode"].items()},
	)
	print(
		"Executed trades/episode:",
		f"{m['executed_trades_mean']:.2f}",
		"buys:",
		f"{m['executed_buys_mean']:.2f}",
		"sells:",
		f"{m['executed_sells_mean']:.2f}",
		"forced_liqs:",
		f"{m['forced_liquidations_mean']:.2f}",
	)
	print("Invalid actions/episode:", f"{m['invalid_mean']:.2f}")

	print("--- Artifacts ---")
	print("Eval metrics:", out["eval_metrics_path"])
	print("Episodes CSV:", out["episodes_csv_path"])


if __name__ == "__main__":
	main()
