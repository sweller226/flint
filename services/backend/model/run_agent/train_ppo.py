from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from ..trade_env.environment import EnvConfig
from ..trade_env.gym_env import ObservationConfig, TradingHourGymEnv
from ..trade_env.sample_builder import build_week_samples
from .model_metadata import PPOModelMeta, write_model_meta


def main() -> None:
	parser = argparse.ArgumentParser(description="Train PPO on the 1-hour trading simulator (discrete actions).")
	parser.add_argument("--contract", default="H", choices=["H", "M", "U", "Z"], help="Contract month dataset")
	parser.add_argument("--max-samples", type=int, default=200, help="Max samples to load from CSV")
	parser.add_argument("--timesteps", type=int, default=50_000, help="Total PPO timesteps")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--bar-minutes",
		type=int,
		default=1,
		help="Bar interval in minutes (1=1m bars, 5=5m bars). 5m reduces noise but changes horizon steps to 12.",
	)
	parser.add_argument(
		"--action-scheme",
		choices=["legacy3", "toggle2"],
		default="legacy3",
		help="Action encoding: legacy3 (HOLD/BUY/SELL) or toggle2 (HOLD/TRADE where TRADE is always legal)",
	)
	parser.add_argument(
		"--use-ict-features",
		action="store_true",
		help="Include price-action / ICT-inspired multi-timeframe features in observations",
	)
	parser.add_argument(
		"--holding-penalty",
		type=float,
		default=0.0,
		help="Per-timestep penalty applied while holding a position (default: 0 to avoid punishing good holds)",
	)
	parser.add_argument(
		"--holding-penalty-start-step",
		type=int,
		default=0,
		help="Delay holding penalty until this minute index (e.g. 20 means after 20 minutes)",
	)
	parser.add_argument(
		"--flat-penalty",
		type=float,
		default=0.0,
		help="Per-timestep penalty applied while flat (position==0)",
	)
	parser.add_argument(
		"--flat-penalty-start-step",
		type=int,
		default=0,
		help="Delay flat penalty until this minute index (e.g. 20 means after 20 minutes)",
	)
	parser.add_argument(
		"--hold-action-penalty",
		type=float,
		default=0.0,
		help="Penalty when the agent chooses HOLD while flat (encourages acting); keep 0 to avoid forcing churn",
	)
	parser.add_argument(
		"--hold-action-penalty-start-step",
		type=int,
		default=20,
		help="Delay HOLD-action penalty until this minute index (e.g. 20 means after 20 minutes)",
	)
	parser.add_argument(
		"--trade-cost",
		type=float,
		default=0.01,
		help="Transaction cost applied when a BUY/SELL executes",
	)
	parser.add_argument(
		"--min-entry-step",
		type=int,
		default=0,
		help="Prevent BUY entries before this step index (e.g. 5 to wait for a 5m opening range)",
	)
	parser.add_argument(
		"--no-trade-penalty",
		type=float,
		default=0.0,
		help="End-of-episode penalty if the agent makes zero trades (encourages at least one BUY)",
	)
	parser.add_argument(
		"--trade-penalty",
		type=float,
		default=0.0,
		help="Shaping-only penalty per executed trade (discourages buy/sell every timestep; does not affect PnL)",
	)
	parser.add_argument(
		"--invalid-action-penalty",
		type=float,
		default=0.0,
		help="Penalty when the agent requests an invalid action (e.g., SELL while flat)",
	)
	parser.add_argument(
		"--save",
		default=None,
		help="Output model path (default: model/simulator/artifacts/ppo_<contract>.zip)",
	)
	args = parser.parse_args()

	try:
		import contextlib
		import io

		with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
			from stable_baselines3 import PPO
			from stable_baselines3.common.vec_env import DummyVecEnv
	except Exception as exc:  # pragma: no cover
		raise RuntimeError(
			"stable-baselines3 is required. Install with `pip install stable-baselines3 gymnasium`."
		) from exc

	model_root = Path(__file__).resolve().parents[1]
	data_dir = model_root / "data"
	csv_path = data_dir / f"df_{args.contract.lower()}.csv"
	if not csv_path.exists():
		raise FileNotFoundError(f"Missing data CSV: {csv_path}")
	print(f"Data CSV: {csv_path}")

	bar_minutes = int(args.bar_minutes)
	if bar_minutes not in {1, 5}:
		raise ValueError("bar_minutes must be 1 or 5")
	samples = list(build_week_samples(csv_path=csv_path, max_samples=args.max_samples, bar_minutes=bar_minutes))
	if len(samples) < 5:
		raise RuntimeError(f"Not enough samples loaded from {csv_path} (got {len(samples)})")

	hour_steps = 60 // bar_minutes
	# Include forecast vector; for 5m bars this is 12 steps.
	obs_cfg = ObservationConfig(
		last_closes=hour_steps,
		forecast_closes=hour_steps,
		use_ict_features=bool(args.use_ict_features),
	)

	def _to_steps(minute_idx: int) -> int:
		return int(math.ceil(float(minute_idx) / float(bar_minutes)))

	def make_env():
		return TradingHourGymEnv(
			samples=samples,
			obs_config=obs_cfg,
			action_scheme=str(args.action_scheme),
			seed=args.seed,
			env_config=EnvConfig(
				invalid_action_penalty=float(args.invalid_action_penalty),
				min_entry_step=int(args.min_entry_step),
				holding_penalty_per_step=float(args.holding_penalty),
				holding_penalty_start_step=_to_steps(int(args.holding_penalty_start_step)),
				flat_penalty_per_step=float(args.flat_penalty),
				flat_penalty_start_step=_to_steps(int(args.flat_penalty_start_step)),
				hold_action_penalty=float(args.hold_action_penalty),
				hold_action_penalty_start_step=_to_steps(int(args.hold_action_penalty_start_step)),
				trade_cost=float(args.trade_cost),
				trade_penalty=float(args.trade_penalty),
				no_trade_penalty=float(args.no_trade_penalty),
			),
		)

	vec_env = DummyVecEnv([make_env])

	model = PPO(
		policy="MlpPolicy",
		env=vec_env,
		verbose=1,
		seed=args.seed,
		n_steps=256,
		batch_size=256,
		gamma=0.99,
		learning_rate=3e-4,
	)

	model.learn(total_timesteps=int(args.timesteps))

	artifacts_dir = model_root / "artifacts"
	artifacts_dir.mkdir(parents=True, exist_ok=True)
	out_path = Path(args.save) if args.save else (artifacts_dir / f"ppo_{args.contract}.zip")
	model.save(str(out_path))
	print(f"Saved PPO model to: {out_path}")
	meta_path = write_model_meta(
		out_path,
		PPOModelMeta(
			contract=str(args.contract),
			action_scheme=str(args.action_scheme),
			bar_minutes=int(bar_minutes),
			obs_config={
				"last_closes": int(obs_cfg.last_closes),
				"forecast_closes": int(obs_cfg.forecast_closes),
				"use_ohlcv": bool(obs_cfg.use_ohlcv),
				"use_position": bool(obs_cfg.use_position),
				"use_entry_price": bool(obs_cfg.use_entry_price),
				"use_time_fraction": bool(obs_cfg.use_time_fraction),
				"normalize_by_current_close": bool(obs_cfg.normalize_by_current_close),
					"use_ict_features": bool(obs_cfg.use_ict_features),
					"ict_config": {
						"lookback_1m": int(obs_cfg.ict_config.lookback_1m),
						"lookback_5m": int(obs_cfg.ict_config.lookback_5m),
						"swing_lookback_1m": int(obs_cfg.ict_config.swing_lookback_1m),
						"swing_lookback_5m": int(obs_cfg.ict_config.swing_lookback_5m),
						"swing_lookback_1h": int(obs_cfg.ict_config.swing_lookback_1h),
						"swing_lookback_4h": int(obs_cfg.ict_config.swing_lookback_4h),
						"orb_minutes": int(obs_cfg.ict_config.orb_minutes),
					},
			},
			env_config={
				"invalid_action_penalty": float(args.invalid_action_penalty),
				"min_entry_step": int(args.min_entry_step),
				"holding_penalty_per_step": float(args.holding_penalty),
				"holding_penalty_start_step": _to_steps(int(args.holding_penalty_start_step)),
				"flat_penalty_per_step": float(args.flat_penalty),
				"flat_penalty_start_step": _to_steps(int(args.flat_penalty_start_step)),
				"hold_action_penalty": float(args.hold_action_penalty),
				"hold_action_penalty_start_step": _to_steps(int(args.hold_action_penalty_start_step)),
				"trade_cost": float(args.trade_cost),
				"trade_penalty": float(args.trade_penalty),
				"no_trade_penalty": float(args.no_trade_penalty),
				"auto_close_on_done": True,
			},
			seed=int(args.seed),
			timesteps=int(args.timesteps),
			created_at_utc=pd.Timestamp.utcnow().isoformat(),
		),
	)
	print(f"Saved PPO metadata to: {meta_path}")


if __name__ == "__main__":
	main()
