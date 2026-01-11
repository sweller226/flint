from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..trade_env.environment import EnvConfig
from ..trade_env.gym_env import ObservationConfig, TradingHourGymEnv
from ..trade_env.sample_builder import build_week_samples


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate a trained PPO model on simulator episodes.")
	parser.add_argument("--contract", default="H", choices=["H", "M", "U", "Z"])
	parser.add_argument("--episodes", type=int, default=20)
	parser.add_argument("--model", default=None, help="Path to PPO .zip model")
	parser.add_argument("--max-samples", type=int, default=200)
	parser.add_argument(
		"--holding-penalty",
		type=float,
		default=0.0,
		help="Per-timestep penalty applied while holding a position (should match training)",
	)
	parser.add_argument(
		"--holding-penalty-start-step",
		type=int,
		default=0,
		help="Delay holding penalty until this step index (0..59)",
	)
	parser.add_argument(
		"--flat-penalty",
		type=float,
		default=0.0,
		help="Per-timestep penalty applied while flat (should match training)",
	)
	parser.add_argument(
		"--flat-penalty-start-step",
		type=int,
		default=0,
		help="Delay flat penalty until this step index (0..59)",
	)
	parser.add_argument(
		"--hold-action-penalty",
		type=float,
		default=0.005,
		help="Penalty when choosing HOLD (should match training)",
	)
	parser.add_argument(
		"--hold-action-penalty-start-step",
		type=int,
		default=20,
		help="Delay HOLD-action penalty until this step index (0..59)",
	)
	parser.add_argument(
		"--trade-cost",
		type=float,
		default=0.0,
		help="Transaction cost applied when a BUY/SELL executes (should match training)",
	)
	parser.add_argument(
		"--no-trade-penalty",
		type=float,
		default=1.0,
		help="End-of-episode penalty if the agent makes zero trades (encourages at least one BUY)",
	)
	args = parser.parse_args()

	try:
		import contextlib
		import io

		# stable-baselines3 may import legacy `gym`, which prints a deprecation banner.
		# We suppress that banner without affecting normal runtime logs.
		with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
			from stable_baselines3 import PPO
	except Exception as exc:  # pragma: no cover
		raise RuntimeError(
			"stable-baselines3 is required. Install with `pip install stable-baselines3 gymnasium`."
		) from exc

	model_root = Path(__file__).resolve().parents[1]
	csv_path = model_root / "data" / f"df_{args.contract.lower()}.csv"
	if not csv_path.exists():
		raise FileNotFoundError(f"Missing data CSV: {csv_path}")
	print(f"Data CSV: {csv_path}")

	samples = list(build_week_samples(csv_path=csv_path, max_samples=args.max_samples))

	artifacts_dir = model_root / "artifacts"
	model_path = Path(args.model) if args.model else (artifacts_dir / f"ppo_{args.contract}.zip")
	model = PPO.load(str(model_path))

	env = TradingHourGymEnv(
		samples=samples,
		obs_config=ObservationConfig(),
		env_config=EnvConfig(
			holding_penalty_per_step=float(args.holding_penalty),
			holding_penalty_start_step=int(args.holding_penalty_start_step),
			flat_penalty_per_step=float(args.flat_penalty),
			flat_penalty_start_step=int(args.flat_penalty_start_step),
			hold_action_penalty=float(args.hold_action_penalty),
			hold_action_penalty_start_step=int(args.hold_action_penalty_start_step),
			trade_cost=float(args.trade_cost),
			no_trade_penalty=float(args.no_trade_penalty),
		),
	)

	pnls = []
	trade_counts = []
	buy_counts = []
	sell_counts = []
	for _ in range(int(args.episodes)):
		obs, _info = env.reset()
		done = False
		pnl = 0.0
		last_info = {}
		while not done:
			action, _state = model.predict(obs, deterministic=True)
			obs, reward, terminated, truncated, _info2 = env.step(int(action))
			last_info = dict(_info2)
			pnl += float(reward)
			done = bool(terminated or truncated)
		pnls.append(pnl)
		trade_counts.append(int(last_info.get("trade_count", 0)))
		buy_counts.append(int(last_info.get("buy_count", 0)))
		sell_counts.append(int(last_info.get("sell_count", 0)))

	pnls_arr = np.asarray(pnls, dtype=np.float32)
	trade_arr = np.asarray(trade_counts, dtype=np.int32)
	no_trade_rate = float((trade_arr == 0).mean()) if trade_arr.size else float("nan")
	print(f"Model: {model_path}")
	print(f"Episodes: {len(pnls)}")
	print(f"No-trade rate: {no_trade_rate*100:.1f}%")
	print(f"Avg trades/ep: {trade_arr.mean():.2f}")
	print(f"Avg buys/ep: {np.mean(buy_counts):.2f}")
	print(f"Avg sells/ep: {np.mean(sell_counts):.2f}")
	print(f"Mean PnL: {pnls_arr.mean():.4f}")
	print(f"Std  PnL: {pnls_arr.std():.4f}")
	print(f"Min  PnL: {pnls_arr.min():.4f}")
	print(f"Max  PnL: {pnls_arr.max():.4f}")


if __name__ == "__main__":
	main()
