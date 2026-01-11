from __future__ import annotations

import argparse
from pathlib import Path

from ..trade_env.environment import EnvConfig
from ..trade_env.gym_env import ObservationConfig, TradingHourGymEnv
from ..trade_env.sample_builder import build_week_samples


def main() -> None:
	parser = argparse.ArgumentParser(description="Train PPO on the 1-hour trading simulator (discrete actions).")
	parser.add_argument("--contract", default="H", choices=["H", "M", "U", "Z"], help="Contract month dataset")
	parser.add_argument("--max-samples", type=int, default=200, help="Max samples to load from CSV")
	parser.add_argument("--timesteps", type=int, default=50_000, help="Total PPO timesteps")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--holding-penalty",
		type=float,
		default=0.0,
		help="Per-timestep penalty applied while holding a position",
	)
	parser.add_argument(
		"--flat-penalty",
		type=float,
		default=0.0,
		help="Per-timestep penalty applied while flat (position==0)",
	)
	parser.add_argument(
		"--hold-action-penalty",
		type=float,
		default=0.0,
		help="Penalty when the agent chooses HOLD (encourages acting)",
	)
	parser.add_argument(
		"--trade-cost",
		type=float,
		default=0.0,
		help="Transaction cost applied when a BUY/SELL executes",
	)
	parser.add_argument(
		"--save",
		default=None,
		help="Output model path (default: model/simulator/artifacts/ppo_<contract>.zip)",
	)
	args = parser.parse_args()

	try:
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

	samples = list(build_week_samples(csv_path=csv_path, max_samples=args.max_samples))
	if len(samples) < 5:
		raise RuntimeError(f"Not enough samples loaded from {csv_path} (got {len(samples)})")

	# Keep this consistent with legacy behavior (includes forecast vector; forecast is currently literal).
	obs_cfg = ObservationConfig(last_closes=60, forecast_closes=60)

	def make_env():
		return TradingHourGymEnv(
			samples=samples,
			obs_config=obs_cfg,
			seed=args.seed,
			env_config=EnvConfig(
				holding_penalty_per_step=float(args.holding_penalty),
				flat_penalty_per_step=float(args.flat_penalty),
				hold_action_penalty=float(args.hold_action_penalty),
				trade_cost=float(args.trade_cost),
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


if __name__ == "__main__":
	main()
