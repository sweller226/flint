from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from ..trade_env.environment import EnvConfig
from ..trade_env.gym_env import ObservationConfig, TradingHourGymEnv
from ..trade_env.sample_builder import WeekSample, build_week_samples


def _load_samples(data_dir: Path, contracts: List[str], max_samples_per_contract: int) -> List[WeekSample]:
	samples: List[WeekSample] = []
	for c in contracts:
		csv_path = data_dir / f"df_{c.lower()}.csv"
		if not csv_path.exists():
			raise FileNotFoundError(f"Missing data CSV: {csv_path}")
		contract_samples = list(build_week_samples(csv_path=csv_path, max_samples=max_samples_per_contract))
		if not contract_samples:
			raise RuntimeError(f"No samples loaded from {csv_path}")
		samples.extend(contract_samples)
	return samples


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Train PPO on mixed 7-trading-day samples from df_h/df_m/df_u/df_z. "
			"Each episode is exactly the last hour (60 minutes) of a 7-day sample; "
			"reward is realized PnL from BUY/SELL actions (max 1 contract)."
		)
	)
	parser.add_argument(
		"--contracts",
		default="H,M,U,Z",
		help="Comma-separated contract list (default: H,M,U,Z)",
	)
	parser.add_argument(
		"--max-samples-per-contract",
		type=int,
		default=200,
		help="How many 7-day samples to load from each df_*.csv",
	)
	parser.add_argument("--timesteps", type=int, default=200_000, help="Total PPO timesteps")
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
		help="Output model path (default: model/simulator/artifacts/ppo_mixed_HMUZ.zip)",
	)
	args = parser.parse_args()

	contracts = [c.strip().upper() for c in str(args.contracts).split(",") if c.strip()]
	for c in contracts:
		if c not in {"H", "M", "U", "Z"}:
			raise ValueError(f"Unsupported contract: {c}")

	try:
		from stable_baselines3 import PPO
		from stable_baselines3.common.vec_env import DummyVecEnv
	except Exception as exc:  # pragma: no cover
		raise RuntimeError(
			"stable-baselines3 is required. Install with `pip install stable-baselines3 gymnasium`."
		) from exc

	model_root = Path(__file__).resolve().parents[1]
	data_dir = model_root / "data"
	print(f"Data dir: {data_dir}")

	samples = _load_samples(data_dir, contracts=contracts, max_samples_per_contract=int(args.max_samples_per_contract))
	if len(samples) < 20:
		raise RuntimeError(f"Not enough samples loaded (got {len(samples)})")

	# No forecasting: exclude forecast from observation entirely.
	obs_cfg = ObservationConfig(last_closes=60, forecast_closes=0)

	def make_env():
		return TradingHourGymEnv(
			samples=samples,
			obs_config=obs_cfg,
			forecast_mode="literal",
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
	out_path = Path(args.save) if args.save else (artifacts_dir / f"ppo_mixed_{''.join(contracts)}.zip")
	model.save(str(out_path))
	print(f"Saved PPO model to: {out_path}")


if __name__ == "__main__":
	main()
