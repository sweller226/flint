from __future__ import annotations

import argparse
from pathlib import Path

from ..trade_env.environment import EnvConfig
from ..trade_env.gym_env import ObservationConfig, TradingHourGymEnv
from ..trade_env.sample_builder import build_week_samples


def main() -> None:
	parser = argparse.ArgumentParser(description="Debug a single PPO rollout to inspect actions/trade frequency.")
	parser.add_argument("--contract", default="H", choices=["H", "M", "U", "Z"])
	parser.add_argument("--model", required=True, help="Path to PPO .zip model")
	parser.add_argument("--max-samples", type=int, default=50)
	parser.add_argument("--holding-penalty", type=float, default=0.0)
	parser.add_argument("--holding-penalty-start-step", type=int, default=0)
	parser.add_argument("--flat-penalty", type=float, default=0.0)
	parser.add_argument("--flat-penalty-start-step", type=int, default=0)
	parser.add_argument("--hold-action-penalty", type=float, default=0.0)
	parser.add_argument("--hold-action-penalty-start-step", type=int, default=0)
	parser.add_argument("--trade-cost", type=float, default=0.0)
	args = parser.parse_args()

	import contextlib
	import io

	with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
		from stable_baselines3 import PPO

	model_root = Path(__file__).resolve().parents[1]
	csv_path = model_root / "data" / f"df_{args.contract.lower()}.csv"
	if not csv_path.exists():
		raise FileNotFoundError(f"Missing data CSV: {csv_path}")
	print(f"Data CSV: {csv_path}")

	samples = list(build_week_samples(csv_path=csv_path, max_samples=args.max_samples))

	obs_cfg = ObservationConfig(last_closes=60, forecast_closes=0)
	env = TradingHourGymEnv(
		samples=samples,
		obs_config=obs_cfg,
		env_config=EnvConfig(
			holding_penalty_per_step=float(args.holding_penalty),
			holding_penalty_start_step=int(args.holding_penalty_start_step),
			flat_penalty_per_step=float(args.flat_penalty),
			flat_penalty_start_step=int(args.flat_penalty_start_step),
			hold_action_penalty=float(args.hold_action_penalty),
			hold_action_penalty_start_step=int(args.hold_action_penalty_start_step),
			trade_cost=float(args.trade_cost),
		),
	)

	model = PPO.load(str(Path(args.model)))

	obs, _ = env.reset()
	terminated = False
	total = 0.0
	actions = []
	while not terminated:
		action, _state = model.predict(obs, deterministic=True)
		actions.append(int(action))
		obs, reward, term, trunc, _info = env.step(int(action))
		total += float(reward)
		terminated = bool(term or trunc)

	counts = {0: 0, 1: 0, 2: 0}
	for a in actions:
		counts[a] += 1
	print(f"Total reward: {total:.4f}")
	print(f"Action counts: HOLD={counts[0]} BUY={counts[1]} SELL={counts[2]}")


if __name__ == "__main__":
	main()
