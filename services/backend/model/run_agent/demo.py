from __future__ import annotations

from pathlib import Path

from ..agents.ppo_agent import PPOTradingAgent
from ..trade_env.environment import EnvConfig
from ..trade_env.gym_env import ObservationConfig, TradingHourGymEnv
from ..trade_env.sample_builder import build_week_samples
from ..trade_env.types import Action


def main() -> None:
	# Data lives in services/backend/model/data
	model_root = Path(__file__).resolve().parents[1]
	data_dir = model_root / "data"
	csv_path = data_dir / "df_h.csv"

	artifacts_dir = model_root / "artifacts"
	zips = list(artifacts_dir.glob("*.zip"))
	if not zips:
		raise RuntimeError(
			f"No PPO model found in {artifacts_dir}. Train one via model.run_agent.run_ppo_pipeline first."
		)
	model_path = max(zips, key=lambda p: p.stat().st_mtime)

	agent = PPOTradingAgent(model_path=model_path)

	obs_cfg = ObservationConfig(last_closes=60, forecast_closes=0)
	env_cfg = EnvConfig(
		holding_penalty_per_step=0.01,
		flat_penalty_per_step=0.02,
		hold_action_penalty=0.005,
		trade_cost=0.01,
		no_trade_penalty=2.0,
	)

	def rollout_one(sample_to_run):
		env_local = TradingHourGymEnv(samples=[sample_to_run], obs_config=obs_cfg, env_config=env_cfg, seed=123)
		obs_vec_local, _ = env_local.reset()
		total_reward_local = 0.0
		forced_liqs_local = 0
		actions_local = []
		for _i in range(60):
			a = agent.act_vec(obs_vec_local, deterministic=True)
			actions_local.append(int(a))
			obs_vec_local, reward, terminated, _truncated, info = env_local.step(int(a))
			total_reward_local += float(reward)
			if bool(info.get("forced_liquidation")):
				forced_liqs_local += 1
			if terminated:
				break
		underlying_local = env_local._env  # type: ignore[attr-defined]
		realized_pnl_local = float(getattr(underlying_local, "_pnl", 0.0)) if underlying_local is not None else 0.0
		trade_count_local = int(len(getattr(underlying_local, "_trades", []))) if underlying_local is not None else 0
		return env_local, actions_local, total_reward_local, realized_pnl_local, trade_count_local, forced_liqs_local

	# Choose a test sample that triggers at least one non-HOLD action (if possible).
	samples = list(build_week_samples(csv_path=csv_path, max_samples=80))
	if not samples:
		raise RuntimeError(f"No samples found in {csv_path}")

	chosen = samples[0]
	for cand in samples:
		_env_tmp, actions_tmp, _r, _pnl, _tc, _fl = rollout_one(cand)
		if any(a != int(Action.HOLD) for a in actions_tmp):
			chosen = cand
			break

	env, _actions, total_reward, realized_pnl, trade_count, forced_liqs = rollout_one(chosen)

	# Re-run one more time to print a full trace with timestamps.
	obs_vec, _ = env.reset()

	print(f"Data CSV: {csv_path}")
	print(f"Model: {model_path}")
	print(f"Test sample contract month: {chosen.contract_month}")
	print("--- Action Trace (60 minutes) ---")
	for i in range(60):
		action = agent.act_vec(obs_vec, deterministic=True)
		obs_vec, reward, terminated, _truncated, info = env.step(int(action))
		ts = info.get("timestamp")
		price = float(info.get("price", 0.0))
		pos = int(info.get("position", 0))
		invalid = bool(info.get("invalid_action"))
		exec_trade = bool(info.get("executed_trade"))
		forced = bool(info.get("forced_liquidation"))
		print(
			f"{i:02d}  {ts}  {Action(int(action)).name:<4}  price={price:8.2f}  "
			f"reward={float(reward):8.4f}  pos={pos}  invalid={int(invalid)}  exec={int(exec_trade)}  forced={int(forced)}"
		)
		if terminated:
			break

	print("--- Summary ---")
	print(f"Realized PnL (trades): {realized_pnl:.4f}")
	print(f"Total reward (PnL - penalties): {total_reward:.4f}")
	print(f"Trades recorded: {trade_count}")
	print(f"Forced liquidations: {forced_liqs}")


if __name__ == "__main__":
	main()
