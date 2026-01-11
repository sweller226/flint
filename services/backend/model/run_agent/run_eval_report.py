from __future__ import annotations

from pathlib import Path

from .ppo_pipeline import PPOEvalConfig, evaluate_ppo_multi


# No-args evaluation settings
EPISODES = 200
MAX_SAMPLES_PER_CONTRACT = 300
CONTRACTS = ("H", "M", "U", "Z")


def _latest_model(artifacts_dir: Path) -> Path:
	zips = list(artifacts_dir.glob("*.zip"))
	if not zips:
		raise RuntimeError(
			f"No PPO model found in {artifacts_dir}. Train one via model.run_agent.run_ppo_pipeline first."
		)
	# Prefer mixed-contract models if present.
	mixed = [p for p in zips if p.stem.startswith("ppo_mixed_")]
	pool = mixed if mixed else zips
	return max(pool, key=lambda p: p.stat().st_mtime)


def main() -> None:
	model_root = Path(__file__).resolve().parents[1]
	artifacts_dir = model_root / "artifacts"
	model_path = _latest_model(artifacts_dir)
	data_dir = model_root / "data"

	eval_cfg = PPOEvalConfig(
		contracts=CONTRACTS,
		max_samples_per_contract=MAX_SAMPLES_PER_CONTRACT,
		episodes=EPISODES,
		# Keep deterministic True for stable comparisons across runs.
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
