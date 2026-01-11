from __future__ import annotations

from .ppo_pipeline import run_default_pipeline


def main() -> None:
	out = run_default_pipeline()
	train = out["train"]
	eval_ = out["eval"]

	print("Saved model:", train["model_path"])
	print("Train metrics:", train["train_metrics_path"])
	print("Eval metrics:", eval_["eval_metrics_path"])
	print("Episode CSV:", eval_["episodes_csv_path"])
	print("Eval summary (realized_pnl_mean):", eval_["eval_metrics"]["realized_pnl_mean"])


if __name__ == "__main__":
	main()
