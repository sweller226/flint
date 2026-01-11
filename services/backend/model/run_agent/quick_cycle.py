from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast train + multi-timestamp eval loop (aims to finish < 1 minute)."
    )
    parser.add_argument("--contract", default="H", choices=["H", "M", "U", "Z"])
    parser.add_argument(
        "--timesteps",
        type=int,
        default=6000,
        help="Training timesteps (keep small for <1 min loops)",
    )
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument(
        "--action-scheme",
        choices=["legacy3", "toggle2"],
        default="legacy3",
    )
    parser.add_argument(
        "--bar-minutes",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--trade-cost",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--trade-penalty",
        type=float,
        default=0.02,
        help="Shaping-only penalty per executed trade (discourages flip-flopping; does not affect PnL)",
    )
    parser.add_argument(
        "--no-trade-penalty",
        type=float,
        default=0.2,
        help="End-of-episode penalty if the agent makes zero trades (prevents the always-HOLD policy)",
    )
    parser.add_argument(
        "--hold-action-penalty",
        type=float,
        default=0.001,
        help="Penalty when choosing HOLD while flat after warmup (nudges post-ORB participation)",
    )
    parser.add_argument(
        "--hold-action-penalty-start-step",
        type=int,
        default=5,
        help="Warmup minutes before penalizing HOLD while flat (default 5 ~= ORB(5m) completes)",
    )
    parser.add_argument(
        "--min-entry-step",
        type=int,
        default=5,
        help="Prevent BUY entries before this step index (default 5 to wait for ORB)",
    )
    parser.add_argument(
        "--use-ict-features",
        action="store_true",
        default=True,
        help="Enable ICT/ORB/sweep features (default true)",
    )
    parser.add_argument(
        "--start-ts",
        action="append",
        required=True,
        help="Evaluation episode start timestamp (repeatable)",
    )
    parser.add_argument(
        "--forecast-csv",
        type=str,
        default=None,
        help="Optional forecast CSV path. If omitted, run_eval_report uses its default per start-ts.",
    )

    args = parser.parse_args()

    forecast_csv: str | None = None
    if args.forecast_csv:
        root = Path(__file__).resolve().parents[4]
        forecast_csv = str((root / args.forecast_csv).resolve())

    train_cmd = [
        sys.executable,
        "-m",
        "services.backend.model.run_agent.train_ppo",
        "--contract",
        args.contract,
        "--timesteps",
        str(args.timesteps),
        "--max-samples",
        str(args.max_samples),
        "--action-scheme",
        args.action_scheme,
        "--bar-minutes",
        str(args.bar_minutes),
        "--trade-cost",
        str(args.trade_cost),
        "--trade-penalty",
        str(args.trade_penalty),
        "--min-entry-step",
        str(args.min_entry_step),
        "--invalid-action-penalty",
        "0.0",
        "--hold-action-penalty",
        str(args.hold_action_penalty),
        "--hold-action-penalty-start-step",
        str(args.hold_action_penalty_start_step),
        "--no-trade-penalty",
        str(args.no_trade_penalty),
    ]
    if args.use_ict_features:
        train_cmd.append("--use-ict-features")

    print("=== TRAIN ===")
    subprocess.run(train_cmd, check=True)

    print("=== EVAL ===")
    for ts in args.start_ts:
        eval_cmd = [
            sys.executable,
            "-m",
            "services.backend.model.run_agent.run_eval_report",
            "--contract",
            args.contract,
            "--start-ts",
            ts,
            "--deterministic",
        ]
        if forecast_csv is not None:
            eval_cmd.extend(["--forecast-csv", forecast_csv])
        print(f"--- {ts} ---")
        subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()
