# Simulator layout

This simulator is organized into three small packages:

- `model.trade_env`: environment + data slicing
- `model.agents`: agent interfaces + implementations
- `model.run_agent`: runnable entrypoints (train/eval/demo)

## Confirming which data is used

All runners load from `services/backend/model/data/`.

For example, the PPO demo prints:
- `Data CSV: .../services/backend/model/data/df_h.csv`

The multi-episode eval report prints:
- `Data dir: .../services/backend/model/data`

## Common commands

From `services/backend`:

- Demo one episode (prints the data CSV):
  - `python -m model.run_agent.demo`

- Multi-episode evaluation report:
  - `python -m model.run_agent.run_eval_report`

- Train+eval default pipeline:
  - `python -m model.run_agent.run_ppo_pipeline`

Advanced (CLI args):
- `python -m model.run_agent.train_ppo --contract H --timesteps 50000`
- `python -m model.run_agent.train_ppo_multi --contracts H,M,U,Z --timesteps 200000`
- `python -m model.run_agent.eval_ppo_multi --episodes 100`
