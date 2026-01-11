from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..trade_env.environment import EnvConfig
from ..trade_env.gym_env import ObservationConfig, TradingHourGymEnv
from ..trade_env.sample_builder import WeekSample, build_week_samples
from ..trade_env.ict_features import ICT_FEATURE_DIM


@dataclass(frozen=True)
class PPOTrainConfig:
	# Data
	contracts: Tuple[str, ...] = ("H", "M", "U", "Z")
	max_samples_per_contract: int = 300

	# PPO
	timesteps: int = 200_000
	seed: int = 42
	policy: str = "MlpPolicy"
	gamma: float = 0.99
	learning_rate: float = 3e-4
	n_steps: int = 256
	batch_size: int = 256

	# Environment / reward shaping
	env_config: EnvConfig = field(
		default_factory=lambda: EnvConfig(
			holding_penalty_per_step=0.01,
			flat_penalty_per_step=0.02,
			hold_action_penalty=0.005,
			trade_cost=0.01,
			no_trade_penalty=2.0,
			invalid_action_penalty=0.0,
		)
	)
	obs_config: ObservationConfig = field(
		default_factory=lambda: ObservationConfig(last_closes=60, forecast_closes=0)
	)

	# Output
	model_name: str = "ppo_mixed_HMUZ"
	artifacts_dir: Optional[Path] = None

	def resolved_artifacts_dir(self) -> Path:
		if self.artifacts_dir is not None:
			return self.artifacts_dir
		# Keep artifacts at model/artifacts
		return Path(__file__).resolve().parents[1] / "artifacts"


@dataclass(frozen=True)
class PPOEvalConfig:
	contracts: Tuple[str, ...] = ("H", "M", "U", "Z")
	max_samples_per_contract: int = 300
	episodes: int = 100
	seed: int = 123
	deterministic: bool = True

	env_config: EnvConfig = field(
		default_factory=lambda: EnvConfig(
			holding_penalty_per_step=0.01,
			flat_penalty_per_step=0.02,
			hold_action_penalty=0.005,
			trade_cost=0.01,
			no_trade_penalty=2.0,
			invalid_action_penalty=0.0,
		)
	)
	obs_config: ObservationConfig = field(
		default_factory=lambda: ObservationConfig(last_closes=60, forecast_closes=0)
	)

	artifacts_dir: Optional[Path] = None

	def resolved_artifacts_dir(self) -> Path:
		if self.artifacts_dir is not None:
			return self.artifacts_dir
		return Path(__file__).resolve().parents[1] / "artifacts"


def _load_samples(data_dir: Path, contracts: Sequence[str], max_samples_per_contract: int) -> List[WeekSample]:
	all_samples: List[WeekSample] = []
	for c in contracts:
		c = c.strip().upper()
		if c not in {"H", "M", "U", "Z"}:
			raise ValueError(f"Unsupported contract: {c}")
		csv_path = data_dir / f"df_{c.lower()}.csv"
		samples = list(build_week_samples(csv_path=csv_path, max_samples=max_samples_per_contract))
		if not samples:
			raise RuntimeError(f"No samples loaded from {csv_path}")
		all_samples.extend(samples)
	return all_samples


def train_ppo_multi(config: PPOTrainConfig, *, data_dir: Optional[Path] = None) -> Dict[str, Any]:
	"""Train PPO across mixed contract samples."""
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

	# This file lives in model/run_agent; parent[1] is model/
	backend_model_root = Path(__file__).resolve().parents[1]
	data_root = data_dir if data_dir is not None else (backend_model_root / "data")

	artifacts_dir = config.resolved_artifacts_dir()
	artifacts_dir.mkdir(parents=True, exist_ok=True)

	samples = _load_samples(data_root, config.contracts, config.max_samples_per_contract)

	def make_env():
		return TradingHourGymEnv(
			samples=samples,
			obs_config=config.obs_config,
			seed=config.seed,
			env_config=config.env_config,
		)

	vec_env = DummyVecEnv([make_env])

	model = PPO(
		policy=config.policy,
		env=vec_env,
		verbose=1,
		seed=config.seed,
		n_steps=config.n_steps,
		batch_size=config.batch_size,
		gamma=config.gamma,
		learning_rate=config.learning_rate,
	)

	t0 = time.time()
	model.learn(total_timesteps=int(config.timesteps))
	train_seconds = float(time.time() - t0)

	model_path = artifacts_dir / f"{config.model_name}.zip"
	model.save(str(model_path))

	train_metrics: Dict[str, Any] = {
		"contracts": list(config.contracts),
		"n_samples": int(len(samples)),
		"max_samples_per_contract": int(config.max_samples_per_contract),
		"timesteps": int(config.timesteps),
		"seed": int(config.seed),
		"train_seconds": train_seconds,
		"reward_shaping": asdict(config.env_config),
		"observation": asdict(config.obs_config),
	}

	(train_path := artifacts_dir / f"{config.model_name}.train.json").write_text(
		json.dumps(train_metrics, indent=2, default=str)
	)

	return {
		"model_path": str(model_path),
		"train_metrics": train_metrics,
		"train_metrics_path": str(train_path),
	}


def evaluate_ppo_multi(
	*,
	model_path: Path,
	config: PPOEvalConfig,
	data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
	"""Evaluate a PPO model; returns metrics and writes artifacts."""
	try:
		import contextlib
		import io

		with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
			from stable_baselines3 import PPO
	except Exception as exc:  # pragma: no cover
		raise RuntimeError(
			"stable-baselines3 is required. Install with `pip install stable-baselines3 gymnasium`."
		) from exc

	backend_model_root = Path(__file__).resolve().parents[1]
	data_root = data_dir if data_dir is not None else (backend_model_root / "data")

	artifacts_dir = config.resolved_artifacts_dir()
	artifacts_dir.mkdir(parents=True, exist_ok=True)

	samples = _load_samples(data_root, config.contracts, config.max_samples_per_contract)
	model = PPO.load(str(model_path))

	def _expected_obs_dim() -> int:
		space = getattr(model, "observation_space", None)
		shape = getattr(space, "shape", None)
		if not shape:
			raise RuntimeError("Loaded PPO model has no observation_space.shape")
		if len(shape) != 1:
			raise RuntimeError(f"Unsupported observation space shape: {shape}")
		return int(shape[0])

	def _base_obs_dim(obs_cfg: ObservationConfig) -> int:
		dim = 0
		if obs_cfg.use_ohlcv:
			dim += 5
		if obs_cfg.use_position:
			dim += 1
		if obs_cfg.use_entry_price:
			dim += 1
		if obs_cfg.use_time_fraction:
			dim += 1
		return dim

	def _total_obs_dim(obs_cfg: ObservationConfig) -> int:
		dim = int(_base_obs_dim(obs_cfg) + int(obs_cfg.last_closes) + int(obs_cfg.forecast_closes))
		if bool(getattr(obs_cfg, "use_ict_features", False)):
			dim += int(ICT_FEATURE_DIM)
		return int(dim)

	def _coerce_obs_config(obs_cfg: ObservationConfig, target_dim: int) -> ObservationConfig:
		use_ict = bool(getattr(obs_cfg, "use_ict_features", False))
		base = int(_base_obs_dim(obs_cfg) + (int(ICT_FEATURE_DIM) if use_ict else 0))
		# If the config thinks ICT is enabled but the model dimension can't support it, disable it.
		if use_ict and int(target_dim) < int(base):
			use_ict = False
			base = int(_base_obs_dim(obs_cfg))
		remaining = int(target_dim) - int(base)
		if remaining < 0:
			raise RuntimeError(f"Model obs dim {target_dim} is smaller than base features {base}")
		# Preserve last_closes when possible; adjust forecast_closes to fit.
		last = int(obs_cfg.last_closes)
		if last > remaining:
			last = remaining
		forecast = remaining - last
		return ObservationConfig(
			use_ohlcv=bool(obs_cfg.use_ohlcv),
			use_position=bool(obs_cfg.use_position),
			use_entry_price=bool(obs_cfg.use_entry_price),
			use_time_fraction=bool(obs_cfg.use_time_fraction),
			last_closes=int(last),
			forecast_closes=int(forecast),
			normalize_by_current_close=bool(obs_cfg.normalize_by_current_close),
			use_ict_features=bool(use_ict),
		)

	expected_dim = _expected_obs_dim()
	obs_cfg_used = config.obs_config
	if _total_obs_dim(obs_cfg_used) != expected_dim:
		obs_cfg_used = _coerce_obs_config(obs_cfg_used, expected_dim)

	env = TradingHourGymEnv(
		samples=samples,
		obs_config=obs_cfg_used,
		seed=config.seed,
		env_config=config.env_config,
	)

	total_rewards: List[float] = []
	realized_pnls_net: List[float] = []
	realized_pnls_gross: List[float] = []
	rois_net: List[float] = []
	rois_gross: List[float] = []
	invalid_counts: List[int] = []
	executed_trade_counts: List[int] = []
	executed_buy_counts: List[int] = []
	executed_sell_counts: List[int] = []
	forced_liquidations: List[int] = []
	action_counts = np.zeros((int(env.action_space.n),), dtype=np.int64)

	per_episode: List[Dict[str, Any]] = []

	for ep in range(int(config.episodes)):
		obs, _info = env.reset()
		done = False
		total_reward = 0.0
		realized_pnl_net = 0.0
		realized_pnl_gross = 0.0
		roi_net = 0.0
		roi_gross = 0.0
		invalid = 0
		exec_trades = 0
		exec_buys = 0
		exec_sells = 0
		forced = 0
		steps = 0
		while not done:
			action, _state = model.predict(obs, deterministic=bool(config.deterministic))
			a = int(action)
			if 0 <= a < len(action_counts):
				action_counts[a] += 1
			obs, reward, terminated, truncated, info = env.step(a)
			total_reward += float(reward)
			steps += 1
			if bool(info.get("invalid_action")):
				invalid += 1
			if bool(info.get("executed_trade")):
				exec_trades += 1
				act = int(info.get("action", -1))
				if act == 1:
					exec_buys += 1
				elif act == 2:
					exec_sells += 1
			elif a != 0 and not bool(info.get("invalid_action")):
				# Back-compat heuristic (should rarely be hit).
				exec_trades += 1
				act = int(info.get("action", -1))
				if act == 1:
					exec_buys += 1
				elif act == 2:
					exec_sells += 1
			if bool(info.get("forced_liquidation")):
				forced += 1
			done = bool(terminated or truncated)
			if done:
				realized_pnl_net = float(info.get("pnl_net", info.get("pnl", realized_pnl_net)))
				realized_pnl_gross = float(info.get("pnl_gross", realized_pnl_gross))
				roi_net = float(info.get("roi_net", roi_net))
				roi_gross = float(info.get("roi_gross", roi_gross))

		total_rewards.append(total_reward)
		realized_pnls_net.append(realized_pnl_net)
		realized_pnls_gross.append(realized_pnl_gross)
		rois_net.append(roi_net)
		rois_gross.append(roi_gross)
		invalid_counts.append(invalid)
		executed_trade_counts.append(exec_trades)
		executed_buy_counts.append(exec_buys)
		executed_sell_counts.append(exec_sells)
		forced_liquidations.append(forced)
		per_episode.append(
			{
				"episode": ep,
				"total_reward": total_reward,
				"realized_pnl_net": realized_pnl_net,
				"realized_pnl_gross": realized_pnl_gross,
				"roi_net": roi_net,
				"roi_gross": roi_gross,
				"steps": steps,
				"invalid_actions": invalid,
				"executed_trades": exec_trades,
				"executed_buys": exec_buys,
				"executed_sells": exec_sells,
				"forced_liquidations": forced,
			}
		)

	total_rewards_arr = np.asarray(total_rewards, dtype=np.float32)
	realized_pnls_net_arr = np.asarray(realized_pnls_net, dtype=np.float32)
	realized_pnls_gross_arr = np.asarray(realized_pnls_gross, dtype=np.float32)
	rois_net_arr = np.asarray(rois_net, dtype=np.float32)
	rois_gross_arr = np.asarray(rois_gross, dtype=np.float32)
	invalid_arr = np.asarray(invalid_counts, dtype=np.int32)
	trades_arr = np.asarray(executed_trade_counts, dtype=np.int32)
	buys_arr = np.asarray(executed_buy_counts, dtype=np.int32)
	sells_arr = np.asarray(executed_sell_counts, dtype=np.int32)
	forced_arr = np.asarray(forced_liquidations, dtype=np.int32)

	# Map policy action indices to labels.
	if int(env.action_space.n) == 2:
		action_labels = ["HOLD", "TRADE"]
	else:
		action_labels = ["HOLD", "BUY", "SELL"]

	metrics: Dict[str, Any] = {
		"model_path": str(model_path),
		"contracts": list(config.contracts),
		"n_samples": int(len(samples)),
		"episodes": int(config.episodes),
		"deterministic": bool(config.deterministic),
		"seed": int(config.seed),
		"reward_shaping": asdict(config.env_config),
		"observation": asdict(obs_cfg_used),
		"total_reward_mean": float(total_rewards_arr.mean()),
		"total_reward_std": float(total_rewards_arr.std()),
		"total_reward_min": float(total_rewards_arr.min()),
		"total_reward_max": float(total_rewards_arr.max()),
		"total_reward_win_rate": float((total_rewards_arr > 0).mean()),
		"realized_pnl_net_mean": float(realized_pnls_net_arr.mean()),
		"realized_pnl_net_std": float(realized_pnls_net_arr.std()),
		"realized_pnl_net_min": float(realized_pnls_net_arr.min()),
		"realized_pnl_net_max": float(realized_pnls_net_arr.max()),
		"realized_pnl_net_win_rate": float((realized_pnls_net_arr > 0).mean()),
		"realized_pnl_gross_mean": float(realized_pnls_gross_arr.mean()),
		"realized_pnl_gross_std": float(realized_pnls_gross_arr.std()),
		"realized_pnl_gross_min": float(realized_pnls_gross_arr.min()),
		"realized_pnl_gross_max": float(realized_pnls_gross_arr.max()),
		"roi_net_mean": float(rois_net_arr.mean()),
		"roi_net_std": float(rois_net_arr.std()),
		"roi_gross_mean": float(rois_gross_arr.mean()),
		"roi_gross_std": float(rois_gross_arr.std()),
		"invalid_mean": float(invalid_arr.mean()),
		"executed_trades_mean": float(trades_arr.mean()),
		"executed_buys_mean": float(buys_arr.mean()),
		"executed_sells_mean": float(sells_arr.mean()),
		"forced_liquidations_mean": float(forced_arr.mean()),
		"policy_action_counts": {k: int(action_counts[i]) for i, k in enumerate(action_labels)},
		"policy_actions_per_episode": {
			k: float(action_counts[i] / max(1, int(config.episodes))) for i, k in enumerate(action_labels)
		},
	}

	metrics_path = artifacts_dir / f"{model_path.stem}.eval.json"
	metrics_path.write_text(json.dumps(metrics, indent=2, default=str))

	csv_path = artifacts_dir / f"{model_path.stem}.episodes.csv"
	lines = [
		"episode,total_reward,realized_pnl_net,realized_pnl_gross,roi_net,roi_gross,steps,invalid_actions,executed_trades,executed_buys,executed_sells,forced_liquidations"
	]
	for row in per_episode:
		lines.append(
			f"{row['episode']},{row['total_reward']:.6f},{row['realized_pnl_net']:.6f},{row['realized_pnl_gross']:.6f},{row['roi_net']:.6f},{row['roi_gross']:.6f},{row['steps']},{row['invalid_actions']},{row['executed_trades']},{row['executed_buys']},{row['executed_sells']},{row['forced_liquidations']}"
		)
	csv_path.write_text("\n".join(lines) + "\n")

	return {
		"eval_metrics": metrics,
		"eval_metrics_path": str(metrics_path),
		"episodes_csv_path": str(csv_path),
	}


def run_default_pipeline() -> Dict[str, Any]:
	"""Train + evaluate using the default configs; writes artifacts to disk."""
	train_cfg = PPOTrainConfig()
	eval_cfg = PPOEvalConfig(env_config=train_cfg.env_config, obs_config=train_cfg.obs_config)

	train_out = train_ppo_multi(train_cfg)
	model_path = Path(train_out["model_path"])
	eval_out = evaluate_ppo_multi(model_path=model_path, config=eval_cfg)

	return {
		"train": train_out,
		"eval": eval_out,
	}
