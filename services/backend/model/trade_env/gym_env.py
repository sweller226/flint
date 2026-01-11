from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

try:
	import gymnasium as gym
	from gymnasium import spaces
except Exception as exc:  # pragma: no cover
	raise RuntimeError(
		"gymnasium is required for PPO training. Install with `pip install gymnasium stable-baselines3`."
	) from exc

from .environment import EnvConfig, TradingHourEnv
from .sample_builder import WeekSample
from .types import Action


@dataclass
class ObservationConfig:
	"""Controls how we convert dict observations into a fixed-size vector."""

	use_ohlcv: bool = True
	use_position: bool = True
	use_entry_price: bool = True
	use_time_fraction: bool = True

	# Include last-N closes from history and the forecast hour closes.
	last_closes: int = 60
	forecast_closes: int = 60  # set to 0 to exclude forecast entirely

	# Optional normalization: subtract current close from close sequences.
	normalize_by_current_close: bool = True


def _tail_1d(values: np.ndarray, n: int) -> np.ndarray:
	if n <= 0:
		return np.zeros((0,), dtype=np.float32)
	values = np.asarray(values, dtype=np.float32).reshape(-1)
	if values.size >= n:
		return values[-n:]
	pad = np.zeros((n - values.size,), dtype=np.float32)
	return np.concatenate([pad, values], axis=0)


class TradingHourGymEnv(gym.Env):
	"""Gymnasium wrapper around `TradingHourEnv` for discrete-action PPO.

	Action space:
	- 0: HOLD
	- 1: BUY (only valid when flat)
	- 2: SELL (only valid when holding)

	Observation space:
	- 1D float vector (fixed length), derived from `TradingHourEnv`'s dict observation.
	"""

	metadata = {"render_modes": []}

	def __init__(
		self,
		*,
		samples: List[WeekSample],
		forecast_mode: str = "literal",  # "literal" for now
		env_config: Optional[EnvConfig] = None,
		obs_config: Optional[ObservationConfig] = None,
		seed: int = 42,
	):
		super().__init__()
		if not samples:
			raise ValueError("samples must be a non-empty list")

		self.samples = samples
		self.forecast_mode = str(forecast_mode)
		self.env_config = env_config or EnvConfig()
		self.obs_config = obs_config or ObservationConfig()
		self._rng = np.random.default_rng(seed)

		self._env: Optional[TradingHourEnv] = None

		self.action_space = spaces.Discrete(3)
		obs_dim = self._obs_dim()
		self.observation_space = spaces.Box(
			low=-np.inf,
			high=np.inf,
			shape=(obs_dim,),
			dtype=np.float32,
		)

	def _obs_dim(self) -> int:
		dim = 0
		if self.obs_config.use_ohlcv:
			dim += 5
		if self.obs_config.use_position:
			dim += 1
		if self.obs_config.use_entry_price:
			dim += 1
		if self.obs_config.use_time_fraction:
			dim += 1
		dim += int(self.obs_config.last_closes)
		dim += int(self.obs_config.forecast_closes)
		return dim

	def _make_forecast_hour(self, sample: WeekSample) -> np.ndarray:
		# For now: use the literal actual hour as the forecast.
		if self.forecast_mode == "literal":
			return sample.actual_hour
		raise ValueError(f"Unknown forecast_mode: {self.forecast_mode}")

	def _obs_to_vec(self, obs: Dict[str, Any]) -> np.ndarray:
		cfg = self.obs_config
		bar = np.asarray(obs["bar"], dtype=np.float32)
		current_close = float(bar[3])

		pieces: List[np.ndarray] = []
		if cfg.use_ohlcv:
			pieces.append(bar.astype(np.float32, copy=False))
		if cfg.use_position:
			pieces.append(np.asarray([float(obs.get("position", 0))], dtype=np.float32))
		if cfg.use_entry_price:
			entry = obs.get("entry_price")
			entry_val = float(entry) if entry is not None else 0.0
			# Store as delta from current close for scale stability.
			pieces.append(np.asarray([entry_val - current_close], dtype=np.float32))
		if cfg.use_time_fraction:
			t = float(obs.get("t", 0))
			pieces.append(np.asarray([t / 59.0], dtype=np.float32))

		# Last-N closes from history (use the env-provided window).
		hw = obs.get("history_windows", {})
		last_1h = np.asarray(hw.get("last_1h", np.zeros((0, 5), dtype=np.float32)), dtype=np.float32)
		history_closes = last_1h[:, 3] if last_1h.size else np.zeros((0,), dtype=np.float32)
		closes_tail = _tail_1d(history_closes, int(cfg.last_closes))

		# Forecast hour closes (fixed shape): use the *full* hour forecast to keep obs size constant.
		forecast_full = np.asarray(obs.get("forecast_remaining", np.zeros((0, 5), dtype=np.float32)), dtype=np.float32)
		# forecast_remaining is variable-length; pad/crop to configured forecast_closes.
		forecast_closes = forecast_full[:, 3] if forecast_full.size else np.zeros((0,), dtype=np.float32)
		forecast_tail = _tail_1d(forecast_closes, int(cfg.forecast_closes))

		if cfg.normalize_by_current_close:
			closes_tail = closes_tail - current_close
			forecast_tail = forecast_tail - current_close

		pieces.append(closes_tail.astype(np.float32, copy=False))
		pieces.append(forecast_tail.astype(np.float32, copy=False))

		vec = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
		if vec.shape != (self._obs_dim(),):
			raise RuntimeError(f"Observation vector has wrong shape {vec.shape}, expected {(self._obs_dim(),)}")
		return vec

	def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
		if seed is not None:
			self._rng = np.random.default_rng(seed)
		sample = self.samples[int(self._rng.integers(0, len(self.samples)))]
		forecast_hour = self._make_forecast_hour(sample)
		self._env = TradingHourEnv(
			history=sample.history,
			actual_hour=sample.actual_hour,
			forecast_hour=forecast_hour,
			timestamps=sample.timestamps_hour,
			config=self.env_config,
		)
		obs = self._env.reset()
		return self._obs_to_vec(obs), {}

	def step(self, action: int):
		if self._env is None:
			raise RuntimeError("Env not initialized; call reset()")
		step = self._env.step(Action(int(action)))
		terminated = bool(step.done)
		truncated = False
		obs_vec = np.zeros((self._obs_dim(),), dtype=np.float32) if terminated else self._obs_to_vec(step.observation)
		info = dict(step.info)
		return obs_vec, float(step.reward), terminated, truncated, info
