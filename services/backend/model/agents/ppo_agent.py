from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..trade_env.types import Action


@dataclass
class PPOTradingAgent:
	"""DecisionAgent-like wrapper around a trained SB3 PPO model."""

	model_path: Path

	def __post_init__(self):
		try:
			from stable_baselines3 import PPO
		except Exception as exc:  # pragma: no cover
			raise RuntimeError(
				"stable-baselines3 is required. Install with `pip install stable-baselines3 gymnasium`."
			) from exc

		self._PPO = PPO
		self._model = PPO.load(str(self.model_path))

	def act_vec(self, obs_vec: np.ndarray, *, deterministic: bool = True) -> Action:
		action, _state = self._model.predict(obs_vec, deterministic=deterministic)
		return Action(int(action))

	def act(self, observation: Dict[str, Any]) -> Action:
		# If you want to use this agent with `TradingHourEnv` directly, you must convert
		# the dict observation to the same vector used during training.
		raise NotImplementedError(
			"Use TradingHourGymEnv for PPO, or convert dict obs to vector consistently."
		)
