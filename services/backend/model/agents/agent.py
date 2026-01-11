from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..trade_env.types import Action


class DecisionAgent:
	"""Agent interface.

	This is intentionally simple so you can later wrap it with PPO/SAC:
	- `act(observation) -> Action`
	"""

	def act(self, observation: Dict[str, Any]) -> Action:
		raise NotImplementedError


@dataclass
class RandomAgent(DecisionAgent):
	seed: int = 42

	def __post_init__(self):
		self._rng = np.random.default_rng(self.seed)

	def act(self, observation: Dict[str, Any]) -> Action:
		position = int(observation.get("position", 0))
		# Only sample valid actions to respect constraints.
		if position == 0:
			choices = [Action.HOLD, Action.BUY]
		else:
			choices = [Action.HOLD, Action.SELL]
		return Action(int(self._rng.choice(choices)))


@dataclass
class HeuristicForecastAgent(DecisionAgent):
	"""Baseline heuristic using forecasted closes.

	Rules:
	- If flat: BUY when the forecasted max close (remaining) exceeds current close by `buy_threshold`.
	- If holding: SELL when current close exceeds entry by `take_profit`, or when forecast suggests downside by `sell_threshold`.
	"""

	buy_threshold: float = 0.25
	take_profit: float = 0.5
	sell_threshold: float = 0.25

	def act(self, observation: Dict[str, Any]) -> Action:
		position = int(observation["position"])
		entry_price: Optional[float] = observation.get("entry_price")
		bar = np.asarray(observation["bar"], dtype=np.float32)
		current_close = float(bar[3])

		forecast_remaining = np.asarray(observation.get("forecast_remaining", []), dtype=np.float32)
		if forecast_remaining.size == 0:
			return Action.HOLD
		forecast_closes = forecast_remaining[:, 3]
		max_future = float(np.max(forecast_closes))
		min_future = float(np.min(forecast_closes))

		if position == 0:
			if (max_future - current_close) >= float(self.buy_threshold):
				return Action.BUY
			return Action.HOLD

		# position == 1
		if entry_price is not None and (current_close - float(entry_price)) >= float(self.take_profit):
			return Action.SELL
		# If forecast indicates likely downside from here, exit.
		if (current_close - min_future) >= float(self.sell_threshold):
			return Action.SELL
		return Action.HOLD
