from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .types import Action, EpisodeResult, StepResult, Trade


if TYPE_CHECKING:
	from ..agents.agent import DecisionAgent


FEATURE_COLS = ["open", "high", "low", "close", "volume"]
CLOSE_IDX = FEATURE_COLS.index("close")


@dataclass
class EnvConfig:
	leverage: float = 1.0
	contract_multiplier: float = 1.0
	invalid_action_penalty: float = 0.0
	# Reward shaping
	# - holding_penalty_per_step: applied each step while in a position (position==1)
	# - flat_penalty_per_step: applied each step while flat (position==0)
	# - hold_action_penalty: applied when the agent chooses HOLD
	# - trade_cost: applied when a BUY/SELL is executed (not invalid)
	holding_penalty_per_step: float = 0.0
	flat_penalty_per_step: float = 0.0
	hold_action_penalty: float = 0.0
	trade_cost: float = 0.0
	no_trade_penalty: float = 0.0
	auto_close_on_done: bool = True
	execution_price: str = "close"  # "close" only for now


class TradingHourEnv:
	"""A minimal single-hour trading environment.

	Episode:
	- Fixed horizon of 60 steps (1 minute each)
	- At each step the agent chooses {HOLD, BUY, SELL}
	- Position constraints:
		- Can hold at most 1 contract (position ∈ {0,1})
		- Can SELL only when holding

	Observations:
	- Provides current minute OHLCV (actual data)
	- Provides forecast for the *remaining* minutes in the hour (can be literal actual hour)
	- Provides a few recent-history windows from the input series
	"""

	def __init__(
		self,
		*,
		history: np.ndarray,
		actual_hour: np.ndarray,
		forecast_hour: np.ndarray,
		timestamps: Optional[List[pd.Timestamp]] = None,
		config: Optional[EnvConfig] = None,
	):
		self.history = np.asarray(history, dtype=np.float32)
		self.actual_hour = np.asarray(actual_hour, dtype=np.float32)
		self.forecast_hour = np.asarray(forecast_hour, dtype=np.float32)
		if self.actual_hour.shape != (60, len(FEATURE_COLS)):
			raise ValueError(
				f"actual_hour must be shape (60, {len(FEATURE_COLS)}), got {self.actual_hour.shape}"
			)
		if self.forecast_hour.shape != (60, len(FEATURE_COLS)):
			raise ValueError(
				f"forecast_hour must be shape (60, {len(FEATURE_COLS)}), got {self.forecast_hour.shape}"
			)
		if self.history.ndim != 2 or self.history.shape[1] != len(FEATURE_COLS):
			raise ValueError(
				f"history must be shape (N, {len(FEATURE_COLS)}), got {self.history.shape}"
			)

		if timestamps is None:
			# If not provided, create dummy minute timestamps.
			base = pd.Timestamp("2000-01-01", tz="UTC")
			self.timestamps = [base + pd.Timedelta(minutes=i) for i in range(60)]
		else:
			if len(timestamps) != 60:
				raise ValueError(f"timestamps must have length 60, got {len(timestamps)}")
			self.timestamps = list(timestamps)

		self.config = config or EnvConfig()

		self._t: int = 0
		self._position: int = 0
		self._entry_price: Optional[float] = None
		self._pnl: float = 0.0
		self._trades: List[Trade] = []
		self._rewards: List[float] = []

	def reset(self) -> Dict[str, Any]:
		self._t = 0
		self._position = 0
		self._entry_price = None
		self._pnl = 0.0
		self._trades = []
		self._rewards = []
		return self._make_observation()

	def _execution_price(self, bar: np.ndarray) -> float:
		# For now always execute at close.
		return float(bar[CLOSE_IDX])

	def _make_observation(self) -> Dict[str, Any]:
		current_bar = self.actual_hour[self._t]
		forecast_remaining = self.forecast_hour[self._t :]

		# Multi-resolution recent history windows (clipped if insufficient).
		def tail(n: int) -> np.ndarray:
			if n <= 0:
				return np.zeros((0, len(FEATURE_COLS)), dtype=np.float32)
			return self.history[-min(len(self.history), n) :]

		obs: Dict[str, Any] = {
			"t": self._t,
			"timestamp": self.timestamps[self._t],
			"position": self._position,
			"entry_price": self._entry_price,
			"pnl": self._pnl,
			"bar": current_bar.copy(),
			"forecast_remaining": forecast_remaining.copy(),
			"history_windows": {
				"last_1h": tail(60).copy(),
				"last_2h": tail(120).copy(),
				"last_4h": tail(240).copy(),
				"last_24h": tail(60 * 24).copy(),
			},
		}
		return obs

	def step(self, action: Action) -> StepResult:
		if self._t < 0 or self._t >= 60:
			raise RuntimeError("Episode is done; call reset()")

		action = Action(int(action))
		position_before = int(self._position)
		ts = self.timestamps[self._t]
		bar = self.actual_hour[self._t]
		price = self._execution_price(bar)

		reward = 0.0
		invalid = False
		executed_trade = False
		forced_liquidation = False
		forced_liquidation_price: Optional[float] = None
		forced_liquidation_timestamp: Optional[pd.Timestamp] = None

		if action == Action.BUY:
			if self._position >= 1:
				invalid = True
			else:
				self._position = 1
				self._entry_price = price
				self._trades.append(Trade(self.timestamps[self._t], Action.BUY, price))
				executed_trade = True
		elif action == Action.SELL:
			if self._position <= 0 or self._entry_price is None:
				invalid = True
			else:
				pnl = (price - float(self._entry_price))
				pnl *= float(self.config.leverage) * float(self.config.contract_multiplier)
				reward += pnl
				self._pnl += pnl
				self._position = 0
				self._entry_price = None
				self._trades.append(Trade(self.timestamps[self._t], Action.SELL, price))
				executed_trade = True
		elif action == Action.HOLD:
			pass
		else:
			raise ValueError(f"Unknown action: {action}")

		if invalid:
			reward -= float(self.config.invalid_action_penalty)

		# Penalize choosing HOLD while flat to encourage taking trades.
		# (We do NOT penalize HOLD while holding; that would just force churn.)
		if action == Action.HOLD and position_before == 0:
			reward -= float(self.config.hold_action_penalty)

		# Transaction cost (commission/slippage proxy) when a trade actually executes.
		if executed_trade:
			reward -= float(self.config.trade_cost)

		# Optional holding penalty to discourage sitting in a position.
		# Applied after action execution at this timestep.
		if self._position == 1:
			reward -= float(self.config.holding_penalty_per_step)
		else:
			reward -= float(self.config.flat_penalty_per_step)

		self._rewards.append(float(reward))

		done = False
		# Advance time
		self._t += 1
		if self._t >= 60:
			done = True
			# Optional end-of-episode penalty to avoid the degenerate policy of never trading.
			# Apply after the final action has been processed.
			if self.config.no_trade_penalty and len(self._trades) == 0:
				self._rewards[-1] -= float(self.config.no_trade_penalty)
				reward -= float(self.config.no_trade_penalty)

			if self.config.auto_close_on_done and self._position == 1 and self._entry_price is not None:
				# Force liquidation at final bar's close.
				final_bar = self.actual_hour[-1]
				final_price = self._execution_price(final_bar)
				pnl = (final_price - float(self._entry_price))
				pnl *= float(self.config.leverage) * float(self.config.contract_multiplier)
				# Apply transaction cost to the forced liquidation for consistency.
				pnl -= float(self.config.trade_cost)
				self._pnl += pnl
				# Treat as an extra reward at the end.
				self._rewards[-1] += float(pnl)
				reward += float(pnl)
				self._position = 0
				self._entry_price = None
				self._trades.append(Trade(self.timestamps[-1], Action.SELL, float(final_price)))
				forced_liquidation = True
				forced_liquidation_price = float(final_price)
				forced_liquidation_timestamp = self.timestamps[-1]

		obs = self._make_observation() if not done else {}
		buy_count = sum(1 for tr in self._trades if tr.action == Action.BUY)
		sell_count = sum(1 for tr in self._trades if tr.action == Action.SELL)
		info: Dict[str, Any] = {
			"action": int(action),
			"invalid_action": invalid,
			"executed_trade": executed_trade,
			"forced_liquidation": forced_liquidation,
			"timestamp": ts,
			"price": float(price),
			"forced_liquidation_timestamp": forced_liquidation_timestamp,
			"forced_liquidation_price": forced_liquidation_price,
			"position": self._position,
			"pnl": self._pnl,
			"t": self._t,
			"trade_count": len(self._trades),
			"buy_count": buy_count,
			"sell_count": sell_count,
		}
		return StepResult(observation=obs, reward=float(reward), done=done, info=info)

	def run_episode(self, agent: "DecisionAgent") -> EpisodeResult:
		# Runtime import to avoid circular imports.
		from ..agents.agent import DecisionAgent

		if not isinstance(agent, DecisionAgent):
			raise TypeError("agent must be a DecisionAgent")

		obs = self.reset()
		done = False
		while not done:
			action = agent.act(obs)
			step = self.step(action)
			done = step.done
			obs = step.observation

		rewards = np.asarray(self._rewards, dtype=np.float32)
		return EpisodeResult(
			pnl=float(self._pnl),
			trades=list(self._trades),
			rewards=rewards,
			final_position=int(self._position),
			final_entry_price=self._entry_price,
		)
