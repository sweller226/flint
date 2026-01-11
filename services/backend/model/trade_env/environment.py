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
	# - trade_penalty: shaping-only penalty applied when a trade executes (does NOT affect pnl_net/pnl_gross)
	holding_penalty_per_step: float = 0.0
	flat_penalty_per_step: float = 0.0
	hold_action_penalty: float = 0.0
	trade_cost: float = 0.0
	trade_penalty: float = 0.0
	no_trade_penalty: float = 0.0
	# Reward mode:
	# - realized: reward only when closing/forced-liquidating (plus costs/penalties)
	# - mark_to_market: dense reward based on next_close - current_close while holding
	reward_mode: str = "mark_to_market"
	# Optional: delay penalties until later in the episode to avoid forcing early trades.
	# All values are in step indices [0..59]. For example, 10 means "start penalizing at minute 10".
	holding_penalty_start_step: int = 0
	flat_penalty_start_step: int = 0
	hold_action_penalty_start_step: int = 0
	# Optional action constraints
	# - min_entry_step: prevent entering a position before this step index (e.g., 5 to wait for 5m ORB)
	min_entry_step: int = 0
	auto_close_on_done: bool = True
	execution_price: str = "close"  # "close" only for now


class TradingHourEnv:
	"""A minimal single-hour trading environment.

	Episode:
	- Fixed horizon of `H` steps (default: 60 one-minute bars)
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
		if self.actual_hour.ndim != 2 or self.actual_hour.shape[1] != len(FEATURE_COLS):
			raise ValueError(
				f"actual_hour must be shape (H, {len(FEATURE_COLS)}), got {self.actual_hour.shape}"
			)
		if self.forecast_hour.ndim != 2 or self.forecast_hour.shape[1] != len(FEATURE_COLS):
			raise ValueError(
				f"forecast_hour must be shape (H, {len(FEATURE_COLS)}), got {self.forecast_hour.shape}"
			)
		if self.actual_hour.shape[0] <= 0:
			raise ValueError("actual_hour must have positive length")
		if self.forecast_hour.shape[0] != self.actual_hour.shape[0]:
			raise ValueError(
				f"forecast_hour length must match actual_hour length, got {self.forecast_hour.shape[0]} vs {self.actual_hour.shape[0]}"
			)
		self.horizon = int(self.actual_hour.shape[0])
		if self.history.ndim != 2 or self.history.shape[1] != len(FEATURE_COLS):
			raise ValueError(
				f"history must be shape (N, {len(FEATURE_COLS)}), got {self.history.shape}"
			)

		if timestamps is None:
			# If not provided, create dummy minute timestamps.
			base = pd.Timestamp("2000-01-01", tz="UTC")
			self.timestamps = [base + pd.Timedelta(minutes=i) for i in range(self.horizon)]
		else:
			if len(timestamps) != self.horizon:
				raise ValueError(
					f"timestamps must have length {self.horizon}, got {len(timestamps)}"
				)
			self.timestamps = list(timestamps)

		self.config = config or EnvConfig()

		self._t: int = 0
		self._position: int = 0
		self._entry_price: Optional[float] = None
		# Realized PnL (excludes shaping penalties). We track both gross and net.
		self._pnl_gross: float = 0.0
		self._pnl_net: float = 0.0
		# For ROI computation
		self._entry_notional: Optional[float] = None
		self._initial_notional: Optional[float] = None
		# Shaped return (sum of per-step rewards)
		self._reward_cum: float = 0.0
		self._trades: List[Trade] = []
		self._rewards: List[float] = []

	def reset(self) -> Dict[str, Any]:
		self._t = 0
		self._position = 0
		self._entry_price = None
		self._pnl_gross = 0.0
		self._pnl_net = 0.0
		self._entry_notional = None
		self._initial_notional = None
		self._reward_cum = 0.0
		self._trades = []
		self._rewards = []
		return self._make_observation()

	def _execution_price(self, bar: np.ndarray) -> float:
		# For now always execute at close.
		return float(bar[CLOSE_IDX])

	def _make_observation(self) -> Dict[str, Any]:
		current_bar = self.actual_hour[self._t]
		forecast_remaining = self.forecast_hour[self._t :]
		hour_so_far = self.actual_hour[: self._t + 1]

		# Provide a rolling context that includes the already-seen portion of the current hour.
		# This is critical for price-action style signals (swing highs/lows, BOS, etc.).
		if self._t > 0:
			ctx = np.concatenate([self.history, self.actual_hour[: self._t]], axis=0)
		else:
			ctx = self.history

		# Multi-resolution recent history windows (clipped if insufficient).
		def tail(n: int) -> np.ndarray:
			if n <= 0:
				return np.zeros((0, len(FEATURE_COLS)), dtype=np.float32)
			return ctx[-min(len(ctx), n) :]

		obs: Dict[str, Any] = {
			"t": self._t,
			"horizon": self.horizon,
			"timestamp": self.timestamps[self._t],
			"position": self._position,
			"entry_price": self._entry_price,
			"pnl": self._pnl_net,
			"pnl_gross": self._pnl_gross,
			"pnl_net": self._pnl_net,
			"shaped_return": self._reward_cum,
			"bar": current_bar.copy(),
			"hour_so_far": hour_so_far.copy(),
			"forecast_remaining": forecast_remaining.copy(),
			"history_windows": {
				"last_1h": tail(self.horizon).copy(),
				"last_2h": tail(self.horizon * 2).copy(),
				"last_4h": tail(self.horizon * 4).copy(),
				"last_24h": tail(self.horizon * 24).copy(),
			},
		}
		return obs

	def step(self, action: Action) -> StepResult:
		if self._t < 0 or self._t >= self.horizon:
			raise RuntimeError("Episode is done; call reset()")

		requested_action = Action(int(action))
		action = requested_action
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

		# Enforce position constraints by mapping invalid requests to HOLD.
		# This prevents the simulator from ever executing an impossible transition.
		if requested_action == Action.BUY and self._position >= 1:
			invalid = True
			action = Action.HOLD
		elif requested_action == Action.BUY and position_before == 0 and self._t < int(self.config.min_entry_step):
			# Optional warmup: block early entries (useful for ORB-style strategies)
			invalid = True
			action = Action.HOLD
		elif requested_action == Action.SELL and (self._position <= 0 or self._entry_price is None):
			invalid = True
			action = Action.HOLD


		if action == Action.BUY:
			self._position = 1
			self._entry_price = price
			self._entry_notional = abs(float(price)) * float(self.config.contract_multiplier)
			if self._initial_notional is None:
				self._initial_notional = float(self._entry_notional)
			self._trades.append(Trade(self.timestamps[self._t], Action.BUY, price))
			executed_trade = True
		elif action == Action.SELL:
			entry = float(self._entry_price)
			gross = (price - entry)
			gross *= float(self.config.leverage) * float(self.config.contract_multiplier)
			# In mark-to-market mode, PnL is represented as per-step deltas.
			if str(self.config.reward_mode) != "mark_to_market":
				reward += float(gross)
			self._pnl_gross += float(gross)
			self._pnl_net += float(gross)
			self._position = 0
			self._entry_price = None
			self._entry_notional = None
			self._trades.append(Trade(self.timestamps[self._t], Action.SELL, price))
			executed_trade = True
		elif action == Action.HOLD:
			pass
		else:
			raise ValueError(f"Unknown action: {action}")

		if invalid:
			reward -= float(self.config.invalid_action_penalty)

		# Dense mark-to-market reward: price change over the next interval while holding.
		# Uses future close for reward only (not exposed to the policy unless you include forecast).
		if str(self.config.reward_mode) == "mark_to_market":
			pos_after = int(self._position)
			if pos_after != 0 and (self._t + 1) < self.horizon:
				next_close = float(self.actual_hour[self._t + 1, CLOSE_IDX])
				cur_close = float(price)
				delta = (next_close - cur_close)
				delta *= float(self.config.leverage) * float(self.config.contract_multiplier)
				reward += float(delta)

		# Penalize choosing HOLD while flat to encourage taking trades.
		# (We do NOT penalize HOLD while holding; that would just force churn.)
		if (
			requested_action == Action.HOLD
			and position_before == 0
			and self._t >= int(self.config.hold_action_penalty_start_step)
		):
			reward -= float(self.config.hold_action_penalty)

		# Transaction cost (commission/slippage proxy) when a trade actually executes.
		# This affects net PnL even if it's a BUY (entry cost).
		if executed_trade and self.config.trade_cost:
			reward -= float(self.config.trade_cost)
			self._pnl_net -= float(self.config.trade_cost)

		# Shaping-only penalty to discourage excessive trading without polluting PnL.
		if executed_trade and self.config.trade_penalty:
			reward -= float(self.config.trade_penalty)

		# Optional per-step penalties (applied after action execution at this timestep).
		if self._position == 1:
			if self._t >= int(self.config.holding_penalty_start_step):
				reward -= float(self.config.holding_penalty_per_step)
		else:
			if self._t >= int(self.config.flat_penalty_start_step):
				reward -= float(self.config.flat_penalty_per_step)

		self._rewards.append(float(reward))
		self._reward_cum += float(reward)

		done = False
		# Advance time
		self._t += 1
		if self._t >= self.horizon:
			done = True
			# Optional end-of-episode penalty to avoid the degenerate policy of never trading.
			# Apply after the final action has been processed.
			if self.config.no_trade_penalty and len(self._trades) == 0:
				self._rewards[-1] -= float(self.config.no_trade_penalty)
				reward -= float(self.config.no_trade_penalty)
				self._reward_cum -= float(self.config.no_trade_penalty)

			if self.config.auto_close_on_done and self._position == 1 and self._entry_price is not None:
				# Force liquidation at final bar's close.
				final_bar = self.actual_hour[-1]
				final_price = self._execution_price(final_bar)
				entry = float(self._entry_price)
				gross = (final_price - entry)
				gross *= float(self.config.leverage) * float(self.config.contract_multiplier)
				self._pnl_gross += float(gross)
				self._pnl_net += float(gross)
				# Treat as an extra reward at the end only for realized reward mode.
				if str(self.config.reward_mode) != "mark_to_market":
					self._rewards[-1] += float(gross)
					reward += float(gross)
					self._reward_cum += float(gross)
				if self.config.trade_cost:
					self._rewards[-1] -= float(self.config.trade_cost)
					reward -= float(self.config.trade_cost)
					self._reward_cum -= float(self.config.trade_cost)
					self._pnl_net -= float(self.config.trade_cost)
				self._position = 0
				self._entry_price = None
				self._entry_notional = None
				self._trades.append(Trade(self.timestamps[-1], Action.SELL, float(final_price)))
				forced_liquidation = True
				forced_liquidation_price = float(final_price)
				forced_liquidation_timestamp = self.timestamps[-1]

		obs = self._make_observation() if not done else {}
		initial_notional = float(self._initial_notional) if self._initial_notional else 0.0
		den = initial_notional if initial_notional > 1e-12 else 0.0
		roi_net = float(self._pnl_net / den) if den else 0.0
		roi_gross = float(self._pnl_gross / den) if den else 0.0
		buy_count = sum(1 for tr in self._trades if tr.action == Action.BUY)
		sell_count = sum(1 for tr in self._trades if tr.action == Action.SELL)
		info: Dict[str, Any] = {
			"action": int(Action.SELL if forced_liquidation else action),
			"requested_action": int(requested_action),
			"invalid_action": invalid,
			"executed_trade": bool(executed_trade or forced_liquidation),
			"forced_liquidation": forced_liquidation,
			"auto_close": forced_liquidation,
			"timestamp": ts,
			"price": float(price),
			"forced_liquidation_timestamp": forced_liquidation_timestamp,
			"forced_liquidation_price": forced_liquidation_price,
			"position": self._position,
			"pnl": self._pnl_net,
			"pnl_net": self._pnl_net,
			"pnl_gross": self._pnl_gross,
			"shaped_return": self._reward_cum,
			"roi_net": roi_net,
			"roi_gross": roi_gross,
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
		initial_notional = float(self._initial_notional) if self._initial_notional else 0.0
		den = initial_notional if initial_notional > 1e-12 else 0.0
		roi_net = float(self._pnl_net / den) if den else 0.0
		roi_gross = float(self._pnl_gross / den) if den else 0.0
		return EpisodeResult(
			pnl=float(self._pnl_net),
			pnl_gross=float(self._pnl_gross),
			shaped_return=float(rewards.sum()),
			roi_net=roi_net,
			roi_gross=roi_gross,
			trades=list(self._trades),
			rewards=rewards,
			final_position=int(self._position),
			final_entry_price=self._entry_price,
		)
