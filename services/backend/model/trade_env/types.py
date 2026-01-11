from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class Action(IntEnum):
	HOLD = 0
	BUY = 1
	SELL = 2


@dataclass(frozen=True)
class Trade:
	timestamp: pd.Timestamp
	action: Action
	price: float


@dataclass(frozen=True)
class StepResult:
	observation: Dict[str, Any]
	reward: float
	done: bool
	info: Dict[str, Any]


@dataclass(frozen=True)
class EpisodeResult:
	# Back-compat: net realized PnL (includes trade_cost, excludes shaping penalties)
	pnl: float
	# New metrics
	pnl_gross: float
	shaped_return: float
	roi_net: float
	roi_gross: float
	trades: List[Trade]
	rewards: np.ndarray
	final_position: int
	final_entry_price: Optional[float]
