from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


@dataclass(frozen=True)
class PPOModelMeta:
	"""Metadata written next to a saved PPO .zip.

	This lets eval/inference recreate the exact ObservationConfig/EnvConfig the model
	was trained with, so we don't rely on guessing dimensions.
	"""

	# What the model was trained for
	contracts: Optional[Sequence[str]] = None
	contract: Optional[str] = None
	action_scheme: str = "legacy3"  # "legacy3" or "toggle2"
	bar_minutes: int = 1

	# Training-time configs (plain dicts to avoid import cycles)
	obs_config: Dict[str, Any] = None  # type: ignore[assignment]
	env_config: Dict[str, Any] = None  # type: ignore[assignment]

	# Provenance
	seed: Optional[int] = None
	timesteps: Optional[int] = None
	created_at_utc: Optional[str] = None

	def to_json(self) -> str:
		return json.dumps(asdict(self), indent=2, sort_keys=True)


def meta_path_for_model(model_path: Path) -> Path:
	# ppo_H.zip -> ppo_H.meta.json
	return model_path.with_suffix(".meta.json")


def write_model_meta(model_path: Path, meta: PPOModelMeta) -> Path:
	path = meta_path_for_model(model_path)
	path.write_text(meta.to_json())
	return path


def load_model_meta(model_path: Path) -> Optional[PPOModelMeta]:
	path = meta_path_for_model(model_path)
	if not path.exists():
		return None
	data = json.loads(path.read_text())
	return PPOModelMeta(**data)
