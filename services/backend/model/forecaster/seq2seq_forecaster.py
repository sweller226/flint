"""Seq2Seq LSTM forecaster utilities (load checkpoint, run next-hour inference, export to CSV).

This module is intentionally self-contained (no notebook imports).

Expected artifacts
- checkpoint: a torch checkpoint with keys:
  - model_state_dict
  - feature_cols (optional)
  - mean, std (optional)
  - config (optional): hidden_size, num_layers, dropout, horizon, days_per_sample
- scaler.json (optional): {feature_cols, mean, std}

If your checkpoint doesn't include mean/std, pass `--scaler-json` to the CLI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required to run the forecaster. Install torch in your environment."
        ) from e


FEATURE_COLS_DEFAULT = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class ForecasterArtifacts:
    model: "torch.nn.Module"
    model_family: str
    feature_cols: Tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    log_volume: bool
    horizon: int
    days_per_sample: int
    day_len: int


def build_seq2seq_lstm_forecaster(
    *,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
):
    _require_torch()
    import torch
    from torch import nn

    class Seq2SeqLSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.decoder = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.proj = nn.Linear(hidden_size, input_size)

        def forward(
            self,
            x,
            y=None,
            *,
            horizon: int = 60,
            teacher_forcing: bool = False,
            target_is_delta: bool = True,
        ):
            # x: (B, Tin, F)
            _enc_out, (h, c) = self.encoder(x)

            last_x = x[:, -1:, :]
            start_token = torch.zeros_like(last_x) if target_is_delta else last_x

            if teacher_forcing and y is not None:
                dec_in = torch.cat([start_token, y[:, :-1, :]], dim=1)
                dec_out, _ = self.decoder(dec_in, (h, c))
                return self.proj(dec_out)

            preds = []
            inp = start_token
            state = (h, c)
            for _ in range(int(horizon)):
                dec_out, state = self.decoder(inp, state)
                step = self.proj(dec_out)
                preds.append(step)
                inp = step
            return torch.cat(preds, dim=1)

    return Seq2SeqLSTMForecaster()


def build_direct_lstm_forecaster(
    *,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    horizon: int,
    output_size: int,
):
    _require_torch()
    from torch import nn

    class DirectLSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, horizon * output_size),
            )
            self.horizon = horizon
            self.output_size = output_size

        def forward(self, x, y=None, *, horizon: Optional[int] = None, teacher_forcing: bool = True):
            _out, (h, _c) = self.lstm(x)
            last = h[-1]
            hzn = int(horizon) if horizon is not None else self.horizon
            return self.head(last).view(x.shape[0], hzn, self.output_size)

    return DirectLSTMForecaster()


def build_attn_lstm_forecaster(
    *,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    horizon: int,
    output_size: int,
):
    """Matches the original notebook's AttnLSTM architecture (MLP scoring + pooled ctx + MLP head)."""

    _require_torch()
    from torch import nn

    class AttnLSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.attn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, horizon * output_size),
            )
            self.horizon = horizon
            self.output_size = output_size

        def forward(self, x, y=None, *, horizon: Optional[int] = None, teacher_forcing: bool = True):
            out, _state = self.lstm(x)  # (B, T, H)
            scores = self.attn(out).squeeze(-1)  # (B, T)
            weights = scores.softmax(dim=1).unsqueeze(-1)  # (B, T, 1)
            ctx = (weights * out).sum(dim=1)  # (B, H)
            hzn = int(horizon) if horizon is not None else self.horizon
            return self.head(ctx).view(x.shape[0], hzn, self.output_size)

    return AttnLSTMForecaster()


def _detect_model_family_from_state_dict(state_dict: Dict[str, "object"]) -> str:
    keys = list(state_dict.keys())
    if any(k.startswith("encoder.") for k in keys) and any(k.startswith("decoder.") for k in keys):
        return "seq2seq_lstm"
    if any(k.startswith("attn.") for k in keys):
        return "attn_lstm"
    if any(k.startswith("conv.") for k in keys):
        return "cnn_lstm"
    if any(k.startswith("lstm.") for k in keys):
        return "direct_lstm"
    return "unknown"


def _load_scaler_json(path: Path) -> Tuple[Tuple[str, ...], np.ndarray, np.ndarray]:
    obj = json.loads(path.read_text())
    feature_cols = tuple(obj.get("feature_cols") or FEATURE_COLS_DEFAULT)
    mean = np.array(obj["mean"], dtype=np.float32)
    std = np.array(obj["std"], dtype=np.float32)
    return feature_cols, mean, std


def load_artifacts(
    *,
    checkpoint_path: Path,
    scaler_json_path: Optional[Path] = None,
    device: Optional[str] = None,
    default_hidden_size: int = 192,
    default_num_layers: int = 2,
    default_dropout: float = 0.1,
    default_horizon: int = 60,
    default_days_per_sample: int = 7,
    default_day_len: int = 390,
    log_volume: bool = True,
) -> ForecasterArtifacts:
    """Load model + scaler artifacts from disk."""
    _require_torch()
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # PyTorch 2.6+ defaults `weights_only=True`, which can fail for checkpoints that
    # include numpy arrays / other pickled objects (common when saving mean/std).
    # We first try the safe default, then fall back to `weights_only=False`.
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
    except Exception as e:  # pragma: no cover
        msg = str(e)
        if "Weights only load failed" in msg or "weights_only" in msg or "UnpicklingError" in msg:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        else:
            raise

    feature_cols = tuple(ckpt.get("feature_cols") or FEATURE_COLS_DEFAULT)

    mean = ckpt.get("mean")
    std = ckpt.get("std")
    if mean is None or std is None:
        if scaler_json_path is None:
            raise ValueError(
                "Checkpoint does not contain mean/std; provide --scaler-json to load the scaler."
            )
        feature_cols2, mean2, std2 = _load_scaler_json(scaler_json_path)
        # Prefer checkpoint feature order if present
        if not ckpt.get("feature_cols"):
            feature_cols = feature_cols2
        mean, std = mean2, std2

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    cfg: Dict[str, object] = ckpt.get("config") or {}
    hidden_size = int(cfg.get("hidden_size", default_hidden_size))
    num_layers = int(cfg.get("num_layers", default_num_layers))
    dropout = float(cfg.get("dropout", default_dropout))
    horizon = int(cfg.get("horizon", default_horizon))
    days_per_sample = int(cfg.get("days_per_sample", default_days_per_sample))
    day_len = int(cfg.get("day_len", default_day_len))

    state_dict = ckpt["model_state_dict"]
    model_family = str(cfg.get("model_family") or _detect_model_family_from_state_dict(state_dict))

    if model_family == "seq2seq_lstm":
        model = build_seq2seq_lstm_forecaster(
            input_size=len(feature_cols),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_family == "attn_lstm":
        model = build_attn_lstm_forecaster(
            input_size=len(feature_cols),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
            output_size=len(feature_cols),
        )
    elif model_family == "direct_lstm":
        model = build_direct_lstm_forecaster(
            input_size=len(feature_cols),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
            output_size=len(feature_cols),
        )
    else:
        raise ValueError(
            f"Unsupported checkpoint model_family={model_family!r}. "
            "Expected seq2seq_lstm, attn_lstm, or direct_lstm."
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return ForecasterArtifacts(
        model=model,
        model_family=model_family,
        feature_cols=feature_cols,
        mean=mean,
        std=std,
        log_volume=log_volume,
        horizon=horizon,
        days_per_sample=days_per_sample,
        day_len=day_len,
    )


def _normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def _denormalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x * std) + mean


def _sanitize_ohlc(pred: np.ndarray, *, open_i: int, high_i: int, low_i: int, close_i: int) -> np.ndarray:
    """Enforce basic OHLC invariants per row (optional but usually helpful)."""
    out = pred.copy()
    o = out[:, open_i]
    h = out[:, high_i]
    l = out[:, low_i]
    c = out[:, close_i]

    hi = np.maximum.reduce([h, o, c, l])
    lo = np.minimum.reduce([l, o, c])

    out[:, high_i] = hi
    out[:, low_i] = lo
    return out


def forecast_next_hour_delta_last(
    *,
    artifacts: ForecasterArtifacts,
    x_abs: np.ndarray,
    start_ts: pd.Timestamp,
    timestamps: pd.DatetimeIndex,
    contract_month: str,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Run seq2seq autoregressive forecast in delta_last mode and return a df-style DataFrame."""

    _require_torch()
    import torch

    if device is None:
        device = next(artifacts.model.parameters()).device.type

    feature_cols = list(artifacts.feature_cols)
    F = len(feature_cols)
    if x_abs.shape[1] != F:
        raise ValueError(f"x_abs has {x_abs.shape[1]} features but model expects {F}")

    x_norm = _normalize(x_abs.astype(np.float32), artifacts.mean, artifacts.std)

    xb = torch.from_numpy(x_norm).to(torch.float32).unsqueeze(0).to(device)  # (1, Tin, F)

    # delta_last: model outputs deltas relative to last input step
    last_x = xb[:, -1:, :]  # (1, 1, F)

    with torch.no_grad():
        # Most non-seq2seq models ignore `y` and `target_is_delta`.
        try:
            pred_delta = artifacts.model(
                xb,
                None,
                horizon=int(artifacts.horizon),
                teacher_forcing=False,
                target_is_delta=True,
            )
        except TypeError:
            pred_delta = artifacts.model(
                xb,
                None,
                horizon=int(artifacts.horizon),
                teacher_forcing=False,
            )
        pred_abs_norm = pred_delta + last_x

    pred_abs_norm_np = pred_abs_norm.squeeze(0).cpu().numpy().astype(np.float32)
    pred_abs = _denormalize(pred_abs_norm_np, artifacts.mean, artifacts.std)

    # Invert log-volume if needed
    if artifacts.log_volume and "volume" in feature_cols:
        vi = feature_cols.index("volume")
        v = np.expm1(pred_abs[:, vi].astype(np.float64))
        pred_abs[:, vi] = np.maximum(v, 0.0).astype(np.float32)

    # Basic OHLC cleanup
    if all(c in feature_cols for c in ("open", "high", "low", "close")):
        pred_abs = _sanitize_ohlc(
            pred_abs,
            open_i=feature_cols.index("open"),
            high_i=feature_cols.index("high"),
            low_i=feature_cols.index("low"),
            close_i=feature_cols.index("close"),
        )

    df_out = pd.DataFrame(pred_abs, columns=feature_cols)

    # Ensure df_h.csv-like column set/order
    df_out.insert(0, "ts_event", [t.isoformat() for t in timestamps])
    df_out["contract_month"] = str(contract_month)

    # If any extra cols exist in feature_cols, keep them, but enforce expected order when possible
    ordered = ["ts_event", "open", "high", "low", "close", "volume", "contract_month"]
    cols = [c for c in ordered if c in df_out.columns] + [c for c in df_out.columns if c not in ordered]
    df_out = df_out[cols]

    return df_out
