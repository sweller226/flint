from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ICTFeatureConfig:
    lookback_1m: int = 120
    lookback_5m: int = 60  # in minutes
    swing_lookback_1m: int = 20
    swing_lookback_5m: int = 12  # in 5m candles
    swing_lookback_1h: int = 12  # in 1h candles
    swing_lookback_4h: int = 12  # in 4h candles
    # Opening Range Breakout (ORB)
    orb_minutes: int = 5


ICT_FEATURE_DIM = 34


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > 1e-12 else 0.0


def _tail2d(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, x.shape[1]), dtype=np.float32)
    if x.shape[0] <= n:
        return x
    return x[-n:]


def _aggregate_n(bars_1m: np.ndarray, n_minutes: int) -> np.ndarray:
    """Aggregate 1m OHLCV bars into N-minute OHLCV bars."""
    n_minutes = int(n_minutes)
    if n_minutes <= 0:
        return np.zeros((0, 5), dtype=np.float32)
    if bars_1m.shape[0] < n_minutes:
        return np.zeros((0, 5), dtype=np.float32)
    n = bars_1m.shape[0] // n_minutes
    out = np.zeros((n, 5), dtype=np.float32)
    for i in range(n):
        c = bars_1m[i * n_minutes : (i + 1) * n_minutes]
        out[i, 0] = float(c[0, 0])
        out[i, 1] = float(np.max(c[:, 1]))
        out[i, 2] = float(np.min(c[:, 2]))
        out[i, 3] = float(c[-1, 3])
        out[i, 4] = float(np.sum(c[:, 4]))
    return out


def _atr14(bars: np.ndarray) -> float:
    if bars.shape[0] < 2:
        return 0.0
    h = bars[:, 1]
    l = bars[:, 2]
    c = bars[:, 3]
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    if tr.size == 0:
        return 0.0
    n = min(14, int(tr.size))
    return float(np.mean(tr[-n:]))


def _rsi14(closes: np.ndarray) -> float:
    closes = closes.astype(np.float32, copy=False).reshape(-1)
    if closes.size < 2:
        return 0.0
    delta = np.diff(closes)
    n = min(14, int(delta.size))
    d = delta[-n:]
    gain = np.mean(np.maximum(d, 0.0))
    loss = np.mean(np.maximum(-d, 0.0))
    rs = _safe_div(float(gain), float(loss))
    return float(100.0 - (100.0 / (1.0 + rs)))


def _last_swing_levels(bars: np.ndarray, lookback: int) -> Tuple[float, float]:
    """Very simple swing high/low over a trailing window."""
    if bars.shape[0] == 0:
        return 0.0, 0.0
    w = _tail2d(bars, int(lookback))
    swing_high = float(np.max(w[:, 1]))
    swing_low = float(np.min(w[:, 2]))
    return swing_high, swing_low


def compute_ict_features(
    *,
    context_1m: np.ndarray,
    current_bar_1m: np.ndarray,
    hour_so_far_1m: np.ndarray | None = None,
    cfg: ICTFeatureConfig = ICTFeatureConfig(),
) -> np.ndarray:
    """Compute a compact set of price-action/ICT-inspired features.

    This is intentionally lightweight and purely based on past+current candles.
    It does NOT require labels and is suitable as extra RL observation features.

    Returns: (ICT_FEATURE_DIM,) float32
    """
    ctx = np.asarray(context_1m, dtype=np.float32)
    cur = np.asarray(current_bar_1m, dtype=np.float32).reshape(5)

    # Build a trailing window including current bar.
    ctx = _tail2d(ctx, int(cfg.lookback_1m))
    bars = np.concatenate([ctx, cur.reshape(1, 5)], axis=0) if ctx.size else cur.reshape(1, 5)

    o, h, l, c, v = [bars[:, i] for i in range(5)]
    cur_o, cur_h, cur_l, cur_c, cur_v = [float(cur[i]) for i in range(5)]

    rng = max(1e-6, float(cur_h - cur_l))
    body = float(abs(cur_c - cur_o))
    upper_wick = float(cur_h - max(cur_c, cur_o))
    lower_wick = float(min(cur_c, cur_o) - cur_l)

    prev_c = float(c[-2]) if c.size >= 2 else cur_c
    ret_1 = float(cur_c - prev_c)
    ret_5 = float(cur_c - float(c[-6])) if c.size >= 6 else 0.0

    atr = _atr14(bars)
    rsi = _rsi14(c)

    # 1m structure
    swing_high_1m, swing_low_1m = _last_swing_levels(bars[:-1], cfg.swing_lookback_1m)
    bos_up_1m = 1.0 if (swing_high_1m and cur_c > swing_high_1m) else 0.0
    bos_down_1m = 1.0 if (swing_low_1m and cur_c < swing_low_1m) else 0.0

    # Liquidity sweep heuristic: wick takes out swing, but close returns inside.
    sweep_high_1m = 1.0 if (swing_high_1m and cur_h > swing_high_1m and cur_c < swing_high_1m) else 0.0
    sweep_low_1m = 1.0 if (swing_low_1m and cur_l < swing_low_1m and cur_c > swing_low_1m) else 0.0

    # Fair Value Gap (very simplified): bar[-1] low > bar[-3] high (up gap) or high < low[-3] (down gap)
    fvg_up = 0.0
    fvg_down = 0.0
    if bars.shape[0] >= 3:
        hi_2 = float(bars[-3, 1])
        lo_2 = float(bars[-3, 2])
        if cur_l > hi_2:
            fvg_up = 1.0
        if cur_h < lo_2:
            fvg_down = 1.0

    # 5m structure from the last 60 minutes (or available)
    bars_60m = _tail2d(bars, int(cfg.lookback_5m))
    bars_5m = _aggregate_n(bars_60m, 5)
    swing_high_5m, swing_low_5m = _last_swing_levels(bars_5m[:-1], cfg.swing_lookback_5m) if bars_5m.size else (0.0, 0.0)
    trend_5m = float(bars_5m[-1, 3] - bars_5m[-4, 3]) if bars_5m.shape[0] >= 4 else 0.0
    range_5m = float(bars_5m[-1, 1] - bars_5m[-1, 2]) if bars_5m.shape[0] >= 1 else 0.0


    # 5m liquidity sweep heuristic relative to 5m swing levels.
    sweep_high_5m = 1.0 if (swing_high_5m and cur_h > swing_high_5m and cur_c < swing_high_5m) else 0.0
    sweep_low_5m = 1.0 if (swing_low_5m and cur_l < swing_low_5m and cur_c > swing_low_5m) else 0.0

    # 1h / 4h structure from longer context (use up to 16 hours of 1m data if available)
    bars_16h = _tail2d(bars, 60 * 16)
    bars_1h = _aggregate_n(bars_16h, 60)
    bars_4h = _aggregate_n(bars_16h, 240)
    swing_high_1h, swing_low_1h = _last_swing_levels(bars_1h[:-1], cfg.swing_lookback_1h) if bars_1h.size else (0.0, 0.0)
    swing_high_4h, swing_low_4h = _last_swing_levels(bars_4h[:-1], cfg.swing_lookback_4h) if bars_4h.size else (0.0, 0.0)
    trend_1h = float(bars_1h[-1, 3] - bars_1h[-4, 3]) if bars_1h.shape[0] >= 4 else 0.0
    trend_4h = float(bars_4h[-1, 3] - bars_4h[-2, 3]) if bars_4h.shape[0] >= 2 else 0.0
    range_1h = float(bars_1h[-1, 1] - bars_1h[-1, 2]) if bars_1h.shape[0] >= 1 else 0.0
    range_4h = float(bars_4h[-1, 1] - bars_4h[-1, 2]) if bars_4h.shape[0] >= 1 else 0.0

    # Normalize some signals by ATR to reduce scale issues.
    atr_n = atr if atr > 1e-6 else 1.0
    dist_sh_1m = float((cur_c - swing_high_1m) / atr_n) if swing_high_1m else 0.0
    dist_sl_1m = float((cur_c - swing_low_1m) / atr_n) if swing_low_1m else 0.0
    dist_sh_5m = float((cur_c - swing_high_5m) / atr_n) if swing_high_5m else 0.0
    dist_sl_5m = float((cur_c - swing_low_5m) / atr_n) if swing_low_5m else 0.0
    dist_sh_1h = float((cur_c - swing_high_1h) / atr_n) if swing_high_1h else 0.0
    dist_sl_1h = float((cur_c - swing_low_1h) / atr_n) if swing_low_1h else 0.0
    dist_sh_4h = float((cur_c - swing_high_4h) / atr_n) if swing_high_4h else 0.0
    dist_sl_4h = float((cur_c - swing_low_4h) / atr_n) if swing_low_4h else 0.0

    # Opening Range Breakout (ORB): uses only the already-seen portion of this hour.
    orb_n = int(getattr(cfg, "orb_minutes", 0) or 0)
    orb_ready = 0.0
    orb_high = 0.0
    orb_low = 0.0
    orb_break_up = 0.0
    orb_break_down = 0.0
    orb_range_atr = 0.0
    orb_dist_high = 0.0
    orb_dist_low = 0.0
    if hour_so_far_1m is not None and orb_n > 0:
        hsf = np.asarray(hour_so_far_1m, dtype=np.float32)
        # Use whatever is available until ORB completes; after completion, ORB is fixed.
        if hsf.ndim == 2 and hsf.shape[1] == 5 and hsf.shape[0] >= 1:
            w = hsf[: min(hsf.shape[0], orb_n)]
            orb_high = float(np.max(w[:, 1]))
            orb_low = float(np.min(w[:, 2]))
            orb_ready = 1.0 if hsf.shape[0] >= orb_n else 0.0
            if orb_high and orb_low:
                orb_range_atr = float((orb_high - orb_low) / atr_n)
                orb_dist_high = float((cur_c - orb_high) / atr_n)
                orb_dist_low = float((cur_c - orb_low) / atr_n)
                if orb_ready:
                    orb_break_up = 1.0 if cur_c > orb_high else 0.0
                    orb_break_down = 1.0 if cur_c < orb_low else 0.0

    feats = np.asarray(
        [
            _safe_div(body, rng),
            _safe_div(upper_wick, rng),
            _safe_div(lower_wick, rng),
            _safe_div(ret_1, atr_n),
            _safe_div(ret_5, atr_n),
            _safe_div(atr, max(1e-6, abs(cur_c))),
            _safe_div(rsi, 100.0),
            dist_sh_1m,
            dist_sl_1m,
            bos_up_1m,
            bos_down_1m,
            sweep_high_1m,
            sweep_low_1m,
            fvg_up,
            fvg_down,
            _safe_div(trend_5m, atr_n),
            _safe_div(range_5m, atr_n),
            1.0 if (cur_v > float(np.mean(v[-20:])) if v.size >= 20 else False) else 0.0,

            # Liquidity sweep (5m)
            sweep_high_5m,
            sweep_low_5m,

            # ORB
            orb_ready,
            orb_range_atr,
            orb_dist_high,
            orb_dist_low,
            orb_break_up,
            orb_break_down,

            # Higher-timeframe context (1h, 4h)
            dist_sh_1h,
            dist_sl_1h,
            _safe_div(trend_1h, atr_n),
            _safe_div(range_1h, atr_n),
            dist_sh_4h,
            dist_sl_4h,
            _safe_div(trend_4h, atr_n),
            _safe_div(range_4h, atr_n),
        ],
        dtype=np.float32,
    )

    if feats.shape != (ICT_FEATURE_DIM,):
        raise RuntimeError(f"ICT feature vector wrong shape {feats.shape}, expected {(ICT_FEATURE_DIM,)}")
    return feats
