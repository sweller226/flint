"""CLI: export a next-hour seq2seq forecast to a CSV with df_h.csv-compatible columns.

Example
    python -m services.backend.model.forecaster.export_forecast \
    --contract H \
        --start-ts "2016-01-10T23:00:00Z" \
    --checkpoint services/backend/model/forecaster/artifacts/lstm_forecaster.pt \
    --scaler-json services/backend/model/forecaster/artifacts/scaler.json \
    --out services/backend/model/forecaster/artifacts/forecast_H_2016-01-10T23-00-00Z.csv

Notes
- start-ts must match an existing row timestamp exactly.
- The script forecasts 60 steps (next hour) by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Supports:
# - python -m services.backend.model.forecaster.export_forecast --contract H --start-ts "2025-09-15T07:00:00Z" --align next
if __package__:
    from .sample_builder import (
        build_forecast_timestamps,
        build_sample_windows,
        load_contract_csv,
        parse_utc_timestamp,
        validate_request_timestamp_range,
    )
    from .seq2seq_forecaster import load_artifacts, forecast_next_hour_delta_last
else:  # pragma: no cover
    from sample_builder import (
        build_forecast_timestamps,
        build_sample_windows,
        load_contract_csv,
        parse_utc_timestamp,
        validate_request_timestamp_range,
    )
    from seq2seq_forecaster import load_artifacts, forecast_next_hour_delta_last


# === Script Parameters (optional) ===
# If you prefer not to pass CLI args, set these and run:
#   python services/backend/model/forecaster/export_forecast.py
# CLI flags still override these defaults when provided.
CONTRACT: str = "H"  # H|M|U|Z
START_TS: str | None = None  # e.g. "2019-02-15T01:04:00Z"
ALIGN: str = "next"  # exact|next|prev|nearest
HORIZON: int = 60

# Leave these as None to use the script defaults.
CHECKPOINT_PATH: str | None = None
SCALER_JSON_PATH: str | None = None
OUT_PATH: str | None = None
LOG_VOLUME: bool = True


def _default_data_dir() -> Path:
    # services/backend/model/forecaster/export_forecast.py -> services/backend/model/data
    return Path(__file__).resolve().parents[1] / "data"


def _default_artifacts_dir() -> Path:
    return Path(__file__).resolve().parent / "artifacts"


def _default_forecasts_dir() -> Path:
    return _default_artifacts_dir() / "forecasts"


def export_forecast(
    *,
    contract: str,
    start_ts: str,
    align: str = "next",
    horizon: int = 60,
    data_dir: Path | None = None,
    checkpoint: Path | None = None,
    scaler_json: Path | None = None,
    out: Path | None = None,
    log_volume: bool = True,
    days: int | None = None,
    day_len: int | None = None,
) -> Path:
    """Programmatic API: export a forecast CSV and return the output path."""
    if data_dir is None:
        data_dir = _default_data_dir()
    if checkpoint is None:
        checkpoint = _default_artifacts_dir() / "lstm_forecaster.pt"
    if scaler_json is None:
        scaler_json = _default_artifacts_dir() / "scaler.json"

    start_ts_parsed = parse_utc_timestamp(start_ts)
    validate_request_timestamp_range(start_ts_parsed)

    csv_path = data_dir / f"df_{contract.lower()}.csv"
    df = load_contract_csv(csv_path, log_volume=log_volume)

    artifacts = load_artifacts(
        checkpoint_path=checkpoint,
        scaler_json_path=scaler_json if scaler_json.exists() else None,
        log_volume=log_volume,
        default_horizon=int(horizon),
        default_days_per_sample=int(days) if days is not None else 7,
        default_day_len=int(day_len) if day_len is not None else 390,
    )

    use_days = int(days) if days is not None else int(artifacts.days_per_sample)
    use_day_len = int(day_len) if day_len is not None else int(artifacts.day_len)

    x_abs, _y_abs, y_ts_from_df, contract_month, _start_idx, aligned_start_ts = build_sample_windows(
        df,
        start_ts=start_ts_parsed,
        horizon=int(horizon),
        days_per_sample=use_days,
        day_len=use_day_len,
        feature_cols=artifacts.feature_cols,
        align=align,
    )

    if aligned_start_ts != start_ts_parsed:
        print(
            f"aligned_start_ts: requested={start_ts_parsed.isoformat()} -> used={aligned_start_ts.isoformat()} (align={align})"
        )

    forecast_ts = build_forecast_timestamps(aligned_start_ts, horizon=int(horizon))

    df_forecast = forecast_next_hour_delta_last(
        artifacts=artifacts,
        x_abs=x_abs,
        start_ts=aligned_start_ts,
        timestamps=forecast_ts,
        contract_month=contract_month,
    )

    if out is None:
        safe = start_ts_parsed.strftime("%Y-%m-%dT%H-%M-%SZ")
        out = _default_forecasts_dir() / f"forecast_{contract}_{safe}.csv"

    out.parent.mkdir(parents=True, exist_ok=True)
    df_forecast.to_csv(out, index=False)
    print(f"saved_forecast_csv: {out}")

    if len(y_ts_from_df) == int(horizon):
        print(f"actual_window_in_df: {y_ts_from_df[0].isoformat()} -> {y_ts_from_df[-1].isoformat()}")

    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Export a seq2seq LSTM next-hour forecast to CSV")
    p.add_argument("--contract", choices=["H", "M", "U", "Z"], default=None, help="Which contract CSV to use")
    p.add_argument("--start-ts", default=None, help="Forecast start timestamp (UTC ISO8601)")
    p.add_argument("--data-dir", type=Path, default=_default_data_dir(), help="Directory containing df_h.csv etc")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=_default_artifacts_dir() / "lstm_forecaster.pt",
        help="Path to trained torch checkpoint",
    )
    p.add_argument(
        "--scaler-json",
        type=Path,
        default=_default_artifacts_dir() / "scaler.json",
        help="Path to scaler.json (used if checkpoint lacks mean/std)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: artifacts/forecasts/forecast_<contract>_<start>.csv)",
    )
    p.add_argument("--horizon", type=int, default=60, help="Forecast horizon in minutes (default: 60)")
    p.add_argument("--days", type=int, default=None, help="Days per sample (default: from checkpoint or 7)")
    p.add_argument("--day-len", type=int, default=None, help="Trading minutes per day override")
    p.add_argument(
        "--align",
        choices=["exact", "next", "prev", "nearest"],
        default="next",
        help="How to align start-ts to available row timestamps (default: next)",
    )
    p.add_argument(
        "--log-volume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to log1p volume on input and expm1 on output (must match training)",
    )
    args = p.parse_args()

    contract = args.contract or CONTRACT
    start_ts = args.start_ts or START_TS
    if start_ts is None:
        raise SystemExit("Missing start timestamp. Provide --start-ts (UTC), e.g. 2025-09-15T07:00:00Z")

    checkpoint = args.checkpoint
    if CHECKPOINT_PATH is not None and args.checkpoint == _default_artifacts_dir() / "lstm_forecaster.pt":
        checkpoint = Path(CHECKPOINT_PATH)

    scaler_json = args.scaler_json
    if SCALER_JSON_PATH is not None and args.scaler_json == _default_artifacts_dir() / "scaler.json":
        scaler_json = Path(SCALER_JSON_PATH)

    out = args.out
    if OUT_PATH is not None and args.out is None:
        out = Path(OUT_PATH)

    try:
        _exported = export_forecast(
            contract=contract,
            start_ts=str(start_ts),
            align=args.align if args.align is not None else ALIGN,
            horizon=int(args.horizon) if args.horizon is not None else HORIZON,
            data_dir=args.data_dir,
            checkpoint=checkpoint,
            scaler_json=scaler_json,
            out=out,
            log_volume=bool(args.log_volume) if args.log_volume is not None else LOG_VOLUME,
            days=args.days,
            day_len=args.day_len,
        )
    except RuntimeError as exc:
        # Common: missing torch. Exit cleanly for CLI usage.
        raise SystemExit(str(exc)) from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
