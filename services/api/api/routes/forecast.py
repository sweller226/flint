import pandas as pd
import subprocess
import os
from pathlib import Path
from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

import sys
VENV_PYTHON = sys.executable

router = APIRouter(tags=["forecast"])

# Configuration
SERVICES_DIR = Path(__file__).parent.parent.parent.parent  # Adjust based on your structure
FORECAST_ARTIFACTS = SERVICES_DIR / "backend/model/forecaster/artifacts/forecasts"
AGENT_ARTIFACTS = SERVICES_DIR / "backend/model/run_agent/artifacts/agent_trades"

executor = ThreadPoolExecutor(max_workers=2)

class ForecastCandle(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class AgentAction(BaseModel):
    timestamp: str
    step: int
    action: str
    executed_trade: bool
    price: float
    reward: float
    pnl_net: float
    pnl_gross: float
    position: int
    trade_count: int

class ForecastResponse(BaseModel):
    forecast: List[ForecastCandle]
    actions: List[AgentAction]
    metrics: dict
    status: str
    execution_time_ms: float

def run_forecast_script(contract: str, start_ts: str) -> tuple[str, str]:
    """Execute the forecast generation script"""
    cmd = [
        VENV_PYTHON,
        "-m", "backend.model.forecaster.export_forecast",
        "--contract", contract,
        "--start-ts", start_ts,
        "--align", "next"
    ]
    
    result = subprocess.run(
        cmd,
        cwd=str(SERVICES_DIR),
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Forecast script failed: {result.stderr}")
    
    return result.stdout, result.stderr


def run_agent_script(contract: str, start_ts: str) -> tuple[str, str, dict]:
    """Execute the agent evaluation script"""
    cmd = [
        VENV_PYTHON,
        "-m", "backend.model.run_agent.run_eval_report",
        "--contract", contract,
        "--start-ts", start_ts,
        "--align", "next"
    ]
    
    result = subprocess.run(
        cmd,
        cwd=str(SERVICES_DIR),
        capture_output=True,
        text=True,
        timeout=90
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Agent script failed: {result.stderr}")
    
    # Parse metrics from stdout
    metrics = {}
    for line in result.stdout.splitlines():
        if ":" in line:
            parts = line.split(":", 1)
            key = parts[0].strip()
            val = parts[1].strip()
            # Try to convert to float/int
            try:
                if "." in val:
                    metrics[key] = float(val)
                else:
                    metrics[key] = int(val)
            except ValueError:
                metrics[key] = val
                
    return result.stdout, result.stderr, metrics

def find_latest_csv(directory: Path, pattern: str) -> Optional[Path]:
    """Find the most recent CSV file matching pattern"""
    matches = list(directory.glob(pattern))
    if not matches:
        return None
    # Sort by modification time, most recent first
    return max(matches, key=lambda p: p.stat().st_mtime)

def parse_forecast_csv(contract: str, start_ts: str) -> List[ForecastCandle]:
    """Parse the generated forecast CSV"""
    # Convert timestamp to filename format
    dt = datetime.fromisoformat(start_ts.replace('Z', '+00:00'))
    filename = f"forecast_{contract}_{dt.strftime('%Y-%m-%dT%H-%M-%SZ')}.csv"
    filepath = FORECAST_ARTIFACTS / filename
    
    # If exact match not found, try to find most recent forecast file
    if not filepath.exists():
        pattern = f"forecast_{contract}_*.csv"
        filepath = find_latest_csv(FORECAST_ARTIFACTS, pattern)
        
    if not filepath or not filepath.exists():
        raise FileNotFoundError(f"Forecast file not found for {contract} at {start_ts}")
    
    df = pd.read_csv(filepath)
    
    # Parse CSV with expected columns: ts_event, open, high, low, close, volume
    candles = []
    for _, row in df.iterrows():
        candles.append(ForecastCandle(
            timestamp=str(row['ts_event']),
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row.get('volume', 0.0))
        ))
    
    return candles

def parse_actions_csv(contract: str, start_ts: str) -> List[AgentAction]:
    """Parse the generated agent actions CSV"""
    # Find the actions file - may have slightly different timestamp due to alignment
    dt = datetime.fromisoformat(start_ts.replace('Z', '+00:00'))
    
    # Try exact match first
    filename = f"actions_{contract}_{dt.strftime('%Y-%m-%dT%H-%M-%SZ')}.csv"
    filepath = AGENT_ARTIFACTS / filename
    
    # If not found, search for most recent file
    if not filepath.exists():
        pattern = f"actions_{contract}_*.csv"
        filepath = find_latest_csv(AGENT_ARTIFACTS, pattern)
    
    if not filepath or not filepath.exists():
        return []  # No actions file found, return empty list
    
    df = pd.read_csv(filepath)
    
    # Parse CSV with columns: ts_event, step, action, executed_trade, price, reward, pnl_net, position, etc.
    actions = []
    for _, row in df.iterrows():
        # Only include rows where an actual trade was executed or action is not HOLD
        action_type = str(row.get('action', 'HOLD'))
        executed = bool(row.get('executed_trade', False))
        
        # Include if it's a trade execution or significant action
        if executed or action_type in ['BUY', 'SELL']:
            actions.append(AgentAction(
                timestamp=str(row['ts_event']),
                step=int(row.get('step', 0)),
                action=action_type,
                executed_trade=executed,
                price=float(row.get('price', 0.0)),
                reward=float(row.get('reward', 0.0)),
                pnl_net=float(row.get('pnl_net', 0.0)),
                pnl_gross=float(row.get('pnl_gross', 0.0)),
                position=int(row.get('position', 0)),
                trade_count=int(row.get('trade_count', 0))
            ))
    
    return actions

@router.get("/forecast", response_model=ForecastResponse)
async def generate_forecast(
    contract: str = Query("H", description="Contract quarter code"),
    start_ts: str = Query(..., description="ISO timestamp to start forecast from"),
):
    import traceback
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info(f"Forecast request: contract={contract}, start_ts={start_ts}")
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        loop = asyncio.get_event_loop()

        # 1. Run forecast script first
        forecast_stdout, forecast_stderr = await loop.run_in_executor(
            executor,
            run_forecast_script,
            contract,
            start_ts
        )

        # 2. Then run agent script
        agent_stdout, agent_stderr, metrics = await loop.run_in_executor(
            executor,
            run_agent_script,
            contract,
            start_ts
        )

        # 3. Parse forecast CSV
        forecast_candles = parse_forecast_csv(contract, start_ts)

        # 4. Parse agent actions CSV
        agent_actions = parse_actions_csv(contract, start_ts)

        end_time = asyncio.get_event_loop().time()
        execution_time_ms = (end_time - start_time) * 1000

        return ForecastResponse(
            forecast=forecast_candles,
            actions=agent_actions,
            metrics=metrics,
            status="success",
            execution_time_ms=execution_time_ms
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Forecast generation timed out")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@router.get("/forecast/status")
async def forecast_status():
    """Check if forecast generation is available"""
    return {
        "available": SERVICES_DIR.exists(),
        "forecast_artifacts": str(FORECAST_ARTIFACTS),
        "agent_artifacts": str(AGENT_ARTIFACTS),
        "forecast_artifacts_exist": FORECAST_ARTIFACTS.exists(),
        "agent_artifacts_exist": AGENT_ARTIFACTS.exists()
    }

@router.get("/forecast/debug")
async def forecast_debug(
    contract: str = Query("H", description="Contract quarter code"),
    start_ts: str = Query(..., description="ISO timestamp to start forecast from"),
):
    """Debug endpoint to check file paths and availability"""
    import os
    
    dt = datetime.fromisoformat(start_ts.replace('Z', '+00:00'))
    
    # Check forecast file
    forecast_filename = f"forecast_{contract}_{dt.strftime('%Y-%m-%dT%H-%M-%SZ')}.csv"
    forecast_filepath = FORECAST_ARTIFACTS / forecast_filename
    forecast_pattern = f"forecast_{contract}_*.csv"
    forecast_matches = list(FORECAST_ARTIFACTS.glob(forecast_pattern)) if FORECAST_ARTIFACTS.exists() else []
    
    # Check actions file
    actions_filename = f"actions_{contract}_{dt.strftime('%Y-%m-%dT%H-%M-%SZ')}.csv"
    actions_filepath = AGENT_ARTIFACTS / actions_filename
    actions_pattern = f"actions_{contract}_*.csv"
    actions_matches = list(AGENT_ARTIFACTS.glob(actions_pattern)) if AGENT_ARTIFACTS.exists() else []
    
    return {
        "request": {
            "contract": contract,
            "start_ts": start_ts,
            "parsed_datetime": dt.isoformat()
        },
        "paths": {
            "services_dir": str(SERVICES_DIR),
            "services_dir_exists": SERVICES_DIR.exists(),
            "forecast_artifacts": str(FORECAST_ARTIFACTS),
            "forecast_artifacts_exists": FORECAST_ARTIFACTS.exists(),
            "agent_artifacts": str(AGENT_ARTIFACTS),
            "agent_artifacts_exists": AGENT_ARTIFACTS.exists()
        },
        "forecast_file": {
            "expected_filename": forecast_filename,
            "expected_path": str(forecast_filepath),
            "exists": forecast_filepath.exists(),
            "pattern": forecast_pattern,
            "matches": [str(f) for f in forecast_matches],
            "match_count": len(forecast_matches)
        },
        "actions_file": {
            "expected_filename": actions_filename,
            "expected_path": str(actions_filepath),
            "exists": actions_filepath.exists(),
            "pattern": actions_pattern,
            "matches": [str(f) for f in actions_matches],
            "match_count": len(actions_matches)
        },
        "python_executable": os.sys.executable,
        "cwd": str(SERVICES_DIR)
    }