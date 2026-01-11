from fastapi import HTTPException
from typing import Optional, Dict
import pandas as pd
import os
from services.market_state import MarketState, load_data

# Contract code to human-readable name mapping
CONTRACT_INFO = {
    "H": {"name": "March (Q1)", "file": "df_h.csv"},
    "M": {"name": "June (Q2)", "file": "df_m.csv"},
    "U": {"name": "September (Q3)", "file": "df_u.csv"},
    "Z": {"name": "December (Q4)", "file": "df_z.csv"},
}

# Registry of loaded MarketState instances per contract
_contract_states: Dict[str, MarketState] = {}
_data_dir: Optional[str] = None


def get_market_state(contract: str = "H") -> MarketState:
    """Get the MarketState instance for a specific contract."""
    global _contract_states, _data_dir
    
    contract = contract.upper()
    if contract not in CONTRACT_INFO:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid contract code: {contract}. Valid codes: {list(CONTRACT_INFO.keys())}"
        )
    
    # Lazy load contract data
    if contract not in _contract_states:
        if _data_dir is None:
            raise HTTPException(
                status_code=503,
                detail="Market data directory not configured."
            )
        file_path = os.path.join(_data_dir, CONTRACT_INFO[contract]["file"])
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"Data file not found for contract {contract}: {file_path}"
            )
        try:
            _contract_states[contract] = load_data(file_path)
            print(f"[OK] Loaded contract {contract} from {file_path}")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load contract {contract}: {e}"
            )
    
    return _contract_states[contract]


def initialize_market_state(data_path: str) -> None:
    """Initialize the market state data directory (for backward compatibility)"""
    global _data_dir, _contract_states
    
    # Convert to absolute path to ensure consistent file access
    data_path = os.path.abspath(data_path)
    
    # Check if data_path is a directory containing contract files
    if os.path.isdir(data_path):
        _data_dir = data_path
    else:
        # Legacy: single file mode - check if es_futures directory exists nearby
        parent_dir = os.path.dirname(data_path)
        es_futures_dir = os.path.join(parent_dir, "data", "es_futures")
        if os.path.isdir(es_futures_dir):
            _data_dir = es_futures_dir
        else:
            # Fallback: use the file's directory
            _data_dir = parent_dir
    
    print(f"[OK] Market data directory set to: {_data_dir}")
    
    # Pre-load the default contract (H)
    try:
        get_market_state("H")
    except Exception as e:
        print(f"[WARN] Could not pre-load default contract: {e}")


def get_available_contracts():
    """Return list of available contracts with metadata."""
    return [
        {"code": code, "name": info["name"]}
        for code, info in CONTRACT_INFO.items()
    ]


def parse_timestamp(ts_str: Optional[str]) -> Optional[pd.Timestamp]:
    """Parse timestamp string to pandas Timestamp"""
    if ts_str is None:
        return None
    try:
        return pd.Timestamp(ts_str, tz="UTC")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timestamp: {ts_str}. Use ISO8601 format."
        )