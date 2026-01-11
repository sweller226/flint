import asyncio
from typing import Dict, Any

async def log_trade_to_solana(trade_data: Dict[str, Any]) -> str:
    """
    Mock function to simulate logging to Solana.
    Returns a mock transaction signature.
    """
    print(f"[MOCK SOLANA] Logging trade: {trade_data}")
    # Simulate network delay
    await asyncio.sleep(0.5)
    return "5xSignAtuReMock123456789Solana"
