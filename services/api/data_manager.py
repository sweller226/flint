import pandas as pd
import asyncio
import yfinance as yf
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class DataManager:
    def __init__(self):
        self.history = pd.DataFrame()
        self.symbol = "ES=F"
        
        # Replay State
        self.replay_index = 0
        self.is_playing = False
        self.replay_speed = 1.0 # 1.0 = 1 candle per tick loop (or based on time)
        self.last_update_time = datetime.now()
        
    async def load_data(self, symbol: str = "ES=F", period: str = "5d"):
        """Load data from yfinance."""
        print(f"Fetching data for {symbol}...")
        try:
            self.symbol = symbol
            ticker = yf.Ticker(symbol)
            # Fetch 1m data
            df = await asyncio.to_thread(ticker.history, period=period, interval="1m")
            
            if df.empty:
                print(f"Warning: No data found for {symbol}.")
                return False
            
            df.reset_index(inplace=True)
            df.columns = [c.lower() for c in df.columns]
            
            # Normalize timestamp column
            if "datetime" in df.columns:
                df.rename(columns={"datetime": "timestamp"}, inplace=True)
            elif "date" in df.columns:
                df.rename(columns={"date": "timestamp"}, inplace=True)
            
            # Ensure proper timezone awareness (remove tz to avoid issues or keep utc)
            if pd.api.types.is_datetime64tz_dtype(df["timestamp"]):
                df["timestamp"] = df["timestamp"].dt.tz_convert(None) # Convert to naive for simplicity

            self.history = df
            self.replay_index = 0
            print(f"Loaded {len(df)} candles for {symbol}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def get_candles(self, start_index: int = 0, limit: int = 1000) -> List[dict]:
        """Return a chunk of candles."""
        if self.history.empty:
            return []
            
        # Ensure bounds
        start = max(0, start_index)
        end = min(len(self.history), start + limit)
        
        subset = self.history.iloc[start:end]
        
        # Convert to list of dicts for JSON response
        # Using string for timestamp to ensure serialization
        result = []
        for _, row in subset.iterrows():
            result.append({
                "time": row["timestamp"].isoformat(), # Lightweight charts likes seconds or ISO
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"]
            })
        return result

    def get_replay_tick(self) -> Optional[dict]:
        """Get next candle in replay if playing."""
        if self.history.empty or not self.is_playing:
            return None
        
        if self.replay_index >= len(self.history):
            self.is_playing = False
            return None
            
        row = self.history.iloc[self.replay_index]
        self.replay_index += 1
        
        return {
            "timestamp": row["timestamp"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"]
        }

    def set_replay_index(self, index: int):
        self.replay_index = max(0, min(index, len(self.history) - 1))

    def set_replay_state(self, playing: bool, speed: float = 1.0):
        self.is_playing = playing
        self.replay_speed = speed
