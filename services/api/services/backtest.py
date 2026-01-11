import asyncio
from typing import Optional, List
from datetime import datetime
import pandas as pd
from services.market_state import MarketState, Candle

class BacktestSession:
    def __init__(self, market_state: MarketState):
        self.market_state = market_state
        self.simulated_time: Optional[pd.Timestamp] = None
        self.speed_ratio: float = 1.0 # 1 real sec = X sim secs
        self.is_playing: bool = False
        self.task: Optional[asyncio.Task] = None
        self.last_update_real_time: float = 0
        self.callbacks = []

    def set_start_time(self, timestamp: str | pd.Timestamp):
        """Set the simulation time to a specific point in history."""
        if isinstance(timestamp, str):
            self.simulated_time = pd.to_datetime(timestamp, utc=True)
        else:
            self.simulated_time = timestamp
        print(f"[Backtest] Time set to {self.simulated_time}")

    def set_speed(self, ratio: float):
        """Set playback speed (1.0 = real-time, 60.0 = 1min/sec, etc)"""
        self.speed_ratio = ratio
        print(f"[Backtest] Speed set to {self.speed_ratio}x")

    async def play(self):
        if self.is_playing:
            return
        
        if self.simulated_time is None:
            # Default to start of data if not set
            self.simulated_time = self.market_state.df['timestamp'].min()

        self.is_playing = True
        print("[Backtest] Play started")
        
        # Start the loop
        while self.is_playing:
            start_loop_time = asyncio.get_event_loop().time()
            
            # Find candles that "happened" in this simulation step
            # For simplicity in this loop, we just grab the next candle after current sim time
            # But to support "time based" playback efficiently, request the next candle:
            
            # Look ahead:
            # How much sim time passes in this tick?
            # Let's target 10 ticks per second (100ms sleep)
            tick_duration_real = 0.1
            sim_delta_seconds = tick_duration_real * self.speed_ratio
            
            next_sim_time = self.simulated_time + pd.Timedelta(seconds=sim_delta_seconds)
            
            # Get candles between current sim time and next sim time
            # We use strictly greater than current, less equal next
            window = self.market_state.load_window_by_time(self.simulated_time, next_sim_time)
            
            # Filter to avoid double emitting the exact start time if it was already sent?
            # load_window_by_time is INCLUSIVE.
            # We should keep track of the last emitted candle time to avoid dupes?
            # Or just filter > self.simulated_time
            mask = (window['timestamp'] > self.simulated_time) & (window['timestamp'] <= next_sim_time)
            new_candles_df = window[mask]
            
            self.simulated_time = next_sim_time
            
            # Emit candles
            if not new_candles_df.empty:
                for _, row in new_candles_df.iterrows():
                    c = Candle(
                        timestamp=row['timestamp'], # Keep as Timestamp object
                        symbol=row['symbol'],
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume']
                    )
                    await self._notify(c)
            
            # Check if we hit end of data
            if self.simulated_time >= self.market_state.df['timestamp'].max():
                print("[Backtest] End of data reached")
                self.is_playing = False
                break

            # Sleep to maintain loop rate
            elapsed = asyncio.get_event_loop().time() - start_loop_time
            sleep_time = max(0, tick_duration_real - elapsed)
            await asyncio.sleep(sleep_time)

    def pause(self):
        self.is_playing = False
        print("[Backtest] Paused")

    async def step(self):
        """Advance by one candle (or minimal time step)"""
        self.pause() # Ensure paused
        
        if self.simulated_time is None:
             self.simulated_time = self.market_state.df['timestamp'].min()
             
        # Find the very next candle after current time
        # We can scan the dataframe
        # Optimization: use searchsorted or just boolean mask with limit 1
        mask = self.market_state.df['timestamp'] > self.simulated_time
        next_rows = self.market_state.df[mask]
        
        if next_rows.empty:
            return # End of data

        row = next_rows.iloc[0]
        self.simulated_time = row['timestamp']
        
        c = Candle(
            timestamp=row['timestamp'],
            symbol=row['symbol'],
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume']
        )
        await self._notify(c)

    def add_callback(self, cb):
        self.callbacks.append(cb)

    async def _notify(self, candle: Candle):
        """Notify all listeners of a new candle"""
        for cb in self.callbacks:
            await cb(candle)
