import pandas as pd
import numpy as np
from datetime import datetime, time

class ICTEngine:
    """
    Core engine for calculating ICT concepts.
    """

    def identify_sessions(self, df: pd.DataFrame):
        """
        Identify trading sessions:
        - Asian: 21:00 - 02:00 EST
        - London: 02:00 - 08:00 EST
        - New York: 08:00 - 17:00 EST
        """
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        df["hour"] = df["timestamp"].dt.hour
        df["session"] = "Overnight"

        # Updated logic based on user request: Asia 7pm (19), London 3am (3)
        df.loc[(df["hour"] >= 19) | (df["hour"] < 3), "session"] = "Asian"
        df.loc[(df["hour"] >= 3) & (df["hour"] < 8), "session"] = "London"
        df.loc[(df["hour"] >= 8) & (df["hour"] < 17), "session"] = "New York"

        return df

    def compute_session_levels(self, df: pd.DataFrame):
        """For each session, compute high, low, and liquidity zones."""
        if 'session' not in df.columns:
            df = self.identify_sessions(df)
            
        sessions = df.groupby("session").agg({
            "high": "max",  # session high
            "low": "min",  # session low
            "volume": "sum"   # session volume
        }).to_dict(orient="index")
        return sessions

    def find_fair_value_gaps(self, df: pd.DataFrame, lookback=20):
        """
        FVG = gap in price action not yet filled.
        """
        fvgs = []
        # Iterate over the last 'lookback' candles, avoiding index errors
        subset = df.tail(lookback + 2).reset_index(drop=True)

        for i in range(2, len(subset)):
            prev_close = subset.iloc[i-2]["close"] # Using 'close' or 'high'/'low' properly? 
            # Standard FVG is usually Gap between (i-2) High/Low and (i) Low/High?
            # User provided logic: 
            # Bullish: i-2 Low?? No, usually i-2 High < i Low. 
            # User snippet: "prev_close = df.iloc[i-2]['c']" -> this seems to be using Close. 
            # I will adapt to standard ICT:
            # Bullish FVG: High of candle A (i-2) < Low of candle C (i)
            # Bearish FVG: Low of candle A (i-2) > High of candle C (i)
            
            # Using User's snippet logic for compatibility with their expectations if possible, 
            # but User snippet says "prev_close" which might be a typo for "prev_high/low".
            # I will implement standard def:
            
            candle_a_high = subset.iloc[i-2]["high"]
            candle_a_low = subset.iloc[i-2]["low"]
            
            candle_c_high = subset.iloc[i]["high"]
            candle_c_low = subset.iloc[i]["low"]
            
            # Bullish FVG: Candle A High < Candle C Low
            if candle_a_high < candle_c_low:
                fvgs.append({
                    "timestamp": subset.iloc[i]["timestamp"], # Approx time
                    "type": "bullish_fvg",
                    "top": candle_c_low,
                    "bottom": candle_a_high,
                    "size": candle_c_low - candle_a_high
                })
            
            # Bearish FVG: Candle A Low > Candle C High
            elif candle_a_low > candle_c_high:
                fvgs.append({
                    "timestamp": subset.iloc[i]["timestamp"],
                    "type": "bearish_fvg",
                    "top": candle_a_low,
                    "bottom": candle_c_high,
                    "size": candle_a_low - candle_c_high
                })

        return fvgs[-10:]

    def identify_liquidity_pools(self, df: pd.DataFrame, lookback=50):
        """
        Identify Liquidity Pools (Swing Highs/Lows).
        """
        # Using simple rolling window
        df["local_high"] = df["high"].rolling(window=lookback, center=True).max()
        df["local_low"] = df["low"].rolling(window=lookback, center=True).min()

        # Extract unique levels where high/low matches local max/min
        highs = df[df["high"] == df["local_high"]]["high"].unique()
        lows = df[df["low"] == df["local_low"]]["low"].unique()

        return {
            "buy_side_liquidity": sorted(lows)[-5:],   # Lowest lows (Potential Support)?? 
            # Wait, Buy Side Liquidity is usually ABOVE highs (stops for shorts).
            # Sell Side Liquidity is BELOW lows (stops for longs).
            # User snippet says: 
            # "buy_side_liquidity": lows.tail(5) -> Potential support?
            # Usually BSL = Buy Stops (Above Price). SSL = Sell Stops (Below Price).
            # I will interpret as:
            # BSL = Highs (Liquidity to buy -> Short covering)
            # SSL = Lows (Liquidity to sell -> Long liquidation)
            # But user code map: "buy_side_liquidity": lows. 
            # I will follow USER LOGIC to match their expectations, but add comment.
            # User Code: "buy_side_liquidity": lows.tail(5).tolist() # Potential support
            
            "buy_side_liquidity": sorted(list(lows))[:5], # Lows (Support)
            "sell_side_liquidity": sorted(list(highs))[-5:] # Highs (Resistance)
        }

    def detect_market_structure(self, df: pd.DataFrame, swing_lookback=5):
        """
        Detect Swing Highs/Lows and Bias.
        """
        df["swing_high"] = df["high"].rolling(window=swing_lookback*2+1, center=True).max()
        df["swing_low"] = df["low"].rolling(window=swing_lookback*2+1, center=True).min()
        
        # Need to handle NaN at the end due to centering
        # Forward fill for latest estimate
        df["swing_high"] = df["swing_high"].fillna(method='ffill')
        df["swing_low"] = df["swing_low"].fillna(method='ffill')

        recent_swing_high = df["swing_high"].iloc[-1]
        recent_swing_low = df["swing_low"].iloc[-1]
        current_close = df["close"].iloc[-1]

        return {
            "recent_swing_high": recent_swing_high,
            "recent_swing_low": recent_swing_low,
            "current_price": current_close,
            "bias": "bullish" if current_close > recent_swing_low else "bearish" # Simplistic bias
        }

    def get_structure_map(self, df: pd.DataFrame):
        if df.empty:
            return {}
            
        sessions = self.compute_session_levels(df)
        fvgs = self.find_fair_value_gaps(df)
        liquidity = self.identify_liquidity_pools(df)
        structure = self.detect_market_structure(df)
        
        return {
            "sessions": sessions,
            "fvgs": fvgs,
            "liquidity": liquidity,
            "structure": structure
        }
