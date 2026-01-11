import pandas as pd
import numpy as np

class VolumeEngine:
    """
    Engine for Volume Profile and Order Flow analytics.
    """

    def compute_volume_profile(self, df: pd.DataFrame, bin_size=1.0, lookback_minutes=60):
        """
        Create volume profile: bucket volume into price bins.
        """
        # Assume df has 'timestamp', 'high', 'low', 'volume'
        # Filter last N minutes
        cutoff_time = df["timestamp"].iloc[-1] - pd.Timedelta(minutes=lookback_minutes)
        recent = df[df["timestamp"] > cutoff_time]
        
        if recent.empty:
            return {}

        min_price = recent["low"].min()
        max_price = recent["high"].max()
        
        # Create bins
        bins = np.arange(start=min_price, stop=max_price + bin_size, step=bin_size)
        
        volume_profile = {}
        
        # Naive distribution: Assign total candle volume to overlapping bins?
        # Better: Distribute volume evenly across price range of the candle
        # User snippet uses simple assignment: "vol_at_level = sum volume where low <= bin < high"
        # This assumes the candle is fully inside the bin or something? 
        # User Logic: "where recent["l"] <= bin_price) & (recent["h"] > bin_price)"
        # This counts volume TWICE if bin size is smaller than candle range? 
        # Actually it counts volume for ALL bins that intersect the candle range.
        # This is a bit "heavy" but acceptable for hackathon approximation. 
        
        for bin_price in bins:
             mask = (recent["low"] <= bin_price) & (recent["high"] > bin_price)
             vol = recent.loc[mask, "volume"].sum()
             if vol > 0:
                 volume_profile[float(bin_price)] = float(vol)

        # POC
        if volume_profile:
            poc = max(volume_profile, key=volume_profile.get)
        else:
            poc = 0

        return {
            "profile": volume_profile,
            "poc": poc
        }

    def compute_delta_volume(self, df: pd.DataFrame):
        """
        Delta = up_volume - down_volume approximation.
        """
        df["delta"] = df.apply(
            lambda row: row["volume"] if row["close"] > row["open"] else -row["volume"],
            axis=1
        )
        
        # Return last few deltas
        return df["delta"].tail(10).tolist()
