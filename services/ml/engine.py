import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MLEngine:
    """
    Machine Learning Engine using Random Forest for Time Series Prediction.
    
    Goal: Predict next candle direction (Up/Down) based on:
    - Lagged returns
    - RSI / Volume Delta (features)
    """

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def prepare_features(self, df: pd.DataFrame):
        """
        Feature Engineering for Time Series.
        """
        data = df.copy()
        
        # Ensure numeric types
        data['close'] = data['close'].astype(float)
        data['volume'] = data['volume'].astype(float)
        
        # 1. Log Returns
        data['log_ret'] = np.log(data['close'] / data['close'].shift(1))
        
        # 2. Lagged Features (Window Size 5)
        for lag in range(1, 6):
            data[f'ret_lag_{lag}'] = data['log_ret'].shift(lag)
            data[f'vol_lag_{lag}'] = np.log(data['volume'] / data['volume'].shift(1)).shift(lag)

        # 3. Simple Volatility
        data['volatility'] = data['log_ret'].rolling(window=10).std()

        # Target: 1 if Next Close > Current Close (Shifted back to align with features)
        data['target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)
        
        data.dropna(inplace=True)
        return data

    def train_model(self, df: pd.DataFrame):
        """
        Train the Random Forest model.
        """
        processed = self.prepare_features(df)
        
        feature_cols = [c for c in processed.columns if 'lag' in c or 'volatility' in c]
        X = processed[feature_cols]
        y = processed['target']
        
        # Time Series Split (No random shuffle)
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        return {"accuracy": acc, "features": feature_cols}

    def predict_next(self, df: pd.DataFrame):
        """
        Predict direction for the live candle.
        """
        if not self.is_trained:
            return None
            
        processed = self.prepare_features(df)
        if processed.empty:
            return None

        # Take the last row features
        feature_cols = [c for c in processed.columns if 'lag' in c or 'volatility' in c]
        latest_features = processed[feature_cols].iloc[[-1]] 
        
        prediction = self.model.predict(latest_features)[0]
        prob = self.model.predict_proba(latest_features)[0][1] # Probability of Class 1 (Up)
        
        return {
            "prediction": "UP" if prediction == 1 else "DOWN",
            "probability": prob
        }
