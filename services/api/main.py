import os
import json
import asyncio
import numpy as np
import pandas as pd
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from trading.ict_engine import ICTEngine
from trading.volume_engine import VolumeEngine
from routes import router as api_router

load_dotenv()

app = FastAPI(title="Flint Backend", version="0.1.0")
# ...

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

from ml.engine import MLEngine

manager = ConnectionManager()
ict_engine = ICTEngine()
volume_engine = VolumeEngine()
ml_engine = MLEngine()

import yfinance as yf

# ... (Previous imports remain, but we replace the mock logic)

# Data Manager
class DataManager:
    def __init__(self):
        self.history = pd.DataFrame()
        self.current_index = 0
        self.symbol = "ES=F"

    async def load_data(self, symbol="ES=F"):
        print(f"Fetching data for {symbol}...")
        try:
            # Fetch 5 days of 1m data in a separate thread to avoid blocking the event loop
            ticker = yf.Ticker(symbol)
            df = await asyncio.to_thread(ticker.history, period="5d", interval="1m")
            
            if df.empty:
                print(f"Warning: No data found via yfinance for {symbol}. Using fallback.")
                return False
            
            df.reset_index(inplace=True)
            # Ensure columns are lower case
            df.columns = [c.lower() for c in df.columns]
            # Rename datetime/date to timestamp if needed
            if "datetime" in df.columns:
                 df.rename(columns={"datetime": "timestamp"}, inplace=True)
            elif "date" in df.columns:
                 df.rename(columns={"date": "timestamp"}, inplace=True)

            self.history = df
            self.current_index = 0
            self.symbol = symbol
            print(f"Loaded {len(df)} candles for {symbol}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def get_next_candle(self):
        if self.history.empty or self.current_index >= len(self.history):
            # Loop back or handle end
            self.current_index = 0
        
        candle = self.history.iloc[self.current_index]
        self.current_index += 1
        return candle

from ml.engine import MLEngine
from gemini import GeminiService

# Service initializations
data_manager = DataManager()
manager = ConnectionManager()
ict_engine = ICTEngine()
volume_engine = VolumeEngine()
ml_engine = MLEngine()
gemini = GeminiService()

# Global DF for analysis (keeps growing window of "live" replayed data)
mock_df = pd.DataFrame() 

@app.websocket("/ws/signals")
async def websocket_endpoint(websocket: WebSocket):
    global mock_df
    await manager.connect(websocket)
    
    current_symbol = "ES=F" # Default to Future
    
    try:
        # Pre-load default data
        if data_manager.history.empty:
            success = await data_manager.load_data(current_symbol)
            if success:
                # Initialize analysis window with first 100 candles
                mock_df = data_manager.history.iloc[:100].copy()
                data_manager.current_index = 100

        async def receive_messages():
            nonlocal current_symbol
            global mock_df
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                
                if msg.get("type") == "SUBSCRIBE":
                    req_symbol = msg.get("symbol", "ES")
                    if req_symbol == "ES": req_symbol = "ES=F"
                    if req_symbol == "BTC": req_symbol = "BTC-USD"
                    
                    current_symbol = req_symbol
                    success = await data_manager.load_data(current_symbol)
                    if success:
                         mock_df = data_manager.history.iloc[:100].copy()
                         data_manager.current_index = 100
                    print(f"Subscribed to {current_symbol}")

                elif msg.get("type") == "CHAT":
                    user_q = msg.get("text", "")
                    # Prepare snapshot for Gemini
                    if not mock_df.empty:
                        struct = ict_engine.get_structure_map(mock_df)
                        snapshot = {
                            "price": float(mock_df.iloc[-1]["close"]),
                            "trend": struct.get("structure", {}).get("bias", "Neutral"),
                            "liquidity": struct.get("liquidity", {}),
                            "signals": ["Recent Bearish FVG" if len(struct.get("fvgs", [])) > 0 else "None"]
                        }
                        response = await gemini.explain_strategy(snapshot, user_q)
                        await websocket.send_text(json.dumps({
                            "type": "CHAT_RESPONSE",
                            "text": response
                        }))

        asyncio.create_task(receive_messages())

        while True:
            await asyncio.sleep(1) # 1 sec per candle replay speed
            
            # Get next candle from history
            if not data_manager.history.empty:
                new_candle = data_manager.get_next_candle()
                
                c_dict = {
                    "timestamp": new_candle["timestamp"],
                    "open": float(new_candle["open"]),
                    "high": float(new_candle["high"]),
                    "low": float(new_candle["low"]),
                    "close": float(new_candle["close"]),
                    "volume": int(new_candle["volume"])
                }
                
                mock_df = pd.concat([mock_df, pd.DataFrame([c_dict])], ignore_index=True)
                if len(mock_df) > 500:
                    mock_df = mock_df.tail(500)
                
                # Analysis
                if len(mock_df) % 50 == 0:
                     ml_engine.train_model(mock_df)
                    
                ict_levels = ict_engine.get_structure_map(mock_df)
                ml_pred = ml_engine.predict_next(mock_df)
                
                update = {
                    "type": "CANDLE",
                    "symbol": current_symbol,
                    "data": {
                        "time": int(pd.to_datetime(c_dict["timestamp"]).timestamp()),
                        "open": c_dict["open"],
                        "high": c_dict["high"],
                        "low": c_dict["low"],
                        "close": c_dict["close"]
                    },
                    "ict": ict_levels,
                    "ml": ml_pred
                }
                await manager.broadcast(json.dumps(update, default=str))

                # Real-ish Signal Logic: Check for FVG creation or Liquidity Sweeps
                fvgs = ict_levels.get("fvgs", [])
                if len(fvgs) > 0 and np.random.random() > 0.95:
                    fvg = fvgs[-1]
                    side = "LONG" if fvg["type"] == "bullish_fvg" else "SHORT"
                    signal = {
                        "type": "SIGNAL",
                        "data": {
                            "side": side,
                            "price": c_dict["close"],
                            "confidence": 0.88,
                            "reason": f"{side} FVG Created at {fvg['bottom']:.2f}-{fvg['top']:.2f}"
                        }
                    }
                    await manager.broadcast(json.dumps(signal))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WS Error: {e}")

@app.get("/api/candles")
async def get_candles(symbol: str = "ES", timeframe: str = "1m", date: str = None):
    print(f"Candle request: symbol={symbol}, timeframe={timeframe}, date={date}")
    # If date is Jan 3, 2017, use mock data as yfinance doesn't provide 1m data that far back
    if date == "2017-01-03":
        print("Generating mock historical data for Jan 3, 2017...")
        base_price = 2250.0 if symbol == "ES" else 1000.0
        start_ts = int(pd.Timestamp("2017-01-03 09:30:00").timestamp())
        mock_candles = []
        for i in range(200):
            ts = start_ts + (i * 60)
            new_candle = {
                "time": ts,
                "open": base_price + np.random.normal(0, 2),
                "high": base_price + 5 + np.random.normal(0, 2),
                "low": base_price - 5 + np.random.normal(0, 2),
                "close": base_price + np.random.normal(0, 2),
            }
            mock_candles.append(new_candle)
            base_price = new_candle["close"]
        return mock_candles

    # Map symbols
    yf_symbol = symbol
    if symbol == "ES": yf_symbol = "ES=F"
    if symbol == "BTC": yf_symbol = "BTC-USD"
    if symbol == "SPX": yf_symbol = "^GSPC"
    
    try:
        ticker = yf.Ticker(yf_symbol)
        # 1m data is limited to last 7 days. For historical, we'd need a larger interval if not mocked
        period = "5d"
        if timeframe not in ["1m", "2m", "5m"]:
            period = "1mo"
            
        df = await asyncio.to_thread(ticker.history, period=period, interval=timeframe)
        
        if df.empty:
            return []
            
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        
        if "datetime" in df.columns:
            df.rename(columns={"datetime": "timestamp"}, inplace=True)
        elif "date" in df.columns:
            df.rename(columns={"date": "timestamp"}, inplace=True)
            
        candles = []
        for _, row in df.iterrows():
            candles.append({
                "time": int(pd.to_datetime(row["timestamp"]).timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"])
            })
        return candles
    except Exception as e:
        print(f"Error fetching candles: {e}")
        return []

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
