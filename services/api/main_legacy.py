import os
import json
import asyncio
import numpy as np
import pandas as pd
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routes import router

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

app.include_router(router)

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

ict_engine = ICTEngine()
volume_engine = VolumeEngine()
ml_engine = MLEngine()

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

# Service initializations
data_manager = DataManager()
ict_engine = ICTEngine()
volume_engine = VolumeEngine()
ml_engine = MLEngine()

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
