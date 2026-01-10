import os
import json
import asyncio
import pandas as pd
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.ict_engine import ICTEngine
from app.volume_engine import VolumeEngine
from app.api_routes import router as api_router

load_dotenv()

app = FastAPI(title="Flint Backend", version="0.1.0")

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

from app.ml_engine import MLEngine

manager = ConnectionManager()
ict_engine = ICTEngine()
volume_engine = VolumeEngine()
ml_engine = MLEngine()

# Mock Data Generator
def generate_mock_candle(last_price):
    import random
    open_p = last_price
    change = random.uniform(-2, 2)
    close_p = open_p + change
    high_p = max(open_p, close_p) + random.uniform(0, 1)
    low_p = min(open_p, close_p) - random.uniform(0, 1)
    vol = random.randint(100, 5000)
    return {
        "timestamp": pd.Timestamp.now(),
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "close": close_p,
        "volume": vol
    }

mock_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
# Init mock df
start_price = 4800.0
for _ in range(100):
   c = generate_mock_candle(start_price)
   mock_df = pd.concat([mock_df, pd.DataFrame([c])], ignore_index=True)
   start_price = c["close"]
   
# Initial Train
if len(mock_df) > 50:
    ml_engine.train_model(mock_df)

@app.websocket("/ws/signals")
async def websocket_endpoint(websocket: WebSocket):
    global mock_df
    await manager.connect(websocket)
    
    current_symbol = "ES"
    current_timeframe = "1m"
    
    try:
        # Create a task to listen for incoming messages
        async def receive_messages():
            nonlocal current_symbol, current_timeframe
            global mock_df
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                if msg.get("type") == "SUBSCRIBE":
                    current_symbol = msg.get("symbol", "ES")
                    current_timeframe = msg.get("timeframe", "1m")
                    print(f"Subscribed to {current_symbol} {current_timeframe}")
                    
                    # Reset Mock Data for new symbol
                    mock_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
                    start_price = 4800.0 if current_symbol == "ES" else (40000.0 if "BTC" in current_symbol else 150.0)
                    for _ in range(100):
                        c = generate_mock_candle(start_price)
                        mock_df = pd.concat([mock_df, pd.DataFrame([c])], ignore_index=True)
                        start_price = c["close"]

        # Run listener in background
        asyncio.create_task(receive_messages())

        while True:
            await asyncio.sleep(1) # Frequency
            
            # Generate new candle
            last_close = mock_df.iloc[-1]["close"]
            new_candle = generate_mock_candle(last_close)
            mock_df = pd.concat([mock_df, pd.DataFrame([new_candle])], ignore_index=True)
            if len(mock_df) > 500:
                mock_df = mock_df.tail(500)
            
            # Re-train occasionally
            if len(mock_df) % 50 == 0:
                 ml_engine.train_model(mock_df)
                
            # Compute ICT
            ict_levels = ict_engine.get_structure_map(mock_df)
            vol_profile = volume_engine.compute_volume_profile(mock_df)
            ml_pred = ml_engine.predict_next(mock_df)
            
            # Broadcast minimal update for chart
            update = {
                "type": "CANDLE",
                "symbol": current_symbol,
                "data": {
                    "time": int(new_candle["timestamp"].timestamp()),
                    "open": new_candle["open"],
                    "high": new_candle["high"],
                    "low": new_candle["low"],
                    "close": new_candle["close"]
                },
                "ict": ict_levels,
                "ml": ml_pred
            }
            await manager.broadcast(json.dumps(update, default=str))
            
            # Random Signal Logic
            if np.random.random() > 0.95: # 5% chance
                 signal = {
                    "type": "SIGNAL",
                    "data": {
                        "side": "LONG" if np.random.random() > 0.5 else "SHORT",
                        "price": new_candle["close"],
                        "confidence": 0.85,
                        "reason": f"ICT Liquidity Sweep ({current_symbol})"
                    }
                 }
                 await manager.broadcast(json.dumps(signal))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WS Error: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
