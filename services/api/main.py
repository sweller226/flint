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
from trading.paper_engine import PaperTradingEngine
from data_manager import DataManager
from inference_service import InferenceService
import routes

# Load Env
load_dotenv()

app = FastAPI(title="Flint Backend", version="0.1.0")

# Service Initialization
data_manager = DataManager()
paper_engine = PaperTradingEngine()

inference_service = InferenceService()
from trading.bot_manager import BotManager
bot_manager = BotManager(paper_engine, inference_service)
ict_engine = ICTEngine()
volume_engine = VolumeEngine()
ml_engine = MLEngine() # Keep old random forest for now as backup

# Inject services into routes
routes.set_services(paper_engine, data_manager, inference_service, bot_manager)

    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router, prefix="/api")

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

import yfinance as yf

# ... (Previous imports remain, but we replace the mock logic)

# DataManager logic moved to services/api/data_manager.py
# Removed old mock DataManager class


from ml.engine import MLEngine

# Service initializations
# WebSocket broadcast loop (Background Task)
@app.on_event("startup")
async def startup_event():
    # Load default data
    await data_manager.load_data("ES=F")
    
    # Start Ticker Loop
    asyncio.create_task(ticker_loop())

async def ticker_loop():
    """Simulates the heartbeat of the application."""
    manager = ConnectionManager() # Needs to be global or attached to app
    # Quick fix: attach manager to app state or make global
    global connection_manager
    connection_manager = manager
    
    while True:
        try:
            await asyncio.sleep(1.0 / data_manager.replay_speed if data_manager.is_playing else 1.0)
            
            # 1. Replay Tick
            if data_manager.is_playing:
                tick = data_manager.get_replay_tick()
                if tick:
                    # Update Paper Engine
                    # Use close price for simplicity
                    paper_engine.process_tick(data_manager.symbol, tick["close"])
                    
                    # Run Inference?
                    # signal = inference_service.analyze_tick(...)
                    
                    # Broadcast
                    msg = {
                        "type": "tick",
                        "candle": {k: str(v) if isinstance(v, (pd.Timestamp, datetime)) else v for k,v in tick.items()},
                        "portfolio": asdict(paper_engine.get_state())
                    }
                    # We need access to the WebSocket manager instance to broadcast
                    # Ideally we refactor ConnectionManager to be a singleton
                    # For now, let's just print
                    # print(f"Tick: {tick['close']}") 
                    # We can't broadcast easily without global instance.
                    await connection_manager.broadcast(json.dumps(msg, default=str))

                    # 3. Auto-Trading processed by BotManager
                    # Get signal (re-analyze or assume tick analysis happened?)
                    # For simplicity, calculate signal here or inside bot
                    signal_data = inference_service.analyze_tick(data_manager.history)
                    
                    trade_result = await bot_manager.execute_strategy(tick, signal_data)
                    if trade_result:
                         await connection_manager.broadcast(json.dumps(trade_result, default=str))

        except Exception as e:
            print(f"Ticker Error: {e}")
            await asyncio.sleep(1)

connection_manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle client messages if any
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
