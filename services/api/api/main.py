from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from api.routes import candles, websocket, ict, volume, contracts
from api.dependencies import initialize_market_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup: Load market data
    data_path = os.getenv("MARKET_DATA_PATH", "data/es_futures")
    
    try:
        initialize_market_state(data_path)
        print(f"[OK] Market data loaded from {data_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load market data: {e}")
    
    yield
    
    print("[OK] Shutdown complete")


app = FastAPI(
    title="Market State API",
    description="FastAPI service for market data, ICT levels, and volume profiles",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(candles.router, prefix='/api')
app.include_router(websocket.router, prefix='/api')
app.include_router(ict.router, prefix='/api')
app.include_router(volume.router, prefix='/api')
app.include_router(contracts.router, prefix='/api')


@app.get("/")
async def root():
    return {
        "service": "Market State API",
        "version": "1.0.0",
        "endpoints": {
            "candles": "/api/candles",
            "websocket": "/ws/candles",
            "ict_levels": "/api/ict/levels",
            "volume_profile": "/api/volume/profile",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)