# Flint - ES Futures HFT Terminal

Hackathon project scaffolding for high-frequency day trading terminal using Electron, React, FastAPI, and ICT Concepts.

## Architecture

- **Desktop (`apps/desktop`)**: Electron + Vite + React + Tailwind. The main trading terminal.
- **Web (`apps/web`)**: Next.js marketing site.
- **Backend (`services/backend`)**: FastAPI python service for ICT logic, Polygon.io data, and Gemini integration.
- **Shared (`packages/*`)**: Shared UI and Config.


## Getting Started

### Quick Start (Windows)
We provide One-Click Batch scripts for easy launching.

**Option 1: Full Stack (Recommended)**
Launches Backend + Marketing Site + Electron Desktop App.
```bash
.\start_flint.bat
```

**Option 2: Web Only**
Launches Backend + Marketing Site (access via browser).
```bash
.\start_web.bat
```

### Manual Mode
If you prefer to run services individually:

1. **Backend**: `cd services/backend && venv/Scripts/activate && python -m uvicorn main:app --reload` (Port 8000)
2. **Desktop**: `npm run electron` (From root)
3. **Web**: `npm run dev:web` (From root, Port 3000)

See `guides/RUN_INSTRUCTIONS.md` for full details.

## Features (Skeleton)

- **ICT Engine**: Stubs for FVG, Liquidity Pools in `services/backend/app/ict_engine.py`.
- **Gemini Strategy**: Prompt templates in `services/backend/app/gemini.py`.
- **Solana Logging**: Stubs in `services/backend/app/api_routes.py`.

## Hackathon Tips

- Focus on the `apps/desktop/src/App.tsx` for visual wow factor.
- Fill in the logic `services/backend/app/ict_engine.py` with real pandas-ta logic.
- Use the Gemini prompt templates to tune the strategy explanation.
