# Flint Terminal - Run Instructions

This guide explains how to run tThe easiest way to run Flint is using the provided launch scripts.

### Option 1: Full Stack (Desktop App)
Use this to run the **Electron Desktop App**, along with the backend and marketing site.

*   **Windows**: `.\start_flint.bat`
*   **Mac**: `./start_flint.sh`

### Option 2: Web Only (Browser)
Use this if you only want to work on the **Marketing Website** or test the terminal in a browser (at `http://localhost:3000`).

*   **Windows**: `.\start_web.bat`
*   **Mac**: `./start_web.sh`

This will open separate terminal windows for each service and launch the Electron app.

---

## Manual Startup

If you prefer to run services individually, ensure you are in the root `flint` directory.

### 1. Backend Service (FastAPI)
The backend manages data feeds, simulated trading logic, and AI endpoints.

```bash
cd services/backend
# Create venv if not exists
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
# Run Server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Marketing Website (Next.js)
Runs on `http://localhost:3000`.

```bash
# From root
npm run dev:web
```

### 3. Desktop Terminal (Electron)
The main trading application.

```bash
# From root
npm run electron
```
*Note: This will launch the Electron window and a local dev server on port 3001.*

## Troubleshooting

- **Blank Electron Window**: Ensure `react-icons` and other dependencies are installed. Run `npm install` in the root.
- **Port Conflicts**: Ensure ports 8000, 3000, and 3001 are free.
- **Backend Errors**: Check the python terminal for missing packages. Ensure you activated the virtual environment.

## Dependencies

- **Node.js**: v18+
- **Python**: 3.10+
- **Rust**: (Optional, for some Solana dependencies)
