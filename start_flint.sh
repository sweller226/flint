#!/bin/bash
set -e

echo "==================================================="
echo "       Flint Trading Terminal - Launchpad (Mac/Linux)"
echo "==================================================="
echo ""

# Function to check dependency
check_dependency() {
    if ! command -v "$1" &> /dev/null; then
        echo "[ERROR] $2 is not installed or not in PATH."
        exit 1
    fi
}

check_dependency "node" "Node.js"
check_dependency "python3" "Python 3"

# Check for VENV
if [ ! -d "services/backend/venv" ]; then
    echo "[INFO] Virtual environment not found. Creating..."
    python3 -m venv services/backend/venv
    echo "[INFO] Installing backend requirements..."
    source services/backend/venv/bin/activate
    pip install -r services/backend/requirements.txt
    deactivate
fi

echo "[1/3] Starting Backend (FastAPI)..."
# Run in background
(cd services/backend && source venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 8000) &

echo "[2/3] Starting Web Frontend (Next.js)..."
(cd apps/web && npm run dev) &

echo "[3/3] Starting Electron Desktop App..."
echo "Waiting 5 seconds for backend to warm up..."
sleep 5
npm run electron &

echo ""
echo "==================================================="
echo "       All Systems Launched!"
echo "==================================================="
echo "Backend API: http://localhost:8000/docs"
echo "Web App:     http://localhost:3000"
echo "Desktop App: Launching..."
echo ""
echo "Press Ctrl+C to stop all services."

# Wait for all background processes
wait
