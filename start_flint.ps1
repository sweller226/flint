$ErrorActionPreference = "Stop"

Write-Host "Starting Flint Terminal Stack..." -ForegroundColor Cyan

# Check if Backend Virtual Environment exists
if (-not (Test-Path "services\.venv")) {
    Write-Host "Virtual environment not found. Creating..." -ForegroundColor Yellow
    python -m venv services\.venv
    Write-Host "Installing backend requirements..." -ForegroundColor Yellow
    .\services\.venv\Scripts\pip install -r services\requirements.txt
}

Write-Host "Launching Backend (FastAPI)..." -ForegroundColor Green
Start-Process -FilePath "cmd.exe" -ArgumentList "/k set MARKET_DATA_PATH=./api/data/es_futures && cd ./services && .venv\Scripts\python -m uvicorn api.api.main:app --host 0.0.0.0 --port 8000 --reload"

Write-Host "Waiting for backend..." -ForegroundColor Cyan
Start-Sleep -Seconds 3  # Wait for backend to start

Write-Host "Launching Marketing Website (Next.js)..." -ForegroundColor Green
Start-Process -FilePath "cmd.exe" -ArgumentList "/k cd ./apps/web && npm run dev"

Write-Host "Launching Desktop App (Electron)..." -ForegroundColor Green
Start-Process -FilePath "cmd.exe" -ArgumentList "/k cd ./apps/desktop && npm run electron"

Write-Host "All services launched!" -ForegroundColor Cyan
