$ErrorActionPreference = "Stop"

Write-Host "Starting Flint Terminal Stack..." -ForegroundColor Cyan

# Check if Backend Virtual Environment exists
if (-not (Test-Path "services\backend\venv")) {
    Write-Host "Virtual environment not found. Creating..." -ForegroundColor Yellow
    python -m venv services\backend\venv
    Write-Host "Installing backend requirements..." -ForegroundColor Yellow
    .\services\backend\venv\Scripts\pip install -r services\backend\requirements.txt
}

Write-Host "Launching Backend (FastAPI)..." -ForegroundColor Green
Start-Process -FilePath "cmd.exe" -ArgumentList "/k cd services\backend && venv\Scripts\python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"

Write-Host "Waiting for backend..." -ForegroundColor Cyan
Start-Sleep -Seconds 3

Write-Host "Launching Marketing Website (Next.js)..." -ForegroundColor Green
Start-Process -FilePath "cmd.exe" -ArgumentList "/k npm run dev -w apps/web"

Write-Host "Launching Desktop App (Electron)..." -ForegroundColor Green
Start-Process -FilePath "cmd.exe" -ArgumentList "/k npm run electron"

Write-Host "All services launched!" -ForegroundColor Cyan
