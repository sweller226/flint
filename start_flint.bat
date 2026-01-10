@echo off
SETLOCAL EnableDelayedExpansion

echo ===================================================
echo       Flint Trading Terminal - Launchpad (BAT)
echo ===================================================
echo.
echo [INFO] Cleaning up previous instances...
call .\kill_flint.bat >nul 2>&1
echo [INFO] Clean start.
echo.


call :CheckDependency "node" "Node.js"
if !errorlevel! neq 0 goto :Error
call :CheckDependency "python" "Python"
if !errorlevel! neq 0 goto :Error

REM Check for VENV (Now in services/api for the core runner)
if not exist "services\api\venv" (
    echo [INFO] Virtual environment not found. Creating...
    python -m venv services\api\venv
    echo [INFO] Installing core requirements...
    call services\api\venv\Scripts\pip install -r services\api\requirements.txt
)

echo [1/3] Starting Backend (FastAPI)...
REM Set PYTHONPATH so api/main.py can find trading, ml, and solana modules
start "Flint Backend" cmd /k "set PYTHONPATH=%CD%\services\api && cd services\api && venv\Scripts\python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 > ..\..\backend_debug.log 2>&1"

echo [2/3] Starting Web Frontend (Next.js)...
start "Flint Web" cmd /k "cd apps\web && npm run dev"

echo [3/3] Starting Electron Desktop App...
echo Waiting 5 seconds for backend to warm up...
timeout /t 5 /nobreak >nul
start "Flint Desktop" cmd /k "npm run electron"

echo.
echo ===================================================
echo       All Systems Launched!
echo ===================================================
echo Backend API: http://localhost:8000/docs
echo Web App:     http://localhost:3000
echo Desktop App: Launching...
echo.
goto :EOF

:CheckDependency
where %~1 >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] %~2 is not installed or not in PATH.
    exit /b 1
)
exit /b 0

:Error
echo.
echo [FATAL] Setup failed. Please fix the errors above.
pause
exit /b 1
