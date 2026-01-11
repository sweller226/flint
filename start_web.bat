@echo off
SETLOCAL EnableDelayedExpansion

echo ===================================================
echo       Flint Web Launchpad (BAT)
echo ===================================================
echo.

call :CheckDependency "node" "Node.js"
if !errorlevel! neq 0 goto :Error
call :CheckDependency "python" "Python"
if !errorlevel! neq 0 goto :Error

REM Check for VENV
if not exist "services\backend\venv" (
    echo [INFO] Virtual environment not found. Creating...
    python -m venv services\backend\venv
    echo [INFO] Installing backend requirements...
    call services\backend\venv\Scripts\pip install -r services\backend\requirements.txt
)

echo [1/2] Starting Backend (FastAPI)...
start "Flint Backend" cmd /k "cd services\backend && venv\Scripts\python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"

echo [2/2] Starting Web Frontend (Next.js)...
start "Flint Web" cmd /k "cd apps\web && npm run dev"

echo.
echo ===================================================
echo       Web Stack Launched!
echo ===================================================
echo Backend API: http://localhost:8000/docs
echo Web App:     http://localhost:3000
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
