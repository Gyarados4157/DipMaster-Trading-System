@echo off
REM DipMaster Trading System - Frontend Development Startup Script (Windows)
REM This script sets up and starts the frontend development environment

setlocal enabledelayedexpansion

echo 🚀 DipMaster Trading System - Frontend Startup
echo ==============================================
echo.

REM Check if Node.js is installed
echo [INFO] Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed. Please install Node.js 18+ from https://nodejs.org/
    pause
    exit /b 1
)

for /f "tokens=1 delims=." %%a in ('node --version') do (
    set NODE_MAJOR=%%a
    set NODE_MAJOR=!NODE_MAJOR:v=!
)

if !NODE_MAJOR! lss 18 (
    echo [ERROR] Node.js version 18+ is required. Current version: 
    node --version
    pause
    exit /b 1
)

echo [SUCCESS] Node.js detected
node --version

REM Check if npm is installed
echo [INFO] Checking npm installation...
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] npm is not installed. Please install npm
    pause
    exit /b 1
)

echo [SUCCESS] npm detected
npm --version

REM Setup environment file
echo [INFO] Setting up environment configuration...
if not exist ".env.local" (
    if exist ".env.example" (
        copy ".env.example" ".env.local" >nul
        echo [SUCCESS] Created .env.local from .env.example
        echo [WARNING] Please review and update .env.local with your configuration
    ) else (
        echo [WARNING] .env.example not found, creating basic .env.local
        (
            echo # DipMaster Frontend Configuration
            echo BACKEND_URL=http://localhost:8000
            echo WS_URL=ws://localhost:8000
            echo JWT_SECRET=dipmaster-secret-key
            echo NEXT_PUBLIC_ENABLE_DEMO_MODE=true
        ) > .env.local
        echo [SUCCESS] Created basic .env.local
    )
) else (
    echo [SUCCESS] .env.local already exists
)

REM Install dependencies
echo [INFO] Installing dependencies...
if not exist "node_modules" (
    npm install
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
    echo [SUCCESS] Dependencies installed successfully
) else (
    echo [INFO] Dependencies already installed, checking for updates...
    npm update
    echo [SUCCESS] Dependencies updated
)

REM Run type checking
echo [INFO] Running TypeScript type checking...
npm run type-check
if %errorlevel% neq 0 (
    echo [WARNING] Type checking found issues, but continuing...
) else (
    echo [SUCCESS] Type checking passed
)

REM Run linting
echo [INFO] Running ESLint...
npm run lint
if %errorlevel% neq 0 (
    echo [WARNING] Linting found issues, but continuing...
) else (
    echo [SUCCESS] Linting passed
)

REM Check backend connectivity
echo [INFO] Checking backend connectivity...
curl -s -f "http://localhost:8000/health" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Backend is not accessible at http://localhost:8000
    echo [WARNING] Make sure the DipMaster backend is running
    echo [WARNING] Frontend will start in demo mode
) else (
    echo [SUCCESS] Backend is accessible at http://localhost:8000
)

echo.
echo [SUCCESS] Setup completed successfully!
echo.

REM Start development server
echo [INFO] Starting development server...
echo [SUCCESS] Frontend will be available at http://localhost:3000
echo [INFO] Press Ctrl+C to stop the server
echo.
echo 🎯 Demo Credentials:
echo    Username: admin
echo    Password: dipmaster123
echo.
echo 📊 Features Available:
echo    • Real-time dashboard
echo    • Position monitoring
echo    • Trading interface
echo    • Risk analysis
echo    • Market data
echo.

npm run dev

pause