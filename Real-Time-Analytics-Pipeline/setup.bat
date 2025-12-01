@echo off
echo.
echo 🚀 Real-Time Analytics Pipeline - Setup Script
echo ==============================================
echo.

REM Check Docker
where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Docker not found. Please install Docker first.
    exit /b 1
)

where docker-compose >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Docker Compose not found. Please install Docker Compose first.
    exit /b 1
)

echo ✅ Docker and Docker Compose found
echo.

REM Create .env file
if not exist .env (
    echo 📝 Creating .env file...
    copy .env.example .env
    echo ✅ .env file created
) else (
    echo ✅ .env file already exists
)

echo.
echo 🐳 Starting services...
docker-compose up -d

echo.
echo ⏳ Waiting for services to be healthy...
timeout /t 10 /nobreak >nul

echo.
echo 🔍 Checking service health...
docker-compose ps

echo.
echo ==============================================
echo ✅ Setup complete!
echo.
echo 📊 Dashboard: http://localhost:3000
echo 🔧 API: http://localhost:8080
echo 📈 Flink UI: http://localhost:8081
echo.
echo To generate sample events:
echo   python scripts/generate_events.py --rate 1000
echo.
echo To view logs:
echo   docker-compose logs -f
echo.
echo To stop services:
echo   docker-compose down
echo ==============================================
pause
