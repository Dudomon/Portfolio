@echo off
echo.
echo =============================================
echo   🤖 ROBOTV7 LEGION - GUI ORIGINAL COMPLETA
echo =============================================
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Install Python 3.10+ first
    pause
    exit /b 1
)

REM Executar GUI ORIGINAL
echo 🚀 Starting RobotV7 with ORIGINAL GUI...
python robotlogin.py

if errorlevel 1 (
    echo.
    echo ❌ Error running RobotV7
    pause
)

exit /b 0