@echo off
echo 🔐 RobotV7 com Sistema de Login
echo ================================
echo.
echo 🚀 Iniciando sistema...
echo.

cd /d "%~dp0"

REM Tentar Python 3.10+
python -c "import sys; print(f'Python {sys.version[:5]}')"
if %ERRORLEVEL% EQU 0 (
    python robotlogin.py
) else (
    echo ❌ Python não encontrado
    echo 💡 Instale Python 3.10+ e adicione ao PATH
    pause
)

echo.
echo 📴 Sistema encerrado
pause