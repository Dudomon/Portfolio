@echo off
echo ========================================
echo 🚀 TRADING ROBOT PPO - MT5 DIRECT
echo ========================================
echo 📡 Sistema usa MetaTrader5 package direto (sem ZMQ/EA)
echo.

cd /d "%~dp0"

echo 🔍 Verificando modelo PPO...
if not exist "Modelo PPO\Legion V1.zip" (
    echo ❌ ERRO: Modelo PPO nao encontrado!
    echo 📁 Esperado: "Modelo PPO\LegionV1.zip"
    echo.
    pause
    exit /b 1
)

echo ✅ Modelo PPO encontrado!
echo 🤖 Iniciando RobotV3 com modelo PPO...
echo.

python -u RobotV3.py 2>&1

if errorlevel 1 (
    echo.
    echo ❌ Ocorreu um erro ao executar o RobotV3
    echo 💡 Verifique:
    echo    - Modelo "Primeiro PPO trader.zip" existe em "Modelo PPO/"
    echo    - MetaTrader 5 esta aberto e conectado
    echo    - MetaTrader5 package instalado: pip install MetaTrader5
    echo    - Simbolo GOLD disponivel no Market Watch
    echo.
    pause
)