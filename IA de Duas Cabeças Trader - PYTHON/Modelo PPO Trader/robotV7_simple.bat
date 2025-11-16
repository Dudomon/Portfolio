@echo off
chcp 65001 > nul 2>&1
echo ========================================
echo LEGION AI TRADER V7 - PPO V7 INTUITION
echo ========================================
echo Iniciando RobotV7...
echo.

cd /d "%~dp0"

echo DEBUG: Pasta atual
cd

echo.
echo DEBUG: Listando arquivos modelo...
dir "Modelo daytrade"

echo.
echo Executando RobotV7.py...
python -u RobotV7.py 2>&1

echo.
echo Programa finalizado.
pause