@echo off
chcp 65001 > nul 2>&1
echo ========================================
echo LEGION AI TRADER V7 - PPO V7 INTUITION
echo ========================================
echo TwoHeadV7Intuition - Backbone Unificado + Gradient Mixing
echo Action Space 8D Otimizado - SEM GATES - Alinhado com daytrader
echo 5.85M Steps - Melhor Modelo (Legion daytrade.zip)
echo Sistema usa MetaTrader5 package direto (sem ZMQ/EA)
echo.

cd /d "%~dp0"

echo Iniciando RobotV7 (verificacao de arquivos no Python)...
echo.

python -u RobotV7.py 2>&1

if errorlevel 1 (
    echo.
    echo ERRO: Falha ao executar o RobotV7
    echo Verifique:
    echo    - Modelo "Legion daytrade.zip" existe em "Modelo daytrade/"
    echo    - Enhanced Normalizer "enhanced_normalizer_final.pkl" existe
    echo    - MetaTrader 5 esta aberto e conectado
    echo    - MetaTrader5 package instalado: pip install MetaTrader5
    echo    - Simbolo GOLD disponivel no Market Watch
    echo    - Python packages: stable-baselines3[extra], torch, numpy
    echo.
    pause
)