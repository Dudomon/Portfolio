@echo off
chcp 65001 > nul
title 🎮 Legion AI Trader V7 - Model Selector

echo.
echo ================================================================
echo 🎮 LEGION AI TRADER V7 - MODEL SELECTOR
echo ================================================================
echo 💰 Portfolio Virtual: $500 inicial
echo 🎯 Seleção de Modelo: Dropdown interface
echo 📊 Monitoramento: Tempo real
echo ================================================================
echo.

REM Verificar se Python está disponível
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python não encontrado!
    echo    Instale Python ou adicione ao PATH
    pause
    exit /b 1
)

REM Verificar se o arquivo existe
if not exist "RobotV7_ModelSelector.py" (
    echo ❌ Arquivo RobotV7_ModelSelector.py não encontrado!
    echo    Certifique-se de estar na pasta correta
    pause
    exit /b 1
)

echo 🚀 Iniciando Model Selector...
echo.

REM Executar o RobotV7 Model Selector
python RobotV7_ModelSelector.py

REM Mostrar mensagem de saída
echo.
echo ================================================================
echo 🛑 Model Selector finalizado
echo ================================================================
echo.

pause