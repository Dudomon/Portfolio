@echo off
chcp 65001 >nul
echo ========================================
echo    🔍 VERIFICADOR DE DEPENDÊNCIAS
echo ========================================
echo.

echo 🔍 Verificando dependências do RobotV3...
echo.

python check_robotv3_dependencies.py

if errorlevel 1 (
    echo.
    echo ❌ VERIFICAÇÃO FALHOU!
    echo.
    echo 💡 Instale as dependências faltantes:
    echo.
    echo pip install torch stable-baselines3 sb3-contrib gym numpy pandas matplotlib seaborn plotly MetaTrader5 requests pyinstaller
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ VERIFICAÇÃO CONCLUÍDA!
echo.
echo 🚀 Agora você pode executar o build:
echo    BuildRobotV3.bat
echo.
pause 