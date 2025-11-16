@echo off
echo.
echo 🤖 ROBOTV7 LEGION - LOGIN SYSTEM
echo ================================
echo.

REM Verificar se Python está disponível
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python não encontrado! Instale o Python primeiro.
    pause
    exit /b 1
)

REM Mudar para o diretório do script
cd /d "%~dp0"

REM Verificar se arquivo existe
if not exist "robotlogin.py" (
    echo ❌ Arquivo robotlogin.py não encontrado!
    pause
    exit /b 1
)

REM Executar o RobotLogin
echo 🚀 Iniciando RobotV7 Login System...
echo.
python robotlogin.py

REM Pausa apenas se houver erro
if errorlevel 1 (
    echo.
    echo ❌ Erro ao executar RobotLogin
    pause
)

REM Exit com código de sucesso
exit /b 0