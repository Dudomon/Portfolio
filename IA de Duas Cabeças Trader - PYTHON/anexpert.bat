@echo off
chcp 65001 >nul
echo.
echo ========================================
echo    ANEXPERT - EXPERTGAIN LAUNCHER
echo ========================================
echo EXPERTGAIN: Treinamento Especializado V7
echo OBJETIVO: Entry Quality 0.488 para 0.55+
echo FASES: 7M steps especializados
echo BASE: Ultimo checkpoint DayTrader
echo ========================================
echo.

REM Verificar se o Python esta disponivel
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python nao encontrado no PATH
    echo       Instale Python ou adicione ao PATH
    pause
    exit /b 1
)

REM Verificar se o arquivo existe
if not exist "expertgain.py" (
    echo ERRO: expertgain.py nao encontrado
    echo       Execute este comando no diretorio correto
    pause
    exit /b 1
)

echo Iniciando ExpertGain...
echo.

REM Executar o ExpertGain
python expertgain.py

REM Verificar se houve erro
if errorlevel 1 (
    echo.
    echo EXPERTGAIN FALHOU COM ERRO
    echo.
) else (
    echo.
    echo EXPERTGAIN CONCLUIDO COM SUCESSO
    echo.
)

echo ========================================
echo    Pressione qualquer tecla para sair
echo ========================================
pause >nul