@echo off
echo.
echo ========================================
echo TRADING FRAMEWORK - AVALIACAO DE MODELOS
echo ========================================
echo.

REM Verificar se está no diretório correto
if not exist "avaliar_modelo_final.py" (
    echo Erro: arquivo avaliar_modelo_final.py nao encontrado!
    echo Execute este .bat no diretorio correto
    pause
    exit /b 1
)

echo Verificando ambiente Python...

REM Tentar ativar ambiente virtual se existir
if exist "venv\Scripts\activate.bat" (
    echo Ativando ambiente virtual...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo Ativando ambiente virtual...
    call .venv\Scripts\activate.bat
) else if exist "..\venv\Scripts\activate.bat" (
    echo Ativando ambiente virtual...
    call ..\venv\Scripts\activate.bat
) else (
    echo Nenhum ambiente virtual encontrado, usando Python global...
)

echo.
echo Iniciando Interface de Avaliacao de Modelos...
echo.

REM Executar o sistema de avaliação
python avaliar_modelo_final.py

echo.
echo ========================================
echo Sistema de Avaliacao Finalizado
echo ========================================
echo.
pause 