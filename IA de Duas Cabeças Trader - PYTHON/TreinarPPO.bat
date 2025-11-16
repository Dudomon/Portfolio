@echo off
cd /d %~dp0

echo ============================
echo  Iniciando Treino Principal
echo ============================

python ppov1.py

if errorlevel 1 (
    echo.
    echo Erro durante a execucao do script.
    echo Verifique o arquivo de log para mais detalhes.
) else (
    echo.
    echo Treinamento concluido com sucesso!
)

echo.
pause 