@echo off
Treino modelo ANDER
echo.

REM Executa o script
echo Executando otimizacao...
python anderv1.py

if errorlevel 1 (
    echo.
    echo Erro durante a execucao do script.
    echo Verifique o arquivo de log para mais detalhes.
) else (
    echo.
    Treinamento Concluído
)

echo.
pause 