@echo off
echo Iniciando otimizacao PPO...
echo.

REM Executa o script
echo Executando otimizacao...
python ppo.py

if errorlevel 1 (
    echo.
    echo Erro durante a execucao do script.
    echo Verifique o arquivo de log para mais detalhes.
) else (
    echo.
    echo Otimizacao concluida com sucesso!
)

echo.
pause 