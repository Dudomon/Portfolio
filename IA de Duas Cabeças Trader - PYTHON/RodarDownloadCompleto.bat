@echo off
echo ========================================
echo    DOWNLOAD COMPLETO DE DADOS DE OURO
echo ========================================
echo.
echo Iniciando download de dados historicos...
echo Timeframes: 5m, 15m, 4h
echo Periodo: Maximo disponivel desde 1990
echo.
echo Pressione qualquer tecla para continuar...
pause >nul

echo.
echo Executando teste do sistema...
python teste_download_completo.py

if %errorlevel% neq 0 (
    echo.
    echo ERRO: Teste falhou! Verifique os problemas acima.
    echo.
    pause
    exit /b 1
)

echo.
echo Teste passou! Executando download completo...
echo.

python downloaddata.py

if %errorlevel% neq 0 (
    echo.
    echo ERRO: Download falhou! Verifique os logs.
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo    DOWNLOAD CONCLUIDO COM SUCESSO!
echo ========================================
echo.
echo Arquivos gerados:
echo - data/GOLD_5m_*.csv
echo - data/GOLD_15m_*.csv  
echo - data/GOLD_4h_*.csv
echo - data/GOLD_COMPLETE_*.csv
echo - data_cache/GOLD_CACHE_*.pkl
echo.
echo Cache otimizado salvo para uso rapido!
echo.
pause 