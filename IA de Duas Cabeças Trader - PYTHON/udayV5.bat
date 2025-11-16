@echo off
echo.
echo ========================================
echo 📈 DAY TRADER V1 - CURRICULUM LEARNING
echo ========================================
echo.
echo 🎯 CONFIGURACAO:
echo - SL: 3-12 pontos ($3-12)
echo - TP: 4-20 pontos ($4-20)  
echo - Ranges otimizados para day trade
echo.
echo 📚 CURRICULUM 2 FASES:
echo - FASE 1: 1m micro-scalping (100k steps)
echo - FASE 2: Multi-timeframe (200k steps)
echo.
echo ⏱️  TEMPO ESTIMADO: ~4-6 horas
echo.
echo 🚀 Iniciando treinamento...
echo.

REM Navegar para diretório do projeto
cd /d "D:\Projeto"

REM Verificar se Python está disponível
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERRO: Python não encontrado!
    echo Instale Python ou adicione ao PATH
    pause
    exit /b 1
)

REM Verificar se dataset 1m existe
if not exist "data\GOLD_1M_MASSIVE_SYNTHETIC_*.pkl" (
    echo.
    echo ⚠️  DATASET 1M NÃO ENCONTRADO!
    echo Executando create_synthetic_1m.py primeiro...
    echo.
    python create_synthetic_1m.py
    if errorlevel 1 (
        echo.
        echo ❌ ERRO: Falha ao criar dataset 1m
        echo Verifique se o dataset 5m existe em data_cache/
        pause
        exit /b 1
    )
    echo.
    echo ✅ Dataset 1m criado com sucesso!
    echo.
)

REM Executar Day Trader
echo 🏋️  INICIANDO TREINAMENTO DAY TRADER...
echo.
python dayv5.py

REM Verificar resultado
if errorlevel 1 (
    echo.
    echo ❌ ERRO: Treinamento falhou!
    echo Verifique logs acima para detalhes
) else (
    echo.
    echo ========================================
    echo ✅ DAY TRADER TREINADO COM SUCESSO!
    echo ========================================
    echo.
    echo 📁 Modelos salvos em: models/daytrading/
    echo 🎯 Pronto para trading real!
    echo.
)

echo.
echo Pressione qualquer tecla para sair...
pause >nul