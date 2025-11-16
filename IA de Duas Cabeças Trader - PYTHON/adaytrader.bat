@echo off
echo.
echo ========================================
echo 🚀 DAY TRADER V7 - DIRECT MULTI-TIMEFRAME
echo ========================================
echo.
echo 🎯 CONFIGURACAO:
echo - SL: 3-12 pontos ($3-12)
echo - TP: 4-20 pontos ($4-20)  
echo - Ranges otimizados para day trade
echo.
echo 🚀 TREINO DIRETO MULTI-TIMEFRAME:
echo - Dataset: 5m + 15m + features
echo - Gates especializados V7
echo - LSTMs aprendem confluencia desde inicio
echo.
echo ⏱️  TEMPO ESTIMADO: ~6-8 horas (2.3M steps)
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

REM Dataset 1m não é mais necessário - treino direto multi-timeframe
echo 🚀 Usando dataset multi-timeframe direto (5m + 15m + features)

REM Executar Day Trader
echo 🏋️  INICIANDO TREINAMENTO DAY TRADER...
echo.
python daytrader.py

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