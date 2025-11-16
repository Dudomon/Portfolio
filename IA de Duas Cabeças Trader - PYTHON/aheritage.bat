@echo off
chcp 65001 >nul
REM ========================================================================
REM HERITAGE V8 - TREINAMENTO PPO TRADER V8HERITAGE  
REM ========================================================================
REM
REM Este script roda o daytrader8dim.py com TwoHeadV8Heritage:
REM - Action Space 8D (sem gates V7)
REM - Position sizing controlado pelo modelo [0.5x - 2.0x]
REM - UnifiedV8FeatureProcessor (combina todos V7 extractors)  
REM - OptimizedV8DecisionMaker (sem temporal/risk/regime gates)
REM - HybridMemorySystem (3 bancos de memoria especializados)
REM - LSTM Critic Heritage Mode habilitado
REM
REM EXPERIMENT_TAG: HeritageV8
REM Checkpoints salvos em: Otimizacao/treino_principal/checkpoints/HeritageV8/
REM ========================================================================

echo.
echo ========================================================================
echo INICIANDO HERITAGE V8 - PPO TRADER V8HERITAGE
echo ========================================================================
echo.
echo Configuracao V8Heritage:
echo    - Action Space: 8D (entry_decision, position_multiplier, 3xSL, 3xTP)  
echo    - Position Sizing: Controlado pelo modelo [0.5x - 2.0x base]
echo    - Gates V7: REMOVIDOS (sem temporal/risk/regime)
echo    - Memory System: HybridMemorySystem (3 bancos)
echo    - Critic: LSTM Heritage Mode
echo    - Experiment: HeritageV8
echo.
echo Iniciando em 3 segundos...
timeout /t 3 /nobreak >nul

echo.
echo EXECUTANDO: python daytrader8dim.py
echo ========================================================================

REM Executar o script Python
python daytrader8dim.py

REM Verificar se houve erro
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERRO durante a execucao!
    echo Codigo de erro: %ERRORLEVEL%
    echo.
    echo Possiveis causas:
    echo    - Python nao encontrado
    echo    - Dependencias ausentes
    echo    - Erro no codigo
    echo    - Dados de treinamento nao encontrados
    echo.
) else (
    echo.
    echo HERITAGE V8 executado com sucesso!
    echo.
)

echo.
echo HERITAGE V8 - SESSAO FINALIZADA
echo ========================================================================
echo.
echo Pressione qualquer tecla para fechar...
pause >nul