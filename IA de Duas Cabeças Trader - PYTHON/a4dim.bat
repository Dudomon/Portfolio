@echo off
echo.
echo ========================================
echo 🚀 V9 OPTIMUS - 4D ACTION SPACE REVOLUTION
echo ========================================
echo.
echo 🎯 CONFIGURACAO V9 OPTIMUS:
echo - Action Space: 4D (entry + confidence + pos1_mgmt + pos2_mgmt)
echo - Obs Space: 450D (83%% menor que V8)
echo - Single LSTM Backbone (256D)
echo - Memory Bank compacto (256 trades)
echo - Market Context focado (4 regimes)
echo.
echo 🧠 ARQUITETURA OPTIMUS:
echo - Entry Head especifico (2D): decision + confidence
echo - Management Head especifico (2D): pos1_mgmt + pos2_mgmt
echo - Maximum efficiency with full control
echo.
echo ⏱️  TEMPO ESTIMADO: ~4-6 horas (1M steps)
echo.
echo 🚀 Iniciando treinamento V9 Optimus...
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

REM Verificar se 4dim.py existe
if not exist "4dim.py" (
    echo ❌ ERRO: 4dim.py não encontrado!
    echo Verifique se o arquivo existe no diretório atual
    pause
    exit /b 1
)

REM Dataset V3 enhanced para V9 Optimus
echo 🚀 Usando dataset V3 Enhanced para V9 Optimus (4D Action Space)

REM Executar V9 Optimus 4D
echo 🏋️  INICIANDO TREINAMENTO V9 OPTIMUS 4D...
echo.
python 4dim.py

REM Verificar resultado
if errorlevel 1 (
    echo.
    echo ❌ ERRO: Treinamento V9 Optimus falhou!
    echo Verifique logs acima para detalhes
) else (
    echo.
    echo ========================================
    echo ✅ V9 OPTIMUS 4D TREINADO COM SUCESSO!
    echo ========================================
    echo.
    echo 📁 Modelos salvos em: Otimizacao/treino_principal/models/Optimus/
    echo 🎯 Action Space: 4D Revolution implementado!
    echo 🧠 Architecture: V9 Optimus - Maximum efficiency
    echo.
)

echo.
echo Pressione qualquer tecla para sair...
pause >nul