@echo off
echo ========================================
echo    🔍 VERIFICAR DEPENDÊNCIAS ROBOANDER
echo ========================================
echo.

echo Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python não encontrado!
    echo Instale o Python 3.8+ primeiro.
    pause
    exit /b 1
)

echo ✅ Python encontrado
echo.

echo Executando verificação completa...
python check_roboander_dependencies.py

echo.
if errorlevel 1 (
    echo ❌ Verificação falhou!
    echo Corrija os problemas antes de fazer o build.
) else (
    echo ✅ Verificação concluída!
    echo Ambiente pronto para build.
)

pause 