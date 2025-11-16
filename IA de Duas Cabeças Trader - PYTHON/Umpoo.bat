@echo off
echo ========================================
echo        UMPOO - HEADV6 TRAINING
echo        TwoHeadV6Intelligent48h
echo ========================================
echo.

echo [%TIME%] Iniciando treinamento HEADV6...
echo.

REM Executar o headv6.py
python headv6.py

REM Verificar se houve erro na execução
if %ERRORLEVEL% neq 0 (
    echo.
    echo [%TIME%] ❌ ERRO no treinamento HEADV6! Código: %ERRORLEVEL%
    echo Pressione qualquer tecla para sair...
    pause >nul
    exit /b %ERRORLEVEL%
)

echo.
echo [%TIME%] ✅ Treinamento HEADV6 concluído com sucesso!
echo.

echo ========================================
echo        COMMITANDO PROJETO
echo ========================================
echo.

REM Adicionar arquivos modificados
echo [%TIME%] Adicionando arquivos ao git...
git add .

REM Verificar se há algo para commitar
git diff --cached --quiet
if %ERRORLEVEL% equ 0 (
    echo [%TIME%] ℹ️ Nenhuma alteração para commitar.
    goto :end
)

REM Criar commit automático
echo [%TIME%] Criando commit...
git commit -m "TREINO HEADV6: Sessão %DATE% %TIME%

- Executado headv6.py (TwoHeadV6Intelligent48h)
- Ranges SL/TP: 10-30 e 15-50 pontos
- Checkpoints salvos em: models/HEADV6/
- Métricas de fim de episódio corrigidas

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

if %ERRORLEVEL% neq 0 (
    echo [%TIME%] ❌ ERRO ao criar commit! Código: %ERRORLEVEL%
    goto :end
)

echo [%TIME%] ✅ Commit criado com sucesso!

:end
echo.
echo ========================================
echo           UMPOO FINALIZADO
echo ========================================
echo Pressione qualquer tecla para sair...
pause >nul