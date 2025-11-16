@echo off
cls
echo.
echo ===============================================
echo 🔥 AVALIADOR AUTOMÁTICO - A CADA 30 MINUTOS
echo ===============================================
echo.
echo 📋 ESTE SCRIPT:
echo    - Avalia o modelo a cada 30 minutos
echo    - Roda em loop infinito
echo    - Mostra resultados no terminal do mainppo1.py
echo.
echo ⚠️  IMPORTANTE: 
echo    - Deixe o mainppo1.py rodando
echo    - Para parar, feche esta janela (Ctrl+C)
echo.
echo 🚀 Iniciando avaliação automática...
echo.

:loop
echo [%date% %time%] 🔄 Criando avaliação automática...
echo avaliacao_automatica_%date%_%time% > eval.txt
echo [%date% %time%] ✅ Arquivo eval.txt criado - aguarde resultados
echo.
echo ⏳ Aguardando 30 minutos para próxima avaliação...
echo    (Para avaliar agora, execute avaliar_modelo.bat)
echo.
timeout /t 1800 /nobreak
goto loop 