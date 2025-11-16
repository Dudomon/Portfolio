@echo off
echo ========================================
echo  EXPERTGAIN V2 - FINE-TUNING INTELIGENTE
echo ========================================
echo.
echo Iniciando ExpertGain V2 com:
echo - Learning Rate Dinamico
echo - Reward Shaping para Entry Quality
echo - 3 Fases Especializadas
echo - Early Stopping Inteligente
echo.
echo Pressione CTRL+C para cancelar...
timeout /t 5

python expertgain.py

echo.
echo ========================================
echo  TREINAMENTO CONCLUIDO
echo ========================================
pause