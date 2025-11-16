@echo off
REM Inicia treinamento do Pescador
cd /d "%~dp0"
python cherry.py
echo.
echo Treinamento encerrado. Pressione qualquer tecla para sair...
pause >nul
