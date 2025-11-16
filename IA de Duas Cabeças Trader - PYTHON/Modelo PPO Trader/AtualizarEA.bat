@echo off
REM Script para atualizar o SimpleDrawingEA.mq5 na pasta do MT5

REM Caminho de origem (pasta do projeto)
set SOURCE_EA="%~dp0SimpleDrawingEA.mq5"

REM Caminho de destino (ajuste conforme necessário para o seu terminal MT5)
set MT5_EXPERTS_DIR="%USERPROFILE%\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Experts\SimpleDrawingEA.mq5"

REM Copiar o arquivo
copy /Y %SOURCE_EA% %MT5_EXPERTS_DIR%

if %ERRORLEVEL%==0 (
    echo EA atualizado com sucesso em %MT5_EXPERTS_DIR%
) else (
    echo Erro ao copiar o EA. Verifique o caminho de destino.
)

pause 