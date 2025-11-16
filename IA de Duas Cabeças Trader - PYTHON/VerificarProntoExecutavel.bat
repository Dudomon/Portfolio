@echo off
chcp 65001 >nul
title 🔍 Verificador de Pré-requisitos - Executável RoboAnder

echo.
echo ===============================================
echo 🔍 VERIFICADOR DE PRÉ-REQUISITOS
echo ===============================================
echo.
echo 📋 Este script verificará se tudo está pronto para criar
echo    o executável do RoboAnder Legion AI:
echo.
echo   🤖 Modelo treinado
echo   🔐 Sistema de login
echo   🌐 Google Drive (opcional)
echo   📦 Dependências Python
echo   🔨 PyInstaller
echo.
echo Iniciando verificação...
echo.

python verificar_pronto_executavel.py

echo.
pause 