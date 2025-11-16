@echo off
echo 🛠️ RobotV7 - Modo Desenvolvimento (SEM LOGIN)
echo ============================================
echo ⚠️ ATENÇÃO: Modo desenvolvimento - pula autenticação
echo.

cd /d "%~dp0"

python robotlogin.py --no-login

echo.
echo 📴 Sistema encerrado
pause