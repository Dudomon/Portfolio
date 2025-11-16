@echo off
REM Abre somente a GUI do RobotV7 (sem console)
cd /d "%~dp0\Modelo PPO Trader"
start "" pythonw.exe RobotV7.py
exit /b 0
