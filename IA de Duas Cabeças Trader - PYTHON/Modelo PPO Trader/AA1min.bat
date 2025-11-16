@echo off
REM Abre somente a GUI do RobotV7 (sem console)
cd /d "%~dp0\Modelo PPO Trader"
start "" pythonw.exe Robot_1min.py
exit /b 0
