@echo off
REM Abre somente a GUI do RobotV7 (sem console)
cd /d "%~dp0\Modelo PPO Trader"
start "" pythonw.exe RobotSwing.py
exit /b 0
