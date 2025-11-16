@echo off
echo 🚀 Gerando RobotV7 executável...

pyinstaller --onefile ^
--windowed ^
--name "RobotV7_FINAL" ^
--add-data "robotlogin.pyc;." ^
--add-data "enhanced_normalizer.pyc;." ^
--add-data "Legion V1.secure;." ^
--add-data "trading_framework;trading_framework" ^
--hidden-import tkinter ^
--hidden-import MetaTrader5 ^
--hidden-import numpy ^
--hidden-import torch ^
--hidden-import stable_baselines3 ^
--clean ^
launcher_final.py

echo ✅ Executável criado em dist\RobotV7_FINAL.exe
pause