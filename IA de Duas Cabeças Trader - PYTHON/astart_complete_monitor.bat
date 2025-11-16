@echo off
chcp 65001 >nul
echo ================================
echo COMPLETE CONVERGENCE MONITOR
echo ================================
echo.
echo This monitor combines:
echo  - Gradient health (zeros debugging)
echo  - Training convergence (loss trends)  
echo  - Performance metrics (rewards)
echo.
echo Updates every 30 seconds
echo Press Ctrl+C to stop
echo.
pause
echo Starting monitor...
echo.
python complete_convergence_monitor.py
pause