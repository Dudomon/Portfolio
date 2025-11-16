@echo off
chcp 65001 >nul
echo INICIANDO TREINAMENTO SILUS - V11 SIGMOID NON-SATURATING
echo ================================================
echo Experiment: SILUS
echo Policy: TwoHeadV11Sigmoid  
echo Activations: SiLU non-saturating
echo ================================================
echo.

python silus.py

echo.
echo ================================================
echo TREINAMENTO SILUS FINALIZADO
echo ================================================
pause