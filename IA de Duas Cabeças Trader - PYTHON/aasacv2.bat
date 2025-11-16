@echo off
chcp 65001 > nul
set PYTHONUTF8=1

echo.
echo ================================================================
echo  SAC v2 TRADING SYSTEM - SOFT ACTOR-CRITIC v2
echo ================================================================
echo.
echo  ALGORITHM: SAC v2 (Soft Actor-Critic with Auto-tuning)
echo  POLICY: MlpPolicy with V12SAC Features Extractor
echo  BUFFER: 1M transitions Experience Replay
echo  FEATURES: Off-Policy + Twin Critics + Auto Entropy
echo  DEVICE: CUDA RTX 4070 Ti
echo  EXPERIMENT: SACVERSION
echo.
echo ============================================================
echo  INICIANDO TREINAMENTO SAC v2...
echo ============================================================
echo.

cd /d "D:\Projeto"
python sacversion.py

echo.
echo ============================================================
echo  TREINAMENTO SAC v2 FINALIZADO
echo ============================================================
echo.
pause