#!/usr/bin/env python3
"""
🚀 RESTART TRAINING - Testar V7Unified com thresholds corrigidos
"""

import subprocess
import sys
import time

def restart_training():
    """Reiniciar treinamento com V7Unified e thresholds corrigidos"""
    
    print("🚀 REINICIANDO TREINAMENTO COM THRESHOLD FIX")
    print("=" * 60)
    print("✅ V7Unified com thresholds equilibrados (-0.33, 0.33)")
    print("✅ Filtros V7 desabilitados no daytrader")
    print("✅ Expectativa: SHORT > 15%, HOLD < 50%")
    print("=" * 60)
    
    # Parar qualquer treinamento existente
    print("🛑 Parando treinamentos existentes...")
    
    try:
        # Tentar parar processos Python que podem estar rodando daytrader
        subprocess.run(["taskkill", "/f", "/im", "python.exe"], 
                      capture_output=True, text=True, check=False)
        time.sleep(2)
    except:
        pass
    
    print("🚀 Iniciando novo treinamento...")
    
    # Iniciar daytrader
    try:
        subprocess.run([sys.executable, "daytrader.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Treinamento interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro no treinamento: {e}")

if __name__ == "__main__":
    restart_training()