#!/usr/bin/env python3
"""
🔍 DEBUG: Verificar por que agente parou de operar após Risk Heat Level
"""

import subprocess
import sys
import time
import re

def debug_trading_decisions():
    print("🔍 DEBUG: Verificando decisões de trading")
    print("=" * 60)
    
    try:
        # Executar daytrader e capturar decisions
        process = subprocess.Popen([
            sys.executable, "daytrader.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
           universal_newlines=True, cwd="D:/Projeto")
        
        start_time = time.time()
        timeout = 60  # 1 minuto
        
        entry_decisions = []
        risk_heat_values = []
        position_counts = []
        
        print("📋 Coletando decisões de trading...")
        
        while time.time() - start_time < timeout:
            line = process.stdout.readline()
            if not line:
                break
                
            # Buscar indicadores de decisão
            if "entry_decision" in line.lower():
                print(f"   📊 {line.strip()}")
                
            if "risk_heat" in line.lower():
                print(f"   🔥 {line.strip()}")
                
            # Buscar posições abertas
            if "positions:" in line.lower() or "open_positions" in line.lower():
                print(f"   💼 {line.strip()}")
                
            # Buscar problemas com SL/TP
            if "sl_points" in line.lower() or "tp_points" in line.lower():
                print(f"   🎯 {line.strip()}")
                
            # Buscar erros
            if "erro" in line.lower() or "error" in line.lower():
                print(f"   ❌ ERRO: {line.strip()}")
                
            # Buscar recompensas
            if "reward" in line.lower() and "total" in line.lower():
                print(f"   💰 {line.strip()}")
                
            # Se chegou ao treinamento, já coletamos dados suficientes
            if "steps/s" in line and time.time() - start_time > 30:
                print("   ✅ Treinamento detectado - parando coleta")
                break
        
        # Terminar processo
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            try:
                process.kill()
            except:
                pass
                
        print(f"\n🎯 ANÁLISE:")
        print("Verifique os logs acima para:")
        print("   1. Se entry_decision está sendo > 0")
        print("   2. Se risk_heat está em range [0,1]")
        print("   3. Se SL/TP não estão com valores inválidos")
        print("   4. Se há posições sendo abertas")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

if __name__ == "__main__":
    debug_trading_decisions()