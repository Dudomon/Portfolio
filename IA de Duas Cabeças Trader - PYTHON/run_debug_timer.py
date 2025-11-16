#!/usr/bin/env python3
"""
⏰ RUN DEBUG TIMER - Rodar daytrader com timer de 30s
"""

import subprocess
import time
import signal
import os

def run_with_timer():
    """Rodar daytrader.py por 30 segundos e capturar debug"""
    
    print("⏰ INICIANDO DAYTRADER COM TIMER DE 30 SEGUNDOS")
    print("=" * 60)
    print("🔍 Procurando por: '🚨 DEBUG THRESHOLD:'")
    print("=" * 60)
    
    # Iniciar processo
    process = subprocess.Popen(
        ["python", "daytrader.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        encoding='utf-8',
        errors='ignore'
    )
    
    start_time = time.time()
    debug_found = False
    
    try:
        while True:
            # Verificar timeout
            if time.time() - start_time > 30:
                print("\n⏰ TIMEOUT DE 30 SEGUNDOS ATINGIDO")
                break
            
            # Ler linha
            line = process.stdout.readline()
            if not line:
                break
            
            # Imprimir linha
            print(line.rstrip())
            
            # Verificar se debug apareceu
            if "🚨 DEBUG THRESHOLD:" in line:
                debug_found = True
                print("\n✅ DEBUG ENCONTRADO!")
                
                # Ler mais algumas linhas do debug
                for _ in range(10):
                    debug_line = process.stdout.readline()
                    if debug_line:
                        print(debug_line.rstrip())
                        if "Final decision:" in debug_line:
                            break
                
                break
    
    except KeyboardInterrupt:
        print("\n⚠️ INTERROMPIDO PELO USUÁRIO")
    
    finally:
        # Terminar processo
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
    
    print("\n" + "=" * 60)
    if debug_found:
        print("✅ DEBUG CAPTURADO COM SUCESSO!")
        print("   Agora podemos analisar onde está o bug")
    else:
        print("❌ DEBUG NÃO APARECEU")
        print("   Problema: Policy não está sendo usada ou debug não está ativo")
    print("=" * 60)

if __name__ == "__main__":
    run_with_timer()