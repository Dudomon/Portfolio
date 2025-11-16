#!/usr/bin/env python3
"""
📊 MONITOR: KL Divergence e Clip Fraction
Monitora se as correções estão funcionando
"""

import re
import time
from pathlib import Path

def monitor_training_metrics():
    """Monitora métricas do treinamento em tempo real"""
    
    print("📊 MONITOR DE TREINAMENTO ATIVO")
    print("Monitorando KL divergence e clip fraction...")
    print("Pressione Ctrl+C para parar")
    print("=" * 50)
    
    last_kl = None
    last_clip = None
    
    try:
        while True:
            # Procurar pelo log mais recente
            log_files = list(Path("logs").glob("ppo_optimization_*.log"))
            if not log_files:
                time.sleep(5)
                continue
            
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            
            try:
                with open(latest_log, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extrair métricas mais recentes
                kl_matches = re.findall(r'approx_kl\s*\|\s*([\d.e-]+)', content)
                clip_matches = re.findall(r'clip_fraction\s*\|\s*([\d.e-]+)', content)
                lr_matches = re.findall(r'learning_rate\s*\|\s*([\d.e-]+)', content)
                
                if kl_matches:
                    current_kl = float(kl_matches[-1])
                    if current_kl != last_kl:
                        status_kl = "✅ BOM" if current_kl > 1e-3 else "⚠️ BAIXO" if current_kl > 1e-4 else "❌ MUITO BAIXO"
                        print(f"KL Divergence: {current_kl:.2e} {status_kl}")
                        last_kl = current_kl
                
                if clip_matches:
                    current_clip = float(clip_matches[-1])
                    if current_clip != last_clip:
                        status_clip = "✅ BOM" if current_clip > 0.05 else "⚠️ BAIXO" if current_clip > 0 else "❌ ZERO"
                        print(f"Clip Fraction: {current_clip:.3f} {status_clip}")
                        last_clip = current_clip
                
                if lr_matches:
                    current_lr = float(lr_matches[-1])
                    print(f"Learning Rate: {current_lr:.2e}")
                
            except Exception as e:
                pass
            
            time.sleep(10)  # Verificar a cada 10 segundos
            
    except KeyboardInterrupt:
        print("\n📊 Monitor interrompido pelo usuário")

if __name__ == "__main__":
    monitor_training_metrics()
