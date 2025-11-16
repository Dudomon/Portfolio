#!/usr/bin/env python3
"""
📊 MONITOR ESPECÍFICO: Learning Rate
Monitora se o LR está sendo modificado indevidamente
"""

import time
import re
from pathlib import Path

def monitor_lr_changes():
    """Monitora mudanças no learning rate"""
    
    print("📊 MONITOR DE LEARNING RATE ATIVO")
    print("Verificando se LR está sendo modificado...")
    print("=" * 50)
    
    last_current_lr = None
    last_config_lr = None
    
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
                
                # Extrair current_lr e learning_rate
                current_lr_matches = re.findall(r'current_lr\s*\|\s*([\d.e-]+)', content)
                config_lr_matches = re.findall(r'learning_rate\s*\|\s*([\d.e-]+)', content)
                
                if current_lr_matches and config_lr_matches:
                    current_lr = float(current_lr_matches[-1])
                    config_lr = float(config_lr_matches[-1])
                    
                    if current_lr != last_current_lr or config_lr != last_config_lr:
                        print(f"\n📊 LEARNING RATES:")
                        print(f"   Current LR (usado):     {current_lr:.2e}")
                        print(f"   Config LR (esperado):   {config_lr:.2e}")
                        
                        if abs(current_lr - config_lr) > 1e-6:
                            print(f"   ⚠️  CONFLITO DETECTADO! Diferença: {abs(current_lr - config_lr):.2e}")
                            print(f"   🔧 Scheduler ainda ativo ou outro sistema modificando LR")
                        else:
                            print(f"   ✅ LRs em sincronia")
                        
                        last_current_lr = current_lr
                        last_config_lr = config_lr
                
            except Exception as e:
                pass
            
            time.sleep(15)  # Verificar a cada 15 segundos
            
    except KeyboardInterrupt:
        print("\n📊 Monitor LR interrompido")

if __name__ == "__main__":
    monitor_lr_changes()
