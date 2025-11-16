#!/usr/bin/env python3
"""
📊 MONITOR SATURAÇÃO CONTÍNUO - V7 Sigmoid Fix
Monitora saturação dos sigmoids após aplicação do fix
"""

import sys
import os
import time
import glob
from datetime import datetime
sys.path.append("D:/Projeto")

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
import json

# Configuração
CHECKPOINTS_DIR = "D:/Projeto/Otimizacao/treino_principal/models/DAYTRADER/"
N_SAMPLES = 500  # Menor para monitoramento mais rápido
CHECK_INTERVAL = 300  # 5 minutos entre verificações

def get_latest_checkpoint():
    """Encontra o checkpoint mais recente"""
    pattern = os.path.join(CHECKPOINTS_DIR, "DAYTRADER_*.zip")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    # Ordenar por data de modificação
    checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return checkpoints[0]

def analyze_entry_quality_fast(checkpoint_path):
    """Análise rápida da Entry Quality"""
    
    try:
        # Carregar modelo
        model = RecurrentPPO.load(checkpoint_path, device='cuda')
        model.policy.set_training_mode(False)
        
        # Executar predições
        lstm_states = None
        entry_qualities = []
        
        for i in range(N_SAMPLES):
            obs = np.random.normal(0, 1.0, (2580,)).astype(np.float32)
            action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
            
            if len(action) >= 2:
                entry_qualities.append(float(action[1]))
        
        # Análise
        if not entry_qualities:
            return None
        
        eq_array = np.array(entry_qualities)
        eq_mean = np.mean(eq_array)
        eq_std = np.std(eq_array)
        
        # Extremos
        eq_near_zero = np.sum(eq_array < 0.1)
        eq_near_one = np.sum(eq_array > 0.9)
        eq_extremes_pct = (eq_near_zero + eq_near_one) / len(eq_array) * 100
        
        # Distribuição por quartis
        quartiles = np.percentile(eq_array, [25, 50, 75])
        
        return {
            'checkpoint': os.path.basename(checkpoint_path),
            'samples': len(entry_qualities),
            'mean': float(eq_mean),
            'std': float(eq_std),
            'min': float(np.min(eq_array)),
            'max': float(np.max(eq_array)),
            'extremes_pct': float(eq_extremes_pct),
            'near_zero': int(eq_near_zero),
            'near_one': int(eq_near_one),
            'q25': float(quartiles[0]),
            'q50': float(quartiles[1]), 
            'q75': float(quartiles[2]),
            'timestamp': datetime.now().isoformat(),
            'analysis_time': datetime.now().strftime('%H:%M:%S')
        }
        
    except Exception as e:
        return {'error': str(e), 'checkpoint': os.path.basename(checkpoint_path)}

def monitor_saturation():
    """Loop principal de monitoramento"""
    
    print("📊 MONITOR SATURAÇÃO CONTÍNUO - V7 FIX")
    print("=" * 60)
    print(f"🔍 Diretório: {CHECKPOINTS_DIR}")
    print(f"⏰ Intervalo: {CHECK_INTERVAL} segundos")
    print(f"📊 Samples por análise: {N_SAMPLES}")
    print("=" * 60)
    
    last_checkpoint = None
    history = []
    
    while True:
        try:
            # Verificar checkpoint mais recente
            latest_checkpoint = get_latest_checkpoint()
            
            if not latest_checkpoint:
                print("⏳ Aguardando checkpoints...")
                time.sleep(CHECK_INTERVAL)
                continue
            
            # Se é um checkpoint novo, analisar
            if latest_checkpoint != last_checkpoint:
                print(f"\n🔍 Novo checkpoint detectado: {os.path.basename(latest_checkpoint)}")
                print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
                
                # Análise rápida
                result = analyze_entry_quality_fast(latest_checkpoint)
                
                if result and 'error' not in result:
                    # Exibir resultados
                    print(f"   📊 Entry Quality: μ={result['mean']:.3f} σ={result['std']:.3f}")
                    print(f"   🚨 Extremos: {result['extremes_pct']:.1f}% (0s:{result['near_zero']}, 1s:{result['near_one']})")
                    print(f"   📈 Quartis: Q1={result['q25']:.3f} Q2={result['q50']:.3f} Q3={result['q75']:.3f}")
                    
                    # Status de melhoria
                    if history:
                        prev = history[-1]
                        if 'extremes_pct' in prev:
                            delta = result['extremes_pct'] - prev['extremes_pct']
                            if delta < -5:
                                status = "🟢 MELHORANDO"
                            elif delta > 5:
                                status = "🔴 PIORANDO"
                            else:
                                status = "🟡 ESTÁVEL"
                            print(f"   📈 Tendência: {status} (Δ{delta:+.1f}%)")
                    
                    # Classificar status
                    if result['extremes_pct'] < 60:
                        print("   ✅ SATURAÇÃO RESOLVIDA!")
                    elif result['extremes_pct'] < 80:
                        print("   🟡 MELHORIA DETECTADA")
                    elif result['extremes_pct'] < 95:
                        print("   ⚠️ AINDA SATURADO")
                    else:
                        print("   🔴 SATURAÇÃO CRÍTICA")
                    
                    # Salvar histórico
                    history.append(result)
                    
                    # Manter últimos 50 registros
                    if len(history) > 50:
                        history = history[-50:]
                    
                    # Salvar relatório
                    report_file = f"D:/Projeto/avaliacoes/saturacao_monitor_{datetime.now().strftime('%Y%m%d')}.json"
                    with open(report_file, 'w') as f:
                        json.dump(history, f, indent=2, default=str)
                
                elif result and 'error' in result:
                    print(f"   ❌ Erro na análise: {result['error']}")
                
                last_checkpoint = latest_checkpoint
            
            else:
                print(f"⏳ Aguardando novo checkpoint... ({datetime.now().strftime('%H:%M:%S')})")
            
            # Aguardar próxima verificação
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n🛑 Monitor interrompido pelo usuário")
            break
        except Exception as e:
            print(f"❌ Erro no monitor: {e}")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    monitor_saturation()