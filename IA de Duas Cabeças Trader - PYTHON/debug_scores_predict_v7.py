#!/usr/bin/env python3
"""
🔍 DEBUG SCORES VIA PREDICT V7 - Análise usando predict
"""

import sys
import os
sys.path.append("D:/Projeto")

import torch
import numpy as np
from sb3_contrib import RecurrentPPO

# ========== CONFIGURAÇÃO ==========
CHECKPOINT_NAME = "checkpoint_7700000_steps_20250808_165028.zip"
CHECKPOINT_PATH = f"D:/Projeto/trading_framework/training/checkpoints/DAYTRADER/{CHECKPOINT_NAME}"
# ==================================

def debug_scores_via_predict():
    """🔍 Analisar scores usando o método predict"""
    
    print("🔍 DEBUG SCORES VIA PREDICT V7")
    print("=" * 60)
    
    try:
        # Carregar modelo
        print(f"🤖 Carregando modelo...")
        model = RecurrentPPO.load(CHECKPOINT_PATH, device='cuda')
        
        # Dimensões
        obs_dim = 2580
        n_samples = 50
        
        print("📊 Gerando observações de teste...")
        
        # Diferentes tipos de input
        test_cases = {
            'zeros': np.zeros((obs_dim,), dtype=np.float32),
            'normal_low': np.random.normal(0, 0.1, (obs_dim,)).astype(np.float32),
            'normal_mid': np.random.normal(0, 0.5, (obs_dim,)).astype(np.float32),  
            'normal_high': np.random.normal(0, 1.0, (obs_dim,)).astype(np.float32),
            'positive': np.abs(np.random.normal(0, 0.5, (obs_dim,))).astype(np.float32),
            'negative': -np.abs(np.random.normal(0, 0.5, (obs_dim,))).astype(np.float32),
            'extreme_pos': np.ones((obs_dim,), dtype=np.float32) * 3.0,
            'extreme_neg': np.ones((obs_dim,), dtype=np.float32) * -3.0,
        }
        
        print("\n🧪 ANALISANDO ENTRY QUALITY PARA DIFERENTES INPUTS...")
        
        for test_name, base_obs in test_cases.items():
            print(f"\n📊 TESTE: {test_name.upper()}")
            print("-" * 40)
            
            # Coletar múltiplas amostras
            entry_decisions = []
            entry_qualities = []
            
            for i in range(n_samples):
                # Adicionar pequeno ruído para variabilidade
                if test_name != 'zeros':
                    obs = base_obs + np.random.normal(0, 0.01, obs_dim).astype(np.float32)
                else:
                    obs = base_obs.copy()
                
                # Predict em modo determinístico
                action, lstm_state = model.predict(obs, deterministic=True)
                
                entry_decisions.append(float(action[0]))
                entry_qualities.append(float(action[1]))
            
            # Análise estatística
            decisions_array = np.array(entry_decisions)
            qualities_array = np.array(entry_qualities)
            
            print(f"  Entry Decisions:")
            print(f"    Min/Mean/Max: {np.min(decisions_array):.3f}/{np.mean(decisions_array):.3f}/{np.max(decisions_array):.3f}")
            print(f"    Std: {np.std(decisions_array):.3f}")
            
            print(f"  Entry Qualities:")
            print(f"    Min/Mean/Max: {np.min(qualities_array):.3f}/{np.mean(qualities_array):.3f}/{np.max(qualities_array):.3f}")
            print(f"    Std: {np.std(qualities_array):.3f}")
            
            # Análise de zeros
            near_zero_qualities = np.sum(qualities_array < 0.001)
            near_one_qualities = np.sum(qualities_array > 0.999)
            
            print(f"  Análise de saturação:")
            print(f"    Near zero (<0.001): {near_zero_qualities}/{n_samples} ({near_zero_qualities/n_samples*100:.1f}%)")
            print(f"    Near one (>0.999): {near_one_qualities}/{n_samples} ({near_one_qualities/n_samples*100:.1f}%)")
            
            # Status
            if near_zero_qualities > n_samples * 0.8:
                print("  🔴 CRÍTICO: Entry Quality sempre próximo de ZERO")
            elif near_one_qualities > n_samples * 0.8:
                print("  🔥 CRÍTICO: Entry Quality sempre próximo de UM")
            elif np.std(qualities_array) < 0.01:
                print("  ❄️ PROBLEMA: Sem variabilidade na Entry Quality")
            else:
                print("  ✅ Entry Quality parece funcional")
        
        print("\n" + "=" * 60)
        print("🎯 TESTE STOCHASTIC vs DETERMINISTIC")
        print("=" * 60)
        
        # Comparar modo determinístico vs estocástico
        test_obs = np.random.normal(0, 0.3, (obs_dim,)).astype(np.float32)
        
        print("🎲 COMPARAÇÃO DE MODOS:")
        
        # Deterministic
        det_qualities = []
        for i in range(10):
            action, _ = model.predict(test_obs, deterministic=True)
            det_qualities.append(float(action[1]))
        
        # Stochastic  
        stoch_qualities = []
        for i in range(10):
            action, _ = model.predict(test_obs, deterministic=False)
            stoch_qualities.append(float(action[1]))
        
        print(f"  Deterministic mode:")
        print(f"    Entry Qualities: {det_qualities[:5]} ...")
        print(f"    Std: {np.std(det_qualities):.6f}")
        
        print(f"  Stochastic mode:")
        print(f"    Entry Qualities: {stoch_qualities[:5]} ...")
        print(f"    Std: {np.std(stoch_qualities):.6f}")
        
        if np.std(det_qualities) < 1e-6:
            print("  🔴 CONFIRMADO: Deterministic sempre retorna o mesmo valor")
        
        if np.std(stoch_qualities) < 1e-6:
            print("  🔴 CRÍTICO: Até modo stochastic sem variabilidade!")
        else:
            print("  ✅ Modo stochastic tem variabilidade normal")
        
        print("\n💡 DIAGNÓSTICO FINAL:")
        
        # Calcular estatísticas gerais
        all_qualities = []
        for test_name, base_obs in test_cases.items():
            for i in range(5):
                if test_name != 'zeros':
                    obs = base_obs + np.random.normal(0, 0.01, obs_dim).astype(np.float32)
                else:
                    obs = base_obs.copy()
                action, _ = model.predict(obs, deterministic=False)
                all_qualities.append(float(action[1]))
        
        all_qualities = np.array(all_qualities)
        overall_zeros = np.sum(all_qualities < 0.001)
        overall_ones = np.sum(all_qualities > 0.999)
        
        print(f"  Resumo geral ({len(all_qualities)} amostras):")
        print(f"    Mean: {np.mean(all_qualities):.3f}")
        print(f"    Std: {np.std(all_qualities):.3f}")
        print(f"    Zeros: {overall_zeros}/{len(all_qualities)} ({overall_zeros/len(all_qualities)*100:.1f}%)")
        print(f"    Ones: {overall_ones}/{len(all_qualities)} ({overall_ones/len(all_qualities)*100:.1f}%)")
        
        if overall_zeros > len(all_qualities) * 0.7:
            print("\n🎯 CONCLUSÃO: PROBLEMA CONFIRMADO!")
            print("  • Entry Quality está saturada para ZERO na maioria dos casos")
            print("  • As redes individuais que geram os scores estão saturadas")
            print("  • Solução: Remover sigmoid final das redes individuais")
        elif overall_ones > len(all_qualities) * 0.7:
            print("\n🎯 CONCLUSÃO: SATURAÇÃO PARA UM!")
            print("  • Entry Quality está saturada para UM na maioria dos casos")
            print("  • Thresholds muito baixos ou scores muito altos")
        else:
            print("\n🎯 CONCLUSÃO: Entry Quality funcionando")
            print("  • Problema pode estar em outro lugar")
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_scores_via_predict()