#!/usr/bin/env python3
"""
🔍 DEBUG THRESHOLDS V7 - Verificar valores dos thresholds adaptativos aprendidos
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

def extract_threshold_values():
    """🔍 Extrair e analisar valores dos thresholds adaptativos"""
    
    print("🔍 DEBUG THRESHOLDS V7 - ANÁLISE COMPLETA")
    print("=" * 60)
    
    try:
        # Carregar modelo
        print(f"🤖 Carregando modelo: {CHECKPOINT_NAME}")
        model = RecurrentPPO.load(CHECKPOINT_PATH, device='cuda')
        
        # Acessar policy
        policy = model.policy
        
        # Encontrar os thresholds adaptativos
        threshold_params = {}
        
        print("\n📊 BUSCANDO THRESHOLDS ADAPTATIVOS...")
        for name, param in policy.named_parameters():
            if 'threshold' in name.lower():
                threshold_params[name] = param.data.item()
                print(f"  ✅ Encontrado: {name} = {param.data.item():.6f}")
        
        if not threshold_params:
            print("⚠️ Nenhum threshold adaptativo encontrado nos parâmetros!")
            print("\n🔍 LISTANDO TODOS OS PARÂMETROS PARA DEBUG:")
            for name, param in policy.named_parameters():
                if param.numel() == 1:  # Parâmetros escalares
                    print(f"  {name}: {param.data.item():.6f}")
            return
        
        print("\n" + "=" * 60)
        print("📊 ANÁLISE DOS THRESHOLDS")
        print("=" * 60)
        
        # Análise detalhada de cada threshold
        for name, value in threshold_params.items():
            print(f"\n🎯 {name.upper()}:")
            print(f"  Valor atual: {value:.6f}")
            
            # Determinar ranges após clamp baseado no nome
            if 'main' in name.lower():
                clamp_min, clamp_max = 0.1, 0.6
                initial_value = 0.25
            elif 'risk' in name.lower():
                clamp_min, clamp_max = 0.05, 0.5
                initial_value = 0.15
            elif 'regime' in name.lower():
                clamp_min, clamp_max = 0.02, 0.4
                initial_value = 0.10
            else:
                clamp_min, clamp_max = 0.0, 1.0
                initial_value = 0.5
            
            # Valor após clamp
            clamped_value = max(clamp_min, min(clamp_max, value))
            print(f"  Valor inicial: {initial_value:.6f}")
            print(f"  Range permitido: [{clamp_min:.3f}, {clamp_max:.3f}]")
            print(f"  Valor após clamp: {clamped_value:.6f}")
            
            # Análise de impacto
            if clamped_value > initial_value * 1.5:
                print("  🚨 THRESHOLD MUITO ALTO - pode estar bloqueando gates!")
            elif clamped_value < initial_value * 0.5:
                print("  🟢 Threshold baixo - favorece ativação")
            else:
                print("  🟡 Threshold em range normal")
        
        print("\n" + "=" * 60)
        print("🧪 SIMULAÇÃO DE GATES")
        print("=" * 60)
        
        # Simular diferentes cenários de scores vs thresholds
        print("\n🎲 CENÁRIOS DE TESTE (score - threshold) * 2.0:")
        
        test_scores = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        for name, threshold_val in threshold_params.items():
            if 'main' in name.lower():
                clamp_min, clamp_max = 0.1, 0.6
            elif 'risk' in name.lower():
                clamp_min, clamp_max = 0.05, 0.5
            elif 'regime' in name.lower():
                clamp_min, clamp_max = 0.02, 0.4
            else:
                continue
                
            clamped_threshold = max(clamp_min, min(clamp_max, threshold_val))
            
            print(f"\n📊 {name.upper()} (threshold = {clamped_threshold:.3f}):")
            print("  Score → Gate Value")
            
            activations_found = 0
            for score in test_scores:
                gate_input = (score - clamped_threshold) * 2.0
                gate_output = torch.sigmoid(torch.tensor(gate_input)).item()
                
                status = "🔥" if gate_output > 0.5 else "❄️"
                print(f"  {score:.1f} → {gate_output:.3f} {status}")
                
                if gate_output > 0.1:  # Conta ativações significativas
                    activations_found += 1
            
            print(f"  📈 Ativações significativas: {activations_found}/{len(test_scores)}")
            if activations_found < 3:
                print("  🚨 CRÍTICO: Quase impossível ativar este gate!")
        
        print("\n" + "=" * 60)
        print("💡 RECOMENDAÇÕES")
        print("=" * 60)
        
        # Gerar recomendações
        critical_thresholds = []
        
        for name, value in threshold_params.items():
            if 'main' in name.lower() and value > 0.4:
                critical_thresholds.append(f"{name}: {value:.3f} (muito alto)")
            elif 'risk' in name.lower() and value > 0.3:
                critical_thresholds.append(f"{name}: {value:.3f} (muito alto)")
            elif 'regime' in name.lower() and value > 0.25:
                critical_thresholds.append(f"{name}: {value:.3f} (muito alto)")
        
        if critical_thresholds:
            print("🚨 THRESHOLDS PROBLEMÁTICOS:")
            for thresh in critical_thresholds:
                print(f"  • {thresh}")
            print("\n💊 SOLUÇÕES:")
            print("  1. Resetar thresholds para valores iniciais mais baixos")
            print("  2. Implementar curriculum learning (thresholds graduais)")
            print("  3. Adicionar regularização nos thresholds")
        else:
            print("✅ Thresholds em ranges aceitáveis")
            print("🔍 Problema pode estar nas sigmoid das redes individuais")
        
        # Estatísticas finais
        avg_threshold = np.mean(list(threshold_params.values()))
        print(f"\n📊 ESTATÍSTICAS FINAIS:")
        print(f"  Threshold médio: {avg_threshold:.3f}")
        print(f"  Thresholds encontrados: {len(threshold_params)}")
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_threshold_values()