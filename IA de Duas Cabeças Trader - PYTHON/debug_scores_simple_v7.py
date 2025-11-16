#!/usr/bin/env python3
"""
🔍 DEBUG SCORES SIMPLES V7 - Análise direta via forward do modelo
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

def debug_scores_via_forward():
    """🔍 Analisar scores via forward normal do modelo"""
    
    print("🔍 DEBUG SCORES VIA FORWARD V7")
    print("=" * 60)
    
    try:
        # Carregar modelo
        print(f"🤖 Carregando modelo...")
        model = RecurrentPPO.load(CHECKPOINT_PATH, device='cuda')
        
        # Gerar observação realista
        obs_dim = 2580  # Observation space do DayTrader
        batch_size = 10
        
        print("📊 Gerando observações de teste...")
        
        # Diferentes tipos de input
        test_cases = {
            'zeros': torch.zeros(batch_size, obs_dim, device='cuda'),
            'normal': torch.randn(batch_size, obs_dim, device='cuda') * 0.3,
            'positive': torch.abs(torch.randn(batch_size, obs_dim, device='cuda')) * 0.5,
            'negative': -torch.abs(torch.randn(batch_size, obs_dim, device='cuda')) * 0.5,
        }
        
        print("\n🧪 ANALISANDO SCORES ATRAVÉS DO FORWARD...")
        
        for test_name, obs in test_cases.items():
            print(f"\n📊 TESTE: {test_name.upper()}")
            print("-" * 40)
            
            with torch.no_grad():
                # Fazer forward normal
                actions, values, log_probs = model.policy(obs)
                
                print(f"  Actions shape: {actions.shape}")
                print(f"  Values shape: {values.shape}")
                
                # Analisar dimensões das actions
                actions_cpu = actions.cpu().numpy()
                
                # V7 tem 11 ações: [entry_decision, entry_quality, sl_type, tp_type, trailing_stop, ...]
                entry_decisions = actions_cpu[:, 0]  # Primeira dimensão
                entry_qualities = actions_cpu[:, 1]  # Segunda dimensão
                
                print(f"  Entry Decisions: min={np.min(entry_decisions):.3f}, mean={np.mean(entry_decisions):.3f}, max={np.max(entry_decisions):.3f}")
                print(f"  Entry Qualities: min={np.min(entry_qualities):.3f}, mean={np.mean(entry_qualities):.3f}, max={np.max(entry_qualities):.3f}")
                
                # Verificar se entry quality está sempre 0
                zero_qualities = np.sum(entry_qualities < 0.001)
                print(f"  Entry Qualities near zero: {zero_qualities}/{batch_size} ({zero_qualities/batch_size*100:.1f}%)")
                
                if zero_qualities > batch_size * 0.8:
                    print("  🚨 PROBLEMA: Entry Quality quase sempre zero!")
                elif np.std(entry_qualities) < 0.01:
                    print("  ⚠️ PROBLEMA: Entry Quality sem variância!")
                else:
                    print("  ✅ Entry Quality parece normal")
        
        print("\n" + "=" * 60)
        print("🎯 TESTE DETALHADO COM HOOK")
        print("=" * 60)
        
        # Hook para capturar valores internos
        gate_values = {}
        
        def hook_entry_head(module, input, output):
            # Tentar capturar valores dos gates se disponível
            if hasattr(module, 'gate_info') and module.gate_info is not None:
                gate_values['last_gates'] = module.gate_info
        
        # Registrar hook (se possível)
        try:
            hook_handle = model.policy.entry_head.register_forward_hook(hook_entry_head)
        except:
            print("⚠️ Não foi possível registrar hook")
            hook_handle = None
        
        # Teste com observação normal
        test_obs = torch.randn(1, obs_dim, device='cuda') * 0.3
        
        with torch.no_grad():
            actions, values, log_probs = model.policy(test_obs)
            
            print(f"🎯 SINGLE TEST RESULT:")
            actions_single = actions.cpu().numpy()[0]
            print(f"  Entry Decision: {actions_single[0]:.3f}")
            print(f"  Entry Quality: {actions_single[1]:.3f}")
            
            if 'last_gates' in gate_values:
                gates_info = gate_values['last_gates']
                print(f"  Gate Info disponível: {type(gates_info)}")
                
                if isinstance(gates_info, dict):
                    for key, value in gates_info.items():
                        if torch.is_tensor(value):
                            print(f"  {key}: {value.cpu().numpy().flatten()}")
            else:
                print("  Gate info não capturado")
        
        if hook_handle:
            hook_handle.remove()
        
        print("\n💡 PRÓXIMOS PASSOS:")
        if zero_qualities > batch_size * 0.8:
            print("  1. ✅ CONFIRMADO: Entry Quality está sempre próximo de zero")
            print("  2. 🔍 Problema está nas redes individuais que geram os scores")
            print("  3. 🛠️ Solução: Remover sigmoid das redes individuais ou usar inicialização diferente")
        else:
            print("  1. Entry Quality parece estar funcionando")
            print("  2. Problema pode estar em outro lugar")
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_scores_via_forward()