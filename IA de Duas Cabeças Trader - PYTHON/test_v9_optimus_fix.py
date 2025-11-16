#!/usr/bin/env python3
"""
🧪 TESTE ESPECÍFICO PARA V9OPTIMUS APÓS CORREÇÃO
Verifica se o problema dos 100% zeros foi resolvido
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('.')

from trading_framework.policies.two_head_v9_optimus import TwoHeadV9Optimus, get_v9_optimus_kwargs
import gym

def test_v9_optimus_full_initialization():
    """Testa inicialização completa da V9Optimus como no 4dim.py"""
    
    print("🧪 TESTE V9OPTIMUS - VERIFICAÇÃO PÓS-CORREÇÃO")
    print("=" * 60)
    
    # Ambiente 450D como no 4dim.py
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
    action_space = gym.spaces.Box(low=np.array([0, 0, -1, -1]), high=np.array([2, 1, 1, 1]), dtype=np.float32)
    
    def dummy_lr_schedule(progress):
        return 3e-5  # Mesmo LR do 4dim.py
    
    print("📋 Configuração:")
    print(f"   Obs Space: {obs_space.shape}")
    print(f"   Action Space: {action_space.shape}")
    
    # Obter kwargs exatos do 4dim.py
    kwargs = get_v9_optimus_kwargs()
    print(f"   ortho_init: {kwargs.get('ortho_init', 'NOT_SET')}")
    print(f"   features_extractor: {kwargs['features_extractor_class'].__name__}")
    
    print("\n🚀 Criando TwoHeadV9Optimus...")
    
    # Criar policy exatamente como no 4dim.py
    policy = TwoHeadV9Optimus(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=dummy_lr_schedule,
        **kwargs
    )
    
    print("\n🔍 ANÁLISE DETALHADA DOS PESOS APÓS INICIALIZAÇÃO COMPLETA:")
    
    # Verificar transformer
    transformer = policy.features_extractor
    critical_weights = {}
    
    print("\n📊 TRANSFORMER WEIGHTS:")
    for name, param in transformer.named_parameters():
        if 'weight' in name and 'projection' in name:
            zeros_pct = (param == 0).float().mean().item() * 100
            critical_weights[name] = zeros_pct
            print(f"   {name}: {zeros_pct:.1f}% zeros")
            
            if zeros_pct > 50:
                print(f"   🚨 CRÍTICO: {name} tem {zeros_pct:.1f}% zeros!")
    
    # Verificar outros componentes críticos
    print("\n📊 POLICY WEIGHTS:")
    for name, param in policy.named_parameters():
        if 'weight' in name and any(key in name for key in ['actor', 'critic', 'lstm']):
            zeros_pct = (param == 0).float().mean().item() * 100
            if zeros_pct > 0:  # Só mostrar se houver zeros
                print(f"   {name}: {zeros_pct:.1f}% zeros")
    
    # Teste funcional básico
    print("\n🏃 TESTE FUNCIONAL:")
    try:
        dummy_obs = torch.FloatTensor(np.random.randn(1, 450))
        dummy_lstm_states = (
            torch.zeros(1, 1, kwargs['lstm_hidden_size']),
            torch.zeros(1, 1, kwargs['lstm_hidden_size'])
        )
        dummy_episode_starts = torch.ones(1, dtype=torch.bool)
        
        with torch.no_grad():
            actions, values, log_probs = policy(dummy_obs, dummy_lstm_states, dummy_episode_starts)
        
        print(f"   ✅ Forward pass OK")
        print(f"   Actions shape: {actions.shape}")
        print(f"   Values shape: {values.shape}")
        print(f"   Action range: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
        
        functional_ok = True
        
    except Exception as e:
        print(f"   ❌ Forward pass FALHOU: {e}")
        functional_ok = False
    
    # Avaliar resultado
    critical_problems = sum(1 for pct in critical_weights.values() if pct > 50)
    
    print(f"\n📊 RESUMO:")
    print(f"   Componentes críticos: {len(critical_weights)}")
    print(f"   Problemas críticos (>50% zeros): {critical_problems}")
    print(f"   Teste funcional: {'PASSOU' if functional_ok else 'FALHOU'}")
    
    if critical_problems == 0 and functional_ok:
        print("\n✅ SUCESSO: V9Optimus funciona corretamente após correção!")
        return True
    else:
        print("\n❌ PROBLEMA: Ainda há issues na V9Optimus")
        return False

if __name__ == "__main__":
    success = test_v9_optimus_full_initialization()
    if success:
        print("\n🎉 CORREÇÃO BEM-SUCEDIDA! V9Optimus pronta para uso no 4dim.py")
    else:
        print("\n💀 Mais investigação necessária")