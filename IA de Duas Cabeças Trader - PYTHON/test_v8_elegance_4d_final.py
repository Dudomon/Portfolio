#!/usr/bin/env python3
"""
🧪 TESTE FINAL V8ELEGANCE_4D COM RECURRENTPPO
Verificação completa antes de executar 4dim.py
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('.')

from trading_framework.policies.two_head_v8_elegance_4d import TwoHeadV8Elegance_4D, get_v8_elegance_4d_kwargs
from sb3_contrib import RecurrentPPO
import gym

def test_final_v8_elegance_4d():
    """Teste final completo com RecurrentPPO"""
    
    print("🧪 TESTE FINAL V8ELEGANCE_4D COM RECURRENTPPO")
    print("=" * 60)
    
    # Environment exato do 4dim.py
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
    action_space = gym.spaces.Box(low=np.array([0, 0, -1, -1]), high=np.array([2, 1, 1, 1]), dtype=np.float32)
    
    # Dummy env completo
    class DummyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = obs_space
            self.action_space = action_space
            self.metadata = {'render.modes': []}
            
        def reset(self):
            return np.random.randn(450).astype(np.float32)
            
        def step(self, action):
            return np.random.randn(450).astype(np.float32), 0.0, False, {}
    
    env = DummyEnv()
    
    # Configuração EXATA do 4dim.py
    model_config = {
        "policy": TwoHeadV8Elegance_4D,
        "env": env,
        "learning_rate": 3e-5,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.12,
        "ent_coef": 0.02,
        "vf_coef": 0.5,
        "max_grad_norm": 0.1,
        "verbose": 1,
        "seed": 42,
        "use_sde": False,
        "policy_kwargs": {
            **get_v8_elegance_4d_kwargs(),
        }
    }
    
    print("📋 Testando configuração EXATA do 4dim.py:")
    print(f"   Policy: TwoHeadV8Elegance_4D")
    print(f"   Learning Rate: {model_config['learning_rate']}")
    print(f"   N Steps: {model_config['n_steps']}")
    print(f"   Batch Size: {model_config['batch_size']}")
    
    try:
        print("\n🚀 Criando RecurrentPPO...")
        model = RecurrentPPO(**model_config)
        print("✅ RecurrentPPO criado com SUCESSO!")
        
        # Verificar pesos do transformer
        print("\n🔍 VERIFICAÇÃO DE PESOS:")
        transformer = model.policy.features_extractor
        zeros_found = 0
        
        for name, param in transformer.named_parameters():
            if 'weight' in name and 'projection' in name:
                zeros_pct = (param == 0).float().mean().item() * 100
                print(f"   {name}: {zeros_pct:.1f}% zeros")
                if zeros_pct > 50:
                    zeros_found += 1
        
        # Teste de forward pass simples
        print("\n🏃 TESTE DE FORWARD PASS:")
        dummy_obs = np.random.randn(1, 450).astype(np.float32)
        
        try:
            with torch.no_grad():
                actions, _states = model.predict(dummy_obs, deterministic=True)
            
            print(f"   ✅ Forward pass OK!")
            print(f"   Actions shape: {actions.shape}")
            print(f"   Actions: {actions}")
            print(f"   Action ranges: [{actions.min():.3f}, {actions.max():.3f}]")
            
            # Validar 4D action space
            if actions.shape[-1] == 4:
                print("   ✅ 4D Action Space VALIDADO!")
                entry_decision = actions[0, 0]  # [batch, action_dim]
                confidence = actions[0, 1] 
                pos1_mgmt = actions[0, 2]
                pos2_mgmt = actions[0, 3]
                
                print(f"   Entry Decision: {entry_decision:.3f} (esperado: 0-2)")
                print(f"   Confidence: {confidence:.3f} (esperado: 0-1)")
                print(f"   Pos1 Mgmt: {pos1_mgmt:.3f} (esperado: -1 a 1)")
                print(f"   Pos2 Mgmt: {pos2_mgmt:.3f} (esperado: -1 a 1)")
                
                forward_ok = True
            else:
                print(f"   ❌ Action shape incorreta: {actions.shape}")
                forward_ok = False
                
        except Exception as e:
            print(f"   ❌ Forward pass FALHOU: {e}")
            forward_ok = False
        
        # Verificar arquitetura específica
        print("\n🏗️ VERIFICAÇÃO DE ARQUITETURA:")
        print(f"   LSTM Hidden: {model.policy.v8_lstm_hidden}D")
        print(f"   Features Dim: {model.policy.v8_features_dim}D") 
        print(f"   Context Dim: {model.policy.v8_context_dim}D")
        print(f"   Entry Head: {type(model.policy.entry_head).__name__}")
        print(f"   Management Head: {type(model.policy.management_head).__name__}")
        print(f"   Market Context: {type(model.policy.market_context_encoder).__name__}")
        
        # Resultado final
        print(f"\n📊 RESULTADO FINAL:")
        print(f"   RecurrentPPO criado: ✅")
        print(f"   Zeros críticos: {zeros_found} (deve ser 0)")
        print(f"   Forward pass: {'✅' if forward_ok else '❌'}")
        print(f"   4D Action Space: {'✅' if forward_ok else '❌'}")
        print(f"   V8 Elegance 4D: ✅")
        
        success = zeros_found == 0 and forward_ok
        
        if success:
            print("\n🎉 SUCESSO TOTAL! V8Elegance_4D está PRONTA!")
            print("🚀 4dim.py pode ser executado com segurança!")
            return True
        else:
            print("\n💀 Ainda há problemas a resolver")
            return False
            
    except Exception as e:
        print(f"❌ ERRO ao criar RecurrentPPO: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_final_v8_elegance_4d()
    if success:
        print("\n✅ PRONTO! Execute: python 4dim.py")
    else:
        print("\n❌ Precisa de mais correções")