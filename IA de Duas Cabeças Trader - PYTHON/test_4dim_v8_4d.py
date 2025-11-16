#!/usr/bin/env python3
"""
🧪 TESTE RÁPIDO 4DIM.PY COM V8_4D
Verifica se o problema dos 100% zeros foi resolvido com a nova policy
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('.')

from trading_framework.policies.two_head_v8_4d import TwoHeadV8_4D, get_v8_4d_kwargs
from sb3_contrib import RecurrentPPO
import gym

def test_4dim_with_v8_4d():
    """Testa 4dim.py config com V8_4D"""
    
    print("🧪 TESTE 4DIM.PY COM V8_4D - VERIFICAÇÃO ZEROS")
    print("=" * 60)
    
    # Environment como no 4dim.py
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
    
    print("📋 Configuração exata do 4dim.py:")
    print(f"   Policy: TwoHeadV8_4D")
    print(f"   Obs Space: {obs_space.shape}")
    print(f"   Action Space: {action_space.shape}")
    
    # Configuração exata do 4dim.py
    model_config = {
        "policy": TwoHeadV8_4D,
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
            **get_v8_4d_kwargs(),
        }
    }
    
    print("\n🚀 Criando RecurrentPPO com V8_4D...")
    
    try:
        model = RecurrentPPO(**model_config)
        print("✅ RecurrentPPO criado com sucesso!")
        
        # Verificar pesos do transformer APÓS criação do modelo
        print("\n🔍 ANÁLISE DOS PESOS APÓS CRIAÇÃO DO MODELO:")
        
        transformer = model.policy.features_extractor
        critical_issues = 0
        
        for name, param in transformer.named_parameters():
            if 'weight' in name and 'projection' in name:
                zeros_pct = (param == 0).float().mean().item() * 100
                print(f"   {name}: {zeros_pct:.1f}% zeros")
                
                if zeros_pct > 50:
                    print(f"   🚨 PROBLEMA: {name} tem {zeros_pct:.1f}% zeros!")
                    critical_issues += 1
                elif zeros_pct > 10:
                    print(f"   ⚠️ ATENÇÃO: {name} tem {zeros_pct:.1f}% zeros")
                else:
                    print(f"   ✅ OK: {name} tem {zeros_pct:.1f}% zeros")
        
        print(f"\n📊 RESULTADO:")
        print(f"   Problemas críticos (>50% zeros): {critical_issues}")
        
        if critical_issues == 0:
            print("\n✅ SUCESSO! Problema dos 100% zeros RESOLVIDO com V8_4D!")
            print("🎉 4dim.py pode ser executado com segurança!")
            return True
        else:
            print("\n❌ PROBLEMA PERSISTE: Ainda há zeros críticos")
            return False
            
    except Exception as e:
        print(f"❌ ERRO ao criar RecurrentPPO: {e}")
        return False

if __name__ == "__main__":
    success = test_4dim_with_v8_4d()
    if success:
        print("\n🚀 PRONTO PARA TREINAR! Execute python 4dim.py")
    else:
        print("\n💀 Mais investigação necessária")