#!/usr/bin/env python3
"""
🧪 TESTE RÁPIDO DE TREINAMENTO
Teste alguns steps de treinamento para verificar se o collect_rollouts funciona
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('.')

from trading_framework.policies.two_head_v8_elegance_4d import TwoHeadV8Elegance_4D, get_v8_elegance_4d_kwargs
from sb3_contrib import RecurrentPPO
import gym

def test_training_steps():
    """Teste de alguns steps de treinamento"""
    
    print("🧪 TESTE RÁPIDO DE TREINAMENTO")
    print("=" * 50)
    
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
            self.step_count = 0
            
        def reset(self):
            self.step_count = 0
            return np.random.randn(450).astype(np.float32)
            
        def step(self, action):
            self.step_count += 1
            done = self.step_count >= 100  # Episode curto
            reward = np.random.randn() * 0.1  # Reward aleatório pequeno
            return np.random.randn(450).astype(np.float32), reward, done, {}
    
    env = DummyEnv()
    
    # Configuração mínima para teste
    model_config = {
        "policy": TwoHeadV8Elegance_4D,
        "env": env,
        "learning_rate": 3e-5,
        "n_steps": 128,  # Poucos steps para teste rápido
        "batch_size": 32,
        "n_epochs": 2,
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
    
    try:
        print("🚀 Criando RecurrentPPO...")
        model = RecurrentPPO(**model_config)
        print("✅ RecurrentPPO criado!")
        
        print("\n🏃 TESTANDO TREINAMENTO (256 steps)...")
        model.learn(total_timesteps=256, progress_bar=True)
        print("✅ Treinamento funcionou!")
        
        print("\n🧪 TESTE DE PREDICT APÓS TREINAMENTO:")
        dummy_obs = np.random.randn(1, 450).astype(np.float32)
        actions, _states = model.predict(dummy_obs, deterministic=True)
        print(f"   Actions: {actions}")
        print(f"   Action ranges: [{actions.min():.3f}, {actions.max():.3f}]")
        
        print("\n🎉 SUCESSO TOTAL! Treinamento funcionando corretamente!")
        return True
        
    except Exception as e:
        print(f"❌ ERRO durante treinamento: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_steps()
    if success:
        print("\n✅ TREINAMENTO OK! 4dim.py pode ser executado.")
    else:
        print("\n❌ Problema no treinamento.")