#!/usr/bin/env python3
"""
🔍 DEBUG ACTION SPACE SAC
Verificar se o SAC está gerando ações corretamente para SHORT
"""

import torch
import numpy as np
import sys
sys.path.append('.')

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import gym

# Mock environment simples
class DebugEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0, -1, -1]), 
            high=np.array([2, 1, 1, 1]), 
            dtype=np.float32
        )
        self.step_count = 0
        
    def reset(self):
        return np.random.randn(450).astype(np.float32) * 0.1
        
    def step(self, action):
        self.step_count += 1
        reward = np.random.randn() * 0.1
        done = self.step_count > 100
        return np.random.randn(450).astype(np.float32) * 0.1, reward, done, {}

def test_sac_action_distribution():
    """Testar se SAC consegue gerar ações SHORT (>=0.67)"""
    
    print("🔍 DEBUG SAC ACTION SPACE")
    print("=" * 50)
    
    # Criar environment 
    env = DebugEnv()
    
    print(f"Action Space: {env.action_space}")
    print(f"Action[0] range: [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")
    print(f"Threshold SHORT: >= 0.67")
    
    # Configuração SAC identica ao sacversion.py
    config = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": 1e-3,
        "buffer_size": 1000,  # Pequeno para teste
        "batch_size": 64,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto_1.0",
        "target_update_interval": 1,
        "verbose": 1,
        "device": "cpu",  # CPU para debug rapido
        "policy_kwargs": {
            "net_arch": dict(pi=[512, 256, 128], qf=[512, 256, 128]),
            "activation_fn": torch.nn.LeakyReLU,
            "log_std_init": -0.5,
        }
    }
    
    print("\\n🚀 Criando modelo SAC...")
    model = SAC(**config)
    
    print("\\n🎯 Testando distribuição de ações ANTES do treinamento:")
    actions_count = {"HOLD": 0, "LONG": 0, "SHORT": 0}
    actions_raw = []
    
    obs = env.reset()
    for i in range(1000):  # 1000 samples
        action, _ = model.predict(obs, deterministic=False)
        raw_decision = float(action[0])
        actions_raw.append(raw_decision)
        
        # Classificar ação usando mesmos thresholds do sacversion.py
        if raw_decision < 0.33:
            actions_count["HOLD"] += 1
        elif raw_decision < 0.67:
            actions_count["LONG"] += 1
        else:
            actions_count["SHORT"] += 1
    
    # Estatísticas
    total = sum(actions_count.values())
    print(f"📊 Distribuição de 1000 ações:")
    for action_type, count in actions_count.items():
        pct = (count / total) * 100
        print(f"   {action_type}: {count} ({pct:.1f}%)")
    
    print(f"\\n📊 Estatísticas action[0]:")
    actions_raw = np.array(actions_raw)
    print(f"   Min: {actions_raw.min():.3f}")
    print(f"   Max: {actions_raw.max():.3f}")
    print(f"   Mean: {actions_raw.mean():.3f}")
    print(f"   Std: {actions_raw.std():.3f}")
    
    # Verificar se consegue atingir threshold SHORT
    short_capable = (actions_raw >= 0.67).sum()
    print(f"   Actions >= 0.67 (SHORT threshold): {short_capable} ({short_capable/len(actions_raw)*100:.1f}%)")
    
    # Treinar brevemente e testar novamente
    print("\\n🏃 Treinamento rápido (200 steps)...")
    model.learn(total_timesteps=200)
    
    print("\\n🎯 Testando distribuição DEPOIS do treinamento:")
    actions_count_after = {"HOLD": 0, "LONG": 0, "SHORT": 0}
    actions_raw_after = []
    
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=False)
        raw_decision = float(action[0])
        actions_raw_after.append(raw_decision)
        
        if raw_decision < 0.33:
            actions_count_after["HOLD"] += 1
        elif raw_decision < 0.67:
            actions_count_after["LONG"] += 1
        else:
            actions_count_after["SHORT"] += 1
    
    total_after = sum(actions_count_after.values())
    print(f"📊 Distribuição de 1000 ações APÓS treinamento:")
    for action_type, count in actions_count_after.items():
        pct = (count / total_after) * 100
        print(f"   {action_type}: {count} ({pct:.1f}%)")
    
    actions_raw_after = np.array(actions_raw_after)
    short_capable_after = (actions_raw_after >= 0.67).sum()
    print(f"   Actions >= 0.67 (SHORT): {short_capable_after} ({short_capable_after/len(actions_raw_after)*100:.1f}%)")
    
    # Diagnóstico
    if actions_count_after["SHORT"] == 0:
        print("\\n❌ PROBLEMA: SAC não consegue gerar ações SHORT!")
        print("   Possíveis causas:")
        print("   1. Action space scaling incorreto")
        print("   2. Entropy coefficient muito baixo")
        print("   3. Policy initialization favorece centro")
        print(f"   4. Max action[0] = {actions_raw_after.max():.3f} < 0.67")
    else:
        print("\\n✅ SUCCESS: SAC consegue gerar ações SHORT!")

if __name__ == "__main__":
    test_sac_action_distribution()