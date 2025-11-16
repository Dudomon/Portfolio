#!/usr/bin/env python3
"""
🔍 REPRODUZIR PROBLEMA 100% ZEROS DURANTE TREINAMENTO
Teste mínimo para identificar quando/onde os pesos são zerados
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('.')

from trading_framework.extractors.transformer_v9_compact import TradingTransformerV9Compact
from trading_framework.policies.two_head_v9_optimus import TwoHeadV9Optimus, get_v9_optimus_kwargs
import gym

# Simular ambiente mínimo
class MinimalEnv:
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    
    def reset(self):
        return np.random.randn(450).astype(np.float32)
    
    def step(self, action):
        obs = np.random.randn(450).astype(np.float32)
        reward = np.random.randn()
        done = False
        info = {}
        return obs, reward, done, info

def check_transformer_weights(transformer, step_name):
    """Verifica pesos do transformer"""
    print(f"\n=== {step_name} ===")
    
    critical_layers = [
        'temporal_projection.weight',
        'final_projection.0.weight'
    ]
    
    # Também verificar bias mas não falhar por causa deles (bias=0 é normal)
    bias_layers = [
        'temporal_projection.bias', 
        'final_projection.0.bias'
    ]
    
    problem_found = False
    
    for layer_name in critical_layers:
        param = transformer
        for attr in layer_name.split('.'):
            param = getattr(param, attr)
        
        zeros_pct = (param == 0).float().mean().item() * 100
        print(f"  {layer_name}: {zeros_pct:.1f}% zeros")
        
        if zeros_pct > 50:
            print(f"  🚨 PROBLEMA CRÍTICO: {layer_name} tem {zeros_pct:.1f}% zeros!")
            problem_found = True
    
    # Verificar bias mas apenas informar
    for layer_name in bias_layers:
        param = transformer
        for attr in layer_name.split('.'):
            param = getattr(param, attr)
        
        zeros_pct = (param == 0).float().mean().item() * 100
        print(f"  {layer_name}: {zeros_pct:.1f}% zeros (bias - normal se 100%)")
    
    return not problem_found

def test_transformer_during_simulated_training():
    """Testa transformer durante 'treinamento' simulado"""
    
    print("🔍 REPRODUZINDO PROBLEMA 100% ZEROS DURANTE TREINAMENTO")
    print("=" * 60)
    
    # Criar environment
    env = MinimalEnv()
    
    def dummy_lr_schedule(progress):
        return 1e-4
    
    # Criar policy V9Optimus
    kwargs = get_v9_optimus_kwargs()
    print(f"ortho_init: {kwargs.get('ortho_init', 'NOT_SET')}")
    
    policy = TwoHeadV9Optimus(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=dummy_lr_schedule,
        **kwargs
    )
    
    transformer = policy.features_extractor
    
    # Check inicial
    if not check_transformer_weights(transformer, "APÓS INICIALIZAÇÃO"):
        return False
    
    # Simular PPO build com orthogonal init
    print("\n🔧 SIMULANDO SB3 MODULE.APPLY() COM ORTHOGONAL INIT...")
    
    def sb3_init_weights(module, gain=1.414):  # sqrt(2)
        """Simula o que SB3 faz em ActorCriticPolicy._build()"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            print(f"    SB3 aplicando orthogonal init em: {type(module).__name__}")
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    
    # APLICAR COMO FEATURES_EXTRACTOR (com sqrt(2) gain)
    from functools import partial
    transformer.apply(partial(sb3_init_weights, gain=np.sqrt(2)))
    
    if not check_transformer_weights(transformer, "APÓS SB3 ORTHOGONAL INIT"):
        return False
    
    # Simular optimizer steps simples (sem forward complexo)
    print("\n🏃 SIMULANDO OPTIMIZER STEPS...")
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    
    for step in range(5):
        # Simular gradientes fake nos parâmetros do transformer
        for param in transformer.parameters():
            if param.requires_grad:
                param.grad = torch.randn_like(param) * 0.001  # Gradientes pequenos
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"  Step {step + 1}: Optimizer step concluído")
        
        # Check a cada step
        if not check_transformer_weights(transformer, f"APÓS OPTIMIZER STEP {step + 1}"):
            print(f"🚨 PROBLEMA DETECTADO NO STEP {step + 1}!")
            return False
    
    print("\n✅ TREINAMENTO SIMULADO CONCLUÍDO SEM PROBLEMAS!")
    return True

if __name__ == "__main__":
    success = test_transformer_during_simulated_training()
    if success:
        print("\n✅ TESTE PASSOU - Transformer manteve pesos corretos")
    else:
        print("\n❌ TESTE FALHOU - Problema com zeros detectado")