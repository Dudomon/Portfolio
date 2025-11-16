#!/usr/bin/env python3
"""
Verificar se as mudanças no critic foram realmente aplicadas
"""

import torch
from trading_framework.policies.two_head_v7_simple import TwoHeadV7Simple, get_v7_kwargs
from trading_framework.extractors.transformer_extractor import TradingTransformerFeatureExtractor
import gym
from gym import spaces
import numpy as np

def verify_architecture():
    """Verifica arquitetura atual do critic"""
    print("VERIFICANDO ARQUITETURA DO CRITIC")
    print("="*50)
    
    # Criar policy
    obs_dim = 100
    action_dim = 3
    
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    action_space = spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
    
    v7_kwargs = get_v7_kwargs()
    
    policy = TwoHeadV7Simple(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
        **v7_kwargs
    )
    
    print(f"1. MODULES NA POLICY:")
    for name, module in policy.named_modules():
        if 'critic' in name.lower():
            print(f"   {name}: {type(module).__name__}")
    
    print(f"\n2. VERIFICANDO SE MLP EXISTE:")
    if hasattr(policy, 'v7_critic_mlp'):
        print(f"   ✅ v7_critic_mlp EXISTS")
        print(f"   Architecture: {policy.v7_critic_mlp}")
    else:
        print(f"   ❌ v7_critic_mlp NOT FOUND")
    
    print(f"\n3. VERIFICANDO MEMORY BUFFER:")
    if hasattr(policy, 'memory_steps'):
        print(f"   ✅ memory_steps: {policy.memory_steps}")
    else:
        print(f"   ❌ memory_steps NOT FOUND")
    
    if hasattr(policy, 'critic_memory_buffer'):
        print(f"   ✅ critic_memory_buffer: {policy.critic_memory_buffer}")
    else:
        print(f"   ❌ critic_memory_buffer NOT FOUND")
    
    print(f"\n4. TESTANDO FORWARD CRITIC:")
    batch_size = 2
    features = torch.randn(batch_size, 256)
    lstm_states = (
        torch.zeros(1, batch_size, 256),
        torch.zeros(1, batch_size, 256),
        torch.zeros(1, batch_size, 256), 
        torch.zeros(1, batch_size, 256)
    )
    episode_starts = torch.zeros(batch_size)
    
    try:
        values, new_states = policy.forward_critic(features, lstm_states, episode_starts)
        print(f"   ✅ Forward critic SUCCESS")
        print(f"   Values shape: {values.shape}")
        print(f"   Memory buffer shape after: {policy.critic_memory_buffer.shape}")
        
    except Exception as e:
        print(f"   ❌ Forward critic FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n5. VERIFICANDO PARAMETROS:")
    total_params = 0
    mlp_params = 0
    lstm_params = 0
    
    for name, param in policy.named_parameters():
        total_params += param.numel()
        if 'v7_critic_mlp' in name:
            mlp_params += param.numel()
            print(f"   MLP: {name} - {param.shape}")
        elif 'lstm_critic' in name:
            lstm_params += param.numel()
            print(f"   LSTM: {name} - {param.shape}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total params: {total_params:,}")
    print(f"   MLP params: {mlp_params:,}")
    print(f"   LSTM params: {lstm_params:,}")
    
    if mlp_params > 0:
        print(f"   ✅ MLP CRITIC IS ACTIVE!")
    else:
        print(f"   ❌ MLP CRITIC NOT FOUND!")

if __name__ == "__main__":
    verify_architecture()