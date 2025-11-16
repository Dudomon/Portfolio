#!/usr/bin/env python3
"""
🔍 DEBUG ZEROS PROBLEM - Investigar problema das redes zeradas
"""

import sys
import os
sys.path.append("D:/Projeto")

import numpy as np
import torch
import gym
from stable_baselines3.common.env_util import make_vec_env

print("🔍 DEBUGGING ZEROS PROBLEM")
print("=" * 50)

try:
    # 1. Test basic torch setup
    print("1. 📊 TORCH SETUP:")
    x = torch.randn(10, 5)
    linear = torch.nn.Linear(5, 3)
    y = linear(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Weight mean: {linear.weight.mean().item():.6f}")
    print(f"   Output mean: {y.mean().item():.6f}")
    print(f"   ✅ PyTorch funcionando")
    
    # 2. Test environment creation
    print("\n2. 🌍 ENVIRONMENT:")
    
    # Simple test environment
    def make_test_env():
        return gym.make('CartPole-v1')
    
    test_env = make_vec_env(make_test_env, n_envs=1)
    print(f"   Obs space: {test_env.observation_space}")
    print(f"   Action space: {test_env.action_space}")
    print(f"   ✅ Environment funcionando")
    
    # 3. Test RecurrentPPO creation
    print("\n3. 🤖 RECURRENT PPO:")
    from sb3_contrib import RecurrentPPO
    
    # Simple model with default policy
    model = RecurrentPPO(
        'MlpLstmPolicy',
        test_env,
        learning_rate=1e-4,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        verbose=1,
        device='cuda'
    )
    
    print(f"   Model created: {type(model).__name__}")
    print(f"   Policy: {type(model.policy).__name__}")
    print(f"   Device: {model.device}")
    
    # Test prediction
    obs = test_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    print(f"   Action shape: {action.shape}")
    print(f"   Action: {action}")
    print(f"   ✅ RecurrentPPO funcionando")
    
    # 4. Test gradient flow
    print("\n4. 🌊 GRADIENT FLOW:")
    
    # Get policy parameters
    total_params = 0
    zero_params = 0
    grad_params = 0
    
    for name, param in model.policy.named_parameters():
        total_params += param.numel()
        if param.grad is not None:
            grad_params += param.numel()
            if torch.abs(param.grad).max() < 1e-8:
                zero_params += param.numel()
    
    print(f"   Total params: {total_params:,}")
    print(f"   Params with grad: {grad_params:,}")
    print(f"   Zero grad params: {zero_params:,}")
    
    if grad_params > 0:
        zero_ratio = zero_params / grad_params
        print(f"   Zero ratio: {zero_ratio:.2%}")
        if zero_ratio > 0.5:
            print("   ⚠️ PROBLEMA: Muitos gradientes zero!")
        else:
            print("   ✅ Gradientes OK")
    else:
        print("   ℹ️ Nenhum gradiente calculado ainda (normal)")
    
    # 5. Test learning step
    print("\n5. 📚 LEARNING STEP:")
    
    try:
        # Small learning step
        model.learn(total_timesteps=128, progress_bar=False)
        print("   ✅ Learning step concluído")
        
        # Check gradients after learning
        grad_params_after = 0
        zero_params_after = 0
        
        for name, param in model.policy.named_parameters():
            if param.grad is not None:
                grad_params_after += param.numel()
                if torch.abs(param.grad).max() < 1e-8:
                    zero_params_after += param.numel()
        
        if grad_params_after > 0:
            zero_ratio_after = zero_params_after / grad_params_after
            print(f"   Zero ratio após learning: {zero_ratio_after:.2%}")
            
            if zero_ratio_after > 0.8:
                print("   🔥 PROBLEMA CONFIRMADO: Gradientes zerados após learning!")
            else:
                print("   ✅ Gradientes normais após learning")
        
    except Exception as e:
        print(f"   ❌ Erro no learning: {e}")
    
    test_env.close()
    
except Exception as e:
    print(f"❌ ERRO GERAL: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("🔍 DEBUG CONCLUÍDO")