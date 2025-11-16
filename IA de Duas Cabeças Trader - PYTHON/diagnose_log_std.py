#!/usr/bin/env python3
"""
🔍 DIAGNÓSTICO COMPLETO: Por que log_std está 100% zero?
"""

import torch
import torch.nn as nn
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.distributions import DiagGaussianDistribution

def check_log_std(model, source=""):
    """Verificar estado do log_std"""
    print(f"\n{'='*60}")
    print(f"🔍 CHECKING LOG_STD {source}")
    print("="*60)
    
    # 1. Procurar log_std nos parâmetros
    for name, param in model.policy.named_parameters():
        if 'log_std' in name.lower():
            data = param.data.cpu().numpy()
            zeros = np.sum(np.abs(data) < 1e-6)
            total = data.size
            print(f"📊 Parameter {name}:")
            print(f"   Shape: {param.shape}")
            print(f"   Values: {param.data[:5]}...")  # Primeiros 5 valores
            print(f"   Zeros: {zeros}/{total} ({100*zeros/total:.1f}%)")
            print(f"   Requires grad: {param.requires_grad}")
            print(f"   Grad: {param.grad}")
            return param
    
    # 2. Verificar action_dist
    if hasattr(model.policy, 'action_dist'):
        dist = model.policy.action_dist
        print(f"📊 action_dist type: {type(dist)}")
        
        if hasattr(dist, 'log_std'):
            log_std = dist.log_std
            data = log_std.data.cpu().numpy()
            zeros = np.sum(np.abs(data) < 1e-6)
            total = data.size
            print(f"📊 action_dist.log_std:")
            print(f"   Shape: {log_std.shape}")
            print(f"   Values: {log_std.data[:5]}...")
            print(f"   Zeros: {zeros}/{total} ({100*zeros/total:.1f}%)")
            print(f"   Requires grad: {log_std.requires_grad}")
            print(f"   Grad: {log_std.grad}")
            return log_std
    
    # 3. Se não encontrou
    print("❌ log_std não encontrado!")
    return None

def test_distribution_creation():
    """Testar criação de distribuição isoladamente"""
    print("\n" + "="*60)
    print("🧪 TESTE: Criação de DiagGaussianDistribution")
    print("="*60)
    
    # Criar distribuição
    dist = DiagGaussianDistribution(11)
    action_net, log_std = dist.proba_distribution_net(latent_dim=256, log_std_init=-0.5)
    
    print(f"✅ log_std criado:")
    print(f"   Type: {type(log_std)}")
    print(f"   Shape: {log_std.shape}")
    print(f"   Values: {log_std.data}")
    print(f"   Requires grad: {log_std.requires_grad}")
    
    # Testar forward
    test_input = torch.randn(1, 256)
    mean = action_net(test_input)
    dist.proba_distribution(mean, log_std)
    
    print(f"✅ Forward pass OK")
    print(f"   Mean shape: {mean.shape}")
    print(f"   Std: {torch.exp(log_std)}")
    
    return log_std

def test_policy_creation():
    """Testar criação de policy V7"""
    print("\n" + "="*60)
    print("🧪 TESTE: Criação de Policy V7")
    print("="*60)
    
    import sys
    sys.path.append('D:\\Projeto')
    
    from trading_framework.policies.two_head_v7_intuition import TwoHeadV7Intuition, get_v7_intuition_kwargs
    import gym
    
    # Criar ambiente dummy
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2580,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1, high=1, shape=(11,), dtype=np.float32)
    
    # Criar policy
    policy_kwargs = get_v7_intuition_kwargs()
    
    # TESTE 1: Com ortho_init=False
    policy_kwargs['ortho_init'] = False
    policy_kwargs['log_std_init'] = -0.5
    
    print("📝 Criando policy com ortho_init=False...")
    policy = TwoHeadV7Intuition(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda x: 0.00005,
        **policy_kwargs
    )
    
    # Verificar log_std
    log_std = None
    for name, param in policy.named_parameters():
        if 'log_std' in name.lower():
            log_std = param
            print(f"✅ log_std encontrado: {name}")
            print(f"   Shape: {param.shape}")
            print(f"   Values: {param.data}")
            print(f"   Requires grad: {param.requires_grad}")
            break
    
    if log_std is None and hasattr(policy, 'action_dist'):
        if hasattr(policy.action_dist, 'log_std'):
            log_std = policy.action_dist.log_std
            print(f"✅ log_std em action_dist")
            print(f"   Shape: {log_std.shape}")
            print(f"   Values: {log_std.data}")
            print(f"   Requires grad: {log_std.requires_grad}")
    
    if log_std is None:
        print("❌ log_std NÃO ENCONTRADO na policy!")
    
    return policy

def main():
    """Diagnóstico principal"""
    print("🔍 DIAGNÓSTICO LOG_STD - Identificando onde está sendo zerado")
    
    # 1. Testar criação isolada
    test_log_std = test_distribution_creation()
    
    # 2. Testar policy V7
    policy = test_policy_creation()
    
    # 3. Criar modelo completo
    print("\n" + "="*60)
    print("🧪 TESTE: Modelo RecurrentPPO completo")
    print("="*60)
    
    import gym
    from trading_framework.policies.two_head_v7_intuition import get_v7_intuition_kwargs
    
    # Ambiente dummy
    env = gym.make('CartPole-v1')
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # Criar modelo
    policy_kwargs = get_v7_intuition_kwargs()
    policy_kwargs['ortho_init'] = False  # IMPORTANTE!
    
    print("📝 Criando RecurrentPPO...")
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=0.00005,
        device='cpu'
    )
    
    # Verificar log_std após criação
    log_std = check_log_std(model, "APÓS CRIAÇÃO")
    
    # 4. Simular um step de treinamento
    print("\n" + "="*60)
    print("🧪 TESTE: Após um step de treinamento")
    print("="*60)
    
    model.learn(total_timesteps=10, log_interval=1)
    
    # Verificar novamente
    log_std = check_log_std(model, "APÓS TREINO")
    
    print("\n" + "="*60)
    print("📊 CONCLUSÃO DO DIAGNÓSTICO")
    print("="*60)
    
    if log_std is not None:
        data = log_std.data.cpu().numpy()
        zeros = np.sum(np.abs(data) < 1e-6)
        if zeros == data.size:
            print("❌ PROBLEMA CONFIRMADO: log_std está 100% ZERO!")
            print("   Possíveis causas:")
            print("   1. Inicialização com ortho_init sobrescrevendo valores")
            print("   2. Callback ou hook modificando durante treino")
            print("   3. Checkpoint corrompido sendo carregado")
            print("   4. Gradientes não propagando (requires_grad=False)")
        else:
            print("✅ log_std está OK - tem valores não-zero")
    else:
        print("❌ log_std NÃO FOI ENCONTRADO!")

if __name__ == "__main__":
    main()