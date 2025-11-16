"""
🔍 TESTE TREINAMENTO SIMULADO: Detectar quando zeros aparecem na V9
"""
import torch
import numpy as np
import gym
from sb3_contrib import RecurrentPPO
from trading_framework.policies.two_head_v9_optimus import TwoHeadV9Optimus, get_v9_optimus_kwargs, fix_v9_optimus_weights

def check_zeros(name, tensor):
    if tensor is not None:
        zeros_pct = (tensor.abs() < 1e-8).float().mean().item() * 100
        status = "✅" if zeros_pct < 50 else "🚨"
        print(f"   {name}: {zeros_pct:.1f}% zeros {status}")
        return zeros_pct
    return 100.0

def simulate_training_steps():
    """Simula o que acontece no treino real"""
    
    print("🔍 SIMULAÇÃO TREINAMENTO V9...")
    
    # 1. Criar ambiente igual ao 4dim.py
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
    action_space = gym.spaces.Box(
        low=np.array([0, 0, -1, -1]), 
        high=np.array([2, 1, 1, 1]), 
        dtype=np.float32
    )
    
    def lr_schedule(progress):
        return 2e-5
    
    # 2. Criar modelo RecurrentPPO igual ao 4dim.py
    model_config = {
        "policy": TwoHeadV9Optimus,
        "env": None,  # Mock
        "learning_rate": lr_schedule,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 8,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.12,
        "ent_coef": 0.1,
        "vf_coef": 0.5,
        "use_sde": False,
        "policy_kwargs": {
            **get_v9_optimus_kwargs(),
        }
    }
    
    print("📊 PASSO 1: Criando RecurrentPPO...")
    # Simular criação sem env real
    policy = TwoHeadV9Optimus(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        **get_v9_optimus_kwargs()
    )
    
    print("\n🔍 VERIFICAÇÃO APÓS CRIAÇÃO POLICY:")
    check_zeros("input_projection.weight", policy.features_extractor.input_projection.weight)
    check_zeros("_residual_projection.weight", policy.features_extractor._residual_projection.weight)
    check_zeros("regime_embedding.weight", policy.market_context_encoder.regime_embedding.weight)
    
    print("\n📊 PASSO 2: Aplicando fix_v9_optimus_weights...")
    
    # Simular modelo fake para fix_v9_optimus_weights
    class FakeModel:
        def __init__(self, policy):
            self.policy = policy
    
    fake_model = FakeModel(policy)
    fix_v9_optimus_weights(fake_model)
    
    print("\n🔍 VERIFICAÇÃO APÓS fix_v9_optimus_weights:")
    check_zeros("input_projection.weight", policy.features_extractor.input_projection.weight)
    check_zeros("_residual_projection.weight", policy.features_extractor._residual_projection.weight)
    check_zeros("regime_embedding.weight", policy.market_context_encoder.regime_embedding.weight)
    
    print("\n📊 PASSO 3: Simulando primeiros forward passes...")
    
    # Simular alguns forward passes
    for step in range(5):
        obs = torch.randn(1, 450)  # Batch size individual
        
        try:
            # Simular forward pass completo
            features = policy.features_extractor(obs)
            lstm_states = (torch.zeros(1, 1, 256), torch.zeros(1, 1, 256))
            episode_starts = torch.zeros(1, dtype=torch.bool)
            
            # Forward actor
            dist = policy.forward_actor(obs, lstm_states, episode_starts)
            actions = dist.sample()
            
            # Forward critic  
            values = policy.forward_critic(features)
            
            print(f"\n🔍 VERIFICAÇÃO APÓS FORWARD PASS {step+1}:")
            check_zeros("input_projection.weight", policy.features_extractor.input_projection.weight)
            check_zeros("_residual_projection.weight", policy.features_extractor._residual_projection.weight)
            check_zeros("regime_embedding.weight", policy.market_context_encoder.regime_embedding.weight)
            
        except Exception as e:
            print(f"❌ Erro no forward pass {step+1}: {e}")
            break
    
    print("\n📊 PASSO 4: Simulando backward pass...")
    
    try:
        # Simular backward pass
        obs = torch.randn(1, 450, requires_grad=True)
        features = policy.features_extractor(obs)
        lstm_states = (torch.zeros(1, 1, 256), torch.zeros(1, 1, 256))
        episode_starts = torch.zeros(1, dtype=torch.bool)
        
        dist = policy.forward_actor(obs, lstm_states, episode_starts)
        actions = dist.sample()
        
        # Simular loss
        loss = actions.sum()
        loss.backward()
        
        print("\n🔍 VERIFICAÇÃO APÓS BACKWARD PASS:")
        check_zeros("input_projection.weight", policy.features_extractor.input_projection.weight)
        check_zeros("_residual_projection.weight", policy.features_extractor._residual_projection.weight)
        check_zeros("regime_embedding.weight", policy.market_context_encoder.regime_embedding.weight)
        
        # Verificar gradientes
        print("\n🔍 VERIFICAÇÃO DE GRADIENTES:")
        if policy.features_extractor.input_projection.weight.grad is not None:
            grad_zeros = (policy.features_extractor.input_projection.weight.grad.abs() < 1e-8).float().mean().item() * 100
            print(f"   input_projection.grad: {grad_zeros:.1f}% zeros")
        
        if policy.features_extractor._residual_projection.weight.grad is not None:
            grad_zeros = (policy.features_extractor._residual_projection.weight.grad.abs() < 1e-8).float().mean().item() * 100
            print(f"   _residual_projection.grad: {grad_zeros:.1f}% zeros")
            
    except Exception as e:
        print(f"❌ Erro no backward pass: {e}")
    
    print("\n🏁 SIMULAÇÃO CONCLUÍDA")

if __name__ == "__main__":
    simulate_training_steps()