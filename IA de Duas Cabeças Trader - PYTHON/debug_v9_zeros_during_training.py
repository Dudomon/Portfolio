"""
🔍 INVESTIGAÇÃO: Quando os zeros aparecem na V9Optimus?

RESULTADO DOS TESTES ANTERIORES:
- Componentes isolados: 0% zeros ✅
- TradingTransformerV9 isolado: 0% zeros ✅  
- TwoHeadV9Optimus recém-criada: 0% zeros ✅

CONCLUSÃO: Os zeros aparecem DURANTE algum processo específico.

PRÓXIMOS TESTES:
1. Verificar zeros após load de checkpoint
2. Verificar zeros após steps de treinamento
3. Verificar zeros após SB3 PPO integration
4. Comparar processo de criação V8 vs V9 completo
"""

import torch
import torch.nn as nn
import numpy as np
import gym
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

def test_v9_with_ppo_creation():
    """Testar V9 quando criada através do PPO (processo completo)"""
    print("🔍 TESTE 1: V9Optimus através de RecurrentPPO")
    
    from trading_framework.policies.two_head_v9_optimus import TwoHeadV9Optimus, get_v9_optimus_kwargs
    
    # Ambiente dummy
    class DummyEnv:
        def __init__(self):
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=np.array([0, 0, -1, -1]), high=np.array([2, 1, 1, 1]), dtype=np.float32)
        
        def reset(self):
            return np.random.randn(450).astype(np.float32)
        
        def step(self, action):
            obs = np.random.randn(450).astype(np.float32)
            reward = np.random.randn() * 0.1
            done = np.random.rand() < 0.01
            info = {}
            return obs, reward, done, info
    
    env = DummyEnv()
    
    print("📋 Criando RecurrentPPO com V9Optimus...")
    
    # Criar PPO com V9
    model = RecurrentPPO(
        TwoHeadV9Optimus,
        env,
        learning_rate=1e-4,
        n_steps=128,
        batch_size=32,
        policy_kwargs=get_v9_optimus_kwargs(),
        verbose=0
    )
    
    # Verificar zeros APÓS criação do PPO
    policy = model.policy
    
    input_proj = policy.features_extractor.input_projection
    regime_emb = policy.market_context_encoder.regime_embedding
    
    input_zeros = (input_proj.weight.abs() < 1e-8).float().mean().item()
    regime_zeros = (regime_emb.weight.abs() < 1e-8).float().mean().item()
    
    print(f"📊 Zeros APÓS criação do RecurrentPPO:")
    print(f"  input_projection: {input_zeros:.1%}")
    print(f"  regime_embedding: {regime_zeros:.1%}")
    
    return model

def test_v8_with_ppo_creation():
    """Testar V8 quando criada através do PPO (processo completo)"""
    print("\n🔍 TESTE 2: V8Elegance através de RecurrentPPO")
    
    from trading_framework.policies.two_head_v8_elegance import TwoHeadV8Elegance, get_v8_elegance_kwargs
    
    # Ambiente dummy
    class DummyEnv:
        def __init__(self):
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2580,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        
        def reset(self):
            return np.random.randn(2580).astype(np.float32)
        
        def step(self, action):
            obs = np.random.randn(2580).astype(np.float32)
            reward = np.random.randn() * 0.1
            done = np.random.rand() < 0.01
            info = {}
            return obs, reward, done, info
    
    env = DummyEnv()
    
    print("📋 Criando RecurrentPPO com V8Elegance...")
    
    # Criar PPO com V8
    model = RecurrentPPO(
        TwoHeadV8Elegance,
        env,
        learning_rate=1e-4,
        n_steps=128,
        batch_size=32,
        policy_kwargs=get_v8_elegance_kwargs(),
        verbose=0
    )
    
    # Verificar zeros APÓS criação do PPO
    policy = model.policy
    
    temporal_proj = policy.features_extractor.temporal_projection
    regime_emb = policy.market_context.regime_embedding
    
    temporal_zeros = (temporal_proj.weight.abs() < 1e-8).float().mean().item()
    regime_zeros = (regime_emb.weight.abs() < 1e-8).float().mean().item()
    
    print(f"📊 Zeros APÓS criação do RecurrentPPO:")
    print(f"  temporal_projection: {temporal_zeros:.1%}")
    print(f"  regime_embedding: {regime_zeros:.1%}")
    
    return model

def test_after_training_steps(model, model_name):
    """Testar zeros após alguns steps de treinamento"""
    print(f"\n🔍 TESTE 3: {model_name} após training steps")
    
    # Executar alguns steps
    print("📋 Executando 10 steps de treinamento...")
    try:
        model.learn(total_timesteps=10, progress_bar=False)
        
        # Verificar zeros APÓS training
        policy = model.policy
        
        if hasattr(policy.features_extractor, 'input_projection'):
            # V9
            input_proj = policy.features_extractor.input_projection
            regime_emb = policy.market_context_encoder.regime_embedding
            
            input_zeros = (input_proj.weight.abs() < 1e-8).float().mean().item()
            regime_zeros = (regime_emb.weight.abs() < 1e-8).float().mean().item()
            
            print(f"📊 Zeros APÓS 10 training steps:")
            print(f"  input_projection: {input_zeros:.1%}")
            print(f"  regime_embedding: {regime_zeros:.1%}")
            
        elif hasattr(policy.features_extractor, 'temporal_projection'):
            # V8
            temporal_proj = policy.features_extractor.temporal_projection
            regime_emb = policy.market_context.regime_embedding
            
            temporal_zeros = (temporal_proj.weight.abs() < 1e-8).float().mean().item()
            regime_zeros = (regime_emb.weight.abs() < 1e-8).float().mean().item()
            
            print(f"📊 Zeros APÓS 10 training steps:")
            print(f"  temporal_projection: {temporal_zeros:.1%}")
            print(f"  regime_embedding: {regime_zeros:.1%}")
            
    except Exception as e:
        print(f"❌ Erro durante training: {e}")

def test_checkpoint_loading():
    """Testar se carregar checkpoint causa zeros"""
    print("\n🔍 TESTE 4: Loading de checkpoint")
    
    # Verificar se há checkpoints V9 disponíveis
    import os
    import glob
    
    checkpoint_patterns = [
        "D:/Projeto/training_checkpoints/*v9*",
        "D:/Projeto/checkpoints/*v9*", 
        "D:/Projeto/*v9*checkpoint*",
        "D:/Projeto/models/*v9*"
    ]
    
    v9_checkpoints = []
    for pattern in checkpoint_patterns:
        v9_checkpoints.extend(glob.glob(pattern))
    
    if v9_checkpoints:
        print(f"📋 Encontrados {len(v9_checkpoints)} checkpoints V9:")
        for cp in v9_checkpoints[:3]:  # Mostrar apenas os primeiros 3
            print(f"  - {cp}")
        
        # Tentar carregar o primeiro
        try:
            print(f"\n📋 Tentando carregar: {v9_checkpoints[0]}")
            model = RecurrentPPO.load(v9_checkpoints[0])
            
            # Verificar zeros no modelo carregado
            policy = model.policy
            
            if hasattr(policy.features_extractor, 'input_projection'):
                input_proj = policy.features_extractor.input_projection
                regime_emb = policy.market_context_encoder.regime_embedding
                
                input_zeros = (input_proj.weight.abs() < 1e-8).float().mean().item()
                regime_zeros = (regime_emb.weight.abs() < 1e-8).float().mean().item()
                
                print(f"📊 Zeros APÓS loading checkpoint:")
                print(f"  input_projection: {input_zeros:.1%}")
                print(f"  regime_embedding: {regime_zeros:.1%}")
                
                return model
                
        except Exception as e:
            print(f"❌ Erro ao carregar checkpoint: {e}")
    else:
        print("📋 Nenhum checkpoint V9 encontrado")
    
    return None

def analyze_initialization_process():
    """Analisar processo de inicialização passo a passo"""
    print("\n🔍 TESTE 5: Análise detalhada do processo de inicialização")
    
    from trading_framework.policies.two_head_v9_optimus import TwoHeadV9Optimus, get_v9_optimus_kwargs
    from trading_framework.extractors.transformer_v9_daytrading import TradingTransformerV9
    
    # Criar observation e action spaces
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
    action_space = gym.spaces.Box(low=np.array([0, 0, -1, -1]), high=np.array([2, 1, 1, 1]), dtype=np.float32)
    
    def dummy_lr_schedule(progress):
        return 1e-4
    
    print("📋 Passo 1: Criar TradingTransformerV9 isoladamente...")
    transformer = TradingTransformerV9(obs_space, features_dim=256)
    input_zeros_1 = (transformer.input_projection.weight.abs() < 1e-8).float().mean().item()
    print(f"  input_projection zeros: {input_zeros_1:.1%}")
    
    print("\n📋 Passo 2: Criar TwoHeadV9Optimus sem super().__init__()...")
    # Simular criação sem chamar super()
    
    print("\n📋 Passo 3: Criar TwoHeadV9Optimus completa...")
    policy = TwoHeadV9Optimus(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=dummy_lr_schedule,
        **get_v9_optimus_kwargs()
    )
    
    input_zeros_3 = (policy.features_extractor.input_projection.weight.abs() < 1e-8).float().mean().item()
    regime_zeros_3 = (policy.market_context_encoder.regime_embedding.weight.abs() < 1e-8).float().mean().item()
    print(f"  input_projection zeros: {input_zeros_3:.1%}")
    print(f"  regime_embedding zeros: {regime_zeros_3:.1%}")
    
    print("\n📋 Passo 4: Simular integração com SB3...")
    # Verificar se o problema pode estar no momento da integração com SB3
    
    return policy

def main():
    """Executar investigação completa"""
    print("🚀 INVESTIGAÇÃO: Quando os zeros aparecem na V9Optimus?")
    print("=" * 80)
    
    # Teste 1: V9 com PPO
    model_v9 = test_v9_with_ppo_creation()
    
    # Teste 2: V8 com PPO (comparação)
    model_v8 = test_v8_with_ppo_creation()
    
    # Teste 3: Zeros após training steps
    if model_v9:
        test_after_training_steps(model_v9, "V9Optimus")
    
    if model_v8:
        test_after_training_steps(model_v8, "V8Elegance")
    
    # Teste 4: Checkpoint loading
    checkpoint_model = test_checkpoint_loading()
    
    # Teste 5: Análise detalhada
    policy = analyze_initialization_process()
    
    print("\n" + "="*80)
    print("🎯 CONCLUSÕES:")
    print("1. Verificamos zeros em diferentes momentos do ciclo de vida")
    print("2. Comparamos V9 vs V8 no mesmo processo")
    print("3. Testamos efeito do training e checkpoint loading")
    print("4. Analisamos o processo de inicialização step-by-step")

if __name__ == "__main__":
    main()