"""
🔍 INVESTIGAÇÃO COMPLETA: Zeros na V9Optimus vs V8Elegance

PROBLEMA IDENTIFICADO:
- V9Optimus: features_extractor.input_projection.weight: 91.4% zeros
- V9Optimus: market_context_encoder.regime_embedding.weight: 65.6% zeros
- V8Elegance: Sem problemas de zeros

HIPÓTESES A INVESTIGAR:
1. SB3 sobrescrevendo inicializações devido a ortho_init=False na V9 vs ortho_init=True na V8
2. Diferenças entre TradingTransformerV9 vs TradingTransformerFeatureExtractor
3. _fix_features_extractor_weights sendo chamado incorretamente
4. Timing de inicialização: V9 vs V8
"""

import torch
import torch.nn as nn
import numpy as np
import gym
from stable_baselines3.common.policies import ActorCriticPolicy

def test_embedding_initialization():
    """Teste específico para nn.Embedding"""
    print("🔍 TESTE 1: Inicialização de nn.Embedding")
    
    # Criar embedding como na V9
    embedding = nn.Embedding(4, 32)
    
    print(f"Embedding weight shape: {embedding.weight.shape}")
    print(f"Zeros iniciais: {(embedding.weight.abs() < 1e-8).float().mean().item():.1%}")
    print(f"Std inicial: {embedding.weight.std().item():.6f}")
    print(f"Range: [{embedding.weight.min().item():.6f}, {embedding.weight.max().item():.6f}]")
    
    # Verificar se alguma re-inicialização afeta
    print("\n🔧 Re-inicializando com Xavier...")
    nn.init.xavier_uniform_(embedding.weight, gain=0.8)
    print(f"Zeros pós-Xavier: {(embedding.weight.abs() < 1e-8).float().mean().item():.1%}")
    
    return embedding

def test_input_projection_initialization():
    """Teste específico para input_projection"""
    print("\n🔍 TESTE 2: Inicialização de input_projection")
    
    # Simular input_projection da V9: features_per_bar (45) -> d_model (128)
    input_projection = nn.Linear(45, 128)
    
    print(f"Input projection weight shape: {input_projection.weight.shape}")
    print(f"Zeros iniciais: {(input_projection.weight.abs() < 1e-8).float().mean().item():.1%}")
    print(f"Std inicial: {input_projection.weight.std().item():.6f}")
    
    # Aplicar inicialização como na V9
    print("\n🔧 Aplicando inicialização V9 (xavier_uniform gain=0.6)...")
    nn.init.xavier_uniform_(input_projection.weight, gain=0.6)
    if input_projection.bias is not None:
        nn.init.zeros_(input_projection.bias)
    
    print(f"Zeros pós-V9-init: {(input_projection.weight.abs() < 1e-8).float().mean().item():.1%}")
    print(f"Std pós-V9-init: {input_projection.weight.std().item():.6f}")
    
    return input_projection

def compare_transformer_initialization():
    """Comparar inicializações entre TradingTransformerV9 e TradingTransformerFeatureExtractor"""
    print("\n🔍 TESTE 3: Comparação de Transformers")
    
    # Importar os dois transformers
    from trading_framework.extractors.transformer_v9_daytrading import TradingTransformerV9
    from trading_framework.extractors.transformer_extractor import TradingTransformerFeatureExtractor
    
    # Observation spaces
    obs_space_v9 = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)  # V9: 450D
    obs_space_v8 = gym.spaces.Box(low=-1, high=1, shape=(2580,), dtype=np.float32)  # V8: 2580D
    
    print("Criando TradingTransformerV9...")
    transformer_v9 = TradingTransformerV9(obs_space_v9, features_dim=256)
    
    print("Criando TradingTransformerFeatureExtractor...")
    transformer_v8 = TradingTransformerFeatureExtractor(obs_space_v8, features_dim=256)
    
    # Verificar zeros nas camadas críticas
    print("\n📊 ANÁLISE DE ZEROS:")
    
    # V9: input_projection
    v9_input_proj = transformer_v9.input_projection
    v9_zeros = (v9_input_proj.weight.abs() < 1e-8).float().mean().item()
    print(f"V9 input_projection zeros: {v9_zeros:.1%}")
    
    # V8: temporal_projection  
    v8_temporal_proj = transformer_v8.temporal_projection
    v8_zeros = (v8_temporal_proj.weight.abs() < 1e-8).float().mean().item()
    print(f"V8 temporal_projection zeros: {v8_zeros:.1%}")
    
    return transformer_v9, transformer_v8

def test_sb3_ortho_init_effect():
    """Testar se ortho_init afeta inicializações não-lineares"""
    print("\n🔍 TESTE 4: Efeito de ortho_init do SB3")
    
    # Criar duas redes idênticas
    class TestNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(45, 128)
            self.embedding1 = nn.Embedding(4, 32)
            
        def _initialize_weights(self):
            nn.init.xavier_uniform_(self.linear1.weight, gain=0.6)
            nn.init.zeros_(self.linear1.bias)
            # Embedding não tem inicialização específica (usar padrão)
    
    # Test 1: Sem SB3 interference
    print("📋 Teste sem SB3...")
    net1 = TestNetwork()
    net1._initialize_weights()
    
    linear1_zeros = (net1.linear1.weight.abs() < 1e-8).float().mean().item()
    embed1_zeros = (net1.embedding1.weight.abs() < 1e-8).float().mean().item()
    
    print(f"Linear zeros: {linear1_zeros:.1%}")
    print(f"Embedding zeros: {embed1_zeros:.1%}")
    
    # Test 2: Com SB3 ortho_init=True (como V8)
    print("\n📋 Simulando SB3 ortho_init=True (V8)...")
    net2 = TestNetwork()
    net2._initialize_weights()
    
    # Simular what SB3 does with ortho_init=True
    for module in net2.modules():
        if isinstance(module, nn.Linear):
            # SB3 aplica orthogonal para Linear layers quando ortho_init=True
            nn.init.orthogonal_(module.weight)
    
    linear2_zeros = (net2.linear1.weight.abs() < 1e-8).float().mean().item()
    embed2_zeros = (net2.embedding1.weight.abs() < 1e-8).float().mean().item()
    
    print(f"Linear zeros pós-ortho: {linear2_zeros:.1%}")
    print(f"Embedding zeros pós-ortho: {embed2_zeros:.1%}")
    
    # Test 3: Com SB3 ortho_init=False (como V9)
    print("\n📋 Simulando SB3 ortho_init=False (V9)...")
    net3 = TestNetwork()
    net3._initialize_weights()
    # ortho_init=False -> SB3 não interfere
    
    linear3_zeros = (net3.linear1.weight.abs() < 1e-8).float().mean().item()
    embed3_zeros = (net3.embedding1.weight.abs() < 1e-8).float().mean().item()
    
    print(f"Linear zeros sem ortho: {linear3_zeros:.1%}")
    print(f"Embedding zeros sem ortho: {embed3_zeros:.1%}")
    
    return net1, net2, net3

def test_fix_features_extractor_timing():
    """Testar timing de _fix_features_extractor_weights"""
    print("\n🔍 TESTE 5: Timing de _fix_features_extractor_weights")
    
    from trading_framework.policies.two_head_v9_optimus import TwoHeadV9Optimus, get_v9_optimus_kwargs
    
    # Criar policy V9
    dummy_obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
    dummy_action_space = gym.spaces.Box(low=np.array([0, 0, -1, -1]), high=np.array([2, 1, 1, 1]), dtype=np.float32)
    
    def dummy_lr_schedule(progress):
        return 1e-4
    
    print("📋 Criando TwoHeadV9Optimus...")
    policy = TwoHeadV9Optimus(
        observation_space=dummy_obs_space,
        action_space=dummy_action_space, 
        lr_schedule=dummy_lr_schedule,
        **get_v9_optimus_kwargs()
    )
    
    # Verificar zeros APÓS criação
    input_proj = policy.features_extractor.input_projection
    regime_emb = policy.market_context_encoder.regime_embedding
    
    input_zeros = (input_proj.weight.abs() < 1e-8).float().mean().item()
    regime_zeros = (regime_emb.weight.abs() < 1e-8).float().mean().item()
    
    print(f"📊 Zeros APÓS criação da policy:")
    print(f"  input_projection: {input_zeros:.1%}")
    print(f"  regime_embedding: {regime_zeros:.1%}")
    
    # Testar se _fix_features_extractor_weights ajuda
    print("\n🔧 Executando _fix_features_extractor_weights...")
    policy._fix_features_extractor_weights()
    
    input_zeros_after = (input_proj.weight.abs() < 1e-8).float().mean().item()
    regime_zeros_after = (regime_emb.weight.abs() < 1e-8).float().mean().item()
    
    print(f"📊 Zeros APÓS _fix_features_extractor_weights:")
    print(f"  input_projection: {input_zeros_after:.1%}")
    print(f"  regime_embedding: {regime_zeros_after:.1%}")
    
    return policy

def main():
    """Executar todos os testes"""
    print("🚀 INVESTIGAÇÃO COMPLETA: V9Optimus Zeros vs V8Elegance")
    print("=" * 80)
    
    # Teste 1: Embedding básico
    embedding = test_embedding_initialization()
    
    # Teste 2: Input projection básico
    input_proj = test_input_projection_initialization()
    
    # Teste 3: Comparação de transformers
    transformer_v9, transformer_v8 = compare_transformer_initialization()
    
    # Teste 4: Efeito ortho_init
    net1, net2, net3 = test_sb3_ortho_init_effect()
    
    # Teste 5: Timing do fix
    policy_v9 = test_fix_features_extractor_timing()
    
    print("\n" + "="*80)
    print("🎯 CONCLUSÕES:")
    print("1. Embeddings e Linear layers inicializam corretamente isoladamente")
    print("2. Diferenças entre V9 vs V8 transformer podem ser a causa")
    print("3. ortho_init=True vs False pode afetar inicializações SB3")
    print("4. _fix_features_extractor_weights pode não estar funcionando")
    print("5. Timing de inicialização pode ser crítico")

if __name__ == "__main__":
    main()