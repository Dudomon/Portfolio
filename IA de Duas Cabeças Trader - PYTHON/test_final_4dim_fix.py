"""
TESTE FINAL: Verificar se usar TradingTransformerFeatureExtractor resolve zeros
"""
import torch
import numpy as np
import gym
from trading_framework.policies.two_head_v9_optimus import TwoHeadV9Optimus, get_v9_optimus_kwargs

def test_final_4dim_fix():
    """Testa se usar TradingTransformerFeatureExtractor resolve zeros na V9"""
    
    print("🔧 TESTE FINAL: V9 com TradingTransformerFeatureExtractor...")
    
    # Simular criação EXATA do 4dim.py
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
    action_space = gym.spaces.Box(
        low=np.array([0, 0, -1, -1]), 
        high=np.array([2, 1, 1, 1]), 
        dtype=np.float32
    )
    
    def lr_schedule(progress):
        return 2e-5  # MESMO LR do 4dim.py
    
    kwargs = get_v9_optimus_kwargs()
    print(f"🔍 Features Extractor: {kwargs['features_extractor_class'].__name__}")
    print(f"🔍 ortho_init: {kwargs['ortho_init']}")
    
    policy = TwoHeadV9Optimus(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        **kwargs
    )
    
    # Verificar que está usando o extractor correto
    extractor_name = policy.features_extractor.__class__.__name__
    print(f"✅ Extractor em uso: {extractor_name}")
    
    # Verificar zeros INICIAL
    def check_zeros(name, tensor):
        if tensor is not None:
            zeros_pct = (tensor.abs() < 1e-8).float().mean().item() * 100
            status = "✅" if zeros_pct < 50 else "🚨"
            print(f"   {name}: {zeros_pct:.1f}% zeros {status}")
            return zeros_pct
        return 100.0
    
    print(f"\n🔍 VERIFICAÇÃO INICIAL:")
    
    # Verificar input/temporal projection
    if hasattr(policy.features_extractor, 'temporal_projection'):
        temporal_zeros = check_zeros("temporal_projection.weight", 
                                   policy.features_extractor.temporal_projection.weight)
    elif hasattr(policy.features_extractor, 'input_projection'):
        temporal_zeros = check_zeros("input_projection.weight", 
                                   policy.features_extractor.input_projection.weight)
    else:
        print("   projection layer: NÃO ENCONTRADO")
        temporal_zeros = 100.0
    
    # Verificar embedding
    embedding_zeros = check_zeros("regime_embedding.weight",
                                 policy.market_context_encoder.regime_embedding.weight)
    
    # Testar forward pass múltiplos
    print(f"\n🚀 TESTANDO FORWARD PASSES...")
    
    zeros_history = []
    for step in range(3):
        obs = torch.randn(4, 450)  # Batch como no PPO
        features = policy.features_extractor(obs)
        
        # Verificar zeros após forward
        if hasattr(policy.features_extractor, 'temporal_projection'):
            zeros_pct = (policy.features_extractor.temporal_projection.weight.abs() < 1e-8).float().mean().item() * 100
        elif hasattr(policy.features_extractor, 'input_projection'):
            zeros_pct = (policy.features_extractor.input_projection.weight.abs() < 1e-8).float().mean().item() * 100
        else:
            zeros_pct = 100.0
        
        zeros_history.append(zeros_pct)
        print(f"   Step {step+1}: {zeros_pct:.1f}% zeros")
    
    # Testar ações
    print(f"\n🎯 TESTANDO DIVERSIDADE DE AÇÕES...")
    actions = []
    for _ in range(5):
        obs = torch.randn(1, 450)
        lstm_states = (torch.zeros(1, 1, 256), torch.zeros(1, 1, 256))
        episode_starts = torch.zeros(1, dtype=torch.bool)
        
        dist = policy.forward_actor(obs, lstm_states, episode_starts)
        action = dist.sample()
        actions.append(action.detach().numpy()[0])
    
    actions = np.array(actions)
    entry_std = actions[:, 0].std()
    confidence_std = actions[:, 1].std()
    
    print(f"   Entry std: {entry_std:.3f}")
    print(f"   Confidence std: {confidence_std:.3f}")
    
    # Resultado final
    final_zeros = zeros_history[-1] if zeros_history else 100.0
    
    print(f"\n📊 RESULTADO FINAL:")
    print(f"   Projection zeros: {final_zeros:.1f}% {'✅' if final_zeros < 50 else '❌'}")
    print(f"   Embedding zeros: {embedding_zeros:.1f}% {'✅' if embedding_zeros < 50 else '❌'}")
    print(f"   Action diversity: {'✅' if entry_std > 0.1 else '❌'}")
    
    success = (final_zeros < 50 and embedding_zeros < 50 and entry_std > 0.1)
    
    if success:
        print("\n🎉 SUCESSO DEFINITIVO!")
        print("   V9 com TradingTransformerFeatureExtractor funciona!")
        print("   4dim.py está CORRIGIDO!")
    else:
        print("\n❌ FALHA!")
        print("   Ainda há problemas...")
    
    return success

if __name__ == "__main__":
    test_final_4dim_fix()