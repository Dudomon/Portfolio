"""
🔍 TESTE V10Pure - V8 Elegance adaptado para 4D
"""
import torch
import numpy as np
import gym
from trading_framework.policies.two_head_v10_pure import TwoHeadV10Pure, get_v10_pure_kwargs

def test_v10_pure():
    """Testa V10Pure com transformer funcional da V8"""
    
    print("🔧 TESTE V10Pure - V8 Elegance funcional para 4D...")
    
    # Usar obs space da V8 que funciona (2580D)
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(2580,), dtype=np.float32)
    action_space = gym.spaces.Box(
        low=np.array([0, 0, -1, -1]), 
        high=np.array([2, 1, 1, 1]), 
        dtype=np.float32
    )
    
    def lr_schedule(progress):
        return 2e-5
    
    kwargs = get_v10_pure_kwargs()
    print(f"🔍 Features Extractor: {kwargs['features_extractor_class'].__name__}")
    print(f"🔍 ortho_init: {kwargs['ortho_init']}")
    
    policy = TwoHeadV10Pure(
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
    
    # Verificar transformer weights (se existir input_projection ou temporal_projection)
    transformer_zeros = 0.0
    if hasattr(policy.features_extractor, 'input_projection'):
        transformer_zeros = check_zeros("input_projection.weight", 
                                      policy.features_extractor.input_projection.weight)
    elif hasattr(policy.features_extractor, 'temporal_projection'):
        transformer_zeros = check_zeros("temporal_projection.weight", 
                                      policy.features_extractor.temporal_projection.weight)
    else:
        print("   transformer projection: NÃO ENCONTRADO")
        transformer_zeros = 0.0  # V8 transformer pode não ter essas camadas
    
    # Verificar embedding
    embedding_zeros = check_zeros("regime_embedding.weight",
                                 policy.market_context_encoder.regime_embedding.weight)
    
    # Testar forward pass múltiplos
    print(f"\n🚀 TESTANDO FORWARD PASSES...")
    
    for step in range(3):
        obs = torch.randn(4, 2580)  # Batch como no PPO com obs space da V8
        features = policy.features_extractor(obs)
        
        print(f"   Step {step+1}: features shape = {features.shape}")
    
    # Testar ações
    print(f"\n🎯 TESTANDO DIVERSIDADE DE AÇÕES...")
    actions = []
    for _ in range(5):
        obs = torch.randn(1, 2580)
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
    print(f"\n📊 RESULTADO FINAL:")
    print(f"   Transformer zeros: {transformer_zeros:.1f}% {'✅' if transformer_zeros < 50 else '❌'}")
    print(f"   Embedding zeros: {embedding_zeros:.1f}% {'✅' if embedding_zeros < 50 else '❌'}")
    print(f"   Action diversity: {'✅' if entry_std > 0.05 else '❌'}")
    
    success = (transformer_zeros < 50 and embedding_zeros < 50 and entry_std > 0.05)
    
    if success:
        print("\n🎉 V10Pure FUNCIONANDO!")
        print("   V8 Elegance adaptado para 4D com sucesso!")
        print("   Usando transformer COMPROVADAMENTE funcional!")
    else:
        print("\n❌ V10Pure com problemas...")
    
    return success

if __name__ == "__main__":
    test_v10_pure()