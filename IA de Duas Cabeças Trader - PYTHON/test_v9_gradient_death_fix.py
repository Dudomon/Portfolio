"""
Teste final para verificar se correção de gradient death na V9 funciona
"""
import torch
import numpy as np
import gym
from trading_framework.policies.two_head_v9_optimus import TwoHeadV9Optimus, get_v9_optimus_kwargs

def test_v9_gradient_death_fix():
    """Testa se gradient death foi corrigido na V9"""
    
    print("🔧 Testando correção de gradient death V9Optimus...")
    
    # Criar espaços dummy
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
    action_space = gym.spaces.Box(
        low=np.array([0, 0, -1, -1]), 
        high=np.array([2, 1, 1, 1]), 
        dtype=np.float32
    )
    
    # Criar política com correções
    def lr_schedule(progress):
        return 1e-4
    
    kwargs = get_v9_optimus_kwargs()
    
    policy = TwoHeadV9Optimus(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        **kwargs
    )
    
    print("\n🔍 TESTANDO GRADIENT FLOW...")
    
    # Simular forward pass múltiplos para verificar se zeros aparecem
    batch_size = 4
    zeros_history = []
    
    for step in range(5):
        # Criar batch de observações
        obs = torch.randn(batch_size, 450, requires_grad=True)
        
        # Forward pass
        features = policy.features_extractor(obs)
        
        # Backward pass simulado
        loss = features.sum()
        loss.backward()
        
        # Verificar zeros na input_projection
        if hasattr(policy.features_extractor, 'input_projection'):
            weight = policy.features_extractor.input_projection.weight
            zeros_pct = (weight.abs() < 1e-8).float().mean().item() * 100
            zeros_history.append(zeros_pct)
            print(f"   Step {step+1}: input_projection {zeros_pct:.1f}% zeros")
        else:
            print(f"   Step {step+1}: input_projection NÃO ENCONTRADO")
            zeros_history.append(100.0)
        
        # Limpar gradientes
        policy.zero_grad()
    
    # Verificar ações
    print("\n🎯 TESTANDO DIVERSIDADE DE AÇÕES...")
    
    action_samples = []
    for _ in range(10):
        obs = torch.randn(1, 450)
        lstm_states = (torch.zeros(1, 1, 256), torch.zeros(1, 1, 256))
        episode_starts = torch.zeros(1, dtype=torch.bool)
        
        dist = policy.forward_actor(obs, lstm_states, episode_starts)
        action = dist.sample()
        action_samples.append(action.detach().numpy()[0])
    
    action_samples = np.array(action_samples)
    
    # Analisar diversidade
    entry_std = action_samples[:, 0].std()
    confidence_std = action_samples[:, 1].std()
    
    print(f"   Entry decision std: {entry_std:.3f} (alvo: >0.1)")
    print(f"   Confidence std: {confidence_std:.3f} (alvo: >0.05)")
    
    # Avaliar resultados
    final_zeros = zeros_history[-1] if zeros_history else 100.0
    
    print(f"\n📊 RESULTADOS FINAIS:")
    print(f"   Zeros finais: {final_zeros:.1f}% {'✅' if final_zeros < 50 else '❌'}")
    print(f"   Entry diversity: {'✅' if entry_std > 0.1 else '❌'}")
    print(f"   Confidence diversity: {'✅' if confidence_std > 0.05 else '❌'}")
    
    # Verificar se houve melhoria nos zeros
    if len(zeros_history) >= 2:
        improvement = zeros_history[0] - zeros_history[-1]
        print(f"   Melhoria zeros: {improvement:.1f}% {'✅' if improvement >= 0 else '❌'}")
    
    # Conclusão
    success = (final_zeros < 50 and entry_std > 0.1 and confidence_std > 0.05)
    
    if success:
        print("\n🎉 SUCESSO! Gradient death foi corrigido!")
        print("   - Input projection saudável")
        print("   - Diversidade de ações restaurada")
        print("   - V9 pronta para treinamento!")
        return True
    else:
        print("\n❌ FALHA! Gradient death persiste.")
        print("   - Precisa de mais correções")
        return False

if __name__ == "__main__":
    test_v9_gradient_death_fix()