"""
Teste rápido para verificar se erro de shape na V9 foi corrigido
"""
import torch
import numpy as np
import gym
from trading_framework.policies.two_head_v9_optimus import TwoHeadV9Optimus, get_v9_optimus_kwargs

def test_v9_shape_fix():
    """Testa se o erro de shape 1x320 x 256x128 foi corrigido"""
    
    print("🔧 Testando correção de shape V9Optimus...")
    
    # Criar espaços dummy
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
    action_space = gym.spaces.Box(
        low=np.array([0, 0, -1, -1]), 
        high=np.array([2, 1, 1, 1]), 
        dtype=np.float32
    )
    
    # Criar política
    def lr_schedule(progress):
        return 1e-4
    
    policy = TwoHeadV9Optimus(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        **get_v9_optimus_kwargs()
    )
    
    # Testar forward pass
    try:
        # Criar batch de observações dummy
        batch_size = 32
        obs = torch.randn(batch_size, 450)
        
        # Extrair features (450D → 256D)
        features = policy.extract_features(obs)
        print(f"✅ Features extraídas: {features.shape}")
        
        # Testar forward_actor - usar batch_size=1 para simplificar LSTM
        batch_size = 1
        obs_single = torch.randn(batch_size, 450)
        features_single = policy.extract_features(obs_single)
        
        lstm_states = (
            torch.zeros(1, batch_size, 256),
            torch.zeros(1, batch_size, 256)
        )
        episode_starts = torch.zeros(batch_size, dtype=torch.bool)
        
        dist = policy.forward_actor(obs_single, lstm_states, episode_starts)
        print(f"✅ Forward actor bem-sucedido! Distribuição criada.")
        
        # Testar amostragem de ações
        actions = dist.sample()
        print(f"✅ Ações amostradas: {actions.shape}")
        
        # Verificar se são 4D
        assert actions.shape[-1] == 4, f"Esperado 4 ações, obtido {actions.shape[-1]}"
        print(f"✅ Ações são 4D como esperado!")
        
        # Testar _get_action_dist_from_latent diretamente com 320D
        combined_input = torch.randn(batch_size, 320)  # Simular input combinado
        dist2 = policy._get_action_dist_from_latent(combined_input)
        actions2 = dist2.sample()
        print(f"✅ _get_action_dist_from_latent funciona com 320D: {actions2.shape}")
        
        print("\n🎉 SUCESSO! Erro de shape corrigido!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_v9_shape_fix()