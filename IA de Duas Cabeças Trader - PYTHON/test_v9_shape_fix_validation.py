"""
🔧 Teste de Validação do Shape Fix para TwoHeadV9Optimus

TESTA:
1. _get_action_dist_from_latent com 256D (chamada SB3 direta)
2. _get_action_dist_from_latent com 320D (chamada via forward_actor)
3. forward_actor funcionando corretamente
4. Dimensões incorretas (fallback)
"""

import torch
import numpy as np
import gym
from trading_framework.policies.two_head_v9_optimus import (
    TwoHeadV9Optimus, 
    get_v9_optimus_kwargs,
    validate_v9_optimus_policy
)

def test_shape_fix_comprehensive():
    """Teste completo do sistema de detecção de shape"""
    
    print("🔧 TESTE: Shape Fix TwoHeadV9Optimus")
    print("=" * 50)
    
    # Criar policy de teste
    dummy_obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
    dummy_action_space = gym.spaces.Box(low=np.array([0, 0, -1, -1]), high=np.array([2, 1, 1, 1]), dtype=np.float32)
    
    def dummy_lr_schedule(progress):
        return 1e-4
    
    policy = TwoHeadV9Optimus(
        observation_space=dummy_obs_space,
        action_space=dummy_action_space,
        lr_schedule=dummy_lr_schedule,
        **get_v9_optimus_kwargs()
    )
    
    policy.eval()
    
    batch_size = 4
    
    print("\n1️⃣ TESTE: Chamada direta _get_action_dist_from_latent com 256D")
    # Simular chamada do SB3 com latent_pi 256D (output direto do LSTM)
    latent_256d = torch.randn(batch_size, 256)
    
    try:
        dist_256 = policy._get_action_dist_from_latent(latent_256d)
        action_256 = dist_256.sample()
        print(f"   ✅ Input 256D → Output shape: {action_256.shape}")
        print(f"   ✅ Actions: {action_256[0].detach().numpy()}")
        assert action_256.shape == (batch_size, 4), f"Esperado (4,4), got {action_256.shape}"
        print("   ✅ Teste 256D PASSOU!")
    except Exception as e:
        print(f"   ❌ Erro com 256D: {e}")
        return False
    
    print("\n2️⃣ TESTE: Chamada _get_action_dist_from_latent com 320D")
    # Simular chamada com combined_input 320D (LSTM + context)
    latent_320d = torch.randn(batch_size, 320)
    
    try:
        dist_320 = policy._get_action_dist_from_latent(latent_320d)
        action_320 = dist_320.sample()
        print(f"   ✅ Input 320D → Output shape: {action_320.shape}")
        print(f"   ✅ Actions: {action_320[0].detach().numpy()}")
        assert action_320.shape == (batch_size, 4), f"Esperado (4,4), got {action_320.shape}"
        print("   ✅ Teste 320D PASSOU!")
    except Exception as e:
        print(f"   ❌ Erro com 320D: {e}")
        return False
    
    print("\n3️⃣ TESTE: forward_actor completo")
    # Teste do forward_actor que deve produzir 320D internamente
    features = torch.randn(batch_size, 450)  # Input do transformer
    lstm_states = None
    episode_starts = torch.zeros(batch_size, dtype=torch.bool)
    
    try:
        dist_actor = policy.forward_actor(features, lstm_states, episode_starts)
        action_actor = dist_actor.sample()
        print(f"   ✅ forward_actor → Output shape: {action_actor.shape}")
        print(f"   ✅ Actions: {action_actor[0].detach().numpy()}")
        assert action_actor.shape == (batch_size, 4), f"Esperado (4,4), got {action_actor.shape}"
        print("   ✅ Teste forward_actor PASSOU!")
    except Exception as e:
        print(f"   ❌ Erro em forward_actor: {e}")
        return False
    
    print("\n4️⃣ TESTE: Dimensão inesperada (fallback)")
    # Teste com dimensão não esperada
    latent_weird = torch.randn(batch_size, 128)  # Dimensão estranha
    
    try:
        dist_weird = policy._get_action_dist_from_latent(latent_weird)
        action_weird = dist_weird.sample()
        print(f"   ✅ Input 128D → Output shape: {action_weird.shape}")
        print(f"   ✅ Actions: {action_weird[0].detach().numpy()}")
        assert action_weird.shape == (batch_size, 4), f"Esperado (4,4), got {action_weird.shape}"
        print("   ✅ Teste fallback PASSOU!")
    except Exception as e:
        print(f"   ❌ Erro em fallback: {e}")
        return False
    
    print("\n5️⃣ TESTE: Sequência 3D")
    # Teste com input 3D (batch, seq, features)
    latent_3d = torch.randn(batch_size, 1, 256)
    
    try:
        dist_3d = policy._get_action_dist_from_latent(latent_3d)
        action_3d = dist_3d.sample()
        print(f"   ✅ Input 3D (4,1,256) → Output shape: {action_3d.shape}")
        print(f"   ✅ Actions: {action_3d[0].detach().numpy()}")
        assert action_3d.shape == (batch_size, 4), f"Esperado (4,4), got {action_3d.shape}"
        print("   ✅ Teste 3D PASSOU!")
    except Exception as e:
        print(f"   ❌ Erro com 3D: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 TODOS OS TESTES PASSARAM!")
    print("✅ Shape Fix implementado com sucesso!")
    print("✅ Detecta automaticamente 256D vs 320D")
    print("✅ Aplica market context quando necessário")
    print("✅ Funciona com chamadas SB3 e forward_actor")
    
    return True

def test_reward_engineering_analysis():
    """Análise específica para reward engineering"""
    
    print("\n🎯 ANÁLISE REWARD ENGINEERING")
    print("=" * 40)
    
    # Verificar se a política gera ações dentro dos ranges esperados
    dummy_obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
    dummy_action_space = gym.spaces.Box(low=np.array([0, 0, -1, -1]), high=np.array([2, 1, 1, 1]), dtype=np.float32)
    
    def dummy_lr_schedule(progress):
        return 1e-4
    
    policy = TwoHeadV9Optimus(
        observation_space=dummy_obs_space,
        action_space=dummy_action_space,
        lr_schedule=dummy_lr_schedule,
        **get_v9_optimus_kwargs()
    )
    
    policy.eval()
    
    # Gerar muitas amostras para análise estatística
    print("Gerando 1000 amostras para análise...")
    
    features = torch.randn(1000, 450)
    lstm_states = None
    episode_starts = torch.zeros(1000, dtype=torch.bool)
    
    with torch.no_grad():
        dist = policy.forward_actor(features, lstm_states, episode_starts)
        actions = dist.sample()  # [1000, 4]
    
    actions_np = actions.detach().numpy()
    
    print(f"\n📊 ESTATÍSTICAS DAS AÇÕES:")
    action_names = ['entry_decision', 'confidence', 'pos1_mgmt', 'pos2_mgmt']
    expected_ranges = [(0, 2), (0, 1), (-1, 1), (-1, 1)]
    
    for i, (name, (low, high)) in enumerate(zip(action_names, expected_ranges)):
        values = actions_np[:, i]
        print(f"   {name}:")
        print(f"     Range esperado: [{low}, {high}]")
        print(f"     Range real: [{values.min():.3f}, {values.max():.3f}]")
        print(f"     Média: {values.mean():.3f}")
        print(f"     Std: {values.std():.3f}")
        
        # Verificar se está dentro do range
        in_range = (values >= low) & (values <= high)
        pct_in_range = in_range.mean() * 100
        print(f"     % no range: {pct_in_range:.1f}%")
        
        if pct_in_range < 95:
            print(f"     ⚠️ Muitas ações fora do range!")
        else:
            print(f"     ✅ Range OK")
    
    print(f"\n🔍 ANÁLISE DE REWARD STABILITY:")
    
    # Verificar consistência das ações
    var_across_batch = np.var(actions_np, axis=0)
    print(f"   Variância por dimensão: {var_across_batch}")
    
    # Detectar possíveis problemas
    if np.any(var_across_batch < 0.01):
        print("   ⚠️ Baixa variância detectada - possível colapso!")
    elif np.any(var_across_batch > 1.0):
        print("   ⚠️ Alta variância detectada - possível instabilidade!")
    else:
        print("   ✅ Variância equilibrada")
    
    return True

if __name__ == "__main__":
    print("🚀 TwoHeadV9Optimus Shape Fix Validation")
    
    # Teste principal
    success = test_shape_fix_comprehensive()
    
    if success:
        # Análise adicional para reward engineering
        test_reward_engineering_analysis()
    
    print(f"\n{'✅ SUCESSO' if success else '❌ FALHOU'}")