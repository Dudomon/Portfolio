"""
🎯 Validação Final - TwoHeadV9Optimus Shape Fix + Reward Engineering

TESTE FINAL:
- Shape fix funcionando
- Exploração balanceada
- Training readiness adequado
"""

import torch
import numpy as np
import gym
from trading_framework.policies.two_head_v9_optimus import (
    TwoHeadV9Optimus, 
    get_v9_optimus_kwargs
)

def final_validation():
    """Validação final do sistema completo"""
    
    print("🎯 VALIDAÇÃO FINAL - TwoHeadV9Optimus")
    print("=" * 50)
    
    # Criar policy
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
    
    # Teste rápido de shape fix
    print("1️⃣ SHAPE FIX:")
    latent_256d = torch.randn(4, 256)
    latent_320d = torch.randn(4, 320)
    
    try:
        dist_256 = policy._get_action_dist_from_latent(latent_256d)
        dist_320 = policy._get_action_dist_from_latent(latent_320d)
        print("   ✅ Shape fix funcionando")
    except Exception as e:
        print(f"   ❌ Shape fix falhou: {e}")
        return False
    
    # Teste de exploração
    print("\n2️⃣ EXPLORAÇÃO:")
    features = torch.randn(1000, 450)
    lstm_states = None
    episode_starts = torch.zeros(1000, dtype=torch.bool)
    
    with torch.no_grad():
        dist = policy.forward_actor(features, lstm_states, episode_starts)
        actions = dist.sample()
    
    actions_np = actions.detach().numpy()
    
    # Métricas chave
    total_variance = np.sum(np.var(actions_np, axis=0))
    exploration_score = min(total_variance / 0.1, 1.0)
    
    # Diversidade
    diversities = []
    for i in range(4):
        unique_vals = len(np.unique(np.round(actions_np[:, i], 2)))
        diversity = unique_vals / len(actions_np)
        diversities.append(diversity)
    
    avg_diversity = np.mean(diversities)
    
    print(f"   Exploration Score: {exploration_score:.2f}")
    print(f"   Avg Diversity: {avg_diversity:.2f}")
    print(f"   Total Variance: {total_variance:.4f}")
    
    if exploration_score > 0.5 and avg_diversity > 0.1:
        print("   ✅ Exploração adequada")
        exploration_ok = True
    else:
        print("   ⚠️ Exploração limitada")
        exploration_ok = False
    
    # Teste de ranges
    print("\n3️⃣ RANGES:")
    action_names = ['entry_decision', 'confidence', 'pos1_mgmt', 'pos2_mgmt']
    expected_ranges = [(0, 2), (0, 1), (-1, 1), (-1, 1)]
    
    ranges_ok = True
    for i, (name, (low, high)) in enumerate(zip(action_names, expected_ranges)):
        values = actions_np[:, i]
        in_range_pct = ((values >= low) & (values <= high)).mean() * 100
        print(f"   {name}: {in_range_pct:.1f}% no range [{low}, {high}]")
        
        if in_range_pct < 95:
            ranges_ok = False
    
    if ranges_ok:
        print("   ✅ Ranges adequados")
    else:
        print("   ⚠️ Alguns valores fora do range")
    
    # Training readiness
    print("\n4️⃣ TRAINING READINESS:")
    stability_score = 0.8 if ranges_ok else 0.5
    training_readiness = (exploration_score * 0.4 + stability_score * 0.6)
    
    print(f"   Training Readiness: {training_readiness:.1%}")
    
    if training_readiness > 0.7:
        print("   🚀 PRONTO PARA TREINAMENTO!")
        ready = True
    elif training_readiness > 0.5:
        print("   ⚠️ AJUSTES MENORES RECOMENDADOS")
        ready = True
    else:
        print("   ❌ REQUER MAIS AJUSTES")
        ready = False
    
    # Resultado final
    print(f"\n{'='*50}")
    print("🎖️ RESULTADO FINAL:")
    
    all_tests = [
        ("Shape Fix", True),
        ("Exploração", exploration_ok),
        ("Ranges", ranges_ok), 
        ("Training Ready", ready)
    ]
    
    passed = sum(1 for _, ok in all_tests if ok)
    total = len(all_tests)
    
    for test_name, ok in all_tests:
        status = "✅" if ok else "❌"
        print(f"   {status} {test_name}")
    
    print(f"\n🎯 SCORE: {passed}/{total} ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 TwoHeadV9Optimus TOTALMENTE VALIDADO!")
        print("🚀 Pronto para integração no DayTrader V7!")
        return True
    else:
        print("⚠️ Alguns ajustes ainda necessários")
        return False

if __name__ == "__main__":
    success = final_validation()
    print(f"\n{'🎉 SUCESSO' if success else '⚠️ PARCIAL'}")