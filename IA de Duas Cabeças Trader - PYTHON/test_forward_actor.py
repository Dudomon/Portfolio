#!/usr/bin/env python3
"""
🧪 TESTE ESPECÍFICO DO FORWARD_ACTOR
"""

import torch
import numpy as np
import sys
sys.path.append('.')

from trading_framework.policies.two_head_v8_elegance_4d import TwoHeadV8Elegance_4D, get_v8_elegance_4d_kwargs
import gym

def test_forward_actor():
    """Teste específico do forward_actor"""
    
    print("🧪 TESTE FORWARD_ACTOR")
    print("=" * 40)
    
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
    action_space = gym.spaces.Box(low=np.array([0, 0, -1, -1]), high=np.array([2, 1, 1, 1]), dtype=np.float32)
    
    def dummy_lr_schedule(progress):
        return 1e-4
    
    try:
        # Criar policy
        policy = TwoHeadV8Elegance_4D(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=dummy_lr_schedule,
            **get_v8_elegance_4d_kwargs()
        )
        
        print("✅ Policy criada")
        
        # Testar forward_actor com features corretas do 4dim.py (450D)
        dummy_features = torch.randn(1, 450).float()  # 4dim.py observation space: 450D
        
        with torch.no_grad():
            print("\n🎯 Testando forward_actor...")
            dummy_lstm_states = None  # LSTM states
            dummy_episode_starts = torch.zeros(1, dtype=torch.bool)  # Episode starts
            distribution = policy.forward_actor(dummy_features, dummy_lstm_states, dummy_episode_starts)
            print(f"   Distribution type: {type(distribution)}")
            print(f"   Has get_actions: {hasattr(distribution, 'get_actions')}")
            
            if hasattr(distribution, 'get_actions'):
                actions = distribution.get_actions(deterministic=True)
                print(f"   Actions shape: {actions.shape}")
                print(f"   Actions: {actions}")
                print("   ✅ forward_actor funcionando!")
                return True
            else:
                print("   ❌ Distribution sem get_actions")
                return False
                
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_forward_actor()
    if success:
        print("\n✅ forward_actor OK!")
    else:
        print("\n❌ forward_actor com problemas.")