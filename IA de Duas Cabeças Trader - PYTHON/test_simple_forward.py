#!/usr/bin/env python3
"""
🧪 TESTE SIMPLES DE FORWARD PASS
Teste direto da policy sem usar model.predict()
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('.')

from trading_framework.policies.two_head_v8_elegance_4d import TwoHeadV8Elegance_4D, get_v8_elegance_4d_kwargs
import gym

def test_direct_forward():
    """Teste direto do forward pass da policy"""
    
    print("🧪 TESTE DIRETO DE FORWARD PASS")
    print("=" * 50)
    
    # Environment exato do 4dim.py
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
    action_space = gym.spaces.Box(low=np.array([0, 0, -1, -1]), high=np.array([2, 1, 1, 1]), dtype=np.float32)
    
    def dummy_lr_schedule(progress):
        return 1e-4
    
    try:
        print("🚀 Criando política V8Elegance_4D...")
        policy = TwoHeadV8Elegance_4D(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=dummy_lr_schedule,
            **get_v8_elegance_4d_kwargs()
        )
        print("✅ Política criada com sucesso!")
        
        # Teste de forward direto
        print("\n🏃 TESTE DE FORWARD DIRETO:")
        dummy_obs = torch.randn(1, 450).float()  # batch=1, obs=450
        
        with torch.no_grad():
            # Testar features extractor
            features = policy.extract_features(dummy_obs)
            print(f"   Features shape: {features.shape}")
            
            # Testar LSTM
            latent_pi, latent_vf, lstm_states = policy._get_latent(features)
            print(f"   LSTM pi shape: {latent_pi.shape}")
            print(f"   LSTM vf shape: {latent_vf.shape}")
            
            # Testar distribution
            distribution = policy._get_action_dist_from_latent(latent_pi)
            print(f"   Distribution type: {type(distribution)}")
            
            # Testar sample
            actions = distribution.sample()
            print(f"   Actions shape: {actions.shape}")
            print(f"   Actions: {actions}")
            
            if actions.shape[-1] == 4:
                print("   ✅ 4D Action Space VALIDADO!")
                entry_decision = actions[0, 0].item()
                confidence = actions[0, 1].item()
                pos1_mgmt = actions[0, 2].item()
                pos2_mgmt = actions[0, 3].item()
                
                print(f"   Entry Decision: {entry_decision:.3f} (esperado: 0-2)")
                print(f"   Confidence: {confidence:.3f} (esperado: 0-1)")
                print(f"   Pos1 Mgmt: {pos1_mgmt:.3f} (esperado: -1 a 1)")
                print(f"   Pos2 Mgmt: {pos2_mgmt:.3f} (esperado: -1 a 1)")
                
                print("\n🎉 SUCESSO! Forward pass funcionando perfeitamente!")
                return True
            else:
                print(f"   ❌ Action shape incorreta: {actions.shape}")
                return False
                
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_forward()
    if success:
        print("\n✅ Forward pass OK! A policy está funcionando.")
    else:
        print("\n❌ Forward pass com problemas.")