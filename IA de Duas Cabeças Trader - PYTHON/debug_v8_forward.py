"""
Debug detalhado do forward pass V8
"""

import torch
import numpy as np
from gym import spaces
from trading_framework.policies.two_head_v8_heritage import TwoHeadV8Heritage

def debug_forward():
    print("🔍 DEBUG V8 Forward Pass")
    
    # Criar policy
    obs_space = spaces.Box(low=-10, high=10, shape=(2580,), dtype=np.float32)
    action_space = spaces.Box(low=-3, high=3, shape=(8,), dtype=np.float32)
    
    def lr_schedule(progress):
        return 3e-4
    
    policy = TwoHeadV8Heritage(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        enable_heritage_mode=True
    )
    
    print("\n1️⃣ Testando UnifiedV8FeatureProcessor...")
    
    # Test input
    obs = torch.randn(1, 2580)
    print(f"Input obs shape: {obs.shape}")
    
    try:
        features, regime_id, info = policy.unified_processor(obs, for_actor=True)
        print(f"✅ Features shape: {features.shape}")
        print(f"✅ Regime ID: {regime_id}")
        print(f"✅ Info: {info}")
        
    except Exception as e:
        print(f"❌ Erro no UnifiedV8FeatureProcessor: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n2️⃣ Testando LSTM processing...")
    
    try:
        lstm_states = (
            torch.zeros(1, 1, policy.v8_lstm_hidden),
            torch.zeros(1, 1, policy.v8_lstm_hidden)
        )
        
        # Process through LSTM first  
        lstm_output, _ = policy.neural_architecture.actor_lstm(
            features.unsqueeze(1), lstm_states
        )
        print(f"✅ LSTM output shape: {lstm_output.shape}")
        
        # Now test decision maker with correct input size
        actions = policy.decision_maker(lstm_output.squeeze(1))
        print(f"✅ Raw actions shape: {actions.shape}")
        
    except Exception as e:
        print(f"❌ Erro no LSTM/DecisionMaker: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n3️⃣ Testando Neural Architecture...")
    
    try:
        lstm_states = (
            torch.zeros(1, 1, policy.v8_lstm_hidden),
            torch.zeros(1, 1, policy.v8_lstm_hidden)
        )
        episode_starts = torch.tensor([True])
        
        # Test with proper 3D input for LSTM (batch, seq, features)
        features_3d = features.unsqueeze(1)  # (1, 1, 512)
        print(f"Features 3D shape: {features_3d.shape}")
        
        actor_output, new_states = policy.neural_architecture.forward_actor(
            features_3d, lstm_states, episode_starts
        )
        print(f"✅ Actor output shape: {actor_output.shape}")
        
    except Exception as e:
        print(f"❌ Erro no Neural Architecture: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n4️⃣ Testando Forward Actor completo...")
    
    try:
        actions, new_lstm_states, info_dict = policy.forward_actor(
            obs, lstm_states, episode_starts
        )
        print(f"✅ Final actions shape: {actions.shape}")
        
    except Exception as e:
        print(f"❌ Erro no Forward Actor: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ Debug concluído com sucesso!")

if __name__ == "__main__":
    debug_forward()