#!/usr/bin/env python3
"""
🚨 DEBUG RADICAL - Investigar por que weight_hh_l0 morre instantaneamente
"""

import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class RadicalDebugCallback(BaseCallback):
    """🚨 Debug radical para descobrir a origem do problema"""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.step_count = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Debug apenas nos primeiros 1000 steps para capturar o problema
        if self.step_count <= 1000 and self.step_count % 100 == 0:
            self.radical_debug()
            
        return True
    
    def radical_debug(self):
        """🚨 Debug ultra-agressivo"""
        print(f"\n🚨 RADICAL DEBUG - Step {self.step_count}")
        
        try:
            # 1. VERIFICAR EPISODE STARTS
            if hasattr(self.training_env, 'get_attr'):
                dones = self.training_env.get_attr('_last_dones')
                print(f"📊 Episode starts: {np.mean(dones) if dones else 'N/A'}")
            
            # 2. VERIFICAR FEATURES DO EXTRACTOR
            policy = self.model.policy
            
            # Pegar uma observação sample
            obs = self.training_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            obs_tensor = torch.FloatTensor(obs).to(policy.device)
            
            with torch.no_grad():
                # Extrair features
                features = policy.features_extractor(obs_tensor)
                
                print(f"🔍 Features stats:")
                print(f"   Shape: {features.shape}")
                print(f"   Mean: {features.mean().item():.6f}")
                print(f"   Std: {features.std().item():.6f}")
                print(f"   Zeros: {(features == 0).float().mean().item()*100:.1f}%")
                print(f"   NaN/Inf: {torch.isnan(features).sum().item()}/{torch.isinf(features).sum().item()}")
                
                # 3. VERIFICAR LSTM STATES
                lstm_states = policy.get_initial_state(1)
                print(f"🧠 LSTM states shape: {[s.shape for s in lstm_states] if lstm_states else 'None'}")
                
                # 4. TESTE FORWARD DO CRITIC
                if hasattr(policy, 'forward_critic'):
                    values = policy.forward_critic(features.unsqueeze(0), lstm_states)
                    print(f"💰 Values:")
                    print(f"   Shape: {values.shape}")
                    print(f"   Mean: {values.mean().item():.6f}")
                    print(f"   Std: {values.std().item():.6f}")
                
                # 5. VERIFICAR WEIGHT_HH_L0 ATUAL
                if hasattr(policy, 'v7_critic_lstm'):
                    weight_hh = policy.v7_critic_lstm.weight_hh_l0
                    print(f"🎯 weight_hh_l0 stats:")
                    print(f"   Shape: {weight_hh.shape}")
                    print(f"   Mean: {weight_hh.mean().item():.6f}")
                    print(f"   Std: {weight_hh.std().item():.6f}")
                    print(f"   Zeros: {(torch.abs(weight_hh) < 1e-6).float().mean().item()*100:.1f}%")
                    
                    # GRADIENT se existir
                    if weight_hh.grad is not None:
                        print(f"📉 weight_hh_l0 gradients:")
                        print(f"   Mean: {weight_hh.grad.mean().item():.6f}")
                        print(f"   Std: {weight_hh.grad.std().item():.6f}")
                        print(f"   Zeros: {(torch.abs(weight_hh.grad) < 1e-6).float().mean().item()*100:.1f}%")
                    else:
                        print("📉 NO GRADIENTS YET")
                
        except Exception as e:
            print(f"❌ Radical debug error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("🚨 Radical Debug Callback criado")
    print("Use: RadicalDebugCallback() no training loop")