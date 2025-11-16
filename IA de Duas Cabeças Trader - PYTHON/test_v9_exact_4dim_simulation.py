"""
🔍 TESTE EXATO: Simular EXATAMENTE o 4dim.py com ambiente completo
"""
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from trading_framework.policies.two_head_v9_optimus import TwoHeadV9Optimus, get_v9_optimus_kwargs, fix_v9_optimus_weights

# Criar ambiente de trading simplificado
class SimpleTradingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Criar dados fake
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='5min')
        self.df = pd.DataFrame({
            'open': np.random.randn(1000).cumsum() + 100,
            'high': np.random.randn(1000).cumsum() + 101,
            'low': np.random.randn(1000).cumsum() + 99,
            'close': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.rand(1000) * 1000000
        }, index=dates)
        
        self.window_size = 10
        self.features_per_bar = 45
        
        # Spaces conforme 4dim.py
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, 
            shape=(self.window_size * self.features_per_bar,),  # 450
            dtype=np.float32
        )
        
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0, -1, -1]),
            high=np.array([2, 1, 1, 1]),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = 100
        
    def reset(self):
        self.current_step = self.window_size
        return self._get_observation()
    
    def _get_observation(self):
        # Gerar 450 features (10 bars × 45 features)
        obs = np.random.randn(self.window_size * self.features_per_bar).astype(np.float32)
        # Normalizar para [-1, 1]
        obs = np.clip(obs / 3, -1, 1)
        return obs
    
    def step(self, action):
        self.current_step += 1
        
        obs = self._get_observation()
        reward = np.random.randn() * 0.01  # Pequeno reward aleatório
        done = self.current_step >= len(self.df) - self.window_size or self.current_step >= self.max_steps
        info = {}
        
        return obs, reward, done, info

def check_weights_status(model, context=""):
    """Verifica status dos pesos críticos"""
    print(f"\n🔍 CHECK: {context}")
    
    if hasattr(model.policy, 'features_extractor'):
        fe = model.policy.features_extractor
        
        # input_projection
        if hasattr(fe, 'input_projection'):
            weight = fe.input_projection.weight
            zeros = (weight.abs() < 1e-8).float().mean().item() * 100
            print(f"   input_projection.weight: {zeros:.1f}% zeros {'🚨' if zeros > 50 else '✅'}")
        
        # _residual_projection
        if hasattr(fe, '_residual_projection'):
            weight = fe._residual_projection.weight
            zeros = (weight.abs() < 1e-8).float().mean().item() * 100
            print(f"   _residual_projection.weight: {zeros:.1f}% zeros {'🚨' if zeros > 50 else '✅'}")
    
    if hasattr(model.policy, 'market_context_encoder'):
        mce = model.policy.market_context_encoder
        
        if hasattr(mce, 'regime_embedding'):
            weight = mce.regime_embedding.weight
            zeros = (weight.abs() < 1e-8).float().mean().item() * 100
            print(f"   regime_embedding.weight: {zeros:.1f}% zeros {'🚨' if zeros > 50 else '✅'}")

def simulate_4dim_training():
    """Simula EXATAMENTE o processo do 4dim.py"""
    
    print("🚀 SIMULAÇÃO EXATA DO 4DIM.PY")
    print("="*60)
    
    # 1. Criar ambiente
    print("\n📊 PASSO 1: Criando ambiente de trading...")
    env = DummyVecEnv([lambda: SimpleTradingEnv()])
    
    # 2. Criar modelo EXATAMENTE como no 4dim.py
    print("\n📊 PASSO 2: Criando RecurrentPPO com V9Optimus...")
    
    def lr_schedule(progress):
        return 2e-5
    
    model_config = {
        "policy": TwoHeadV9Optimus,
        "env": env,
        "learning_rate": lr_schedule,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 8,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.12,
        "ent_coef": 0.1,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "policy_kwargs": get_v9_optimus_kwargs(),
        "verbose": 0,
        "device": "cpu"  # Usar CPU para evitar problemas de device
    }
    
    model = RecurrentPPO(**model_config)
    
    check_weights_status(model, "Após RecurrentPPO.__init__")
    
    # 3. Aplicar fix_v9_optimus_weights (como no 4dim.py)
    print("\n📊 PASSO 3: Aplicando fix_v9_optimus_weights...")
    fix_v9_optimus_weights(model)
    
    check_weights_status(model, "Após fix_v9_optimus_weights")
    
    # 4. Verificar se _fix_lstm_initialization seria chamado
    print("\n📊 PASSO 4: Verificando _fix_lstm_initialization...")
    policy_class_name = model.policy.__class__.__name__
    print(f"   Policy: {policy_class_name}")
    
    if "V9" in policy_class_name:
        print("   ✅ V9 detectada - _fix_lstm_initialization será PULADO")
    else:
        print("   ⚠️ _fix_lstm_initialization seria aplicado")
    
    # 5. Simular início do treino com callbacks
    print("\n📊 PASSO 5: Simulando início do treino com learn()...")
    
    from stable_baselines3.common.callbacks import BaseCallback
    
    class MonitorCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.step_count = 0
            
        def _on_step(self):
            self.step_count += 1
            
            # Monitorar a cada 100 steps
            if self.step_count % 100 == 0:
                print(f"\n   Step {self.step_count}:")
                
                # Verificar pesos
                if hasattr(self.model.policy, 'features_extractor'):
                    fe = self.model.policy.features_extractor
                    
                    if hasattr(fe, 'input_projection'):
                        weight = fe.input_projection.weight
                        zeros = (weight.abs() < 1e-8).float().mean().item() * 100
                        print(f"      input_projection: {zeros:.1f}% zeros")
                    
                    if hasattr(fe, '_residual_projection'):
                        weight = fe._residual_projection.weight
                        zeros = (weight.abs() < 1e-8).float().mean().item() * 100
                        print(f"      _residual_projection: {zeros:.1f}% zeros")
                
                if hasattr(self.model.policy, 'market_context_encoder'):
                    mce = self.model.policy.market_context_encoder
                    
                    if hasattr(mce, 'regime_embedding'):
                        weight = mce.regime_embedding.weight
                        zeros = (weight.abs() < 1e-8).float().mean().item() * 100
                        print(f"      regime_embedding: {zeros:.1f}% zeros")
            
            return True
    
    monitor_callback = MonitorCallback()
    
    try:
        # Treinar por poucos steps para ver o que acontece
        print("\n   Iniciando learn() por 500 steps...")
        model.learn(total_timesteps=500, callback=monitor_callback)
        
    except Exception as e:
        print(f"\n❌ Erro durante learn(): {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Verificação final
    print("\n" + "="*60)
    print("📊 VERIFICAÇÃO FINAL")
    print("="*60)
    
    check_weights_status(model, "Estado final após treino")
    
    # Análise detalhada
    print("\n🔍 ANÁLISE DETALHADA:")
    
    if hasattr(model.policy, 'features_extractor'):
        fe = model.policy.features_extractor
        
        if hasattr(fe, 'input_projection'):
            weight = fe.input_projection.weight
            print(f"\n   input_projection.weight:")
            print(f"      Shape: {weight.shape}")
            print(f"      Mean: {weight.mean().item():.6f}")
            print(f"      Std: {weight.std().item():.6f}")
            print(f"      Min: {weight.min().item():.6f}")
            print(f"      Max: {weight.max().item():.6f}")
            
            # Verificar quantos valores são EXATAMENTE zero
            exact_zeros = (weight == 0).float().mean().item() * 100
            near_zeros = (weight.abs() < 1e-8).float().mean().item() * 100
            print(f"      Exact zeros: {exact_zeros:.1f}%")
            print(f"      Near zeros (<1e-8): {near_zeros:.1f}%")

if __name__ == "__main__":
    simulate_4dim_training()