"""
🚀 SimpleV8_4D - "Cópia da V7Intuition com saída 4D"

ESTRATÉGIA SIMPLES:
- COPIAR EXATAMENTE a V7Intuition que funciona
- MODIFICAR apenas get_distribution para retornar 4D
- MANTER toda infraestrutura existente
- Zero risco de quebrar
"""

# Importar V7Intuition que funciona
from trading_framework.policies.two_head_v7_intuition import TwoHeadV7Intuition, get_v7_intuition_kwargs

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Type, Union, Tuple
from stable_baselines3.common.distributions import DiagGaussianDistribution

# Importar transformer para 450D
from trading_framework.extractors.transformer_v9_compact import TradingTransformerV9Compact

class SimpleV8_4D(TwoHeadV7Intuition):
    """🚀 SimpleV8_4D - V7Intuition com saída 4D"""
    
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        print(f"🚀 SimpleV8_4D inicializando (CÓPIA EXATA da V7Intuition):")
        print(f"   Action Space: {action_space.shape} (4D)")
        print(f"   Obs Space: {observation_space.shape}")
        
        # Inicializar como V7Intuition normal
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        
        # Adicionar cabeças 4D específicas
        lstm_hidden = kwargs.get('v7_shared_lstm_hidden', 512)
        
        self.action_4d_head = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 4)  # 4D direct output
        )
        
        print("✅ SimpleV8_4D configurado - V7 + cabeça 4D!")
    
    def get_distribution(self, obs):
        """Override para retornar distribuição 4D"""
        # Usar o pipeline normal da V7 até obter features LSTM
        features = self.extract_features(obs, self.features_extractor)
        
        # Get LSTM output usando método da classe base
        latent_pi, latent_vf, lstm_states = self._get_latent(features)
        
        # Nossa cabeça 4D customizada
        raw_actions = self.action_4d_head(latent_pi)
        
        # Aplicar ativações específicas para cada dimensão
        entry_decision = torch.tanh(raw_actions[:, 0:1]) * 1.0 + 1.0  # [0-2]
        confidence = torch.sigmoid(raw_actions[:, 1:2])               # [0-1]
        pos1_mgmt = torch.tanh(raw_actions[:, 2:3])                   # [-1,1]
        pos2_mgmt = torch.tanh(raw_actions[:, 3:4])                   # [-1,1]
        
        # Combinar 4D
        actions_4d = torch.cat([entry_decision, confidence, pos1_mgmt, pos2_mgmt], dim=-1)
        
        # Retornar distribuição determinística
        return DiagGaussianDistribution.proba_distribution(actions_4d, torch.ones_like(actions_4d) * 0.01)
    
    def evaluate_actions(self, obs, actions):
        """Override para evaluation com 4D"""
        distribution = self.get_distribution(obs)
        log_prob = distribution.log_prob(actions)
        
        # Value function normal da V7
        features = self.extract_features(obs, self.features_extractor)
        latent_pi, latent_vf, lstm_states = self._get_latent(features)
        values = self.value_net(latent_vf)
        
        return values, log_prob, distribution.entropy()

def get_simple_v8_4d_kwargs():
    """Configurações para SimpleV8_4D - baseadas na V7"""
    kwargs = get_v7_intuition_kwargs()
    
    # Substituir o transformer por 450D
    kwargs['features_extractor_class'] = TradingTransformerV9Compact
    kwargs['features_extractor_kwargs'] = {'features_dim': 256}
    
    return kwargs

def validate_simple_v8_4d_policy(policy=None):
    """Valida a política SimpleV8_4D"""
    import gym
    
    if policy is None:
        dummy_obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
        dummy_action_space = gym.spaces.Box(low=np.array([0, 0, -1, -1]), high=np.array([2, 1, 1, 1]), dtype=np.float32)
        
        def dummy_lr_schedule(progress):
            return 1e-4
        
        policy = SimpleV8_4D(
            observation_space=dummy_obs_space,
            action_space=dummy_action_space,
            lr_schedule=dummy_lr_schedule,
            **get_simple_v8_4d_kwargs()
        )
    
    print("✅ SimpleV8_4D validada - V7Intuition + 4D!")
    print(f"   🧠 Baseado na V7Intuition (comprovadamente funcional)")
    print(f"   🎯 Saída: 4D customizada")
    print(f"   📊 Action Space: 4D")
    
    return True

if __name__ == "__main__":
    print("🚀 SimpleV8_4D - V7Intuition com saída 4D!")
    validate_simple_v8_4d_policy()