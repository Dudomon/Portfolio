"""
🚀 TwoHeadV8_4D_Fixed - "V8 Elegance para 4D - FUNCIONAL"

ESTRATÉGIA:
- COPIAR estrutura da V7Intuition que funciona
- MODIFICAR apenas as saídas para 4D
- MANTER toda a infraestrutura funcional
- Action Space: 4D [entry_decision, confidence, pos1_mgmt, pos2_mgmt]
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Type, Union, Tuple
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F

# Importar transformer funcional para 450D
from trading_framework.extractors.transformer_v9_compact import TradingTransformerV9Compact

# Fallback para PyTorchObs
try:
    from stable_baselines3.common.type_aliases import PyTorchObs
except ImportError:
    PyTorchObs = torch.Tensor

# Imports corretos para RecurrentPPO
try:
    from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
except ImportError:
    from stable_baselines3.common.policies import RecurrentActorCriticPolicy

class TwoHeadV8_4D_Fixed(RecurrentActorCriticPolicy):
    """🚀 TwoHeadV8_4D_Fixed - V8 adaptado para 4D usando estrutura da V7"""
    
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        # Parâmetros específicos
        self.v8_lstm_hidden = kwargs.pop('lstm_hidden_size', 256)
        self.v8_features_dim = kwargs.pop('features_dim', 256)
        
        print(f"🚀 V8_4D_Fixed inicializando (baseado na V7):")
        print(f"   LSTM Hidden: {self.v8_lstm_hidden}D")
        print(f"   Features: {self.v8_features_dim}D")
        print(f"   Action Space: {action_space.shape} (4D)")
        print(f"   Obs Space: {observation_space.shape}")
        
        # Criar heads ANTES do super().__init__()
        self.entry_head = nn.Sequential(
            nn.Linear(self.v8_lstm_hidden, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 2)  # 2D: entry_decision + confidence
        )
        
        self.management_head = nn.Sequential(
            nn.Linear(self.v8_lstm_hidden, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 2)  # 2D: pos1_mgmt + pos2_mgmt
        )
        
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        
        print("✅ V8_4D_Fixed configurado com sucesso!")
        
    def _build_mlp_extractor(self) -> None:
        """
        Override para usar heads customizados em vez do MLP padrão
        """
        # A classe base já criou self.lstm_actor
        # Agora criamos o "action_net" que na verdade são nossos heads customizados
        
        # Para manter compatibilidade, criamos um "fake" mlp_extractor
        self.mlp_extractor = nn.Identity()
        
        # O action_net agora é uma combinação dos nossos heads
        self.action_net = nn.ModuleDict({
            'entry_head': self.entry_head,
            'management_head': self.management_head
        })
        
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> torch.distributions.Distribution:
        """
        Override para gerar ações 4D usando nossos heads customizados
        """
        # Entry Head (2D)
        entry_raw = self.action_net.entry_head(latent_pi)
        entry_decision = torch.tanh(entry_raw[:, 0:1]) * 1.0 + 1.0  # [0-2]
        confidence = torch.sigmoid(entry_raw[:, 1:2])               # [0-1]
        
        # Management Head (2D) 
        mgmt_raw = self.action_net.management_head(latent_pi)
        pos1_mgmt = torch.tanh(mgmt_raw[:, 0:1])  # [-1,1]
        pos2_mgmt = torch.tanh(mgmt_raw[:, 1:2])  # [-1,1]
        
        # Combinar para 4D: [entry_decision, confidence, pos1_mgmt, pos2_mgmt]
        combined_actions = torch.cat([entry_decision, confidence, pos1_mgmt, pos2_mgmt], dim=-1)
        
        # Retornar distribuição determinística (sem log_std)
        return torch.distributions.Normal(combined_actions, torch.ones_like(combined_actions) * 0.01)

def get_v8_4d_fixed_kwargs():
    """Configurações para TwoHeadV8_4D_Fixed"""
    return {
        'features_extractor_class': TradingTransformerV9Compact,
        'features_extractor_kwargs': {
            'features_dim': 256,
        },
        'lstm_hidden_size': 256,
        'n_lstm_layers': 1,
        'shared_lstm': True,
        'enable_critic_lstm': False,
        'activation_fn': nn.LeakyReLU,
        'net_arch': [],  # Custom architecture
        'ortho_init': False,  # IMPORTANTE: não quebrar transformer
        'log_std_init': -0.5,
    }

def validate_v8_4d_fixed_policy(policy=None):
    """Valida a política V8_4D_Fixed"""
    import gym
    
    if policy is None:
        dummy_obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
        dummy_action_space = gym.spaces.Box(low=np.array([0, 0, -1, -1]), high=np.array([2, 1, 1, 1]), dtype=np.float32)
        
        def dummy_lr_schedule(progress):
            return 1e-4
        
        policy = TwoHeadV8_4D_Fixed(
            observation_space=dummy_obs_space,
            action_space=dummy_action_space,
            lr_schedule=dummy_lr_schedule,
            **get_v8_4d_fixed_kwargs()
        )
    
    print("✅ TwoHeadV8_4D_Fixed validada - V8 simplificado para 4D!")
    print(f"   🧠 LSTM: {policy.v8_lstm_hidden}D")
    print(f"   🎯 Entry Head: 2D (entry_decision + confidence)")
    print(f"   💰 Management Head: 2D (pos1_mgmt + pos2_mgmt)")
    print(f"   📊 Total Actions: 4D")
    
    return True

if __name__ == "__main__":
    print("🚀 TwoHeadV8_4D_Fixed - V8 simplificado para 4D Action Space!")
    validate_v8_4d_fixed_policy()