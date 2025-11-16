"""
🔥 V12SAC - "SAC v2 Trading Features Extractor"

FILOSOFIA V12SAC:
- Features extractor para SAC v2
- MLP-based architecture (sem LSTM)
- Ações contínuas nativas
- Maximum entropy exploration
- Off-policy sample efficiency

ARQUITETURA SAC:
- Custom Features Extractor (256D → 512D → 256D)
- Trading-specific feature engineering
- Continuous action space [position_size, stop_loss, take_profit, hold_time]
- Twin Q-networks (automático no SAC)
- Automatic entropy tuning
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Type, Union, Tuple
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F
import gym
import math
from .sac_trading_transformer import SACTradingTransformerExtractor, get_sac_trading_transformer_kwargs

class V12SACFeatureExtractor(BaseFeaturesExtractor):
    """
    🔥 V12SAC Features Extractor for SAC v2
    
    Custom feature extractor optimized for trading with SAC:
    - MLP-based (no LSTM memory)
    - Trading-specific layers
    - Continuous action friendly
    - Maximum information extraction
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Get input dimension
        if hasattr(observation_space, 'shape'):
            input_dim = observation_space.shape[0]
        else:
            # Fallback for Box spaces
            input_dim = np.prod(observation_space.shape)
        
        self.input_dim = input_dim
        
        # 🚀 SAC-OPTIMIZED ARCHITECTURE WITH PROPER INITIALIZATION
        
        # Layer 1: Raw features processing with SAC-optimized init
        self.features_layer1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(alpha=1.0),  # ELU instead of ReLU (no dying neuron problem)
            nn.LayerNorm(512),  # LayerNorm for gradient stability
            nn.Dropout(0.05),   # Reduced dropout for SAC
        )
        
        # Layer 2: Market analysis layer with residual connection
        self.market_analysis = nn.Sequential(
            nn.Linear(512, 384),
            nn.ELU(alpha=1.0),  # ELU for non-zero gradients
            nn.LayerNorm(384),  # Stabilize gradients
            nn.Dropout(0.05),
        )
        
        # Layer 3: Trading decision layer with careful initialization
        self.trading_layer = nn.Sequential(
            nn.Linear(384, features_dim),
            nn.Tanh(),  # Tanh for bounded outputs (SAC-friendly)
        )
        
        # 🎯 NORMALIZATION LAYER (crítico para SAC)
        self.feature_norm = nn.LayerNorm(features_dim)
        
        # 🔥 CRITICAL: SAC-SPECIFIC INITIALIZATION
        self._initialize_weights()
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features for SAC policy and critic networks"""
        
        # Ensure correct input shape
        if observations.dim() > 2:
            observations = observations.view(observations.size(0), -1)
            
        # Feature processing pipeline
        x = self.features_layer1(observations)
        x = self.market_analysis(x)
        x = self.trading_layer(x)
        
        # Normalize features (importante para SAC stability)
        x = self.feature_norm(x)
        
        return x
    
    def _initialize_weights(self):
        """
        🔥 CRITICAL: SAC-specific weight initialization to prevent zero gradients
        
        This initialization is crucial for SAC because:
        1. SAC is sensitive to weight initialization due to entropy regularization
        2. Poor initialization leads to saturation and zero gradients
        3. ELU requires different initialization than ReLU
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization optimized for ELU/Tanh
                # Using fan_in mode for better gradient flow in SAC
                nn.init.xavier_normal_(module.weight, gain=1.0)
                
                if module.bias is not None:
                    # Small positive bias to avoid initial saturation
                    nn.init.constant_(module.bias, 0.01)
            
            elif isinstance(module, nn.LayerNorm):
                # Proper LayerNorm initialization
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)

def get_v12_sac_kwargs():
    """
    🔥 Get SAC v2 policy kwargs with V12 features extractor - ZERO GRADIENT FIX
    """
    return {
        "features_extractor_class": V12SACFeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": 256,
        },
        # 🚀 CRITICAL: SAC-optimized network architecture 
        "net_arch": dict(
            pi=[256, 128],    # Actor: Smaller, more stable
            qf=[256, 128]     # Critic: Twin Q-networks will be smaller
        ),
        # 🔥 ELU activation for all SAC networks (prevents dying neurons)
        "activation_fn": torch.nn.ELU,
        
        # 🎯 SAC-specific policy kwargs for gradient stability
        "log_std_init": -3.0,  # Conservative initial exploration
        "use_sde": False,      # Disable state-dependent exploration for stability
    }

# Compatibility class (para manter compatibilidade com código existente)
class V12SACPolicy:
    """Compatibility wrapper - SAC usa MlpPolicy internamente"""
    pass

# Para debug e informação
def get_v12_sac_info():
    """Return information about V12SAC configuration - SAC TRADING TRANSFORMER VERSION"""
    return {
        "algorithm": "SAC v2",
        "policy_type": "MlpPolicy with Trading Transformer Features",
        "features_extractor": "SACTradingTransformerExtractor (Gradient-Stable)",
        "features_dim": 256,
        "action_space": "Continuous Box",
        "memory": "Experience Replay Buffer",
        "transformer_features": [
            "Stable multi-head attention",
            "Positional encoding for temporal data",
            "Gradient-stable initialization (gain=0.05)",
            "LayerNorm for stability",
            "Attention pooling for global context"
        ],
        "stability_improvements": [
            "Very small initialization gains",
            "Post-norm transformer layers",
            "Residual connections",
            "Dropout for regularization",
            "GELU activation for smooth gradients"
        ],
        "advantages": [
            "Temporal sequence understanding",
            "Off-policy sample efficiency",
            "Automatic entropy tuning",
            "Twin Q-networks", 
            "Continuous actions native",
            "Gradient explosion prevention"
        ]
    }


def get_v12_trading_transformer_kwargs(features_dim: int = 256) -> Dict[str, Any]:
    """Get V12 SAC Trading Transformer kwargs"""
    return get_sac_trading_transformer_kwargs(features_dim)

def apply_sac_initialization_fix(model):
    """
    🔥 CRITICAL: Apply additional SAC initialization fixes to trained model
    
    Call this after model creation to ensure all networks have proper initialization
    """
    if not hasattr(model, 'policy'):
        return
        
    def init_sac_network(module):
        if isinstance(module, nn.Linear):
            # Xavier initialization for SAC networks
            nn.init.xavier_normal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)
    
    # Apply to policy networks (actor)
    if hasattr(model.policy, 'actor'):
        model.policy.actor.apply(init_sac_network)
    
    # Apply to critic networks (twin Q-networks)
    if hasattr(model.policy, 'critic'):
        model.policy.critic.apply(init_sac_network)
    
    # Apply to critic target networks
    if hasattr(model.policy, 'critic_target'):
        model.policy.critic_target.apply(init_sac_network)
    
    print("✅ SAC initialization fix applied to all networks")

# Main exports
__all__ = [
    "V12SACFeatureExtractor",
    "get_v12_sac_kwargs", 
    "V12SACPolicy",
    "get_v12_sac_info",
    "apply_sac_initialization_fix"
]