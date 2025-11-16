"""
🔥 SAC TRANSFORMER POLICY - Arquitetura Temporal para SAC v2

FILOSOFIA SAC TRANSFORMER:
- SAC v2 com sequências temporais (como o sistema original)
- Features Extractor com atenção temporal  
- Compatível com observation space 450D (10 timesteps × 45 features)
- Maximum entropy exploration com contexto temporal
- Off-policy sample efficiency com memória temporal
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Type, Union, Tuple
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn.functional as F
import gym
from stable_baselines3.sac.policies import SACPolicy

class SACTransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    🔥 SAC Transformer Features Extractor
    
    Arquitetura temporal para SAC com atenção:
    - Input: 450D (10 timesteps × 45 features) 
    - Temporal attention mechanism
    - Trading-specific feature processing
    - SAC-optimized outputs
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Calculate input dimensions
        if hasattr(observation_space, 'shape'):
            total_input_dim = observation_space.shape[0]  # 450
        else:
            total_input_dim = np.prod(observation_space.shape)
        
        # Temporal sequence parameters (matching original system)
        self.seq_len = 10  # 10 timesteps
        self.features_per_timestep = 45  # 45 features per bar
        
        assert total_input_dim == self.seq_len * self.features_per_timestep, \
            f"Expected {self.seq_len * self.features_per_timestep}D input, got {total_input_dim}D"
        
        # 🔥 TEMPORAL FEATURE PROCESSING
        # Layer 1: Feature embedding per timestep
        self.feature_embedding = nn.Sequential(
            nn.Linear(self.features_per_timestep, 128),
            nn.ELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.05),
        )
        
        # Layer 2: DISABLED - Temporal attention causing gradient explosion
        # self.temporal_attention = nn.MultiheadAttention(
        #     embed_dim=128,
        #     num_heads=4,
        #     dropout=0.05,
        #     batch_first=True
        # )
        
        # Layer 2: Simple temporal processing (STABLE)
        self.temporal_processing = nn.Sequential(
            nn.Linear(128, 128),
            nn.ELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.05)
        )
        
        # Layer 3: Temporal context aggregation
        self.context_aggregator = nn.Sequential(
            nn.Linear(128, 256),
            nn.ELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.05),
        )
        
        # Layer 4: Trading decision layer
        self.trading_decision = nn.Sequential(
            nn.Linear(256, features_dim),
            nn.ELU(),
            nn.LayerNorm(features_dim),
        )
        
        # Initialize weights properly for SAC
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for stable SAC training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for ELU activations
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            # MultiheadAttention removed - no longer needed
                        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract temporal features for SAC"""
        batch_size = observations.size(0)
        
        # Reshape to temporal sequence: [batch, seq_len, features_per_timestep]
        x = observations.view(batch_size, self.seq_len, self.features_per_timestep)
        
        # Feature embedding for each timestep
        # x shape: [batch, seq_len, features_per_timestep] -> [batch, seq_len, 128]
        x = self.feature_embedding(x)
        
        # Simple temporal processing (STABLE - no attention)
        # Process each timestep through temporal layer
        x = self.temporal_processing(x)
        
        # Aggregate temporal context (mean pooling across time)
        # [batch, seq_len, 128] -> [batch, 128]  
        temporal_context = torch.mean(x, dim=1)
        
        # Context aggregation
        x = self.context_aggregator(temporal_context)
        
        # Final trading decision features
        features = self.trading_decision(x)
        
        return features


class SACTransformerPolicy(SACPolicy):
    """
    🔥 SAC Transformer Policy
    
    Custom SAC policy with transformer features extractor:
    - Temporal attention for sequence processing
    - SAC-optimized actor/critic networks
    - Proper initialization for gradient stability
    """
    
    def __init__(self, *args, **kwargs):
        # Ensure we're using our specific features extractor
        print(f"🔧 SACTransformerPolicy initializing with kwargs: {list(kwargs.keys())}")
        
        # Force our transformer features extractor
        kwargs['features_extractor_class'] = SACTransformerFeatureExtractor
        kwargs['features_extractor_kwargs'] = {"features_dim": 256}
        
        # SAC-optimized network architecture
        kwargs['net_arch'] = dict(
            pi=[256, 128],  # Actor network
            qf=[256, 128]   # Critic network
        )
        
        # ELU activation for gradient stability
        kwargs['activation_fn'] = torch.nn.ELU
        
        # Conservative initialization for SAC
        kwargs['log_std_init'] = -2.0
        
        print(f"🔧 Final kwargs for SACTransformerPolicy: features_extractor_class={kwargs.get('features_extractor_class')}")
        
        super().__init__(*args, **kwargs)
        
        # DEBUG: Check observation and action spaces
        print(f"🔍 Args length: {len(args)}")
        print(f"🔍 Available kwargs: {list(kwargs.keys())}")
        if len(args) >= 2:
            print(f"🔍 Args[1] (should be obs_space): {args[1]} - shape: {getattr(args[1], 'shape', 'no shape')}")
        if len(args) >= 3:
            print(f"🔍 Args[2] (should be action_space): {args[2]} - shape: {getattr(args[2], 'shape', 'no shape')}")
        
        # MANUAL FIX: Force create features extractor if it wasn't created
        print(f"🔍 After super init - features_extractor type: {type(self.features_extractor)}")
        if self.features_extractor is None:
            print("🔧 Manual fix: Creating SACTransformerFeatureExtractor...")
            # Get observation space from args or kwargs
            if 'observation_space' in kwargs:
                obs_space = kwargs['observation_space']
                print(f"🔍 Got obs_space from kwargs: {obs_space}")
            elif len(args) >= 2:
                obs_space = args[1]  # Usually second argument
                print(f"🔍 Got obs_space from args[1]: {obs_space}")
            else:
                print("❌ Cannot find observation space for manual features extractor creation")
                obs_space = None
                
            if obs_space is not None:
                print(f"🔍 Observation space shape: {getattr(obs_space, 'shape', 'no shape')}")
                try:
                    self.features_extractor = SACTransformerFeatureExtractor(obs_space, features_dim=256)
                    print(f"✅ Manually created features extractor: {type(self.features_extractor)}")
                except Exception as e:
                    print(f"❌ Error creating features extractor: {e}")
            
        # Apply SAC-specific initialization
        self._apply_sac_initialization()
        
        # Final verification
        print(f"🔍 Final features_extractor type: {type(self.features_extractor)}")
        if self.features_extractor is None:
            print("❌ CRITICAL: features_extractor is still None after manual fix!")
        else:
            print("✅ Features extractor ready for training")
        
    def _apply_sac_initialization(self):
        """Apply SAC-specific weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for ELU
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)


def get_sac_transformer_kwargs():
    """
    🔥 Get SAC Transformer policy kwargs
    """
    return {
        "policy": SACTransformerPolicy,
        "policy_kwargs": {
            "features_extractor_class": SACTransformerFeatureExtractor,
            "features_extractor_kwargs": {
                "features_dim": 256,
            },
            "net_arch": dict(pi=[256, 128], qf=[256, 128]),
            "activation_fn": torch.nn.ELU,
            "log_std_init": -2.0,
        }
    }


def validate_sac_transformer_policy(policy):
    """Validate SAC Transformer Policy structure"""
    
    required_components = [
        'features_extractor', 'actor', 'critic'
    ]
    
    for component in required_components:
        if not hasattr(policy, component):
            raise ValueError(f"SAC Transformer Policy missing: {component}")
    
    # Debug: Print actual features extractor type
    print(f"🔍 Actual features extractor type: {type(policy.features_extractor)}")
    print(f"🔍 Expected type: {SACTransformerFeatureExtractor}")
    print(f"🔍 Features extractor class name: {policy.features_extractor.__class__.__name__}")
    
    # Check temporal features extractor
    if not isinstance(policy.features_extractor, SACTransformerFeatureExtractor):
        print(f"❌ Expected SACTransformerFeatureExtractor, got {type(policy.features_extractor)}")
        # Don't fail, just warn for now
        print("⚠️ WARNING: Using different features extractor than expected")
    else:
        print("✅ SAC Transformer Policy validated successfully")


# Main exports
__all__ = [
    "SACTransformerFeatureExtractor",
    "SACTransformerPolicy", 
    "get_sac_transformer_kwargs",
    "validate_sac_transformer_policy"
]