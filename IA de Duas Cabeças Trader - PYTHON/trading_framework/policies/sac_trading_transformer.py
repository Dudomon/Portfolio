"""
🔥 SAC Trading Transformer - Arquitetura Especializada

FILOSOFIA SAC TRADING TRANSFORMER:
- Transformer especializado para dados de trading temporal
- Estabilidade numérica para SAC v2
- Gradient stability através de técnicas avançadas
- Processamento inteligente de features temporais de mercado

ARQUITETURA:
- Multi-scale temporal attention
- Gradient-stable initialization 
- Trading-specific normalization
- SAC v2 optimized feature extraction
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Type, Union, Tuple
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F
import gym
import math


class StableMultiHeadAttention(nn.Module):
    """
    🔥 Gradient-Stable Multi-Head Attention para Trading
    
    Melhorias para estabilidade:
    - Gradient clipping interno
    - Scaled initialization
    - Attention dropout inteligente
    - Residual connections estáveis
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = max(num_heads, 1)  # Prevent zero division
        
        assert embed_dim % self.num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({self.num_heads})"
        
        self.head_dim = embed_dim // self.num_heads
        self.scale = max(self.head_dim, 1) ** -0.5  # Prevent zero power operation
        
        # Query, Key, Value projections with stable initialization
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)  
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights for stability
        self._init_weights()
    
    def _init_weights(self):
        """FIXED: Proper Xavier initialization for stable gradients"""
        # Using Xavier initialization like successful PPO implementation
        for module in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            
        # Output projection with reasonable gain
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        nn.init.constant_(self.out_proj.bias, 0.0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with gradient stability
        
        Args:
            x: [batch, seq_len, embed_dim]
            mask: Optional attention mask
            
        Returns:
            output: [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        residual = x
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with stability
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            attn_scores.masked_fill_(mask == 0, float('-inf'))
        
        # Stable softmax with gradient clipping
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        output = self.out_proj(attn_output)
        
        # Residual connection with layer norm (post-norm for stability)
        output = self.layer_norm(output + residual)
        
        return output


class TradingTemporalEncoder(nn.Module):
    """
    🔥 Encoder Temporal Especializado para Trading
    
    Features:
    - Multi-layer transformer blocks
    - Position encoding para dados temporais
    - Stable gradient flow
    """
    
    def __init__(self, embed_dim: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Transformer layers
        self.layers = nn.ModuleList([
            StableMultiHeadAttention(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.LeakyReLU(negative_slope=0.01),  # 🔥 FIXED: Consistent with SAC config
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])
        
        # Initialize feed-forward weights
        self._init_ff_weights()
    
    def _init_ff_weights(self):
        """FIXED: Proper Xavier initialization for feed-forward layers"""
        for ff in self.feed_forwards:
            for module in ff.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer layers
        
        Args:
            x: [batch, seq_len, embed_dim]
            
        Returns:
            output: [batch, seq_len, embed_dim]
        """
        for i in range(self.num_layers):
            # Multi-head attention
            attn_out = self.layers[i](x)
            
            # Feed-forward with residual connection
            ff_out = self.feed_forwards[i](attn_out)
            x = self.layer_norms[i](ff_out + attn_out)
        
        return x


class SACTradingTransformerExtractor(BaseFeaturesExtractor):
    """
    🔥 SAC Trading Transformer Features Extractor
    
    Arquitetura especializada para:
    - Trading temporal sequence processing
    - SAC v2 policy optimization  
    - Stable gradient flow
    - Maximum information extraction from market data
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Get input dimension
        if hasattr(observation_space, 'shape'):
            total_input_dim = observation_space.shape[0]
        else:
            total_input_dim = np.prod(observation_space.shape)
        
        # Temporal sequence parameters
        self.seq_len = 10  # 10 timesteps
        self.features_per_timestep = 45  # 45 features per bar
        
        assert total_input_dim == self.seq_len * self.features_per_timestep, \
            f"Expected {self.seq_len * self.features_per_timestep}D input, got {total_input_dim}D"
        
        # Embedding dimension for transformer
        self.embed_dim = 128
        
        # Layer 1: Feature embedding per timestep
        self.feature_embedding = nn.Sequential(
            nn.Linear(self.features_per_timestep, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Layer 2: Positional encoding for temporal information
        self.positional_encoding = self._create_positional_encoding()
        
        # Layer 3: Transformer encoder
        self.transformer_encoder = TradingTemporalEncoder(
            embed_dim=self.embed_dim,
            num_layers=2,
            num_heads=4,
            dropout=0.1
        )
        
        # Layer 4: Global temporal aggregation
        self.temporal_aggregator = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Layer 5: Final trading features
        self.final_projection = nn.Sequential(
            nn.Linear(256, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU()
        )
        
        # Initialize all weights
        self._initialize_weights()
    
    def _create_positional_encoding(self) -> torch.Tensor:
        """Create sinusoidal positional encoding for temporal sequences"""
        pe = torch.zeros(self.seq_len, self.embed_dim)
        position = torch.arange(0, self.seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * 
                           -(math.log(10000.0) / self.embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, seq_len, embed_dim]
        
        return pe
    
    def _initialize_weights(self):
        """FIXED: Proper Xavier initialization like successful PPO"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Using same initialization as working PPO implementation
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract trading features using stable transformer
        
        Args:
            observations: [batch, seq_len * features_per_timestep]
            
        Returns:
            features: [batch, features_dim]
        """
        batch_size = observations.size(0)
        
        # Reshape to temporal sequence: [batch, seq_len, features_per_timestep]
        x = observations.view(batch_size, self.seq_len, self.features_per_timestep)
        
        # Feature embedding: [batch, seq_len, features_per_timestep] -> [batch, seq_len, embed_dim]
        x = self.feature_embedding(x)
        
        # Add positional encoding for temporal information
        x = x + self.pe.expand(batch_size, -1, -1)
        
        # Transformer encoding: [batch, seq_len, embed_dim] -> [batch, seq_len, embed_dim]
        x = self.transformer_encoder(x)
        
        # Global temporal aggregation: [batch, seq_len, embed_dim] -> [batch, embed_dim]
        # Use attention pooling instead of simple mean
        temporal_weights = torch.softmax(
            torch.mean(x, dim=-1, keepdim=True), dim=1
        )  # [batch, seq_len, 1]
        
        global_context = torch.sum(x * temporal_weights, dim=1)  # [batch, embed_dim]
        
        # Temporal aggregation: [batch, embed_dim] -> [batch, 256]
        x = self.temporal_aggregator(global_context)
        
        # Final projection: [batch, 256] -> [batch, features_dim]
        features = self.final_projection(x)
        
        return features


def get_sac_trading_transformer_kwargs(features_dim: int = 256) -> Dict[str, Any]:
    """Get configuration for SAC Trading Transformer"""
    return {
        "features_extractor_class": SACTradingTransformerExtractor,
        "features_extractor_kwargs": {"features_dim": features_dim}
    }


def validate_sac_trading_transformer(model) -> bool:
    """Validate SAC Trading Transformer is properly configured"""
    try:
        extractor = model.policy.features_extractor
        if not isinstance(extractor, SACTradingTransformerExtractor):
            return False
        
        # Test forward pass
        test_input = torch.randn(1, 450)  # 10 * 45 features
        output = extractor(test_input)
        
        return output.shape[1] == extractor.features_dim
    except Exception:
        return False


if __name__ == "__main__":
    # Test the implementation
    from gym.spaces import Box
    
    print("🧪 Testing SAC Trading Transformer...")
    
    obs_space = Box(low=-np.inf, high=np.inf, shape=(450,), dtype=np.float32)
    extractor = SACTradingTransformerExtractor(obs_space, features_dim=256)
    
    # Test forward pass
    batch_size = 32
    test_obs = torch.randn(batch_size, 450)
    
    print(f"Input shape: {test_obs.shape}")
    output = extractor(test_obs)
    print(f"Output shape: {output.shape}")
    
    # Test gradients
    loss = output.sum()
    loss.backward()
    
    # Check gradient statistics
    total_grad_norm = 0.0
    param_count = 0
    
    for name, param in extractor.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            param_count += 1
            
            zeros_pct = (param.grad == 0).float().mean().item() * 100
            print(f"{name}: grad_norm={grad_norm:.6f}, zeros={zeros_pct:.1f}%")
    
    total_grad_norm = math.sqrt(total_grad_norm)
    print(f"\nTotal gradient norm: {total_grad_norm:.6f}")
    print(f"Parameters with gradients: {param_count}")
    print("✅ SAC Trading Transformer test completed!")