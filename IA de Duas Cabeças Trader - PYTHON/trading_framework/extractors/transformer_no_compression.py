import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces


class TradingTransformerNoCompression(BaseFeaturesExtractor):
    """
    🎯 TRANSFORMER SEM COMPRESSÃO - 450D → 450D (100% FIDELIDADE)

    Arquitetura ZERO-LOSS mantendo todas as 450 dimensões:
    - Input: 450D (10 barras × 45 features)
    - Processing: 450D → 450D (sem perda de informação)
    - Output: 450D (mesma dimensionalidade)

    VANTAGENS:
    ✅ 100% fidelidade de informação
    ✅ Transformer aprende relações temporais SEM comprimir
    ✅ Todas as features críticas preservadas
    ✅ Permite policy processar informação completa

    TRADE-OFFS:
    ⚠️ Mais parâmetros (450D vs 128D interno)
    ⚠️ Mais memória GPU
    ⚠️ Treinamento ligeiramente mais lento
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 450, seq_len: int = 10):
        # Force features_dim = 450 (no compression)
        features_dim = 450
        super().__init__(observation_space, features_dim)

        # 🎯 CONFIGURAÇÕES PARA TRADING SEM COMPRESSÃO
        total_features = observation_space.shape[0]  # 450 total features
        self.seq_len = 10  # 10 barras históricas
        self.input_dim = total_features // self.seq_len  # 450 / 10 = 45 features por barra

        # 🔥 NO-COMPRESSION PARAMETERS
        self.d_model = 450  # 🚀 MANTÉM 450D (100% fidelidade)
        self.n_heads = 10  # 10 heads (450D / 45D per head)
        # Note: self.features_dim já foi setado pelo super().__init__ com 450

        # 🎯 PROJECTION: 45D → 450D (expande features de cada barra)
        self.temporal_projection = nn.Linear(self.input_dim, self.d_model)

        # 🎯 POSITIONAL ENCODING
        self.pos_encoding = self._create_temporal_positional_encoding()

        # 🎯 TRANSFORMER LAYER (mantém 450D)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,  # 450D
            nhead=self.n_heads,  # 10 heads (45D cada)
            dim_feedforward=512,  # Feedforward pode ser maior que d_model
            dropout=0.01,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        # 🎯 AGGREGATOR: 450D → 450D (identity-like, apenas normaliza)
        # Pode ser identity ou linear simples
        self.temporal_aggregator = nn.Linear(self.d_model, self.features_dim, bias=False)
        # Inicializar como identity
        nn.init.eye_(self.temporal_aggregator.weight)

        # 🔥 LEARNABLE POOLING
        self.learnable_pooling = self._create_learnable_pooling()

        # 🎯 LAYER NORMALIZATION
        self.input_layer_norm = nn.LayerNorm(self.d_model)
        self.output_layer_norm = nn.LayerNorm(self.d_model)

        # 🔥 INITIALIZE WEIGHTS
        self._initialize_weights()

        # 🔍 DEBUG COUNTER
        self._debug_step = 0

    def _create_temporal_positional_encoding(self):
        """Positional encoding para 10 timesteps × 450D"""
        pe = torch.zeros(self.seq_len, self.d_model)
        position = torch.arange(0, self.seq_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() *
                           -(np.log(10000.0) / self.d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def _create_learnable_pooling(self):
        """Learnable pooling com recent bias"""
        weights = torch.ones(self.seq_len) / self.seq_len

        # Recent bias
        recent_boost = torch.linspace(0.8, 1.2, self.seq_len)
        weights = weights * recent_boost
        weights = weights / weights.sum()

        # Small noise
        noise = torch.normal(0, 0.01, size=(self.seq_len,))
        weights = weights + noise
        weights = torch.clamp(weights, min=0.01)
        weights = weights / weights.sum()

        return nn.Parameter(weights)

    def _initialize_weights(self):
        """Inicialização otimizada para 450D"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier com gain menor para alta dimensionalidade
                nn.init.xavier_uniform_(module.weight, gain=0.2)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.MultiheadAttention):
                if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight, gain=0.5)
                if hasattr(module, 'in_proj_bias') and module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if hasattr(module, 'out_proj'):
                    nn.init.xavier_uniform_(module.out_proj.weight, gain=0.5)
                    if module.out_proj.bias is not None:
                        nn.init.zeros_(module.out_proj.bias)

    def _create_temporal_sequence(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Cria sequência temporal preservando dimensionalidade

        Args:
            observations: [batch_size, 450] - 10 barras × 45 features

        Returns:
            temporal_seq: [batch_size, 10, 450] - Sequência 450D por timestep
        """
        batch_size = observations.shape[0]
        device = observations.device

        # Validate input
        expected_features = self.seq_len * self.input_dim  # 450
        assert observations.shape[1] == expected_features, \
            f"Expected {expected_features} features, got {observations.shape[1]}"

        # Reshape: [batch, 450] → [batch, 10, 45]
        temporal_obs = observations.view(batch_size, self.seq_len, self.input_dim)

        # Project cada timestep: 45D → 450D
        temporal_variations = []

        for t in range(self.seq_len):
            bar_features = temporal_obs[:, t, :]  # [batch, 45]

            # Normalize input
            bar_features_norm = F.layer_norm(bar_features, bar_features.shape[-1:])

            # Project: 45D → 450D
            projected_features = self.temporal_projection(bar_features_norm)  # [batch, 450]

            # Light dropout
            if self.training:
                projected_features = F.dropout(projected_features, p=0.05, training=True)

            temporal_variations.append(projected_features)

        # Stack: [batch, 10, 450]
        temporal_seq = torch.stack(temporal_variations, dim=1)

        assert temporal_seq.shape == (batch_size, self.seq_len, self.d_model)

        return temporal_seq

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass SEM COMPRESSÃO: 450D → 450D

        Args:
            observations: [batch_size, 450]

        Returns:
            features: [batch_size, 450] - Mesma dimensionalidade!
        """
        batch_size = observations.shape[0]
        device = observations.device

        self._debug_step += 1

        # Create temporal sequence: [batch, 10, 450]
        x = self._create_temporal_sequence(observations)

        # Add positional encoding
        pos_encoding = self.pos_encoding.to(device)
        x = x + pos_encoding.repeat(batch_size, 1, 1)

        # Input normalization
        x = self.input_layer_norm(x)

        # Transformer layer (450D → 450D)
        x_residual = x
        x = self.transformer_layer(x)  # [batch, 10, 450]

        # Residual connection
        x = x_residual + 0.5 * (x - x_residual)

        # Output normalization
        x = self.output_layer_norm(x)

        # Temporal aggregation: [batch, 10, 450] → [batch, 450]
        pooling_weights = F.softmax(self.learnable_pooling, dim=0)
        x_pooled = torch.einsum('s,bsd->bd', pooling_weights, x)  # [batch, 450]

        # Final projection (identity-like): 450D → 450D
        features = self.temporal_aggregator(x_pooled)  # [batch, 450]

        # 🔍 DEBUG: Log shape periodically
        if self._debug_step % 10000 == 0:
            print(f"[NO-COMPRESS] Step {self._debug_step}: Input {observations.shape} → Output {features.shape}")
            print(f"[NO-COMPRESS] Pooling weights recent_bias: {pooling_weights[-3:].sum().item():.3f}")

        return features


# Alias
TransformerNoCompression = TradingTransformerNoCompression


def get_no_compression_kwargs():
    """
    🎯 Retorna policy_kwargs para Transformer SEM COMPRESSÃO

    Usage:
        from trading_framework.extractors.transformer_no_compression import get_no_compression_kwargs

        policy_kwargs = get_no_compression_kwargs()
        model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=policy_kwargs, ...)
    """
    return dict(
        features_extractor_class=TradingTransformerNoCompression,
        features_extractor_kwargs=dict(
            features_dim=450,  # NO COMPRESSION
            seq_len=10
        ),
        net_arch=dict(
            pi=[256, 256],  # Policy pode processar 450D completo
            vf=[256, 256]   # Value function também
        ),
        # Shared Backbone processa 450D nativo
        share_features_extractor=True
    )
