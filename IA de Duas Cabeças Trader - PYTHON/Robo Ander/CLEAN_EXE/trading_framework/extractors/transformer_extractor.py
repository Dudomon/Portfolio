import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces


class TradingTransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    🎯 TRANSFORMER OTIMIZADO COM VISÃO TEMPORAL REAL PARA TRADING
    
    Arquitetura otimizada com processamento de dados sequenciais:
    - Processa 20 barras históricas consecutivas (129 features cada)
    - Sistema otimizado: 8220→2580 dimensões (68% redução)
    - Attention heads especializados em padrões temporais
    - Features avançadas: microestrutura, volatilidade, correlação, momentum
    - Visão temporal real em vez de simulação
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64, seq_len: int = 20):
        super().__init__(observation_space, features_dim)
        
        # 🎯 CONFIGURAÇÕES ESPECÍFICAS PARA TRADING TEMPORAL OTIMIZADO
        # Input agora é sequência temporal otimizada: seq_len * features_per_bar
        total_features = observation_space.shape[0]  # 450 total features (V10Pure)
        self.seq_len = 10  # 10 barras históricas para V10Pure
        self.input_dim = total_features // self.seq_len  # 450 / 10 = 45 features por barra
        
        # 🔥 TRADING-SPECIFIC PARAMETERS
        self.d_model = 128  # Dimensão interna otimizada
        self.n_heads = 4  # 4 heads: price, volume, momentum, volatility
        
        # 🚨 DEPTH REDUCTION: Single layer projection (4 layers → 1 layer)
        self.temporal_projection = nn.Linear(self.input_dim, self.d_model)  # Direct: 45 → 128
        
        # 🎯 POSITIONAL ENCODING - Específico para sequência temporal real
        self.pos_encoding = self._create_temporal_positional_encoding()
        
        # 🚨 DEPTH REDUCTION: Single transformer layer (2 layers → 1 layer)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,  # MANTIDO: 4 heads especializados (price, volume, momentum, volatility)
            dim_feedforward=128,  # REDUZIDO: 256 → 128 (mais eficiente)
            dropout=0.01,  # REDUZIDO: 0.05 → 0.01 (menos regularização)
            activation='gelu',
            batch_first=True,
            norm_first=True  # OTIMIZADO: norm_first=True (mais estável)
        )
        
        # 🚨 DEPTH REDUCTION: Simple aggregation (remove 6 layers de attentions extras)
        self.temporal_aggregator = nn.Linear(self.d_model, self.features_dim)  # Direct: 128 → 64
        
        # 🔥 LEARNABLE POOLING: Replace mean pooling that kills gradients
        self.learnable_pooling = self._create_learnable_pooling()
        
        # 🎯 LAYER NORMALIZATION - Apenas para input (não duplicar)
        self.input_layer_norm = nn.LayerNorm(self.d_model)
        self.output_layer_norm = nn.LayerNorm(self.d_model)
        
        # 🔥 INITIALIZE WEIGHTS FOR TEMPORAL TRADING
        self._initialize_temporal_weights()
        
        # 🔍 DEBUG COUNTER
        self._debug_step = 0
        
        # 🔥 DEAD NEURONS FIX: Track para aplicar fix se necessário
        self._dead_features_detected = False
        self._dead_features_mask = None
    
    def _create_temporal_positional_encoding(self):
        """Cria positional encoding específico para sequência temporal real"""
        pe = torch.zeros(self.seq_len, self.d_model)
        position = torch.arange(0, self.seq_len).unsqueeze(1).float()
        
        # 🎯 Trading-specific frequencies (diferentes escalas temporais)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           -(np.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def _create_learnable_pooling(self):
        """🔥 Cria learnable pooling para preservar gradients"""
        # 🚨 STRONGER INITIALIZATION: Recent bias para trading + noise
        weights = torch.ones(self.seq_len) / self.seq_len
        
        # Add recent bias (trading principle: recent data mais importante)
        recent_boost = torch.linspace(0.8, 1.2, self.seq_len)  # 0.8 → 1.2 linear
        weights = weights * recent_boost
        
        # Normalize to sum=1
        weights = weights / weights.sum()
        
        # Add small noise to break symmetry
        noise = torch.normal(0, 0.01, size=(self.seq_len,))
        weights = weights + noise
        weights = torch.clamp(weights, min=0.01)  # Prevent zeros
        weights = weights / weights.sum()  # Re-normalize
        
        return nn.Parameter(weights)
    
    def _initialize_temporal_weights(self):
        """🔥 CORREÇÃO: Inicialização específica incluindo MultiheadAttention"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 🔥 FIX: Xavier com gain MUITO MENOR para prevenir zeros (0.6 → 0.3)
                nn.init.xavier_uniform_(module.weight, gain=0.3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)  # 🔥 FIX: Zeros para bias (padrão estável)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.MultiheadAttention):
                # 🔥 CORREÇÃO CRÍTICA: Inicializar MultiheadAttention com Xavier
                if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight, gain=0.8)
                if hasattr(module, 'in_proj_bias') and module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)  # 🔥 FIX: Zeros para estabilidade
                if hasattr(module, 'out_proj'):
                    nn.init.xavier_uniform_(module.out_proj.weight, gain=0.8)
                    if module.out_proj.bias is not None:
                        nn.init.zeros_(module.out_proj.bias)
    
    def _create_temporal_sequence(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Cria sequência temporal REAL a partir das observações históricas
        
        Args:
            observations: [batch_size, 450] - Features históricas V10Pure (45 features × 10 barras)
            
        Returns:
            temporal_seq: [batch_size, 10, 128] - Sequência temporal processada
        """
        batch_size = observations.shape[0]
        device = observations.device  # 🔧 DEVICE SAFETY
        
        # 🔧 SHAPE VALIDATION OTIMIZADO - Agora esperamos 450 features (45 * 10)
        expected_total_features = self.seq_len * self.input_dim  # 10 * 45 = 450
        assert observations.shape[1] == expected_total_features, f"Expected {expected_total_features} features (45 features/bar × 10 bars), got {observations.shape[1]}"
        
        # 🎯 RESHAPE PARA SEQUÊNCIA TEMPORAL REAL
        # observations: [batch_size, seq_len * input_dim] → [batch_size, seq_len, input_dim]
        temporal_obs = observations.view(batch_size, self.seq_len, self.input_dim)
        
        # 🎯 PROJECT CADA BARRA HISTÓRICA INDIVIDUALMENTE
        temporal_variations = []
        
        for t in range(self.seq_len):
            # Processa cada barra histórica individual
            bar_features = temporal_obs[:, t, :]  # [batch_size, input_dim]
            
            # 🎯 CONVERGENCE MONITORING: Only essential checks (REDUCED FREQUENCY)
            if hasattr(self, '_debug_step') and self._debug_step % 50000 == 0 and t == 0:
                # Minimal convergence tracking - no verbose output
                input_health = {
                    'mean_abs': bar_features.abs().mean().item(),
                    'saturation': (bar_features.abs() > 3.0).float().mean().item(),
                    'dead_ratio': (bar_features.abs() < 1e-8).float().mean().item()
                }
                # Store for convergence analysis (no print)
            
            # 🎯 CONVERGENCE MONITORING: Minimal health check (REDUCED FREQUENCY)
            if hasattr(self, '_debug_step') and self._debug_step % 25000 == 0 and t == 0:
                # Silent convergence monitoring
                bar_features_norm_check = F.layer_norm(bar_features, bar_features.shape[-1:])
                pre_projection = self.temporal_projection(bar_features_norm_check)
                projection_health = (pre_projection.abs() > 3.0).float().mean().item()
                
                # Store convergence metrics (no print output)
                if not hasattr(self, '_convergence_metrics'):
                    self._convergence_metrics = []
                self._convergence_metrics.append({
                    'step': self._debug_step,
                    'projection_saturation': projection_health,
                    'input_stability': bar_features.std().item()
                })
            
            # 🔥 FIX GRADIENT DEATH: SEMPRE normalizar inputs para prevenir zeros
            # Intermittent normalization pode estar causando instabilidade
            bar_features_norm = F.layer_norm(bar_features, bar_features.shape[-1:])
            
            projected_features = self.temporal_projection(bar_features_norm)  # [batch_size, d_model]
            
            # Apply small dropout to prevent co-adaptation (REDUCED FREQUENCY)
            if self.training and hasattr(self, '_debug_step') and self._debug_step % 10 == 0:
                projected_features = F.dropout(projected_features, p=0.1, training=True)
                
            temporal_variations.append(projected_features)
        
        # 🎯 STACK TEMPORAL SEQUENCE REAL
        temporal_seq = torch.stack(temporal_variations, dim=1)  # [batch, seq_len, d_model]
        
        # 🔧 FINAL VALIDATION
        expected_shape = (batch_size, self.seq_len, self.d_model)
        assert temporal_seq.shape == expected_shape, f"Expected shape {expected_shape}, got {temporal_seq.shape}"
        
        return temporal_seq
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass com visão temporal REAL
        
        Args:
            observations: [batch_size, 2580] - Features históricas otimizadas (129 features × 20 barras)
            
        Returns:
            features: [batch_size, features_dim] - Features processadas com sequência temporal real
        """
        batch_size = observations.shape[0]
        device = observations.device  # 🔧 DEVICE SAFETY
        
        # 🔍 INCREMENT DEBUG COUNTER
        self._debug_step += 1
        
        # 🎯 CREATE REAL TEMPORAL SEQUENCE
        x = self._create_temporal_sequence(observations)  # [batch, seq_len, d_model]
        
        # 🔧 DEVICE SAFETY: Garantir que positional encoding está no mesmo device
        pos_encoding = self.pos_encoding.to(device)
        
        # 🎯 ADD POSITIONAL ENCODING
        x = x + pos_encoding.repeat(batch_size, 1, 1)
        
        # 🚨 GRADIENT DEATH FIX: Apply strong regularization from the start
        
        # 1. Input normalization with scaling
        x = self.input_layer_norm(x)  # [batch, seq_len, d_model]
        
        # 🔥 CRITICAL FIX: Apply dropout to input to prevent saturation (REDUCED FREQUENCY)
        if self.training and hasattr(self, '_debug_step') and self._debug_step % 5 == 0:
            x = F.dropout(x, p=0.1, training=True)  # Moderate dropout (era 0.3)
        
        # 2. Single transformer layer (MANTÉM 4 heads especializados)
        x_residual = x  # Store for residual scaling
        x = self.transformer_layer(x)  # [batch, seq_len, d_model] - 6 sublayers
        
        # 🔥 RESIDUAL SCALING: Prevent explosion (AUMENTADO para melhor gradient flow)
        x = x_residual + 0.5 * (x - x_residual)  # Scale residual by 0.5 (era 0.1)
        
        # 3. Output normalization  
        x = self.output_layer_norm(x)  # [batch, seq_len, d_model]
        
        # 4. Temporal aggregation (learnable pooling instead of mean)
        # 🔥 GRADIENT FIX: Use learnable weights instead of uniform mean
        pooling_weights = F.softmax(self.learnable_pooling, dim=0)  # Normalize weights
        
        # 🚨 AUXILIARY LOSS: Force recent bias learning (REDUCED TO EVERY 1000 STEPS)
        if self.training and hasattr(self, '_debug_step') and self._debug_step % 1000 == 0:
            # Recent timesteps (8-9) should have higher weights for trading
            recent_target = torch.zeros_like(pooling_weights)
            recent_target[-2:] = 0.6  # Last 2 timesteps get 60% weight
            recent_target[:-2] = 0.4 / (self.seq_len - 2)  # Remaining distributed
            
            # Compute auxiliary loss (KL divergence to encourage recent bias)
            kl_loss = F.kl_div(F.log_softmax(self.learnable_pooling, dim=0), 
                              recent_target, reduction='sum')
            # Scale and store for monitoring
            self._auxiliary_loss = 0.1 * kl_loss  # Small weight
            
            # Manual gradient injection (REDUCED FREQUENCY)
            if hasattr(self, '_debug_step') and self._debug_step % 5000 == 0:
                # Add manual gradient to pooling weights
                recent_gradient = (recent_target - pooling_weights) * 0.01
                if self.learnable_pooling.grad is None:
                    self.learnable_pooling.grad = recent_gradient.detach()
                else:
                    self.learnable_pooling.grad += recent_gradient.detach()
        
        # Weighted aggregation: [batch, seq_len, d_model] → [batch, d_model]
        # Using einsum for efficiency: 's,bsd->bd' (seq, batch×seq×dim -> batch×dim)
        x_pooled = torch.einsum('s,bsd->bd', pooling_weights, x)
        
        # 🎯 CONVERGENCE: Position gradient scaling DESABILITADO para testar zeros
        # Gradient scaling pode estar causando instabilidade
        if False:  # self.training and hasattr(self, '_debug_step') and self._debug_step % 10000 == 0:
            self._apply_position_gradient_scaling()
        
        # 🎯 CONVERGENCE: Pooling weights monitoring (REDUCED FREQUENCY)
        if hasattr(self, '_debug_step') and self._debug_step % 50000 == 0:
            weight_divergence = pooling_weights.std().item()
            recent_bias = pooling_weights[-3:].sum().item()
            
            # Store convergence data (no verbose print)
            if not hasattr(self, '_pooling_convergence'):
                self._pooling_convergence = []
            self._pooling_convergence.append({
                'step': self._debug_step,
                'weight_std': weight_divergence,
                'recent_bias': recent_bias
            })
        
        # 5. Final projection
        features = self.temporal_aggregator(x_pooled)  # [batch, features_dim]
        
        return features

    def _apply_position_gradient_scaling(self):
        """🔥 Scale gradients for position-related features to prevent explosion"""
        if not hasattr(self.temporal_projection, 'weight') or self.temporal_projection.weight.grad is None:
            return
            
        # Position features are typically in the last portion of the input
        # Assuming positions features are last 20% of input_dim (rough estimate)
        position_start_idx = int(0.8 * self.input_dim)  # Last 20% are position features
        
        # GRADIENT SCALING DESABILITADO para testar zeros
        # Scaling agressivo (0.05) pode estar causando zeros
        if False:  # self.temporal_projection.weight.grad.size(1) > position_start_idx:
            # Scale position-related gradients by 0.05 (more aggressive dampening)
            self.temporal_projection.weight.grad[:, position_start_idx:] *= 0.05
            
            # Silent gradient balance monitoring (REDUCED FREQUENCY)
            if hasattr(self, '_debug_step') and self._debug_step % 25000 == 0:
                pos_grad_norm = self.temporal_projection.weight.grad[:, position_start_idx:].norm().item()
                market_grad_norm = self.temporal_projection.weight.grad[:, :position_start_idx].norm().item()
                
                # Store balance metrics (no print)
                if not hasattr(self, '_gradient_balance'):
                    self._gradient_balance = []
                self._gradient_balance.append({
                    'step': self._debug_step,
                    'market_norm': market_grad_norm,
                    'position_norm': pos_grad_norm
                })


# 🔥 ALIAS PARA COMPATIBILIDADE
TransformerFeatureExtractor = TradingTransformerFeatureExtractor

"""
🎯 EXEMPLO DE USO NO POLICY_KWARGS (OTIMIZADO):

policy_kwargs = dict(
    features_extractor_class=TransformerFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128, seq_len=20),
    # Sistema otimizado: 8220→2580 dimensões (68% redução)
    # Input: 129 features/barra × 20 barras = 2580 features
    # Features avançadas: microestrutura + volatilidade + correlação + momentum
)
"""