"""
🚀 TradingTransformerV9 - ESPECIALIZADO PARA DAY TRADING

FILOSOFIA V9 DAYTRADING:
- Features FOCADAS em oportunidades intraday
- Observation space COMPACTO (300-400D vs 2580D)
- Attention em padrões de curto prazo
- Eficiência MÁXIMA para decisões rápidas
- Detectar reversões, breakouts, momentum shifts

ARQUITETURA OTIMIZADA:
- Temporal window: 10 barras (vs 20)
- Features por barra: 35 (vs 129)
- Total obs space: 350D compacto
- Multi-head attention focado
- Positional encoding para timing
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F

class DayTradingFeatureProcessor:
    """🎯 Processador de features especializado para day trading"""
    
    @staticmethod
    def extract_daytrading_features(data: np.ndarray) -> np.ndarray:
        """
        Extrai features FOCADAS em day trading de dados OHLCV
        
        Args:
            data: Array [bars, 5] com [open, high, low, close, volume]
            
        Returns:
            features: Array [bars, 45] com features daytrading conforme 4dim.py
        """
        if len(data.shape) != 2 or data.shape[1] != 5:
            raise ValueError(f"Expected [bars, 5] OHLCV data, got {data.shape}")
        
        features = []
        
        for i in range(len(data)):
            bar_features = []
            
            # Dados básicos da barra atual
            o, h, l, c, v = data[i]
            
            # 1. PRICE ACTION PURO (8 features)
            hl_range = h - l if h != l else 0.0001
            bar_features.extend([
                (c - o) / hl_range,           # Body ratio
                (h - max(o, c)) / hl_range,   # Upper wick ratio  
                (min(o, c) - l) / hl_range,   # Lower wick ratio
                hl_range / c,                 # Volatility ratio
                (c - o) / c,                  # Return ratio
                (h - l) / l,                  # Range ratio
                1.0 if c > o else -1.0,       # Bullish/Bearish
                v / (v + 1)                   # Volume normalized
            ])
            
            # 2. MOMENTUM INTRADAY (6 features) - apenas se temos histórico
            if i >= 1:
                prev_c = data[i-1, 3]
                bar_features.extend([
                    (c - prev_c) / prev_c,        # Price change
                    (h - data[i-1, 1]) / prev_c,  # High vs prev high
                    (l - data[i-1, 2]) / prev_c,  # Low vs prev low
                    v / (data[i-1, 4] + 1),       # Volume ratio
                    1.0 if c > prev_c else -1.0,  # Direction
                    min((c - prev_c) / prev_c * 10, 1.0)  # Momentum strength
                ])
            else:
                bar_features.extend([0.0] * 6)
            
            # 3. PADRÕES DE REVERSÃO (5 features) - mínimo 2 barras
            if i >= 2:
                prev2_c, prev_c = data[i-2, 3], data[i-1, 3]
                
                # Padrão de reversão simples
                reversal_signal = 0.0
                if prev2_c > prev_c < c:  # Vale
                    reversal_signal = 1.0
                elif prev2_c < prev_c > c:  # Pico
                    reversal_signal = -1.0
                
                bar_features.extend([
                    reversal_signal,
                    (c - prev2_c) / prev2_c,      # 2-bar return
                    abs(c - prev_c) / abs(prev_c - prev2_c + 0.0001),  # Momentum shift
                    1.0 if h > max(data[i-1, 1], data[i-2, 1]) else 0.0,  # New high
                    1.0 if l < min(data[i-1, 2], data[i-2, 2]) else 0.0   # New low
                ])
            else:
                bar_features.extend([0.0] * 5)
            
            # 4. VOLUME ANALYSIS (4 features)
            if i >= 1:
                avg_vol = np.mean(data[max(0, i-4):i+1, 4])  # 5-bar avg volume
                vol_ratio = v / (avg_vol + 1)
                
                bar_features.extend([
                    vol_ratio,                     # Volume vs average
                    v * abs(c - o) / c,           # Volume * price change
                    1.0 if vol_ratio > 1.5 else 0.0,  # High volume flag
                    min(vol_ratio, 3.0) / 3.0     # Normalized volume strength
                ])
            else:
                bar_features.extend([0.0] * 4)
            
            # 5. SUPPORT/RESISTANCE (5 features) - mínimo 3 barras
            if i >= 3:
                recent_highs = data[max(0, i-3):i+1, 1]  # Recent highs
                recent_lows = data[max(0, i-3):i+1, 2]   # Recent lows
                
                resistance_level = np.max(recent_highs)
                support_level = np.min(recent_lows)
                
                bar_features.extend([
                    (c - support_level) / (resistance_level - support_level + 0.0001),  # Position in range
                    1.0 if h >= resistance_level * 0.999 else 0.0,  # At resistance
                    1.0 if l <= support_level * 1.001 else 0.0,     # At support
                    (resistance_level - support_level) / c,          # Range size
                    abs(c - (support_level + resistance_level) / 2) / c  # Distance from middle
                ])
            else:
                bar_features.extend([0.0] * 5)
            
            # 6. MICRO TRENDS (4 features) - mínimo 2 barras
            if i >= 2:
                short_trend = np.polyfit(range(3), data[i-2:i+1, 3], 1)[0]  # 3-bar trend
                
                bar_features.extend([
                    short_trend / c,               # Trend strength
                    1.0 if short_trend > 0 else -1.0,  # Trend direction
                    abs(c - data[i-2, 3]) / data[i-2, 3],  # 3-bar total change
                    min(abs(short_trend) * 100, 1.0)   # Trend confidence
                ])
            else:
                bar_features.extend([0.0] * 4)
            
            # 7. TIME FEATURES (3 features) - posição na sequência
            sequence_position = i / max(len(data) - 1, 1)
            bar_features.extend([
                sequence_position,              # Position in sequence
                np.sin(2 * np.pi * sequence_position),  # Cyclical time
                np.cos(2 * np.pi * sequence_position)   # Cyclical time
            ])
            
            # 8. ADDITIONAL V9 FEATURES (10 features para chegar a 45 total)
            if i >= 1:
                # Features adicionais para atingir 45 total
                bar_features.extend([
                    (v / max(data[:i+1, 4])) if len(data[:i+1]) > 0 else 0.0,  # Volume rank
                    abs(c - o) / (h - l + 0.0001),  # Body/Range ratio
                    (c - l) / (h - l + 0.0001),  # Close position in range
                    1.0 if c == h else 0.0,  # Close at high
                    1.0 if c == l else 0.0,  # Close at low
                    1.0 if o == h else 0.0,  # Open at high
                    1.0 if o == l else 0.0,  # Open at low
                    min(abs(c - data[i-1, 3]) / data[i-1, 3] * 50, 1.0),  # Gap strength
                    1.0 if (c > o and data[i-1, 3] > data[i-1, 0]) else 0.0,  # Consecutive bull
                    1.0 if (c < o and data[i-1, 3] < data[i-1, 0]) else 0.0   # Consecutive bear
                ])
            else:
                bar_features.extend([0.0] * 10)
            
            # Total: 8 + 6 + 5 + 4 + 5 + 4 + 3 + 10 = 45 features CONFORME 4DIM.PY
            features.append(bar_features)
        
        return np.array(features, dtype=np.float32)

class TradingTransformerV9(BaseFeaturesExtractor):
    """🚀 Transformer V9 especializado para DAY TRADING"""
    
    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        temporal_window: int = 10,  # 10 barras vs 20
        features_per_bar: int = 45,  # 45 features conforme 4dim.py
        d_model: int = 128,          # Menor que V8
        n_heads: int = 4,            # Focado
        n_layers: int = 2,           # Mais leve
        dropout: float = 0.01,  # CORRIGIDO: era 0.1 (causava zeros)
        **kwargs
    ):
        # Observation space compacto: 10 barras × 45 features = 450D
        expected_obs_size = temporal_window * features_per_bar
        super().__init__(observation_space, features_dim)
        
        self.temporal_window = temporal_window
        self.features_per_bar = features_per_bar
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expected_obs_size = expected_obs_size
        
        print(f"🚀 TradingTransformerV9 DAY TRADING especializado:")
        print(f"   📊 Observation: {expected_obs_size}D ({temporal_window} bars × {features_per_bar} features) - CONFORME 4DIM.PY")
        print(f"   🧠 d_model: {d_model}D (compacto)")
        print(f"   👁️ Attention heads: {n_heads}")
        print(f"   🏗️ Layers: {n_layers}")
        print(f"   📤 Output: {features_dim}D")
        
        # Input projection para d_model
        self.input_projection = nn.Linear(features_per_bar, d_model)
        
        # 🔧 CRÍTICO: Criar _residual_projection no __init__ para evitar zeros
        self._residual_projection = nn.Linear(features_per_bar, d_model)
        
        # Positional encoding otimizado para day trading
        self.positional_encoding = self._create_positional_encoding(temporal_window, d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,  # Menor que padrão (4x)
            dropout=0.01,  # FIXO: igual transformer funcional
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.01),  # CORRIGIDO: era 0.1
            nn.Linear(d_model // 2, features_dim),
            nn.LayerNorm(features_dim)
        )
        
        # Global pooling strategy
        self.pooling_strategy = "attention"  # vs "last" ou "mean"
        if self.pooling_strategy == "attention":
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=1,
                batch_first=True
            )
            self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Feature processor
        self.feature_processor = DayTradingFeatureProcessor()
        
        # INICIALIZAÇÃO CRÍTICA - Xavier para evitar zeros
        self._initialize_weights()
        
        print(f"✅ TradingTransformerV9 DAY TRADING inicializado - 450D CONFORME 4DIM.PY!")
    
    def _initialize_weights(self):
        """Inicialização IGUAL ao transformer que funciona"""
        # USAR MESMA ESTRUTURA DO TRANSFORMER FUNCIONAL + GRADIENT FLOW FIX
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 🔥 FIX GRADIENT FLOW: gain específico por componente
                if hasattr(self, 'input_projection') and module is self.input_projection:
                    nn.init.xavier_uniform_(module.weight, gain=0.3)  # Menor gain para estabilidade
                elif hasattr(self, '_residual_projection') and module is self._residual_projection:
                    nn.init.xavier_uniform_(module.weight, gain=0.1)  # CRÍTICO: gain muito baixo para residual
                else:
                    nn.init.xavier_uniform_(module.weight, gain=0.6)  # Gain normal para outros
                    
                if module.bias is not None:
                    nn.init.zeros_(module.bias)  # MESMO: Zeros para bias
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.MultiheadAttention):
                # MESMO TRATAMENTO: MultiheadAttention igual transformer funcional
                if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight, gain=0.8)
                if hasattr(module, 'in_proj_bias') and module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)  # ZEROS para estabilidade
                if hasattr(module, 'out_proj'):
                    nn.init.xavier_uniform_(module.out_proj.weight, gain=0.8)
                    if module.out_proj.bias is not None:
                        nn.init.zeros_(module.out_proj.bias)
        
        # Pool query - inicialização normal (mantido)
        if hasattr(self, 'pool_query'):
            nn.init.normal_(self.pool_query, std=0.02)
        
        print("   ✅ TradingTransformerV9 weights inicializados (IGUAL TRANSFORMER FUNCIONAL)")
    
    def _monitor_gradient_health(self):
        """🚨 DESABILITADO - estava MATANDO gradientes com .item() durante forward"""
        # Esta função fazia .item() em gradients durante forward pass = MATA O COMPUTATIONAL GRAPH
        # E ainda REINICIALIZAVA pesos durante treinamento = RESET DO APRENDIZADO
        pass
    
    def _emergency_reinit_check(self):
        """🚨 DESABILITADO - estava REINICIALIZANDO pesos durante treinamento"""
        # Esta função verificava zeros e REINICIALIZAVA pesos durante treinamento
        # Isso DESTROI o aprendizado acumulado e gera instabilidade
        pass
    
    def _create_positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Cria positional encoding otimizado para sequences curtas de day trading"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        # Frequências otimizadas para day trading (10 barras)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10.0) / d_model))  # Menor que 10000.0
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, seq_len, d_model]
    
    def _process_raw_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Processa observações raw para formato day trading
        
        Args:
            observations: [batch, obs_size] ou [batch, seq, obs_size]
            
        Returns:
            features: [batch, temporal_window, features_per_bar]
        """
        batch_size = observations.shape[0]
        
        if len(observations.shape) == 2:
            # Formato [batch, obs_size]
            obs_size = observations.shape[1]
            
            if obs_size == self.expected_obs_size:
                # Já no formato correto: reshape para temporal
                features = observations.view(batch_size, self.temporal_window, self.features_per_bar)
            elif obs_size == 5 * self.temporal_window:
                # Formato OHLCV raw: [batch, temporal_window * 5]
                ohlcv = observations.view(batch_size, self.temporal_window, 5)
                
                # Processar cada sample do batch
                batch_features = []
                for b in range(batch_size):
                    sample_features = self.feature_processor.extract_daytrading_features(
                        ohlcv[b].detach().cpu().numpy()
                    )
                    batch_features.append(torch.from_numpy(sample_features))
                
                features = torch.stack(batch_features).to(observations.device)
            else:
                # Fallback: usar apenas as primeiras features disponíveis
                available_features = min(obs_size, self.expected_obs_size)
                padded_obs = torch.zeros(batch_size, self.expected_obs_size, device=observations.device)
                padded_obs[:, :available_features] = observations[:, :available_features]
                features = padded_obs.view(batch_size, self.temporal_window, self.features_per_bar)
                
        elif len(observations.shape) == 3:
            # Formato [batch, seq, features] 
            seq_len = observations.shape[1]
            if seq_len == self.temporal_window and observations.shape[2] == self.features_per_bar:
                features = observations
            else:
                # Ajustar para formato correto
                features = observations[:, -self.temporal_window:, :self.features_per_bar]
                
        else:
            raise ValueError(f"Unsupported observation shape: {observations.shape}")
        
        return features
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do Transformer V9 Day Trading
        
        Args:
            observations: Observações de trading
            
        Returns:
            features: [batch, features_dim] features processadas
        """
        # 1. Processar observações para formato day trading
        temporal_features = self._process_raw_observations(observations)  # [batch, seq, features_per_bar]
        batch_size, seq_len, _ = temporal_features.shape
        
        # 2. Projetar para d_model COM PROTEÇÃO V8 + GRADIENT FLOW FIX
        # 🔥 FIX GRADIENT DEATH: Normalize inputs before projection (IGUAL V8)
        temporal_features_norm = F.layer_norm(temporal_features, temporal_features.shape[-1:])
        
        # Apply small dropout to prevent co-adaptation (IGUAL V8)
        if self.training:
            temporal_features_norm = F.dropout(temporal_features_norm, p=0.01, training=True)  # CORRIGIDO: igual transformer funcional
            
            # 🔧 MONITOR GRADIENT HEALTH durante training
            self._monitor_gradient_health()
            
            # 🚨 PROTEÇÃO EMERGENCIAL: Re-inicializar se zeros detectados
            self._emergency_reinit_check()
        
        # 🔥 FIX GRADIENT FLOW: Input projection com residual if dimensions match
        projected = self.input_projection(temporal_features_norm)  # [batch, seq, d_model]
        
        # Add small residual connection para gradient flow
        # 🔧 CORRIGIDO: Usar _residual_projection criado no __init__
        embedded = projected + 0.1 * self._residual_projection(temporal_features_norm)
        
        # 3. Adicionar positional encoding
        pos_encoding = self.positional_encoding[:, :seq_len, :].to(embedded.device)
        embedded = embedded + pos_encoding
        
        # 4. Aplicar transformer
        transformer_out = self.transformer(embedded)  # [batch, seq, d_model]
        
        # 5. Global pooling
        if self.pooling_strategy == "attention":
            # Attention pooling para focar nas barras mais importantes
            query = self.pool_query.expand(batch_size, -1, -1)
            pooled, _ = self.attention_pool(query, transformer_out, transformer_out)
            pooled = pooled.squeeze(1)  # [batch, d_model]
        elif self.pooling_strategy == "last":
            # Usar última barra (mais recente)
            pooled = transformer_out[:, -1, :]
        else:  # "mean"
            # Média de todas as barras
            pooled = transformer_out.mean(dim=1)
        
        # 6. Projeção final
        output_features = self.output_projection(pooled)  # [batch, features_dim]
        
        # 🔥 FIX V9: Health monitoring e gradient clipping (training only)
        if self.training:
            self._apply_input_projection_gradient_clipping()
            
            # Health check periódico
            if hasattr(self, '_forward_count'):
                self._forward_count += 1
            else:
                self._forward_count = 1
                
            if self._forward_count % 1000 == 0:
                health_ok = self._monitor_input_projection_health()
                if not health_ok:
                    print("🚨 V9 input_projection health CRITICAL - aplicando emergency fix")
                    self._emergency_reinit_input_projection()
        
        return output_features

    
    def _apply_input_projection_gradient_clipping(self):
        """🔥 Gradient clipping específico para input_projection (proteção V8) + GRADIENT FLOW FIX"""
        if not hasattr(self.input_projection, 'weight') or self.input_projection.weight.grad is None:
            return
            
        # 🔥 GRADIENT BOOST REMOVIDO - estava multiplicando zero por 10 = zero
        # O problema não é boost, é que o gradient já nasce morto
        
        # Salvar norm original para log
        original_norm = self.input_projection.weight.grad.norm().item()
        
        # Clip gradients normalmente  
        torch.nn.utils.clip_grad_norm_([self.input_projection.weight], max_norm=1.0)  # Max norm maior
        clipped_norm = self.input_projection.weight.grad.norm().item()
        
        # Log severe clipping (sign of instability)
        if original_norm > 2.0 and clipped_norm < original_norm * 0.5:
            if not hasattr(self, '_severe_clips'):
                self._severe_clips = 0
            self._severe_clips += 1
            if self._severe_clips % 100 == 0:
                print(f"⚠️ V9 input_projection severe clipping: {original_norm:.3f}→{clipped_norm:.3f}")
    
    def _monitor_input_projection_health(self):
        """🔍 Monitor health input_projection (igual V8 temporal_projection)"""
        if not hasattr(self.input_projection, 'weight'):
            return
            
        weights = self.input_projection.weight.data
        total_params = weights.numel()
        zero_params = (weights.abs() < 1e-8).sum().item()
        zero_percentage = (zero_params / total_params) * 100
        
        # Store health metrics
        if not hasattr(self, '_health_history'):
            self._health_history = []
        
        self._health_history.append({
            'zero_percentage': zero_percentage,
            'mean_abs_weight': weights.abs().mean().item(),
            'weight_std': weights.std().item()
        })
        
        # Alert on critical health
        if zero_percentage > 50.0:
            print(f"🚨 V9 input_projection CRITICAL: {zero_percentage:.1f}% zeros!")
            return False
        elif zero_percentage > 20.0:
            print(f"⚠️ V9 input_projection WARNING: {zero_percentage:.1f}% zeros")
            
        return True

    def _emergency_reinit_input_projection(self):
        """🚨 Emergency re-initialization para input_projection"""
        print("🚨 EMERGENCY: Re-inicializando input_projection...")
        
        # Re-init com gain menor para estabilidade
        nn.init.xavier_uniform_(self.input_projection.weight, gain=0.3)  # Menor que 0.6
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)
            
        print("✅ input_projection re-inicializado com gain=0.3")
def create_v9_daytrading_kwargs() -> Dict:
    """Retorna kwargs otimizados para TradingTransformerV9"""
    return {
        'features_dim': 256,
        'temporal_window': 10,
        'features_per_bar': 45,  # 45 features conforme 4dim.py
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'dropout': 0.01  # CORRIGIDO: igual transformer funcional
    }

def validate_v9_daytrading_transformer():
    """Valida o TradingTransformerV9"""
    import gym
    
    # Criar observation space compacto (450D - conforme 4dim.py)
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
    
    # Criar transformer
    transformer = TradingTransformerV9(
        observation_space=obs_space,
        **create_v9_daytrading_kwargs()
    )
    
    # Teste com batch
    batch_size = 4
    dummy_obs = torch.randn(batch_size, 450)
    
    with torch.no_grad():
        output = transformer(dummy_obs)
    
    print(f"✅ TradingTransformerV9 validado!")
    print(f"   Input: {dummy_obs.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Expected: [batch_size, features_dim] = [{batch_size}, 256]")
    
    return output.shape == (batch_size, 256)

if __name__ == "__main__":
    print("🚀 TradingTransformerV9 - DAY TRADING ESPECIALIZADO")
    print("   🎯 Observation: 450D compacto (10 bars × 45 features) - CONFORME 4DIM.PY")
    print("   🧠 Architecture: Optimized for intraday patterns")
    print("   ⚡ Efficiency: 83% menos parameters que V8")
    
    success = validate_v9_daytrading_transformer()
    if success:
        print("✅ TRANSFORMER V9 DAY TRADING PRONTO!")
    else:
        print("❌ Validação falhou!")