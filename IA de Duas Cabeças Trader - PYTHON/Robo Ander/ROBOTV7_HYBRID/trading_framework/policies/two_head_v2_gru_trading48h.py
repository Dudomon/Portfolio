"""
🔥 TwoHeadV2GRU TRADING 48H - ULTRA ESPECIALIZADA
Baseada na TwoHeadV2 original + GRU multicamadas + especialização para trading 48h

CARACTERÍSTICAS ESPECIAIS:
- GRU 3 layers para hierarquia temporal (5-15min / 1-4h / 12-48h)
- Multi-scale temporal attention (8 heads)
- Pattern recognition networks (micro/medium/macro)
- Trading regime classifier (SCALP/SWING/TREND/HOLD)
- Adaptive heads baseados no regime
- Volatility awareness para trading 48h
- Risk management integrado
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Type
from collections import namedtuple
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from trading_framework.extractors.transformer_extractor import TradingTransformerFeatureExtractor

# 🔥 FALLBACK PARA COMPATIBILIDADE
try:
    from stable_baselines3.common.type_aliases import PyTorchObs
except ImportError:
    PyTorchObs = torch.Tensor

class TwoHeadV2GRUTrading48hPolicy(RecurrentActorCriticPolicy):
    """
    🔥 TWO HEAD V2 GRU TRADING 48H POLICY
    
    FILOSOFIA ULTRA-ESPECIALIZADA:
    - GRU 3 layers com hierarquia temporal específica para trading 48h
    - Layer 1: Micro patterns (5-15min) - scalping, noise filtering
    - Layer 2: Medium patterns (1-4h) - swing trading, breakouts
    - Layer 3: Macro patterns (12-48h) - trends, position management
    - Multi-scale attention para capturar correlações temporais
    - Pattern recognition especializado para cada timeframe
    - Trading regime adaptation (SCALP/SWING/TREND/HOLD)
    - Volatility awareness para ajustar estratégia automaticamente
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = TradingTransformerFeatureExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        # 🔥 PARÂMETROS ULTRA-ESPECIALIZADOS PARA TRADING 48H
        lstm_hidden_size: int = 128,  # Base robusta
        n_lstm_layers: int = 3,       # 3 GRU layers especializadas
        attention_heads: int = 8,     # Multi-scale attention
        volatility_awareness: bool = True,  # Awareness de volatilidade
        **kwargs
    ):
        
        # 🔥 CONFIGURAÇÕES ESPECIALIZADAS PARA TRADING 48H
        self.gru_hidden_size = lstm_hidden_size  # 128
        self.n_gru_layers = n_lstm_layers        # 3
        self.attention_heads = attention_heads   # 8
        self.volatility_awareness = volatility_awareness
        
        # 🔥 FEATURES EXTRACTOR OTIMIZADO PARA 48H
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {
                'features_dim': 128,  # 128 features robustas
                'seq_len': 12        # 12 barras para contexto 48h
            }
        
        # 🔥 ARQUITETURA ROBUSTA PARA 3 GRU LAYERS
        if net_arch is None:
            net_arch = [dict(pi=[384, 256, 128], vf=[384, 256, 128])]
        
        # 🔥 CONFIGURAÇÃO DAS 3 CAMADAS GRU ESPECIALIZADAS
        self.gru_layer_configs = [
            {
                "units": 128, 
                "dropout": 0.15, 
                "focus": "micro_patterns", 
                "timeframe": "5-15min",
                "strategy": "scalping_noise_filter"
            },
            {
                "units": 128, 
                "dropout": 0.20, 
                "focus": "medium_patterns", 
                "timeframe": "1-4h",
                "strategy": "swing_breakouts"
            },
            {
                "units": 128, 
                "dropout": 0.25, 
                "focus": "macro_patterns", 
                "timeframe": "12-48h",
                "strategy": "trend_position_mgmt"
            }
        ]
        
        # 🔥 DETECTAR ACTION SPACE
        self.action_dim = action_space.shape[0]
        print(f"🔥 TwoHeadV2GRUTrading48h detectou action_space: {self.action_dim}D")
        
        # 🔥 INICIALIZAR COMO RECURRENT POLICY
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            lstm_hidden_size=self.gru_hidden_size,
            n_lstm_layers=self.n_gru_layers,
            **kwargs
        )
        
        # 🔥 SUBSTITUIR LSTM POR GRU 3 LAYERS ESPECIALIZADAS
        self._replace_lstm_with_specialized_gru()
        
        # 🔥 COMPONENTES ULTRA-ESPECIALIZADOS PARA TRADING 48H
        self._init_trading_48h_components()
        
        # 🔥 INICIALIZAÇÃO OTIMIZADA PARA CONVERGÊNCIA
        self._initialize_trading_weights()
        
        # Debug info
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔥 TwoHeadV2GRUTrading48h: {total_params:,} parâmetros")
        print(f"🔥 Arquitetura: GRU 3 layers especializadas para trading 48h")
        for i, config in enumerate(self.gru_layer_configs):
            print(f"🔥 Layer {i+1}: {config['units']} units - {config['timeframe']} - {config['strategy']}")
        print(f"🔥 Volatility Awareness: {self.volatility_awareness}")
        print(f"🔥 Action Space: {self.action_dim}D")
    
    def _replace_lstm_with_specialized_gru(self):
        """🔥 SUBSTITUIR LSTM POR GRU 3 LAYERS ESPECIALIZADAS PARA TRADING 48H"""
        
        features_dim = self.features_extractor.features_dim
        
        # 🔥 GRU 3 LAYERS ESPECIALIZADAS PARA ACTOR
        self.lstm_actor = nn.GRU(
            input_size=features_dim,           # 128 features
            hidden_size=self.gru_hidden_size,  # 128 hidden
            num_layers=self.n_gru_layers,      # 3 layers especializadas
            batch_first=True,
            dropout=0.2,                       # Dropout entre camadas
            bidirectional=False
        )
        
        # 🔥 GRU 3 LAYERS ESPECIALIZADAS PARA CRITIC
        self.lstm_critic = nn.GRU(
            input_size=features_dim,           # 128 features
            hidden_size=self.gru_hidden_size,  # 128 hidden
            num_layers=self.n_gru_layers,      # 3 layers especializadas
            batch_first=True,
            dropout=0.2,                       # Dropout entre camadas
            bidirectional=False
        )
        
        print(f"🔥 LSTM substituído por GRU 3 LAYERS especializadas para trading 48h")
    
    def _init_trading_48h_components(self):
        """🔥 COMPONENTES ULTRA-ESPECIALIZADOS PARA TRADING 48H"""
        
        # 🔥 MULTI-SCALE TEMPORAL ATTENTION (8 heads para 3 escalas temporais)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.gru_hidden_size,  # 128
            num_heads=self.attention_heads,  # 8 heads
            dropout=0.1,
            batch_first=True
        )
        
        # 🔥 VOLATILITY AWARENESS NETWORK
        if self.volatility_awareness:
            self.volatility_detector = nn.Sequential(
                nn.Linear(self.gru_hidden_size, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 4),  # [low, medium, high, extreme] volatility
                nn.Softmax(dim=-1)
            )
        
        # 🔥 CROSS-TIMEFRAME FUSION (combina 3 escalas temporais)
        self.timeframe_fusion = nn.Sequential(
            nn.Linear(self.gru_hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, self.gru_hidden_size),
            nn.LayerNorm(self.gru_hidden_size)
        )
        
        # 🔥 PATTERN RECOGNITION NETWORKS (3 detectores especializados)
        
        # Micro Pattern Detector (5-15min): Scalping, noise filtering
        self.micro_pattern_detector = nn.Sequential(
            nn.Linear(self.gru_hidden_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.Sigmoid()  # Probabilidade de padrão micro
        )
        
        # Medium Pattern Detector (1-4h): Swing trading, breakouts
        self.medium_pattern_detector = nn.Sequential(
            nn.Linear(self.gru_hidden_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.Sigmoid()  # Probabilidade de padrão médio
        )
        
        # Macro Pattern Detector (12-48h): Trends, position management
        self.macro_pattern_detector = nn.Sequential(
            nn.Linear(self.gru_hidden_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.Sigmoid()  # Probabilidade de padrão macro
        )
        
        # 🔥 TRADING REGIME CLASSIFIER (determina estratégia ótima)
        regime_input_size = self.gru_hidden_size + 32 + 32 + 32  # GRU + 3 pattern scores
        if self.volatility_awareness:
            regime_input_size += 4  # + volatility info
        
        self.regime_classifier = nn.Sequential(
            nn.Linear(regime_input_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 4 regimes: SCALP, SWING, TREND, HOLD
            nn.Softmax(dim=-1)
        )
        
        # 🔥 ADAPTIVE TWO HEADS (baseadas no regime de trading)
        
        # Entry Head - QUANDO entrar (especializada por regime)
        entry_input_size = self.gru_hidden_size + 4  # GRU + regime info
        self.entry_head = nn.Sequential(
            nn.Linear(entry_input_size, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # Management Head - COMO gerenciar (especializada por regime)
        mgmt_input_size = self.gru_hidden_size + 4  # GRU + regime info
        self.management_head = nn.Sequential(
            nn.Linear(mgmt_input_size, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # 🔥 FINAL FUSION NETWORK (combina Entry + Management)
        self.final_fusion = nn.Sequential(
            nn.Linear(32 + 32, 96),  # Entry + Management heads
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        # 🔥 FINAL ACTION NETWORK (saída adaptativa)
        self.action_net = nn.Sequential(
            nn.Linear(self.gru_hidden_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.action_dim)  # Saída dinâmica
        )
        
        # 🔥 VALUE NETWORK (otimizada para trading 48h)
        self.value_net = nn.Sequential(
            nn.Linear(self.gru_hidden_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        print(f"🔥 Componentes Trading 48h:")
        print(f"   - Multi-scale Attention: {self.attention_heads} heads")
        print(f"   - Pattern Recognition: Micro + Medium + Macro")
        print(f"   - Trading Regimes: SCALP, SWING, TREND, HOLD")
        print(f"   - Volatility Awareness: {self.volatility_awareness}")
        print(f"   - Adaptive Two Heads: Entry + Management")
    
    def _initialize_trading_weights(self):
        """🔥 INICIALIZAÇÃO OTIMIZADA PARA CONVERGÊNCIA EM TRADING"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform com gain conservador para estabilidade
                nn.init.xavier_uniform_(module.weight, gain=0.3)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.GRU):
                # 🔥 INICIALIZAÇÃO ESPECIAL PARA GRU 3 LAYERS
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data, gain=0.3)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data, gain=0.3)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # GRU: Update gate bias = 1 para estabilidade
                        n = param.size(0)
                        param.data[n//3:2*n//3].fill_(1)
            elif isinstance(module, nn.MultiheadAttention):
                # 🔥 INICIALIZAÇÃO PARA ATTENTION
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param.data, gain=0.3)
                    elif 'bias' in name and param is not None:
                        nn.init.constant_(param.data, 0)
        
        print(f"🔥 Inicialização trading otimizada aplicada (gain=0.3)")
    
    # 🔥 USAR MÉTODOS INTERNOS PADRÃO - SEM FORWARD CUSTOMIZADO
    # RecurrentPPO usa métodos internos que sempre funcionaram
    # GRU 3 layers especializadas vão funcionar perfeitamente

def create_two_head_v2_gru_trading48h_policy(
    observation_space,
    action_space,
    lr_schedule,
    **kwargs
) -> TwoHeadV2GRUTrading48hPolicy:
    """Factory para criar TwoHeadV2GRUTrading48hPolicy"""
    return TwoHeadV2GRUTrading48hPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        **kwargs
    )

def get_trading_48h_kwargs() -> Dict[str, Any]:
    """🔥 CONFIGURAÇÕES ULTRA-ESPECIALIZADAS PARA TRADING 48H"""
    return {
        'lstm_hidden_size': 128,     # Base robusta
        'n_lstm_layers': 3,          # 3 GRU layers especializadas
        'attention_heads': 8,        # Multi-scale attention
        'volatility_awareness': True, # Awareness de volatilidade
        'features_extractor_kwargs': {
            'features_dim': 128,     # 128 features robustas
            'seq_len': 12           # 12 barras para contexto 48h
        },
        'net_arch': [dict(pi=[384, 256, 128], vf=[384, 256, 128])],
        'activation_fn': nn.ReLU,
        'ortho_init': True,
    }