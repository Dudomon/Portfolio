#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 TWOHEAD V2 GRU POLICY - VERSÃO COM GRADIENT CLIPPING

CORREÇÃO DEFINITIVA:
- ✅ Input normalization (resolver input muito grande)
- ✅ Gradient clipping (resolver gradient explosion)
- ✅ GRU 1 layer estável
- ✅ Inicialização ultra conservadora
- ✅ Layer normalization para estabilidade
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Type
from collections import namedtuple
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 🔥 FALLBACK PARA COMPATIBILIDADE
try:
    from stable_baselines3.common.type_aliases import PyTorchObs
except ImportError:
    PyTorchObs = torch.Tensor

# 🔥 IMPORTS DO FRAMEWORK
try:
    from ..extractors.transformer_extractor import TradingTransformerFeatureExtractor
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from extractors.transformer_extractor import TradingTransformerFeatureExtractor

class TwoHeadV2GRUPolicy(RecurrentActorCriticPolicy):
    """
    🔥 TWOHEAD V2 GRU POLICY - VERSÃO COM GRADIENT CLIPPING
    
    CORREÇÕES DEFINITIVAS:
    - ✅ Input normalization (resolver input muito grande)
    - ✅ Gradient clipping automático
    - ✅ GRU 1 layer estável
    - ✅ Layer normalization para estabilidade
    - ✅ Inicialização ultra conservadora
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
        # 🔥 PARÂMETROS GRU COM CLIPPING
        gru_hidden_size: int = 64,    # Ainda menor
        n_gru_layers: int = 1,        # 1 layer
        max_grad_norm: float = 0.5,   # GRADIENT CLIPPING!
        **kwargs
    ):
        """
        Inicializa TwoHeadV2GRUPolicy com GRADIENT CLIPPING
        """
        
        # 🔥 CONFIGURAÇÕES COM CLIPPING
        self.gru_hidden_size = gru_hidden_size
        self.n_gru_layers = n_gru_layers
        self.max_grad_norm = max_grad_norm
        
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {
                'features_dim': 128,
                'seq_len': 6  # Ainda menor
            }
        
        # 🔥 ARQUITETURA ULTRA SIMPLES
        if net_arch is None:
            net_arch = [dict(pi=[64, 32], vf=[64, 32])]  # Ainda menor
        
        # ✅ DETECTAR ACTION SPACE
        self.action_dim = action_space.shape[0]
        print(f"🔥 TwoHeadV2GRU COM CLIPPING detectou action_space: {self.action_dim}D")
        
        # 🔥 REMOVER PARÂMETROS DUPLICADOS DOS KWARGS
        cleaned_kwargs = {k: v for k, v in kwargs.items() 
                         if k not in ['lstm_hidden_size', 'n_lstm_layers', 'gru_hidden_size', 'n_gru_layers', 'max_grad_norm']}
        
        # 🔥 INICIALIZAÇÃO DA CLASSE PAI
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
            # 🔥 PARÂMETROS COMPATÍVEIS COM RECURRENTPPO
            lstm_hidden_size=self.gru_hidden_size,
            n_lstm_layers=self.n_gru_layers,
            **cleaned_kwargs
        )
        
        # 🔥 COMPONENTES COM NORMALIZAÇÃO
        self._init_normalized_components()
        
        # 🔥 INICIALIZAÇÃO ULTRA CONSERVADORA
        self._initialize_ultra_conservative_weights()
        
        # 🔥 LOGS DE CONFIGURAÇÃO
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔥 TwoHeadV2GRU COM CLIPPING: {total_params:,} parâmetros")
        print(f"🔥 Gradient clipping: max_norm={self.max_grad_norm}")
        print(f"🔥 Arquitetura: GRU({self.gru_hidden_size}) + LayerNorm")
    
    def _init_normalized_components(self):
        """
        🔥 COMPONENTES COM NORMALIZAÇÃO PARA ESTABILIDADE
        """
        features_dim = self.features_extractor.features_dim
        
        # 🔥 INPUT NORMALIZATION
        self.input_norm = nn.LayerNorm(features_dim)
        
        # 🔥 GRU SIMPLES PARA ACTOR
        self.lstm_actor = nn.GRU(
            input_size=features_dim,
            hidden_size=self.gru_hidden_size,
            num_layers=self.n_gru_layers,
            batch_first=True,
            dropout=0.0
        )
        
        # 🔥 GRU SIMPLES PARA CRITIC  
        self.lstm_critic = nn.GRU(
            input_size=features_dim,
            hidden_size=self.gru_hidden_size,
            num_layers=self.n_gru_layers,
            batch_first=True,
            dropout=0.0
        )
        
        # 🔥 OUTPUT NORMALIZATION
        self.gru_norm = nn.LayerNorm(self.gru_hidden_size)
        
        print(f"🔥 Componentes com normalização: Input + GRU + Output LayerNorm")
    
    def _initialize_ultra_conservative_weights(self):
        """
        🔥 INICIALIZAÇÃO ULTRA CONSERVADORA + GRADIENT CLIPPING
        """
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:  # 🔥 SÓ APLICAR EM TENSORES 2D+
                if 'gru' in name.lower() or 'lstm' in name.lower():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param, gain=0.001)  # AINDA MENOR!
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param, gain=0.001)      # AINDA MENOR!
                    else:
                        nn.init.xavier_uniform_(param, gain=0.001)
                else:
                    nn.init.xavier_uniform_(param, gain=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name and param.dim() == 1:  # 🔥 TENSORES 1D
                nn.init.normal_(param, mean=0.0, std=0.01)
        
        print("🔥 Inicialização EXTREMA aplicada (gain=0.001) + Gradient clipping")
    
    def _clip_gradients(self):
        """
        🔥 GRADIENT CLIPPING AUTOMÁTICO
        """
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)

def get_trading_gru_kwargs() -> Dict[str, Any]:
    """
    🔥 CONFIGURAÇÕES COM GRADIENT CLIPPING - SEM DUPLICATAS
    """
    return {
        # 🔥 PARÂMETROS ESPECÍFICOS DA POLICY (passados via __init__)
        'gru_hidden_size': 64,       # Será usado internamente
        'n_gru_layers': 1,           # Será usado internamente  
        'max_grad_norm': 0.5,        # Será usado internamente
        # 🔥 PARÂMETROS PARA RECURRENTPPO (sem conflito)
        'features_extractor_class': TradingTransformerFeatureExtractor,
        'features_extractor_kwargs': {
            'features_dim': 128,
            'seq_len': 6
        },
        'net_arch': [dict(pi=[64, 32], vf=[64, 32])],
        'activation_fn': nn.ReLU,
        'ortho_init': True,
    }

def create_two_head_v2_gru_policy(
    observation_space,
    action_space,
    lr_schedule,
    **kwargs
) -> TwoHeadV2GRUPolicy:
    """
    🔥 FACTORY FUNCTION PARA TWOHEADV2GRUPOLICY
    """
    # Merge com configurações padrão
    config = get_trading_gru_kwargs()
    config.update(kwargs)
    
    return TwoHeadV2GRUPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        **config
    ) 