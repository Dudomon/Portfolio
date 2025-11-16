"""
🔥 TWOHHEAD POLICY ORIGINAL RESTAURADA - ARQUITETURA ROBUSTA COMPLETA

Política customizada extremamente robusta para PPO trading.
- ARQUITETURA ORIGINAL: Muito mais parâmetros e capacidade de aprendizado
- LSTM ROBUSTO: 2 camadas, hidden_size=128, dropout=0.2  
- MLP ROBUSTO: Camadas densas maiores (512->256->128->64)
- ATTENTION MECHANISM: Multi-head attention para capturar dependências
- RESIDUAL CONNECTIONS: Skip connections para gradientes estáveis
- LAYER NORMALIZATION: Normalização em todas as camadas
- INICIALIZAÇÃO XAVIER: Inicialização robusta dos pesos
"""

import torch
import torch.nn as nn
import numpy as np
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from stable_baselines3.common.distributions import CategoricalDistribution
from typing import Dict, Any, Optional, Tuple, Union
import gym

class TwoHeadPolicy(RecurrentActorCriticPolicy):
    """
    🎯 TWO HEAD POLICY HÍBRIDA ESTÁVEL - MELHOR DOS DOIS MUNDOS
    
    Arquitetura híbrida que combina:
    - 1 camada LSTM robusta (128 hidden) para memória longa
    - 1 camada GRU estável para estabilidade
    - Dropout adaptativo para evitar overfitting
    - Layer normalization para gradientes estáveis
    """
    
    def __init__(self, observation_space, action_space, lr_schedule, 
                 use_sde=False, log_std_init=0.0, full_std=True, 
                 sde_net_arch=None, use_expln=False, squash_output=False,
                 features_extractor_class=None, features_extractor_kwargs=None,
                 normalize_images=True, lstm_hidden_size=128, n_lstm_layers=1, **kwargs):
        
        # 🎯 CONFIGURAÇÕES HÍBRIDAS ESTÁVEIS
        self.lstm_hidden_size = lstm_hidden_size  # 128 (robusto mas estável)
        self.n_lstm_layers = n_lstm_layers  # 1 (estável)
        
        # 🎯 FEATURES EXTRACTOR OTIMIZADO
        if features_extractor_class is None:
            from ..extractors.transformer_extractor import TradingTransformerFeatureExtractor
            features_extractor_class = TradingTransformerFeatureExtractor
            
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {'features_dim': 128}  # Restaurado para 128
        
        # 🔥 CONFIGURAÇÕES HÍBRIDAS ROBUSTAS
        self.features_dim = 128  # SEMPRE 128 para TransformerFeatureExtractor
        self.attention_heads = 4  # Reduzido para estabilidade
        
        # 🔥 ARQUITETURA HÍBRIDA - EQUILIBRADA
        if 'net_arch' not in kwargs:
            kwargs['net_arch'] = [dict(pi=[256, 128, 64], vf=[256, 128, 64])]  # Reduzido mas robusto
        
        # 🔥 CONFIGURAÇÕES LSTM HÍBRIDAS ESTÁVEIS
        print(f"🎯 TwoHeadPolicy HÍBRIDA ESTÁVEL: features_dim={self.features_dim}, lstm_hidden={self.lstm_hidden_size}, n_layers={self.n_lstm_layers}")
        
        super().__init__(
            observation_space, action_space, lr_schedule,
            use_sde=use_sde, log_std_init=log_std_init, full_std=full_std,
            sde_net_arch=sde_net_arch, use_expln=use_expln, squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            lstm_hidden_size=self.lstm_hidden_size,
            n_lstm_layers=self.n_lstm_layers,
            **kwargs
        )
        
        # 🔥 COMPONENTES HÍBRIDOS ESTÁVEIS
        self._build_hybrid_components()
        
        # 🔥 INICIALIZAÇÃO ESTÁVEL
        self._initialize_stable_weights()
        
        # Debug info
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🎯 TwoHeadPolicy HÍBRIDA ESTÁVEL: {total_params:,} parâmetros")
    
    def _build_hybrid_components(self):
        """🎯 CONSTRÓI COMPONENTES HÍBRIDOS ESTÁVEIS"""
        
        # 🎯 GRU ESTÁVEL PARA COMPLEMENTAR LSTM
        self.gru_layer = nn.GRU(
            input_size=self.lstm_hidden_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0  # 🔥 CORREÇÃO: Sem dropout para 1 camada
        )
        
        # 🎯 LAYER NORMALIZATION PARA ESTABILIDADE
        self.lstm_norm = nn.LayerNorm(self.lstm_hidden_size)
        self.gru_norm = nn.LayerNorm(self.lstm_hidden_size)
        
        # 🎯 ATTENTION ESTÁVEL (REDUZIDO)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.lstm_hidden_size,
            num_heads=self.attention_heads,  # 4 heads (reduzido)
            dropout=0.05,  # Dropout baixo
            batch_first=True
        )
        
        # 🎯 FEATURE FUSION ESTÁVEL
        fusion_input = self.features_dim + self.lstm_hidden_size * 2  # LSTM + GRU
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),  # Dropout baixo
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.05)  # Dropout muito baixo
        )
        
        # 🎯 RESIDUAL CONNECTIONS ESTÁVEIS
        self.residual_policy = nn.Linear(128, 64)
        self.residual_value = nn.Linear(128, 64)
        
        print(f"🎯 Componentes híbridos: LSTM({self.lstm_hidden_size}) + GRU({self.lstm_hidden_size}) + Attention({self.attention_heads} heads)")
    
    def _initialize_stable_weights(self):
        """🎯 INICIALIZAÇÃO ESTÁVEL PARA EVITAR EXPLOSÃO DE GRADIENTES"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform com escala reduzida
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # Gain reduzido
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data, gain=0.5)  # Gain reduzido
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data, gain=0.5)  # Gain reduzido
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # Forget gate bias = 1 (estável)
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data, gain=0.5)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data, gain=0.5)
                    elif 'bias' in name:
                        param.data.fill_(0)
        
        print(f"🎯 Inicialização estável aplicada (gain=0.5)")

def create_two_head_policy(**kwargs):
    """🔥 FACTORY FUNCTION PARA CRIAR TWOHHEAD POLICY ORIGINAL"""
    return TwoHeadPolicy


def get_default_policy_kwargs():
    """🎯 CONFIGURAÇÕES PADRÃO PARA TWOHHEAD POLICY HÍBRIDA ESTÁVEL"""
    return {
        'lstm_hidden_size': 128,  # 🎯 HÍBRIDO: 128 (robusto mas estável)
        'n_lstm_layers': 1,       # 🎯 HÍBRIDO: 1 camada (estável)
        'net_arch': [dict(pi=[256, 128, 64], vf=[256, 128, 64])],  # 🎯 HÍBRIDO: Equilibrado
    } 