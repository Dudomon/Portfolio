#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 TWOHEAD V3 HYBRID POLICY - MELHOR DOS DOIS MUNDOS

Política híbrida que combina o melhor das TwoHead original e V2:
- ARQUITETURA HÍBRIDA: 2 layers LSTM + 1 layer GRU para máxima estabilidade
- PATTERN RECOGNITION: Detectores simplificados (micro/macro patterns)
- TEMPORAL AWARENESS: Otimizado para trading 48h com hierarquia temporal
- TWO HEADS ESPECIALIZADAS: Entry Head + Management Head otimizadas
- ESTABILIDADE BALANCEADA: Inicialização equilibrada (gain=0.2)
- ATTENTION MODERADO: 6 heads para capturar dependências sem complexidade excessiva
- COMPATIBILIDADE TOTAL: Funciona com qualquer action space (6D, 7D, 10D)
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
from stable_baselines3.common.distributions import (
    Distribution,
    DiagGaussianDistribution,
    CategoricalDistribution,
    MultiCategoricalDistribution,
    BernoulliDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import get_action_dim

# 🔥 FALLBACK PARA COMPATIBILIDADE
try:
    from stable_baselines3.common.type_aliases import PyTorchObs
except ImportError:
    PyTorchObs = torch.Tensor

# 🔥 LSTM STATES FORMAT PARA RECURRENTPPO
LSTMStates = namedtuple('LSTMStates', ['pi', 'vf'])

# 🔥 IMPORTS DO FRAMEWORK
try:
    from ..extractors.transformer_extractor import TradingTransformerFeatureExtractor
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from extractors.transformer_extractor import TradingTransformerFeatureExtractor

class TwoHeadV3HybridPolicy(RecurrentActorCriticPolicy):
    """
    🚀 TWOHEAD V3 HYBRID POLICY - ARQUITETURA HÍBRIDA OTIMIZADA
    
    CARACTERÍSTICAS HÍBRIDAS:
    - ✅ LSTM ROBUSTO: 2 layers (128 hidden) para hierarquia temporal
    - ✅ GRU ESTÁVEL: 1 layer adicional para estabilização
    - ✅ PATTERN RECOGNITION: Detectores micro/macro simplificados
    - ✅ TEMPORAL ATTENTION: 6 heads para capturar dependências 48h
    - ✅ TWO HEADS ESPECIALIZADAS: Entry + Management otimizadas
    - ✅ ESTABILIDADE BALANCEADA: Inicialização equilibrada (gain=0.2)
    - ✅ COMPATIBILIDADE TOTAL: Action spaces 6D/7D/10D
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
        # 🚀 PARÂMETROS HÍBRIDOS OTIMIZADOS
        lstm_hidden_size: int = 128,  # Base robusta
        n_lstm_layers: int = 2,       # Equilíbrio: 2 layers (entre 1 e 3)
        attention_heads: int = 8,     # Otimizado: 8 heads (128 % 8 = 0)
        gru_enabled: bool = True,     # GRU adicional para estabilidade
        pattern_recognition: bool = True,  # Pattern detection simplificado
        **kwargs
    ):
        """
        Inicializa a TwoHeadV3HybridPolicy com arquitetura híbrida LSTM+GRU
        
        ARQUITETURA HÍBRIDA:
        - 2 layers LSTM para hierarquia temporal (short-term + medium-term)
        - 1 layer GRU para estabilização e noise reduction
        - Pattern recognition para micro/macro patterns
        - 6 heads attention para dependências temporais
        - Two heads especializadas para Entry/Management
        """
        
        # 🚀 CONFIGURAÇÕES HÍBRIDAS OTIMIZADAS
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {
                'features_dim': 128,  # Base robusta
                'seq_len': 10         # Contexto temporal otimizado para 48h
            }
        
        # 🚀 ARQUITETURA HÍBRIDA EQUILIBRADA
        if net_arch is None:
            net_arch = [dict(pi=[320, 256, 128], vf=[320, 256, 128])]  # Maior para LSTM+GRU
        
        # 🚀 PARÂMETROS DA ARQUITETURA HÍBRIDA
        self.lstm_hidden_size = lstm_hidden_size      # 128 (base)
        self.n_lstm_layers = n_lstm_layers            # 2 (hierarquia temporal)
        self.attention_heads = attention_heads        # 6 (moderado)
        self.gru_enabled = gru_enabled               # True (estabilidade)
        self.pattern_recognition = pattern_recognition # True (pattern detection)
        
        # 🚀 CONFIGURAÇÃO DAS 2 CAMADAS LSTM
        self.lstm_layer_configs = [
            {"units": 128, "dropout": 0.15, "focus": "short_term", "timeframe": "5min-1h"},
            {"units": 128, "dropout": 0.20, "focus": "medium_term", "timeframe": "1h-24h"}
        ]
        
        # ✅ CORREÇÃO: Detectar action space automaticamente
        self.action_dim = action_space.shape[0]
        print(f"🚀 TwoHeadV3HybridPolicy detectou action_space: {self.action_dim}D")
        
        # 🚀 INICIALIZAÇÃO DA CLASSE PAI
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
            # 🚀 PARÂMETROS LSTM PARA O PAI
            lstm_hidden_size=self.lstm_hidden_size,
            n_lstm_layers=self.n_lstm_layers,
            **kwargs
        )
        
        # 🚀 INICIALIZAÇÃO DOS COMPONENTES HÍBRIDOS
        self._init_hybrid_components()
        
        # 🚀 INICIALIZAÇÃO BALANCEADA
        self._initialize_balanced_weights()
        
        # 🚀 LOGS DE CONFIGURAÇÃO HÍBRIDA
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🚀 TwoHeadV3HybridPolicy: {total_params:,} parâmetros")
        print(f"🚀 Arquitetura Híbrida: 2-LSTM + 1-GRU + 6-Head Attention")
        print(f"🚀 LSTM Layer 1: {self.lstm_layer_configs[0]['units']} units - {self.lstm_layer_configs[0]['timeframe']}")
        print(f"🚀 LSTM Layer 2: {self.lstm_layer_configs[1]['units']} units - {self.lstm_layer_configs[1]['timeframe']}")
        print(f"🚀 GRU Stabilizer: {self.lstm_hidden_size} units - Noise reduction")
        print(f"🚀 Pattern Recognition: {'Enabled' if self.pattern_recognition else 'Disabled'}")
        print(f"🚀 Compatível com action_space: {self.action_dim}D")
    
    def _init_hybrid_components(self):
        """
        🚀 INICIALIZA COMPONENTES HÍBRIDOS LSTM+GRU+ATTENTION
        CORREÇÃO: Não criar LSTM duplicado - usar apenas o da classe pai
        """
        features_dim = self.features_extractor.features_dim  # 128
        
        # 🚀 CORREÇÃO CRÍTICA: NÃO CRIAR LSTM DUPLICADO
        # A classe pai RecurrentActorCriticPolicy já cria o LSTM
        # Vamos apenas usar processamento adicional
        
        # 🚀 GRU ESTABILIZADOR (da TwoHeadPolicy original)
        if self.gru_enabled:
            self.gru_stabilizer = nn.GRU(
                input_size=self.lstm_hidden_size,  # Input do LSTM output
                hidden_size=self.lstm_hidden_size, # Mesmo tamanho
                num_layers=1,                      # 1 layer para estabilidade
                batch_first=True,
                dropout=0.0  # Sem dropout para 1 camada
            )
            
            # Layer normalization para GRU
            self.gru_norm = nn.LayerNorm(self.lstm_hidden_size)
        
        # 🚀 TEMPORAL MULTI-HEAD ATTENTION (moderado - 6 heads)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.lstm_hidden_size,
            num_heads=self.attention_heads,  # 6 heads (moderado)
            dropout=0.10,
            batch_first=True
        )
        
        # 🚀 PATTERN RECOGNITION SIMPLIFICADO (da V2, mas simplificado)
        if self.pattern_recognition:
            # Micro Pattern Detector (Scalping, Short-term)
            self.micro_pattern_detector = nn.Sequential(
                nn.Linear(self.lstm_hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(64, 32),
                nn.Sigmoid()
            )
            
            # Macro Pattern Detector (Trends, Long-term)
            self.macro_pattern_detector = nn.Sequential(
                nn.Linear(self.lstm_hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(64, 32),
                nn.Sigmoid()
            )
            
            pattern_features = 32 + 32  # micro + macro
        else:
            pattern_features = 0
        
        # 🚀 FEATURE FUSION HÍBRIDA
        # Combina LSTM (da classe pai) + GRU + Attention + Patterns
        fusion_input_size = self.lstm_hidden_size  # Base LSTM (da classe pai)
        if self.gru_enabled:
            fusion_input_size += self.lstm_hidden_size  # + GRU
        fusion_input_size += pattern_features  # + Patterns
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 320),  # Maior para combinar tudo
            nn.LayerNorm(320),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(320, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # 🚀 TWO HEADS ESPECIALIZADAS (da V2, mas otimizadas)
        
        # Entry Head - Especializada em QUANDO/COMO entrar
        self.entry_head = nn.Sequential(
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # Management Head - Especializada em gerenciamento de posições
        self.management_head = nn.Sequential(
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # 🚀 CORREÇÃO CRÍTICA: NÃO CRIAR ACTION_NET/VALUE_NET CUSTOMIZADAS
        # A classe pai RecurrentActorCriticPolicy já cria essas redes
        # Vamos apenas usar o processamento híbrido para modificar as features
        
        # Salvar tamanho final para validação
        self.combined_head_size = 64 + 64  # entry + management = 128
        
        # 🚀 RESIDUAL CONNECTIONS (da TwoHeadPolicy original)
        self.residual_connection = nn.Linear(128, self.combined_head_size)
        
        print(f"🚀 Componentes Híbridos Inicializados:")
        print(f"🚀 - LSTM: {self.n_lstm_layers} layers × {self.lstm_hidden_size} units")
        print(f"🚀 - GRU: {'Enabled' if self.gru_enabled else 'Disabled'}")
        print(f"🚀 - Attention: {self.attention_heads} heads")
        print(f"🚀 - Pattern Recognition: {'Enabled' if self.pattern_recognition else 'Disabled'}")
        print(f"🚀 - Feature Fusion: {fusion_input_size} → 320 → 256 → 128")
        print(f"🚀 - Combined Features: {self.combined_head_size} (Entry + Management)")
    
    def _initialize_balanced_weights(self):
        """
        🚀 INICIALIZAÇÃO BALANCEADA (gain=0.2)
        Equilibra estabilidade da TwoHead original com robustez da V2
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform com gain mais agressivo para V3
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # Mais agressivo
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data, gain=0.2)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data, gain=0.2)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # Forget gate bias = 1 para estabilidade
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data, gain=0.2)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data, gain=0.2)
                    elif 'bias' in name:
                        param.data.fill_(0)
        
        print(f"🚀 Inicialização agressiva aplicada (gain=0.5) para V3")
    
    # 🚀 CORREÇÃO CRÍTICA: REMOVER FORWARD CUSTOMIZADO
    # O RecurrentPPO usa métodos internos (_get_latent_from_obs, _get_action_dist_from_latent)
    # Forward customizado causa incompatibilidades de dimensões
    
    def _get_latent_from_obs(
        self,
        obs: PyTorchObs,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor
    ) -> torch.Tensor:
        """
        🚀 MÉTODO INTERNO CORRIGIDO - COMPATÍVEL COM RECURRENTPPO
        """
        # 1. Extrair features usando o método da classe pai
        features = self.extract_features(obs)  # [batch, features_dim]
        
        # 2. Simular processamento LSTM básico (a classe pai fará o processamento real)
        # Por enquanto, vamos apenas aplicar nosso processamento híbrido
        return self._apply_hybrid_processing(features)
    
    def _apply_hybrid_processing(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        🚀 APLICA PROCESSAMENTO HÍBRIDO ADICIONAL AOS FEATURES EXTRAÍDOS
        """
        # input_features vem do features extractor [batch, features_dim]
        
        # 1. Primeiro projetar features para dimensão LSTM se necessário
        if input_features.shape[-1] != self.lstm_hidden_size:
            if not hasattr(self, 'input_projector'):
                self.input_projector = nn.Linear(
                    input_features.shape[-1], 
                    self.lstm_hidden_size
                ).to(input_features.device)
            projected_features = self.input_projector(input_features)
        else:
            projected_features = input_features
        
        # 2. GRU estabilizador (se habilitado)
        if self.gru_enabled:
            gru_input = projected_features.unsqueeze(1)  # [batch, 1, lstm_hidden_size]
            gru_out, _ = self.gru_stabilizer(gru_input)
            gru_out = self.gru_norm(gru_out.squeeze(1))  # [batch, lstm_hidden_size]
        else:
            gru_out = projected_features
        
        # 3. Temporal Attention
        attn_input = projected_features.unsqueeze(1)  # [batch, 1, lstm_hidden_size]
        attn_out, _ = self.temporal_attention(attn_input, attn_input, attn_input)
        attn_out = attn_out.squeeze(1)  # [batch, lstm_hidden_size]
        
        # 4. Pattern Recognition (se habilitado)
        feature_list = [attn_out]
        
        if self.gru_enabled:
            feature_list.append(gru_out)
        
        if self.pattern_recognition:
            micro_patterns = self.micro_pattern_detector(attn_out)  # [batch, 32]
            macro_patterns = self.macro_pattern_detector(attn_out)  # [batch, 32]
            feature_list.extend([micro_patterns, macro_patterns])
        
        # 5. Feature Fusion
        fused_features = torch.cat(feature_list, dim=-1)
        fused_features = self.feature_fusion(fused_features)  # [batch, 128]
        
        # 6. Two Heads Especializadas
        entry_features = self.entry_head(fused_features)      # [batch, 64]
        management_features = self.management_head(fused_features)  # [batch, 64]
        
        # 7. Residual Connection
        residual = self.residual_connection(fused_features)   # [batch, 128]
        
        # 8. Combinar heads + residual
        combined_features = torch.cat([entry_features, management_features], dim=-1)  # [batch, 128]
        combined_features = combined_features + residual  # Residual connection
        
        # 9. CORREÇÃO CRÍTICA: Retornar features com dimensão que a classe pai espera
        # A classe pai espera features com lstm_hidden_size (que passamos no __init__)
        # Vamos projetar de volta para a dimensão correta
        final_features = self._project_to_parent_dimension(combined_features)
        
        return final_features  # [batch, lstm_hidden_size]
    
    def _project_to_parent_dimension(self, combined_features: torch.Tensor) -> torch.Tensor:
        """
        🚀 PROJETA FEATURES COMBINADAS PARA A DIMENSÃO QUE A CLASSE PAI ESPERA
        """
        # combined_features: [batch, 128] (entry + management)
        # Precisamos retornar: [batch, lstm_hidden_size]
        
        if combined_features.shape[-1] == self.lstm_hidden_size:
            # Já tem a dimensão correta
            return combined_features
        else:
            # Criar projeção linear se necessário
            if not hasattr(self, 'dimension_projector'):
                self.dimension_projector = nn.Linear(
                    combined_features.shape[-1], 
                    self.lstm_hidden_size
                ).to(combined_features.device)
            
            return self.dimension_projector(combined_features)
    
    # 🚀 CORREÇÃO CRÍTICA: REMOVER EVALUATE_ACTIONS CUSTOMIZADO
    # Usar apenas a implementação da classe pai que funciona corretamente
    # O processamento híbrido é aplicado automaticamente via _get_latent_from_obs
    
    # 🚀 CORREÇÃO CRÍTICA: USAR APENAS IMPLEMENTAÇÃO DA CLASSE PAI
    # Remover _get_action_dist_from_latent customizado para evitar conflitos

def get_hybrid_trading_kwargs() -> Dict[str, Any]:
    """
    🚀 CONFIGURAÇÕES HÍBRIDAS OTIMIZADAS PARA TWOHEADV3
    """
    return {
        # 🚀 ARQUITETURA HÍBRIDA
        'lstm_hidden_size': 128,     # Base robusta
        'n_lstm_layers': 2,          # Hierarquia temporal (short + medium)
        'attention_heads': 8,        # Otimizado (128 % 8 = 0)
        'gru_enabled': True,         # GRU para estabilidade
        'pattern_recognition': True, # Pattern detection
        
        # 🚀 FEATURES EXTRACTOR
        'features_extractor_class': TradingTransformerFeatureExtractor,
        'features_extractor_kwargs': {
            'features_dim': 128,     # Base robusta
            'seq_len': 10,           # Contexto otimizado para 48h
        },
        
        # 🚀 NETWORK ARCHITECTURE
        'net_arch': [dict(pi=[320, 256, 128], vf=[320, 256, 128])],
        
        # 🚀 OTIMIZAÇÕES
        'ortho_init': True,
        'activation_fn': nn.ReLU,
    }

def create_two_head_v3_hybrid_policy(
    observation_space,
    action_space,
    lr_schedule,
    **kwargs
) -> TwoHeadV3HybridPolicy:
    """
    🚀 FUNÇÃO AUXILIAR PARA CRIAÇÃO DA TWOHEAD V3 HYBRID POLICY
    """
    hybrid_kwargs = get_hybrid_trading_kwargs()
    hybrid_kwargs.update(kwargs)
    
    return TwoHeadV3HybridPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        **hybrid_kwargs
    ) 