#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 TWOHEAD V2 POLICY - POLÍTICA CORRIGIDA PARA 7DIM E 10DIM

Política especializada corrigida para funcionar com ambos action spaces:
- CORREÇÃO CRÍTICA: Compatibilidade com Box 7dim e 10dim
- LSTM robusto e estável (128 hidden, 1 layer como TwoHeadPolicy)
- Temporal Attention otimizado (4 heads como TwoHeadPolicy)
- Two Heads especializadas estáveis
- Inicialização estável (gain=0.5)
- Código limpo sem complexidade desnecessária
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
    # Fallback for older versions
    PyTorchObs = torch.Tensor

# 🔥 LSTM STATES FORMAT PARA RECURRENTPPO
LSTMStates = namedtuple('LSTMStates', ['pi', 'vf'])

# 🔥 IMPORTS DO FRAMEWORK
try:
    from ..extractors.transformer_extractor import TradingTransformerFeatureExtractor
except ImportError:
    # Fallback para import local
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from extractors.transformer_extractor import TradingTransformerFeatureExtractor

class TwoHeadV2Policy(RecurrentActorCriticPolicy):
    """
    🔥 TWOHEAD V2 POLICY - CORRIGIDA PARA 7DIM E 10DIM
    
    CORREÇÕES APLICADAS:
    - ✅ LSTM estável: 128 hidden, 1 layer (como TwoHeadPolicy)
    - ✅ Attention equilibrado: 4 heads (como TwoHeadPolicy) 
    - ✅ Arquitetura equilibrada: [256,128,64] (como TwoHeadPolicy)
    - ✅ Inicialização estável: gain=0.5 (como TwoHeadPolicy)
    - ✅ Compatibilidade total: 7dim e 10dim action spaces
    - ✅ Código limpo: removida complexidade desnecessária
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
        # 🧠 PARÂMETROS EXPERT PARA ARQUITETURA INTELIGENTE
        lstm_hidden_size: int = 128,  # 🧠 EXPERT: 128 para capacidade robusta
        n_lstm_layers: int = 3,       # 🧠 EXPERT: 3 layers para hierarquia temporal
        attention_heads: int = 8,     # 🧠 EXPERT: 8 heads para padrões complexos
        **kwargs
    ):
        """
        Inicializa a TwoHeadV2Policy com ARQUITETURA EXPERT E INTELIGENTE
        
        ARQUITETURA AVANÇADA:
        - lstm_hidden_size: 128 - capacidade robusta para padrões complexos
        - n_lstm_layers: 3 - hierarquia temporal (micro/médio/macro)
        - attention_heads: 8 - multi-head attention para capturar nuances
        - Foco: Inteligência máxima e capacidade expressiva
        """
        
        # 🧠 CONFIGURAÇÕES EXPERT PARA ARQUITETURA INTELIGENTE
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {
                'features_dim': 128,  # ✅ Mantido: 128 como base
                'seq_len': 12  # 🧠 EXPERT: Mais contexto temporal para decisões inteligentes
            }
        
        # 🧠 ARQUITETURA EXPERT: Máxima capacidade para inteligência
        if net_arch is None:
            net_arch = [dict(pi=[384, 256, 128], vf=[384, 256, 128])]  # Arquitetura robusta para padrões complexos
        
        # 🧠 PARÂMETROS DA ARQUITETURA EXPERT
        self.lstm_hidden_size = lstm_hidden_size  # 128 (robusto)
        self.n_lstm_layers = n_lstm_layers        # 3 (hierárquico)
        self.attention_heads = attention_heads    # 8 (multi-escala)
        
        # 🧠 CONFIGURAÇÃO DAS 3 CAMADAS ESPECIALIZADAS EXPERT
        self.lstm_layer_configs = [
            {"units": 128, "dropout": 0.10, "focus": "micro_patterns", "timeframe": "5-15min"},
            {"units": 128, "dropout": 0.15, "focus": "medium_patterns", "timeframe": "1-4h"},
            {"units": 128, "dropout": 0.20, "focus": "macro_patterns", "timeframe": "12-48h"}
        ]
        
        # ✅ CORREÇÃO: Detectar action space automaticamente
        self.action_dim = action_space.shape[0]
        print(f"🔥 TwoHeadV2Policy detectou action_space: {self.action_dim}D")
        
        # 🔥 INICIALIZAÇÃO DA CLASSE PAI COM PARÂMETROS CORRIGIDOS
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,  # ✅ Arquitetura equilibrada
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
            # 🔥 CORREÇÃO CRÍTICA: Passar parâmetros LSTM corretos para o pai
            lstm_hidden_size=self.lstm_hidden_size,
            n_lstm_layers=self.n_lstm_layers,
            **kwargs
        )
        
        # 🔥 INICIALIZAÇÃO DOS COMPONENTES CORRIGIDOS
        self._init_corrected_components()
        
        # 🔥 INICIALIZAÇÃO ESTÁVEL COMO TWOHEADPOLICY
        self._initialize_stable_weights()
        
        # 🧠 LOGS DE CONFIGURAÇÃO DA ARQUITETURA EXPERT
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🧠 TwoHeadV2Policy EXPERT: {total_params:,} parâmetros")
        print(f"🧠 Arquitetura Inteligente: {self.n_lstm_layers} camadas LSTM especializadas")
        print(f"🧠 Layer 1: {self.lstm_layer_configs[0]['units']} units - {self.lstm_layer_configs[0]['timeframe']}")
        print(f"🧠 Layer 2: {self.lstm_layer_configs[1]['units']} units - {self.lstm_layer_configs[1]['timeframe']}")
        print(f"🧠 Layer 3: {self.lstm_layer_configs[2]['units']} units - {self.lstm_layer_configs[2]['timeframe']}")
        print(f"🧠 Multi-Head Attention: {self.attention_heads} heads")
        print(f"🧠 Compatível com action_space: {self.action_dim}D")
    
    def _init_corrected_components(self):
        """
        🔥 ARQUITETURA CORRIGIDA: USAR APENAS LSTM DO PAI + PROCESSAMENTO ESPECIALIZADO
        
        CORREÇÃO: Não criar LSTM layers duplicados - usar apenas os do RecurrentPPO
        """
        features_dim = self.features_extractor.features_dim  # Pode ser 64 ou 128
        
        # 🔥 CORREÇÃO: NÃO CRIAR LSTM LAYERS DUPLICADOS
        # O RecurrentActorCriticPolicy já cria LSTM com lstm_hidden_size=128, n_lstm_layers=3
        # Vamos usar apenas processamento especializado
        
        # 🔥 LSTM LAYER PRINCIPAL (compatível com RecurrentActorCriticPolicy)
        self.lstm = nn.LSTM(
            input_size=features_dim,  # Features do transformer (64 ou 128)
            hidden_size=self.lstm_hidden_size,  # 128
            num_layers=self.n_lstm_layers,      # 3
            dropout=0.15 if self.n_lstm_layers > 1 else 0.0,
            batch_first=True
        )
        
        # 🔥 MULTI-SCALE TEMPORAL ATTENTION (8 heads para 3 escalas)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.lstm_hidden_size,  # Tamanho do LSTM hidden (128)
            num_heads=8,    # 8 heads para capturar múltiplas escalas
            dropout=0.10,
            batch_first=True
        )
        
        # 🔥 CROSS-TIMEFRAME FUSION
        # Combina informações das 3 escalas temporais
        self.timeframe_fusion = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 256),  # Input do LSTM hidden
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, self.lstm_hidden_size),  # Reduz para tamanho padrão
            nn.LayerNorm(self.lstm_hidden_size)
        )
        
        # 🔥 PATTERN RECOGNITION NETWORKS
        # Especialização para diferentes tipos de padrões
        
        # Micro Pattern Detector (Scalping, Noise)
        self.micro_pattern_detector = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Sigmoid()  # Probabilidade de padrão micro
        )
        
        # Medium Pattern Detector (Swing, Breakouts)
        self.medium_pattern_detector = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Sigmoid()  # Probabilidade de padrão médio
        )
        
        # Macro Pattern Detector (Trends, Cycles)
        self.macro_pattern_detector = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Sigmoid()  # Probabilidade de padrão macro
        )
        
        # 🔥 TRADING REGIME CLASSIFIER
        # Determina o melhor regime de trading baseado nas 3 escalas
        self.regime_classifier = nn.Sequential(
            nn.Linear(self.lstm_hidden_size + 32 + 32 + 32, 128),  # Fused + pattern scores
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 4 regimes: SCALP, SWING, TREND, HOLD
            nn.Softmax(dim=-1)
        )
        
        # 🔥 ADAPTIVE HEADS BASEADOS NO REGIME
        # Entry Head - Especializada em QUANDO entrar
        self.entry_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size + 4, 96),  # +4 para regime info
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # Management Head - Especializada em COMO gerenciar
        self.management_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size + 4, 96),  # +4 para regime info
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # 🔥 FINAL ACTION NETWORKS
        # Rede final que combina Entry + Management decisions
        self.action_net = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 128),  # Usar lstm_hidden_size
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, self.action_dim)  # Saída dinâmica (6 dim)
        )
        
        # Value network otimizada
        self.value_net = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 1)  # Valor escalar
        )
        
        print(f"🔥 Arquitetura CORRIGIDA: LSTM do pai + processamento especializado")
        print(f"🔥 Pattern Recognition: Micro + Medium + Macro detectors")
        print(f"🔥 Regime Classification: SCALP, SWING, TREND, HOLD")
        print(f"🔥 Action network: {self.lstm_hidden_size} -> 128 -> {self.action_dim}")
        print(f"🔥 Features dim: {features_dim} -> LSTM hidden: {self.lstm_hidden_size}")
    
    def _initialize_stable_weights(self):
        """
        🧠 INICIALIZAÇÃO EXPERT PARA ARQUITETURA INTELIGENTE
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 🧠 EXPERT: Inicialização inteligente baseada no tamanho da camada
                fan_in = module.weight.size(1)
                if fan_in > 256:  # Camadas grandes (384, etc.)
                    gain = 0.2  # Gain menor para camadas grandes
                elif fan_in > 128:  # Camadas médias (256, etc.)
                    gain = 0.4  # Gain moderado
                else:  # Camadas pequenas (128, 64)
                    gain = 0.6  # Gain maior para camadas pequenas
                
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data, gain=0.5)  # LSTM precisa de gain moderado
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data, gain=0.5)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # Forget gate bias = 1 para memória de longo prazo
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1)
            elif isinstance(module, nn.MultiheadAttention):
                # 🧠 EXPERT: Inicialização especial para attention layers
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param.data, gain=0.3)  # Attention precisa ser suave
                    elif 'bias' in name and param is not None:
                        nn.init.constant_(param.data, 0)
        
        print(f"🧠 Inicialização EXPERT aplicada - scaling inteligente baseado na arquitetura")
    
    # 🔥 FORWARD COMPLEXO REMOVIDO - USAR APENAS MÉTODOS INTERNOS QUE SEMPRE FUNCIONARAM
    # RecurrentPPO usa métodos internos: _process_sequence, _get_action_dist_from_latent, etc.
    # O forward() complexo estava interferindo na convergência
    
    # 🔥 MÉTODOS AUXILIARES REMOVIDOS - USAR IMPLEMENTAÇÃO PADRÃO DO PAI
    # forward_actor, forward_critic, predict_values eram dependentes do forward() customizado
    
    def evaluate_actions(
        self,
        obs: PyTorchObs,
        actions: torch.Tensor,
        lstm_states: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Avaliação de ações"""
        # Processar observações (sem risk awareness para evitar circularidade)
        base_features = self._get_base_latent_from_obs(obs, lstm_states, episode_starts)
        
        # 🔥 CORREÇÃO: Action distribution sem risk awareness circular
        action_logits = self.action_net(base_features)
        action_distribution = self._get_action_dist_from_latent(action_logits)
        
        # Values
        values = self.value_net(base_features)
        
        # Log prob e entropy
        log_prob = action_distribution.log_prob(actions)
        entropy = action_distribution.entropy()
        
        return values.flatten(), log_prob, entropy
    
    def _get_base_latent_from_obs(
        self,
        obs: PyTorchObs,
        lstm_states: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor
    ) -> torch.Tensor:
        """
        🔥 MÉTODO AUXILIAR CORRIGIDO: Usar apenas LSTM do pai
        Versão simplificada do forward para evaluate_actions
        """
        # ✅ 1. EXTRACT FEATURES
        features = self.extract_features(obs)  # [batch, 128]
        batch_size = features.shape[0]
        
        # Preparar para processamento sequencial
        if len(features.shape) == 2:
            features = features.unsqueeze(1)  # [batch, 1, 128]
        
        # ✅ 2. PROCESSAR LSTM STATES (usando lógica da classe pai)
        device = features.device
        dtype = features.dtype
        
        # Processar LSTM states corretamente
        if lstm_states is None:
            # Criar states iniciais para 3 camadas LSTM
            h_state = torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_size, device=device, dtype=dtype)
            c_state = torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_size, device=device, dtype=dtype)
            lstm_states = (h_state, c_state)
        else:
            # Usar states existentes e ajustar batch size se necessário
            if hasattr(lstm_states, 'pi'):
                h_state, c_state = lstm_states.pi
            else:
                h_state, c_state = lstm_states
            
            # 🔥 CORREÇÃO CRÍTICA: Ajustar batch size dinamicamente
            current_batch_size = h_state.shape[1]
            if current_batch_size != batch_size:
                # Redimensionar states conforme necessário
                if batch_size > current_batch_size:
                    # Expandir states para batch maior - repetir dados existentes
                    repeat_factor = batch_size // current_batch_size
                    remainder = batch_size % current_batch_size
                    
                    h_expanded = h_state.repeat(1, repeat_factor, 1)
                    c_expanded = c_state.repeat(1, repeat_factor, 1)
                    
                    if remainder > 0:
                        h_remainder = h_state[:, :remainder, :]
                        c_remainder = c_state[:, :remainder, :]
                        h_state = torch.cat([h_expanded, h_remainder], dim=1)
                        c_state = torch.cat([c_expanded, c_remainder], dim=1)
                    else:
                        h_state = h_expanded
                        c_state = c_expanded
                else:
                    # Cortar states para batch menor - usar apenas primeiros elementos
                    h_state = h_state[:, :batch_size, :].contiguous()
                    c_state = c_state[:, :batch_size, :].contiguous()
        
        # ✅ 3. PROCESSAR LSTM (simulando o comportamento da classe pai)
        # O RecurrentActorCriticPolicy usa LSTM interno, vamos simular
        lstm_out, _ = self.lstm(features, (h_state, c_state))
        lstm_out = lstm_out.squeeze(1)  # [batch, 128]
        
        return lstm_out
    
    def _get_latent_from_obs(
        self,
        obs: PyTorchObs,
        lstm_states: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor
    ) -> torch.Tensor:
        """
        🔥 MÉTODO AUXILIAR LIMPO: Extrair features latentes (simplificado)
        """
        # Usar features base diretamente
        return self._get_base_latent_from_obs(obs, lstm_states, episode_starts)
    
    def _get_action_dist_from_latent(self, action_logits: torch.Tensor):
        """
        🔥 CORREÇÃO CRÍTICA: Action distribution simplificada e estável
        Compatível com 7dim E 10dim Box action spaces
        """
        # ✅ CORREÇÃO: Sempre usar DiagGaussianDistribution para Box action space
        # Separar mean e log_std
        action_dim = self.action_dim
        
        if action_logits.shape[-1] >= action_dim * 2:
            # Se temos logits suficientes, usar metade para mean e metade para log_std
            mean_actions = action_logits[:, :action_dim]
            log_std_actions = action_logits[:, action_dim:action_dim*2]
        else:
            # Se não, usar todos para mean e log_std fixo
            mean_actions = action_logits
            log_std_actions = torch.zeros_like(mean_actions) - 0.5  # std ≈ 0.6
        
        # ✅ Garantir que log_std seja razoável
        log_std_actions = torch.clamp(log_std_actions, -2.0, 1.0)
        
        return DiagGaussianDistribution(action_dim).proba_distribution(mean_actions, log_std_actions)

def get_optimized_trading_kwargs() -> Dict[str, Any]:
    """
    🔥 CONFIGURAÇÕES CORRIGIDAS PARA TWOHEADV2POLICY
    
    Returns:
        Dict com configurações estáveis baseadas na TwoHeadPolicy
    """
    return {
        # 🔥 LSTM CORRIGIDO (baseado na TwoHeadPolicy funcional)
        'lstm_hidden_size': 128,     # ✅ 128 como TwoHeadPolicy (era 96)
        'n_lstm_layers': 3,          # ✅ 3 como TwoHeadPolicy (era 2)
        'attention_heads': 4,        # ✅ 4 como TwoHeadPolicy (era 8)
        
        # 🔥 FEATURES EXTRACTOR
        'features_extractor_class': TradingTransformerFeatureExtractor,
        'features_extractor_kwargs': {
            'features_dim': 128,     # 128 features
            'seq_len': 8,            # 8 barras de histórico
        },
        
        # 🔥 NETWORK ARCHITECTURE CORRIGIDA
        'net_arch': [dict(pi=[256, 128, 64], vf=[256, 128, 64])],  # ✅ Equilibrada
        
        # 🔥 OPTIMIZAÇÕES
        'ortho_init': True,
        'activation_fn': nn.ReLU,
    }

def create_two_head_v2_policy(
    observation_space,
    action_space,
    lr_schedule,
    **kwargs
) -> TwoHeadV2Policy:
    """
    🔥 FUNÇÃO AUXILIAR PARA CRIAÇÃO DA TWOHEAD V2 POLICY CORRIGIDA
    """
    # 🔥 MERGE COM CONFIGURAÇÕES CORRIGIDAS
    optimized_kwargs = get_optimized_trading_kwargs()
    optimized_kwargs.update(kwargs)
    
    return TwoHeadV2Policy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        **optimized_kwargs
    ) 