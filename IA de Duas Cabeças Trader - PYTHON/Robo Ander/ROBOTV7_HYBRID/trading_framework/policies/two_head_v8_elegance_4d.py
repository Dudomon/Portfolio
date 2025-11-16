"""
🚀 TwoHeadV8Elegance_4D - "V8 Elegance Adaptado para 4D Action Space"

ESTRATÉGIA:
- COPIAR estrutura da V8Elegance (comprovadamente funcional)
- MODIFICAR apenas as saídas para 4D
- MANTER toda a infraestrutura funcional
- Action Space: 4D [entry_decision, confidence, pos1_mgmt, pos2_mgmt]
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Type, Union, Tuple
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution
import torch.nn.functional as F

# Importar transformer que o 4dim.py realmente usa
from trading_framework.extractors.transformer_v9_daytrading import TradingTransformerV9

# Fallback para PyTorchObs
try:
    from stable_baselines3.common.type_aliases import PyTorchObs
except ImportError:
    PyTorchObs = torch.Tensor

# Imports corretos para RecurrentPPO
try:
    from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
except ImportError:
    from stable_baselines3.common.policies import RecurrentActorCriticPolicy

class MarketContextEncoder_4D(nn.Module):
    """🌍 Market Context Encoder - detector de regime simplificado"""
    
    def __init__(self, input_dim: int = 256, context_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.context_dim = context_dim
        
        # Detector de regime simples
        self.regime_detector = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(128),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 4)  # 4 regimes
        )
        
        # Context embedding
        self.regime_embedding = nn.Embedding(4, 32)
        
        # Context processor
        self.context_processor = nn.Sequential(
            nn.Linear(input_dim + 32, context_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(context_dim)
        )
        
    def forward(self, lstm_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Processa features do LSTM e retorna contexto de mercado
        Returns: (context_features, regime_id, info)
        """
        # Detectar regime
        regime_logits = self.regime_detector(lstm_features)
        
        # Handle batch dimension properly
        if len(regime_logits.shape) == 3:  # batch, seq, classes
            regime_logits_last = regime_logits[:, -1, :]  # batch, classes
            regime_id_tensor = torch.argmax(regime_logits_last[0], dim=-1)
        elif len(regime_logits.shape) == 2:  # batch, classes
            regime_id_tensor = torch.argmax(regime_logits[0], dim=-1)
        else:
            regime_id_tensor = torch.argmax(regime_logits, dim=-1)
        
        # Embedding do regime
        regime_emb = self.regime_embedding(regime_id_tensor)
        if len(lstm_features.shape) == 3:  # batch, seq, features
            batch_size, seq_len = lstm_features.shape[:2]
            regime_emb = regime_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, seq_len, -1)
        else:  # batch, features
            batch_size = lstm_features.shape[0]
            regime_emb = regime_emb.unsqueeze(0).expand(batch_size, -1)
        
        # Combinar features + regime
        combined = torch.cat([lstm_features, regime_emb], dim=-1)
        context_features = self.context_processor(combined)
        
        info = {'regime_id': regime_id_tensor}
        
        return context_features, regime_id_tensor, info

class DaytradeEntryHead_4D(nn.Module):
    """🎯 Entry Head para 4D - entry decision + confidence"""
    
    def __init__(self, input_dim: int = 320):  # LSTM(256) + context(64)
        super().__init__()
        
        self.input_dim = input_dim
        
        # Entry Decision Network
        self.entry_decision_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1)  # Raw logit for entry decision
        )
        
        # Entry Confidence Network
        self.entry_confidence_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Tanh()  # Tanh output for confidence
        )
        
    def forward(self, combined_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processa entrada e retorna entry decision + confidence
        Returns: (entry_decision, entry_confidence)
        """
        # Entry decision (0-2 range)
        raw_entry = self.entry_decision_net(combined_input)
        entry_decision = torch.tanh(raw_entry) * 1.0 + 1.0  # [0-2]
        
        # Entry confidence (0-1 range)
        raw_confidence = self.entry_confidence_net(combined_input)
        entry_confidence = (raw_confidence + 1.0) / 2.0  # [-1,1] → [0,1]
        
        return entry_decision, entry_confidence

class ManagementHead_4D(nn.Module):
    """💰 Management Head para 4D - 2 posições apenas"""
    
    def __init__(self, input_dim: int = 320):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Position 1 Management
        self.pos1_mgmt_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Tanh()  # [-1,1] output
        )
        
        # Position 2 Management
        self.pos2_mgmt_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Tanh()  # [-1,1] output
        )
        
    def forward(self, combined_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processa entrada e retorna management para 2 posições
        Returns: (pos1_mgmt, pos2_mgmt)
        """
        pos1_mgmt = self.pos1_mgmt_net(combined_input)
        pos2_mgmt = self.pos2_mgmt_net(combined_input)
        
        return pos1_mgmt, pos2_mgmt

class TwoHeadV8Elegance_4D(RecurrentActorCriticPolicy):
    """
    🚀 TwoHeadV8Elegance_4D - "V8 Elegance Adaptado para 4D"
    
    ARQUITETURA 4D:
    - UMA LSTM compartilhada (256D) - elegância da V8
    - Entry Head específico (entry + confidence) - 2D
    - Management Head específico (pos1 + pos2) - 2D  
    - Market Context único (4 regimes) - simplicidade
    - 4D action space: [entry_decision, confidence, pos1_mgmt, pos2_mgmt]
    """
    
    def __init__(
        self,
        observation_space,
        action_space, 
        *args,
        # V8 4D PARAMETERS
        v8_lstm_hidden: int = 256,
        v8_features_dim: int = 256,
        v8_context_dim: int = 64,
        **kwargs
    ):
        # Parâmetros específicos
        self.v8_lstm_hidden = v8_lstm_hidden
        self.v8_features_dim = v8_features_dim  
        self.v8_context_dim = v8_context_dim
        
        print(f"🚀 V8Elegance_4D inicializando:")
        print(f"   LSTM Hidden: {v8_lstm_hidden}D")
        print(f"   Features: {v8_features_dim}D")
        print(f"   Context: {v8_context_dim}D")
        print(f"   Action Space: {action_space.shape} (4D)")
        print(f"   Obs Space: {observation_space.shape}")
        
        # Configurar heads ANTES do super().__init__() para _build_mlp_extractor
        combined_dim = v8_lstm_hidden + v8_context_dim  # 256 + 64 = 320
        
        # Inicializar como None primeiro para evitar erro no super().__init__()
        self.market_context_encoder = None
        self.entry_head = None
        self.management_head = None
        
        # Chamar super().__init__() 
        super().__init__(observation_space, action_space, *args, **kwargs)
        
        # Configurar heads reais DEPOIS do super().__init__()
        self.market_context_encoder = MarketContextEncoder_4D(
            input_dim=v8_lstm_hidden, 
            context_dim=v8_context_dim
        )
        
        self.entry_head = DaytradeEntryHead_4D(input_dim=combined_dim)
        self.management_head = ManagementHead_4D(input_dim=combined_dim)
        
        print("✅ V8Elegance_4D configurado com sucesso!")
        
    def _build_mlp_extractor(self) -> None:
        """
        Override para usar heads customizados em vez do MLP padrão
        """
        # Criar um mlp_extractor simples com atributos necessários
        class SimpleMLP(nn.Module):
            def __init__(self, feature_dim):
                super().__init__()
                self.latent_dim_pi = feature_dim
                self.latent_dim_vf = feature_dim
                self.forward_actor = nn.Identity()
                self.forward_critic = nn.Identity()
            
            def forward(self, features):
                return self.forward_actor(features), self.forward_critic(features)
        
        self.mlp_extractor = SimpleMLP(self.v8_lstm_hidden)
        
        # Durante super().__init__(), os heads podem não estar prontos ainda
        # Criar action_net vazio que será populado depois
        self.action_net = nn.ModuleDict()
        
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> torch.distributions.Distribution:
        """
        Override para gerar ações 4D usando nossos heads customizados
        latent_pi é 256D do LSTM, preciso fazer market context aqui
        """
        # Debug: verificar dimensões (REMOVIDO - spam no treinamento)
        # print(f"DEBUG latent_pi shape: {latent_pi.shape}")
        
        # Se latent_pi tem seq dimension, remover
        if len(latent_pi.shape) == 3:  # [batch, seq, features]
            latent_pi = latent_pi.squeeze(1)  # [batch, features]
        
        # Market context (256D → 64D)
        context_features, regime_id, _ = self.market_context_encoder(latent_pi)
        
        # Combinar LSTM + context (256D + 64D = 320D)
        combined_input = torch.cat([latent_pi, context_features], dim=-1)
        
        # Entry Head (2D)
        entry_decision, entry_confidence = self.entry_head(combined_input)
        
        # Management Head (2D) 
        pos1_mgmt, pos2_mgmt = self.management_head(combined_input)
        
        # Combinar para 4D: [entry_decision, confidence, pos1_mgmt, pos2_mgmt]
        combined_actions = torch.cat([entry_decision, entry_confidence, pos1_mgmt, pos2_mgmt], dim=-1)
        
        # Retornar distribuição compatível com SB3
        from stable_baselines3.common.distributions import DiagGaussianDistribution
        log_std = torch.log(torch.ones_like(combined_actions) * 0.01)  # Convert std to log_std
        return DiagGaussianDistribution(combined_actions.shape[-1]).proba_distribution(combined_actions, log_std)
    
    def forward_actor(self, features: torch.Tensor, lstm_states, episode_starts: torch.Tensor) -> torch.distributions.Distribution:
        """
        Forward Actor V8_4D - EXATAMENTE como V8 mas output 4D
        """
        # 1. Extract features first (450D → 256D via TradingTransformerV9)
        extracted_features = self.extract_features(features)  # [batch, 256]
        
        # 2. Add sequence dimension for LSTM (single timestep)
        extracted_features = extracted_features.unsqueeze(1)  # [batch, 1, 256]
        
        # 3. Processar através da LSTM compartilhada (IGUAL V8)
        lstm_out, new_lstm_states = self.lstm_actor(extracted_features, lstm_states)
        
        # 4. Market context (IGUAL V8)
        context_features, regime_id, context_info = self.market_context_encoder(lstm_out)
        
        # 5. Combinar LSTM + context (IGUAL V8)
        lstm_features_2d = lstm_out.squeeze(1)  # [batch, 256]
        context_features_2d = context_features.squeeze(1) if len(context_features.shape) == 3 else context_features
        combined_input = torch.cat([lstm_features_2d, context_features_2d], dim=-1)  # [batch, 320]
        
        # 6. Gerar ações 4D usando heads (DIFERENTE: 4D em vez de 8D)
        return self._get_action_dist_from_latent(combined_input)
    
    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass através do critic para obter valores
        """
        # Usar o value network da classe base
        return self.value_net(features)

def get_v8_elegance_4d_kwargs():
    """Configurações para TwoHeadV8Elegance_4D"""
    return {
        'features_extractor_class': TradingTransformerV9,
        'features_extractor_kwargs': {
            'features_dim': 256,
        },
        'v8_lstm_hidden': 256,
        'v8_features_dim': 256,
        'v8_context_dim': 64,
        'n_lstm_layers': 1,
        'shared_lstm': True,
        'enable_critic_lstm': False,
        'activation_fn': nn.LeakyReLU,
        'net_arch': [],  # Custom architecture
        'ortho_init': False,  # IMPORTANTE: não quebrar transformer
        'log_std_init': -0.5,
    }

def validate_v8_elegance_4d_policy(policy=None):
    """Valida a política V8Elegance_4D"""
    import gym
    
    if policy is None:
        dummy_obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
        dummy_action_space = gym.spaces.Box(low=np.array([0, 0, -1, -1]), high=np.array([2, 1, 1, 1]), dtype=np.float32)
        
        def dummy_lr_schedule(progress):
            return 1e-4
        
        policy = TwoHeadV8Elegance_4D(
            observation_space=dummy_obs_space,
            action_space=dummy_action_space,
            lr_schedule=dummy_lr_schedule,
            **get_v8_elegance_4d_kwargs()
        )
    
    print("✅ TwoHeadV8Elegance_4D validada - V8 elegante para 4D!")
    print(f"   🧠 LSTM: {policy.v8_lstm_hidden}D")
    print(f"   🎯 Entry Head: 2D (entry_decision + confidence)")
    print(f"   💰 Management Head: 2D (pos1_mgmt + pos2_mgmt)")
    print(f"   📊 Total Actions: 4D")
    
    return True

if __name__ == "__main__":
    print("🚀 TwoHeadV8Elegance_4D - V8 Elegance adaptado para 4D Action Space!")
    validate_v8_elegance_4d_policy()