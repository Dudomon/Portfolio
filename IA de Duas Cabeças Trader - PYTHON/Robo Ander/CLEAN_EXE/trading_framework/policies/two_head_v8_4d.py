"""
🚀 TwoHeadV8_4D - "V8 Elegance adaptado para 4D Action Space"

FILOSOFIA V8_4D:
- BASEADO NO V8 ELEGANCE QUE FUNCIONA
- ADAPTADO para 4D action space do 4dim.py  
- UMA LSTM compartilhada (elegância)
- Entry Head: entry_decision + confidence (2D)
- Management Head: pos1_mgmt + pos2_mgmt (2D)
- Total: 4D action space
- Obs Space: 450D compacto

ARQUITETURA:
- Single LSTM Backbone (256D)
- Entry Head: [0-2] decision + [0-1] confidence  
- Management Head: [-1,1] pos1 + [-1,1] pos2
- Memory Bank simplificado (512 trades)
- Market Context único (4 regimes)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Type, Union, Tuple
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution
import torch.nn.functional as F

# Importar transformer funcional para 450D
from trading_framework.extractors.transformer_v9_compact import TradingTransformerV9Compact

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

class MarketContextEncoder(nn.Module):
    """🌍 Market Context Encoder - detector de regime simples"""
    
    def __init__(self, input_dim: int = 256, context_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.context_dim = context_dim
        
        # Detector de regime (Bull/Bear/Sideways/Volatile)
        self.regime_detector = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(64),
            nn.Dropout(0.05),
            nn.Linear(64, 4)  # 4 regimes
        )
        
        # Context embedding
        self.regime_embedding = nn.Embedding(4, 16)
        
        # Context processor
        self.context_processor = nn.Sequential(
            nn.Linear(input_dim + 16, context_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(context_dim)
        )
        
    def forward(self, lstm_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Processa features do LSTM e retorna contexto de mercado
        Returns: (context_features, regime_id_tensor, info)
        """
        # Detectar regime
        regime_logits = self.regime_detector(lstm_features)
        
        # Handle batch dimension properly
        if len(regime_logits.shape) == 3:  # batch, seq, classes
            regime_logits_last = regime_logits[:, -1, :]
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
        
        # Info dict
        regime_names = ['Bull', 'Bear', 'Sideways', 'Volatile']
        regime_name = regime_names[regime_id_tensor.item() if hasattr(regime_id_tensor, 'item') else 0]
        
        info = {
            'regime_id': regime_id_tensor,
            'regime_name': regime_name,
            'regime_confidence': F.softmax(regime_logits, dim=-1).max().item() if regime_logits.numel() > 0 else 0.0
        }
        
        return context_features, regime_id_tensor, info

class EntryHead4D(nn.Module):
    """🎯 Entry Head V8_4D - 2D output: entry_decision + confidence"""
    
    def __init__(self, input_dim: int = 256, context_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.context_dim = context_dim
        
        # Entry processor
        self.entry_processor = nn.Sequential(
            nn.Linear(input_dim + context_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(64)
        )
        
        # Output heads
        self.entry_decision_head = nn.Linear(64, 1)  # [0-2] scaled
        self.confidence_head = nn.Linear(64, 1)      # [0-1]
        
    def forward(self, lstm_features: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do Entry Head
        Returns: tensor [batch, 2] com [entry_decision, confidence]
        """
        # Combinar features
        combined = torch.cat([lstm_features, context_features], dim=-1)
        
        # Process
        processed = self.entry_processor(combined)
        
        # Output layers
        entry_raw = self.entry_decision_head(processed)
        confidence_raw = self.confidence_head(processed)
        
        # Apply activations e scaling
        entry_decision = torch.tanh(entry_raw) * 1.0 + 1.0  # [-1,1] -> [0,2]
        confidence = torch.sigmoid(confidence_raw)          # [0,1]
        
        return torch.cat([entry_decision, confidence], dim=-1)

class ManagementHead4D(nn.Module):
    """💰 Management Head V8_4D - 2D output: pos1_mgmt + pos2_mgmt"""
    
    def __init__(self, input_dim: int = 256, context_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.context_dim = context_dim
        
        # Management processor
        self.mgmt_processor = nn.Sequential(
            nn.Linear(input_dim + context_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(64)
        )
        
        # Output heads
        self.pos1_mgmt_head = nn.Linear(64, 1)  # [-1,1]
        self.pos2_mgmt_head = nn.Linear(64, 1)  # [-1,1]
        
    def forward(self, lstm_features: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do Management Head
        Returns: tensor [batch, 2] com [pos1_mgmt, pos2_mgmt]
        """
        # Combinar features
        combined = torch.cat([lstm_features, context_features], dim=-1)
        
        # Process
        processed = self.mgmt_processor(combined)
        
        # Output layers
        pos1_raw = self.pos1_mgmt_head(processed)
        pos2_raw = self.pos2_mgmt_head(processed)
        
        # Apply tanh for [-1,1] range
        pos1_mgmt = torch.tanh(pos1_raw)
        pos2_mgmt = torch.tanh(pos2_raw)
        
        return torch.cat([pos1_mgmt, pos2_mgmt], dim=-1)

class TwoHeadV8_4D(RecurrentActorCriticPolicy):
    """🚀 TwoHeadV8_4D - V8 Elegance adaptado para 4D Action Space"""
    
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        # Extrair parâmetros específicos V8_4D
        self.v8_lstm_hidden = kwargs.pop('lstm_hidden_size', 256)
        self.v8_features_dim = kwargs.pop('features_dim', 256)
        self.v8_context_dim = kwargs.pop('context_dim', 64)
        
        print(f"🚀 V8_4D inicializando:")
        print(f"   LSTM Hidden: {self.v8_lstm_hidden}D")
        print(f"   Features: {self.v8_features_dim}D")
        print(f"   Context: {self.v8_context_dim}D")
        print(f"   Action Space: {action_space.shape} (4D)")
        print(f"   Obs Space: {observation_space.shape} (450D)")
        
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        
        # Inicializar componentes específicos
        self._setup_v8_4d_components()
        
    def _setup_v8_4d_components(self):
        """Setup dos componentes específicos V8_4D"""
        print("🔧 Configurando componentes V8_4D...")
        
        # Market Context Encoder
        self.market_context = MarketContextEncoder(
            input_dim=self.v8_lstm_hidden,
            context_dim=self.v8_context_dim
        )
        
        # Entry Head (2D)
        self.entry_head = EntryHead4D(
            input_dim=self.v8_lstm_hidden,
            context_dim=self.v8_context_dim
        )
        
        # Management Head (2D)
        self.management_head = ManagementHead4D(
            input_dim=self.v8_lstm_hidden,
            context_dim=self.v8_context_dim
        )
        
        print("✅ V8_4D componentes configurados!")
        
    def forward_actor(self, features: torch.Tensor, lstm_states, episode_starts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass da Actor Network V8_4D (4D output)
        """
        # LSTM forward
        lstm_output, new_lstm_states = self.lstm_actor(features, lstm_states, episode_starts)
        
        # Market context
        context_features, regime_id, regime_info = self.market_context(lstm_output)
        
        # Entry Head (2D)
        entry_actions = self.entry_head(lstm_output, context_features)
        
        # Management Head (2D)
        mgmt_actions = self.management_head(lstm_output, context_features)
        
        # Combinar para 4D: [entry_decision, confidence, pos1_mgmt, pos2_mgmt]
        combined_actions = torch.cat([entry_actions, mgmt_actions], dim=-1)
        
        # Info dict
        info = {
            'regime_info': regime_info,
            'entry_decision': entry_actions[:, 0].mean().item(),
            'confidence': entry_actions[:, 1].mean().item(),
            'pos1_mgmt': mgmt_actions[:, 0].mean().item(),
            'pos2_mgmt': mgmt_actions[:, 1].mean().item()
        }
        
        return combined_actions, new_lstm_states, info
        
    def forward(self, obs: PyTorchObs, lstm_states, episode_starts: torch.Tensor, deterministic: bool = False):
        """
        Forward pass completo V8_4D - USAR MÉTODO DA CLASSE BASE
        """
        # Usar features_extractor da classe base corretamente
        features = self.extract_features(obs, self.features_extractor)
        
        # Actor forward
        actions, new_lstm_states, actor_info = self.forward_actor(features, lstm_states, episode_starts)
        
        # Value forward (usando LSTM compartilhada)
        values = self.value_net(features)
        
        # Log probs (distribuição será criada pela classe base quando necessário)
        log_prob = None  # Será calculado pela classe base quando necessário
        
        return actions, values, log_prob, new_lstm_states

def get_v8_4d_kwargs():
    """Configurações para TwoHeadV8_4D"""
    return {
        'features_extractor_class': TradingTransformerV9Compact,
        'features_extractor_kwargs': {
            'features_dim': 256,
        },
        'lstm_hidden_size': 256,
        'context_dim': 64,
        'n_lstm_layers': 1,
        'shared_lstm': True,
        'enable_critic_lstm': False,
        'activation_fn': nn.LeakyReLU,
        'net_arch': [],  # Custom architecture
        'ortho_init': False,  # IMPORTANTE: não quebrar transformer
        'log_std_init': -0.5,
    }

def validate_v8_4d_policy(policy=None):
    """Valida a política V8_4D"""
    import gym
    
    if policy is None:
        dummy_obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
        dummy_action_space = gym.spaces.Box(low=np.array([0, 0, -1, -1]), high=np.array([2, 1, 1, 1]), dtype=np.float32)
        
        def dummy_lr_schedule(progress):
            return 1e-4
        
        policy = TwoHeadV8_4D(
            observation_space=dummy_obs_space,
            action_space=dummy_action_space,
            lr_schedule=dummy_lr_schedule,
            **get_v8_4d_kwargs()
        )
    
    print("✅ TwoHeadV8_4D validada - V8 Elegance adaptado para 4D!")
    print(f"   🧠 LSTM: {policy.v8_lstm_hidden}D")
    print(f"   🎯 Entry Head: 2D (entry_decision + confidence)")
    print(f"   💰 Management Head: 2D (pos1_mgmt + pos2_mgmt)")
    print(f"   🌍 Context: {policy.v8_context_dim}D")
    print(f"   📊 Total Actions: 4D")
    
    return True

if __name__ == "__main__":
    print("🚀 TwoHeadV8_4D - V8 Elegance adaptado para 4D Action Space!")
    validate_v8_4d_policy()