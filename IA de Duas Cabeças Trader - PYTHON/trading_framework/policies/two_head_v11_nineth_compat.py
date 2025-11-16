"""
V11 Nineth Compatible Policy - Baseada na arquitetura real do checkpoint.

Arquitetura detectada do Nineth 4.75M:
- v11_shared_gru (não LSTM)
- hybrid_fusion
- management_head com pos1_mgmt_net e pos2_mgmt_net
- memory_bank.memory_processor
- v8_critic com 6 camadas (0, 2, 4, 6, 8, 11)
- critic.weight/bias (value head final)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Type, Union, Tuple
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution
import torch.nn.functional as F

from trading_framework.extractors.transformer_extractor import TradingTransformerFeatureExtractor

try:
    from stable_baselines3.common.type_aliases import PyTorchObs
except ImportError:
    PyTorchObs = torch.Tensor

try:
    from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
except ImportError:
    from stable_baselines3.common.policies import RecurrentActorCriticPolicy


class MarketContextEncoder(nn.Module):
    """Market context sem raw features bypass."""

    def __init__(self, input_dim: int = 256, context_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim

        self.regime_detector = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(128),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 4),
        )

        self.regime_embedding = nn.Embedding(4, 32)

        self.context_processor = nn.Sequential(
            nn.Linear(input_dim + 32, context_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(context_dim),
        )

    def forward(self, gru_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        regime_logits = self.regime_detector(gru_features)

        if len(regime_logits.shape) == 3:
            regime_logits_last = regime_logits[:, -1, :]
            regime_id_tensor = torch.argmax(regime_logits_last[0], dim=-1)
        elif len(regime_logits.shape) == 2:
            regime_id_tensor = torch.argmax(regime_logits[0], dim=-1)
        else:
            regime_id_tensor = torch.argmax(regime_logits, dim=-1)

        regime_emb = self.regime_embedding(regime_id_tensor)
        if len(gru_features.shape) == 3:
            batch_size, seq_len = gru_features.shape[:2]
            regime_emb = regime_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, seq_len, -1)
        else:
            batch_size = gru_features.shape[0]
            regime_emb = regime_emb.unsqueeze(0).expand(batch_size, -1)

        combined = torch.cat([gru_features, regime_emb], dim=-1)
        context_features = self.context_processor(combined)

        info = {"regime_id": regime_id_tensor}
        return context_features, regime_id_tensor, info


class DaytradeManagementHead(nn.Module):
    """Management head com 2 redes separadas (pos1 e pos2)."""

    def __init__(self, input_dim: int = 320):
        super().__init__()

        # Posição 1
        self.pos1_mgmt_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 2),  # SL/TP para pos1
        )

        # Posição 2
        self.pos2_mgmt_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 2),  # SL/TP para pos2
        )

    def forward(self, combined_features: torch.Tensor):
        pos1_actions = torch.tanh(self.pos1_mgmt_net(combined_features))
        pos2_actions = torch.tanh(self.pos2_mgmt_net(combined_features))

        # Concatenar [batch, 2, 2]
        management_actions = torch.stack([pos1_actions, pos2_actions], dim=-2)

        gate_info = {"sl_tp_adjustments": management_actions}
        return management_actions, gate_info


class ElegantMemoryBank(nn.Module):
    """Memory bank com processor."""

    def __init__(self, memory_size: int = 512, trade_dim: int = 8):
        super().__init__()
        self.memory_size = memory_size
        self.trade_dim = trade_dim

        self.register_buffer("trade_memory", torch.zeros(memory_size, trade_dim))
        self.register_buffer("memory_ptr", torch.zeros(1, dtype=torch.long))

        # Memory processor do Nineth
        self.memory_processor = nn.Sequential(
            nn.Linear(trade_dim, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(16, 8),
            nn.LeakyReLU(negative_slope=0.01),
        )

    def add_trade(self, trade_data: torch.Tensor):
        if trade_data.dim() == 1:
            trade_data = trade_data.unsqueeze(0)

        num_new = trade_data.shape[0]
        insert_pos = self.memory_ptr.item() % self.memory_size
        end_pos = min(insert_pos + num_new, self.memory_size)
        to_insert = end_pos - insert_pos

        self.trade_memory[insert_pos:end_pos] = trade_data[:to_insert]
        self.memory_ptr += to_insert

        if to_insert < num_new:
            remaining = num_new - to_insert
            self.trade_memory[:remaining] = trade_data[to_insert:to_insert + remaining]
            self.memory_ptr += remaining

    def get_memory(self):
        return self.trade_memory.clone()


class TwoHeadV11NinethCompat(RecurrentActorCriticPolicy):
    """
    Policy compatível com arquitetura Nineth 4.75M.
    Baseada na arquitetura real detectada do checkpoint.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        v8_gru_hidden: int = 256,
        v8_features_dim: int = 256,
        v8_context_dim: int = 64,
        v8_memory_size: int = 512,
        critic_learning_rate: float = 4e-5,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        self.v8_gru_hidden = v8_gru_hidden
        self.v8_features_dim = v8_features_dim
        self.v8_context_dim = v8_context_dim
        self.v8_memory_size = v8_memory_size
        self.critic_learning_rate = critic_learning_rate

        # GRU ao invés de LSTM (detectado no checkpoint)
        gru_input_dim = self.features_dim
        self.v11_shared_gru = nn.GRU(gru_input_dim, v8_gru_hidden, batch_first=True)

        # Market context
        self.market_context = MarketContextEncoder(input_dim=v8_gru_hidden, context_dim=v8_context_dim)

        # Hybrid fusion (detectado no checkpoint)
        fusion_input_dim = v8_gru_hidden + v8_context_dim
        self.hybrid_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.01),
        )

        # Management head com pos1/pos2 separados
        self.management_head = DaytradeManagementHead(256)

        # Memory bank
        self.memory_bank = ElegantMemoryBank(memory_size=v8_memory_size, trade_dim=8)

        # Critic V8 com 6 camadas (0, 2, 4, 6, 8, 11)
        self.v8_critic = nn.Sequential(
            nn.Linear(256, 256),  # 0
            nn.LeakyReLU(negative_slope=0.01),  # 1
            nn.Linear(256, 256),  # 2
            nn.LeakyReLU(negative_slope=0.01),  # 3
            nn.Linear(256, 128),  # 4
            nn.LeakyReLU(negative_slope=0.01),  # 5
            nn.Linear(128, 64),   # 6
            nn.LeakyReLU(negative_slope=0.01),  # 7
            nn.Linear(64, 32),    # 8
            nn.LeakyReLU(negative_slope=0.01),  # 9
            nn.Dropout(0.1),      # 10
            nn.Linear(32, 16),    # 11
        )

        # Value head final (critic)
        self.critic = nn.Linear(16, 1)

        # Action distribution
        self.action_dist = DiagGaussianDistribution(self.action_space.shape[0])
        self.current_regime = torch.tensor(0)
        self.training_step = 0
        self.last_context_info = {}

        # Actor head (simples - usa hybrid_fusion output)
        # Dimensão de ação baseada no action_space
        self.actor_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, self.action_space.shape[0]),
        )

        self.log_std = nn.Parameter(torch.ones(self.action_space.shape[0]) * -0.5)
        nn.init.constant_(self.log_std, -0.5)

        self._setup_critic_optimizer()

    def forward_actor(self, obs: torch.Tensor, lstm_states=None, dones=None):
        features = obs
        if features.dim() == 2:
            features = features.unsqueeze(1)

        gru_output, _ = self.v11_shared_gru(features)
        context_features, regime_id, info = self.market_context(gru_output)

        combined_features = torch.cat([gru_output, context_features], dim=-1)
        fused_features = self.hybrid_fusion(combined_features)

        action_means = self.actor_head(fused_features)

        self.last_context_info = info
        self.current_regime = regime_id

        return action_means, lstm_states

    def forward_critic(self, obs: torch.Tensor, lstm_states=None, dones=None):
        features = obs
        if features.dim() == 2:
            features = features.unsqueeze(1)

        gru_output, _ = self.v11_shared_gru(features)
        context_features, regime_id, info = self.market_context(gru_output)

        combined_features = torch.cat([gru_output, context_features], dim=-1)
        fused_features = self.hybrid_fusion(combined_features)

        critic_features = self.v8_critic(fused_features)
        value = self.critic(critic_features)

        return value, lstm_states

    def forward(self, obs: torch.Tensor, lstm_states=None, episode_starts=None, deterministic=False):
        action_means, lstm_states = self.forward_actor(obs, lstm_states, episode_starts)

        log_std = self.log_std
        action_distribution = self.action_dist.proba_distribution(action_means, log_std)
        actions = action_distribution.mode() if deterministic else action_distribution.sample()
        log_prob = action_distribution.log_prob(actions)

        values, lstm_states = self.forward_critic(obs, lstm_states, episode_starts)
        return actions, values, log_prob, lstm_states

    def _setup_critic_optimizer(self):
        import torch.optim as optim

        critic_params = list(self.v8_critic.parameters()) + list(self.critic.parameters())

        if critic_params and self.critic_learning_rate is not None:
            self.critic_optimizer = optim.Adam(
                critic_params,
                lr=self.critic_learning_rate,
                weight_decay=1e-5,
            )
            print(f"✅ Critic optimizer created with LR: {self.critic_learning_rate}")
        else:
            print("⚠️ No critic parameters found or LR not set")


def get_v11_nineth_compat_kwargs():
    return {
        "v8_gru_hidden": 256,
        "v8_features_dim": 256,
        "v8_context_dim": 64,
        "v8_memory_size": 512,
        "features_extractor_class": TradingTransformerFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [256, 128],
        "lstm_hidden_size": 256,
        "n_lstm_layers": 1,
        "activation_fn": torch.nn.LeakyReLU,
        "ortho_init": True,
        "log_std_init": -0.5,
        "full_std": True,
        "use_expln": False,
        "squash_output": False,
    }
