"""
🚀 SAC TwoHead Policy - Política customizada para SAC
Adaptada da TwoHeadPolicy do mainppo1.py para funcionar com SAC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.sac.policies import Actor, ContinuousCritic
from torch.cuda.amp import autocast
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import gym

# 🔥 CONFIGURAÇÃO AMP (AUTOMATIC MIXED PRECISION)
ENABLE_AMP = torch.cuda.is_available()

class TwoHeadActor(Actor):
    """
    Actor customizado para SAC com arquitetura TwoHead
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            clip_mean,
            normalize_images,
        )
        
        # 🔥 ARQUITETURA TWOHEAD CUSTOMIZADA
        self.features_dim = features_dim
        
        # Encoder de mercado
        self.market_encoder = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Transformer para posições
        self.position_transformer = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Combinar features
        self.feature_combiner = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Heads de ação customizados
        action_dim = action_space.shape[0]  # 10 ações
        
        # Mean e log_std para distribuição contínua
        self.mu = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)
        
        print(f"[SAC TwoHead] Actor criado: features_dim={features_dim}, action_dim={action_dim}")

    def get_action_dist_params(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the parameters for the action distribution.
        """
        with autocast(enabled=ENABLE_AMP):
            # Extrair features
            features = self.extract_features(obs)
            
            # Processar com arquitetura TwoHead
            market_features = self.market_encoder(features)
            position_features = self.position_transformer(features)
            
            # Combinar features
            combined = torch.cat([market_features, position_features], dim=1)
            latent = self.feature_combiner(combined)
            
            # Gerar mean e log_std
            mean = self.mu(latent)
            log_std = self.log_std(latent)
            
            # Clamp log_std
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            
            return mean, log_std

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean, log_std, deterministic=deterministic, reparameterize=True)

    def action_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean, log_std)

class TwoHeadCritic(ContinuousCritic):
    """
    Critic customizado para SAC com arquitetura TwoHead
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            normalize_images,
            n_critics,
            share_features_extractor,
        )
        
        # 🔥 ARQUITETURA TWOHEAD CUSTOMIZADA PARA CRITIC
        self.features_dim = features_dim
        action_dim = action_space.shape[0]
        
        # Criar critics customizados
        self.q_networks = nn.ModuleList()
        for idx in range(n_critics):
            q_net = nn.Sequential(
                nn.Linear(features_dim + action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            self.q_networks.append(q_net)
        
        print(f"[SAC TwoHead] Critic criado: features_dim={features_dim}, action_dim={action_dim}, n_critics={n_critics}")

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        with autocast(enabled=ENABLE_AMP):
            # Extrair features
            features = self.extract_features(obs)
            
            # Concatenar obs features com actions
            qvalue_input = torch.cat([features, actions], dim=1)
            
            # Calcular Q-values para cada critic
            q_values = []
            for q_net in self.q_networks:
                q_val = q_net(qvalue_input)
                q_values.append(q_val)
            
            return tuple(q_values)

class SACTwoHeadPolicy(BasePolicy):
    """
    Policy class for SAC with TwoHead architecture
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: callable,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        if net_arch is None:
            net_arch = [256, 256]

        actor_arch, critic_arch = net_arch, net_arch
        
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update({"n_critics": n_critics, "net_arch": critic_arch, "share_features_extractor": share_features_extractor})

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: callable) -> None:
        # Create actor and target
        self.actor = self.make_actor()
        self.actor_target = self.make_actor()
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Create critic and target  
        self.critic = self.make_critic()
        self.critic_target = self.make_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Target networks should always be in eval mode
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)
        
        # 🔥 INICIALIZAR OTIMIZADORES
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> TwoHeadActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return TwoHeadActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> TwoHeadCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return TwoHeadCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode 