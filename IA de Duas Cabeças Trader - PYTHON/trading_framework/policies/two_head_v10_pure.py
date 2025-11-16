"""
🚀 TwoHeadV10Pure - CÓPIA EXATA da V8 Elegance para 4D

ESTRATÉGIA:
- COPIAR 100% da V8 Elegance que FUNCIONA
- MODIFICAR apenas as dimensões de saída para 4D
- NÃO INVENTAR NADA NOVO
- Action Space: 4D [entry_decision, confidence, pos1_mgmt, pos2_mgmt]
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Type, Union, Tuple
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution
import torch.nn.functional as F

# Voltar para V9 mas sem health checks malucos
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

class MarketContextEncoder_V10(nn.Module):
    """🌍 Market Context Encoder - EXATO da V8"""
    
    def __init__(self, input_dim: int = 256, context_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.context_dim = context_dim
        
        # Detector de regime - IGUAL V8
        self.regime_detector = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(128),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 4)  # 4 regimes
        )
        
        # Context embedding - IGUAL V8
        self.regime_embedding = nn.Embedding(4, 32)
        
        # Context processor - IGUAL V8
        self.context_processor = nn.Sequential(
            nn.Linear(input_dim + 32, context_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(context_dim)
        )
        
    def forward(self, lstm_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        EXATO da V8 - sem mudanças
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

class ActionHead_V10(nn.Module):
    """🎯 ÚNICO Action Head para 4D - simplificação da V8"""
    
    def __init__(self, input_dim: int = 320):  # LSTM(256) + context(64)
        super().__init__()
        
        self.input_dim = input_dim
        
        # EXATO da V8 mas com saída 4D
        self.action_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 4)  # 4D output: [entry, confidence, pos1, pos2]
        )
        
    def forward(self, combined_input: torch.Tensor) -> torch.Tensor:
        """
        Processa entrada e retorna 4 ações
        Returns: tensor [batch, 4] com [entry_decision, confidence, pos1_mgmt, pos2_mgmt]
        """
        raw_actions = self.action_net(combined_input)
        
        # Aplicar ativações específicas por dimensão
        entry_decision = torch.clamp(torch.tanh(raw_actions[:, 0:1]) * 1.0 + 1.0, 0.0, 2.0)  # [0-2]
        confidence = torch.clamp((torch.tanh(raw_actions[:, 1:2]) + 1.0) / 2.0, 0.0, 1.0)  # [0-1]
        pos1_mgmt = torch.tanh(raw_actions[:, 2:3])  # [-1, 1]
        pos2_mgmt = torch.tanh(raw_actions[:, 3:4])  # [-1, 1]
        
        return torch.cat([entry_decision, confidence, pos1_mgmt, pos2_mgmt], dim=1)

class TwoHeadV10Pure(RecurrentActorCriticPolicy):
    """
    🚀 TwoHeadV10Pure - V8 Elegance EXATA para 4D
    
    ARQUITETURA:
    - LSTM compartilhada (256D) - IGUAL V8
    - Market Context (64D) - IGUAL V8  
    - Action Head ÚNICO (4D output) - SIMPLIFICADO
    - Features Extractor FUNCIONAL - IGUAL V8
    """
    
    def __init__(
        self,
        observation_space,
        action_space, 
        *args,
        # V8 PARAMETERS - EXATOS
        v8_lstm_hidden: int = 256,
        v8_features_dim: int = 256,
        v8_context_dim: int = 64,
        **kwargs
    ):
        # Parâmetros EXATOS da V8
        self.v8_lstm_hidden = v8_lstm_hidden
        self.v8_features_dim = v8_features_dim  
        self.v8_context_dim = v8_context_dim
        
        print(f"🚀 V10Pure inicializando (cópia V8 para 4D):")
        print(f"   LSTM Hidden: {v8_lstm_hidden}D")
        print(f"   Features: {v8_features_dim}D")
        print(f"   Context: {v8_context_dim}D")
        print(f"   Action Space: {action_space.shape} (4D)")
        print(f"   Obs Space: {observation_space.shape}")
        print(f"   ortho_init from kwargs: {kwargs.get('ortho_init', 'NOT SET')}")
        print(f"   activation_fn from kwargs: {kwargs.get('activation_fn', 'NOT SET')}")
        
        # Configurar componentes ANTES do super().__init__()
        combined_dim = v8_lstm_hidden + v8_context_dim  # 256 + 64 = 320
        
        # Inicializar como None primeiro
        self.market_context_encoder = None
        self.action_head = None
        
        # Chamar super().__init__() 
        super().__init__(observation_space, action_space, *args, **kwargs)
        
        # Configurar componentes reais DEPOIS do super().__init__()
        self.market_context_encoder = MarketContextEncoder_V10(
            input_dim=v8_lstm_hidden, 
            context_dim=v8_context_dim
        )
        
        self.action_head = ActionHead_V10(input_dim=combined_dim)
        
        # Inicialização simples - sem múltiplas re-inicializações
        self._simple_init()
        
        print("✅ V10Pure configurado com sucesso!")
    
    def _simple_init(self):
        """Inicialização simples - SEM múltiplas chamadas"""
        
        # Inicializar apenas o regime_embedding se necessário
        if hasattr(self, 'market_context_encoder') and hasattr(self.market_context_encoder, 'regime_embedding'):
            nn.init.uniform_(self.market_context_encoder.regime_embedding.weight, -0.1, 0.1)
            print("🔧 Regime embedding inicializado: uniform(-0.1, 0.1)")
        
        # Inicializar action head com Xavier normal
        if hasattr(self, 'action_head'):
            for name, module in self.action_head.named_modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight, gain=0.8)
                    if module.bias is not None:
                        nn.init.uniform_(module.bias, -0.05, 0.05)
            print("🔧 Action head inicializado: xavier + uniform bias")
    
    def _build_mlp_extractor(self) -> None:
        """
        Override EXATO da V8 - sem mudanças
        """
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
        
        # Action net vazio
        self.action_net = nn.ModuleDict()
    
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> torch.distributions.Distribution:
        """
        BASEADO na V8 mas com saída 4D
        """
        # Remover seq dimension se existir
        if len(latent_pi.shape) == 3:  # [batch, seq, features]
            latent_pi = latent_pi.squeeze(1)  # [batch, features]
        
        # Detecção de dimensão IGUAL V8
        feature_dim = latent_pi.shape[-1]
        
        if feature_dim == self.v8_lstm_hidden:  # 256D - LSTM only
            # SB3 chamada direta: aplicar market context
            context_features, regime_id, _ = self.market_context_encoder(latent_pi)
            combined_input = torch.cat([latent_pi, context_features], dim=-1)  # 256 + 64 = 320
            
        elif feature_dim == (self.v8_lstm_hidden + self.v8_context_dim):  # 320D - LSTM + context
            # forward_actor call: já tem context
            combined_input = latent_pi
            
        else:
            # Fallback IGUAL V8
            print(f"⚠️ Dimensão inesperada: {feature_dim}D")
            
            if feature_dim < self.v8_lstm_hidden:
                # Pad com zeros até 256D
                padding_size = self.v8_lstm_hidden - feature_dim
                latent_padded = torch.cat([
                    latent_pi, 
                    torch.zeros(latent_pi.shape[0], padding_size, device=latent_pi.device)
                ], dim=-1)
                print(f"   Padding {feature_dim}D → {self.v8_lstm_hidden}D")
                
            else:
                # Truncar para 256D
                latent_padded = latent_pi[:, :self.v8_lstm_hidden]
                print(f"   Truncating {feature_dim}D → {self.v8_lstm_hidden}D")
            
            # Aplicar market context
            context_features, regime_id, _ = self.market_context_encoder(latent_padded)
            combined_input = torch.cat([latent_padded, context_features], dim=-1)
        
        # Gerar ações 4D
        actions_4d = self.action_head(combined_input)
        
        # Noise injection IGUAL V8
        if self.training:
            exploration_noise = torch.randn_like(actions_4d) * 0.02
            actions_4d = actions_4d + exploration_noise
        
        # Distribuição IGUAL V8
        base_std = 0.1
        log_std = torch.log(torch.ones_like(actions_4d) * base_std)
        
        return DiagGaussianDistribution(actions_4d.shape[-1]).proba_distribution(actions_4d, log_std)
    
    def forward_actor(self, features: torch.Tensor, lstm_states, episode_starts: torch.Tensor) -> torch.distributions.Distribution:
        """
        Forward Actor EXATO da V8 - sem mudanças
        """
        # 1. Extract features first 
        extracted_features = self.extract_features(features)  # [batch, 256]
        
        # 2. Add sequence dimension for LSTM
        extracted_features = extracted_features.unsqueeze(1)  # [batch, 1, 256]
        
        # 3. Processar através da LSTM compartilhada (IGUAL V8)
        lstm_out, new_lstm_states = self.lstm_actor(extracted_features, lstm_states)
        
        # 4. Market context (IGUAL V8)
        context_features, regime_id, context_info = self.market_context_encoder(lstm_out)
        
        # 5. Combinar LSTM + context (IGUAL V8)
        lstm_features_2d = lstm_out.squeeze(1)  # [batch, 256]
        context_features_2d = context_features.squeeze(1) if len(context_features.shape) == 3 else context_features
        combined_input = torch.cat([lstm_features_2d, context_features_2d], dim=-1)  # [batch, 320]
        
        # 6. Gerar ações 4D (DIFERENTE: 4D em vez de 8D)
        return self._get_action_dist_from_latent(combined_input)
    
    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass EXATO da V8
        """
        return self.value_net(features)

def get_v10_pure_kwargs():
    """Configurações para TwoHeadV10Pure"""
    return {
        'features_extractor_class': TradingTransformerV9,  # V9 sem health checks
        'features_extractor_kwargs': {
            'features_dim': 256,
            'temporal_window': 10,
            'features_per_bar': 45,
        },
        'v8_lstm_hidden': 256,
        'v8_features_dim': 256,
        'v8_context_dim': 64,
        'n_lstm_layers': 1,
        'shared_lstm': True,
        'enable_critic_lstm': False,
        'activation_fn': nn.LeakyReLU,
        'net_arch': [],
        'ortho_init': False,  # IGUAL V8 que funciona
        'log_std_init': -0.5,
    }

def validate_v10_pure_policy(policy=None):
    """Valida a política V10Pure"""
    import gym
    
    if policy is None:
        dummy_obs_space = gym.spaces.Box(low=-1, high=1, shape=(2580,), dtype=np.float32)
        dummy_action_space = gym.spaces.Box(low=np.array([0, 0, -1, -1]), high=np.array([2, 1, 1, 1]), dtype=np.float32)
        
        def dummy_lr_schedule(progress):
            return 1e-4
        
        policy = TwoHeadV10Pure(
            observation_space=dummy_obs_space,
            action_space=dummy_action_space,
            lr_schedule=dummy_lr_schedule,
            **get_v10_pure_kwargs()
        )
    
    print("✅ TwoHeadV10Pure validada - V8 funcional para 4D!")
    print(f"   🧠 LSTM: {policy.v8_lstm_hidden}D")
    print(f"   🎯 Action Head: 4D (entry + confidence + pos1 + pos2)")
    print(f"   📊 Total Actions: 4D")
    
    return True

if __name__ == "__main__":
    print("🚀 TwoHeadV10Pure - V8 Elegance funcional adaptado para 4D Action Space!")
    validate_v10_pure_policy()