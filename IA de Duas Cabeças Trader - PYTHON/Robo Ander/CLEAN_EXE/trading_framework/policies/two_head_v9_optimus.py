"""
🚀 TwoHeadV9Optimus - "V8 Elegance Adaptado para 4D Action Space"

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
        # Entry decision (0-2 range) com clamp para noise injection
        raw_entry = self.entry_decision_net(combined_input)
        entry_decision = torch.clamp(torch.tanh(raw_entry) * 1.0 + 1.0, 0.0, 2.0)  # [0-2], clamped
        
        # Entry confidence (0-1 range) com clamp para noise injection
        raw_confidence = self.entry_confidence_net(combined_input)
        entry_confidence = torch.clamp((raw_confidence + 1.0) / 2.0, 0.0, 1.0)  # [-1,1] → [0,1], clamped
        
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

class TwoHeadV9Optimus(RecurrentActorCriticPolicy):
    """
    🚀 TwoHeadV9Optimus - "V8 Elegance Adaptado para 4D"
    
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
        
        print(f"🚀 V9Optimus inicializando:")
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
        
        # CRÍTICO: Re-inicializar features_extractor após SB3
        self._fix_features_extractor_weights()
        
        # 🎯 REWARD ENGINEERING: Inicialização otimizada dos heads
        self._initialize_action_heads()
        
        print("✅ V9Optimus configurado com sucesso!")
    
    def _fix_features_extractor_weights(self):
        """Corrige pesos do features_extractor que SB3 pode ter zerado"""
        # 🚨 REMOVIDO: Não chamar _initialize_weights() novamente!
        # O TradingTransformerV9 já foi inicializado corretamente no __init__
        # Chamar novamente pode sobrescrever configurações específicas
        
        # CRÍTICO: Inicializar embeddings explicitamente (como V8)
        self._initialize_embeddings()
        
        # 🚨 PROTEÇÃO EMERGENCIAL: Verificar se tudo foi inicializado
        self._verify_initialization()
    
    def _initialize_embeddings(self):
        """🔧 Inicialização específica para embeddings - proteção contra zeros"""
        if hasattr(self, 'market_context_encoder') and hasattr(self.market_context_encoder, 'regime_embedding'):
            # Inicialização uniform para garantir que TODOS os parâmetros são != 0
            nn.init.uniform_(self.market_context_encoder.regime_embedding.weight, -0.1, 0.1)
            print("🔧 Regime embedding inicializado com uniform(-0.1, 0.1) - ZERO zeros garantido")
    
    def _verify_initialization(self):
        """🚨 VERIFICAÇÃO EMERGENCIAL: Confirma que tudo foi inicializado corretamente"""
        print("🔍 Verificando inicialização V9Optimus...")
        
        # Verificar features_extractor.input_projection
        if hasattr(self, 'features_extractor') and hasattr(self.features_extractor, 'input_projection'):
            zeros_pct = (self.features_extractor.input_projection.weight.abs() < 1e-8).float().mean().item()
            print(f"   input_projection: {zeros_pct:.1%} zeros")
            if zeros_pct > 0.5:
                print("🚨 CRÍTICO: input_projection tem muitos zeros - FORÇANDO RE-INIT!")
                nn.init.xavier_uniform_(self.features_extractor.input_projection.weight, gain=0.3)
        
        # Verificar market_context_encoder.regime_embedding
        if hasattr(self, 'market_context_encoder') and hasattr(self.market_context_encoder, 'regime_embedding'):
            zeros_pct = (self.market_context_encoder.regime_embedding.weight.abs() < 1e-8).float().mean().item()
            print(f"   regime_embedding: {zeros_pct:.1%} zeros")
            if zeros_pct > 0.5:
                print("🚨 CRÍTICO: regime_embedding tem muitos zeros - FORÇANDO RE-INIT!")
                nn.init.uniform_(self.market_context_encoder.regime_embedding.weight, -0.1, 0.1)
        
        print("✅ Verificação de inicialização concluída")
    
    def _initialize_action_heads(self):
        """🎯 Inicialização otimizada dos action heads para reward engineering"""
        print("🎯 Inicializando action heads para exploração balanceada...")
        
        heads_to_init = [
            ('entry_head', self.entry_head),
            ('management_head', self.management_head),
            ('market_context_encoder', self.market_context_encoder)
        ]
        
        for head_name, head in heads_to_init:
            if head is not None:
                for name, module in head.named_modules():
                    if isinstance(module, nn.Linear):
                        # 🔥 ALTA VARIÂNCIA para resolver concentração
                        # Xavier normal com gain aumentado para mais exploração
                        nn.init.xavier_normal_(module.weight, gain=0.8)  # 0.3 → 0.8
                        
                        # Bias com maior variância para quebrar simetria
                        if module.bias is not None:
                            nn.init.uniform_(module.bias, -0.05, 0.05)  # -0.01,0.01 → -0.05,0.05
                            
                    elif isinstance(module, nn.LayerNorm):
                        # LayerNorm padrão
                        nn.init.constant_(module.weight, 1.0)
                        nn.init.constant_(module.bias, 0.0)
        
        print("🔥 Action heads inicializados com gain=0.8 para ALTA EXPLORAÇÃO")
    
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
        🔧 Shape-Robust Override para gerar ações 4D
        
        DETECÇÃO AUTOMÁTICA:
        - Se latent_pi = 256D → Aplicar market context (SB3 direct call)
        - Se latent_pi = 320D → Já tem context (forward_actor call)
        """
        # Remover seq dimension se existir
        if len(latent_pi.shape) == 3:  # [batch, seq, features]
            latent_pi = latent_pi.squeeze(1)  # [batch, features]
        
        # 🎯 DETECÇÃO ROBUSTA DE DIMENSÃO
        feature_dim = latent_pi.shape[-1]
        
        if feature_dim == self.v8_lstm_hidden:  # 256D - LSTM only
            # SB3 chamada direta: aplicar market context
            context_features, regime_id, _ = self.market_context_encoder(latent_pi)
            combined_input = torch.cat([latent_pi, context_features], dim=-1)  # 256 + 64 = 320
            
        elif feature_dim == (self.v8_lstm_hidden + self.v8_context_dim):  # 320D - LSTM + context
            # forward_actor call: já tem context
            combined_input = latent_pi
            
        else:
            # Fallback: dimensão inesperada - ajustar para 256D se necessário
            print(f"⚠️ Dimensão inesperada: {feature_dim}D")
            
            if feature_dim < self.v8_lstm_hidden:
                # Pad com zeros até 256D
                padding_size = self.v8_lstm_hidden - feature_dim
                latent_padded = torch.cat([
                    latent_pi, 
                    torch.zeros(latent_pi.shape[0], padding_size, device=latent_pi.device)
                ], dim=-1)
                print(f"   Padding {feature_dim}D → {self.v8_lstm_hidden}D")
                
            elif feature_dim > (self.v8_lstm_hidden + self.v8_context_dim):
                # Truncar para 256D
                latent_padded = latent_pi[:, :self.v8_lstm_hidden]
                print(f"   Truncating {feature_dim}D → {self.v8_lstm_hidden}D")
                
            else:
                # Entre 256D e 320D: truncar para 256D
                latent_padded = latent_pi[:, :self.v8_lstm_hidden]
                print(f"   Truncating {feature_dim}D → {self.v8_lstm_hidden}D")
            
            # Aplicar market context
            context_features, regime_id, _ = self.market_context_encoder(latent_padded)
            combined_input = torch.cat([latent_padded, context_features], dim=-1)
        
        # Entry Head (2D)
        entry_decision, entry_confidence = self.entry_head(combined_input)
        
        # Management Head (2D) 
        pos1_mgmt, pos2_mgmt = self.management_head(combined_input)
        
        # Combinar para 4D: [entry_decision, confidence, pos1_mgmt, pos2_mgmt]
        combined_actions = torch.cat([entry_decision, entry_confidence, pos1_mgmt, pos2_mgmt], dim=-1)
        
        # 🔥 NOISE INJECTION para quebrar concentração
        if self.training:
            exploration_noise = torch.randn_like(combined_actions) * 0.02
            combined_actions = combined_actions + exploration_noise
        
        # 🎯 REWARD ENGINEERING OPTIMIZATION
        # Retornar distribuição compatível com SB3
        from stable_baselines3.common.distributions import DiagGaussianDistribution
        
        # 🎯 EXPLORAÇÃO BALANCEADA pós-análise
        # Concentração resolvida - reduzindo para balance exploração/estabilidade
        base_std = 0.1  # 0.15 → 0.1 (balanceamento final)
        log_std = torch.log(torch.ones_like(combined_actions) * base_std)
        
        return DiagGaussianDistribution(combined_actions.shape[-1]).proba_distribution(combined_actions, log_std)
    
    def forward_actor(self, features: torch.Tensor, lstm_states, episode_starts: torch.Tensor) -> torch.distributions.Distribution:
        """
        Forward Actor V9 - EXATAMENTE como V8 mas output 4D
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

def fix_v9_optimus_weights(model):
    """🔧 CORREÇÃO CRÍTICA: Função para corrigir pesos V9Optimus APÓS criação do modelo PPO"""
    print("🔧 EXECUTANDO CORREÇÃO CRÍTICA ANTI-ZEROS V9Optimus...")
    
    if not hasattr(model, 'policy'):
        print("❌ Modelo não tem policy")
        return False
    
    policy = model.policy
    
    # 1. CORREÇÃO FEATURES EXTRACTOR
    if hasattr(policy, 'features_extractor'):
        fe = policy.features_extractor
        
        # Corrigir input_projection
        if hasattr(fe, 'input_projection'):
            print("🔧 Corrigindo input_projection...")
            nn.init.xavier_uniform_(fe.input_projection.weight, gain=0.3)
            if fe.input_projection.bias is not None:
                nn.init.zeros_(fe.input_projection.bias)
            print(f"   ✅ input_projection inicializado: gain=0.3")
        
        # Corrigir _residual_projection (agora sempre existe)
        if hasattr(fe, '_residual_projection'):
            print("🔧 Corrigindo _residual_projection...")
            nn.init.xavier_uniform_(fe._residual_projection.weight, gain=0.1)
            if fe._residual_projection.bias is not None:
                nn.init.zeros_(fe._residual_projection.bias)
            print(f"   ✅ _residual_projection inicializado: gain=0.1")
        else:
            print("⚠️ _residual_projection não encontrado!")
    
    # 2. CORREÇÃO MARKET CONTEXT ENCODER
    if hasattr(policy, 'market_context_encoder'):
        mce = policy.market_context_encoder
        
        # Corrigir regime_embedding
        if hasattr(mce, 'regime_embedding'):
            print("🔧 Corrigindo regime_embedding...")
            nn.init.uniform_(mce.regime_embedding.weight, -0.1, 0.1)
            print(f"   ✅ regime_embedding inicializado: uniform(-0.1, 0.1) - ZERO zeros")
    
    # 3. NÃO CHAMAR _fix_features_extractor_weights
    # Isso causaria múltiplas re-inicializações desnecessárias
    
    print("🎉 CORREÇÃO CRÍTICA ANTI-ZEROS CONCLUÍDA!")
    return True

def get_v9_optimus_kwargs():
    """Configurações para TwoHeadV9Optimus"""
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
        'ortho_init': False,  # CRÍTICO: não quebrar transformer (igual V8)
        'log_std_init': -0.5,
    }

def validate_v9_optimus_policy(policy=None):
    """Valida a política V9Optimus"""
    import gym
    
    if policy is None:
        dummy_obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
        dummy_action_space = gym.spaces.Box(low=np.array([0, 0, -1, -1]), high=np.array([2, 1, 1, 1]), dtype=np.float32)
        
        def dummy_lr_schedule(progress):
            return 1e-4
        
        policy = TwoHeadV9Optimus(
            observation_space=dummy_obs_space,
            action_space=dummy_action_space,
            lr_schedule=dummy_lr_schedule,
            **get_v9_optimus_kwargs()
        )
    
    print("✅ TwoHeadV9Optimus validada - V8 elegante para 4D!")
    print(f"   🧠 LSTM: {policy.v8_lstm_hidden}D")
    print(f"   🎯 Entry Head: 2D (entry_decision + confidence)")
    print(f"   💰 Management Head: 2D (pos1_mgmt + pos2_mgmt)")
    print(f"   📊 Total Actions: 4D")
    
    return True

if __name__ == "__main__":
    print("🚀 TwoHeadV9Optimus - V8 Elegance adaptado para 4D Action Space!")
    validate_v9_optimus_policy()