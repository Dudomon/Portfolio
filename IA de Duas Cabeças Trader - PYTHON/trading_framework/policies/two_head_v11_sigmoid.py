"""
🚀 TwoHeadV8Elegance - "Simplicidade Focada no Daytrade"

FILOSOFIA V8:
- UMA LSTM compartilhada (elegância)
- Entry Head e Management Head ESPECÍFICOS (especialização)
- 4D action space (otimização para trading eficiente)
- Simplicidade sem perder funcionalidade

ARQUITETURA ELEGANTE:
- Single LSTM Backbone (256D)
- Entry Head específico (entry + confidence)
- Management Head específico (SL/TP por posição)
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

# Importar base funcional
from trading_framework.extractors.transformer_extractor import TradingTransformerFeatureExtractor

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
    """🌍 Market Context Encoder - ÚNICO detector de regime + RAW FEATURES BYPASS"""

    def __init__(self, input_dim: int = 256, context_dim: int = 64):
        super().__init__()

        self.input_dim = input_dim
        self.context_dim = context_dim

        # Detector de regime (Bull/Bear/Sideways/Volatile)
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

        # ⚡ RAW FEATURES PROCESSOR - Bypass direto para features críticas
        # Input: [volume_momentum, price_position, breakout_strength,
        #         trend_consistency, support_resistance, volatility_regime, market_structure]
        self.raw_features_processor = nn.Sequential(
            nn.Linear(7, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(32)
        )

        # Context processor (LSTM + Regime + Raw Features)
        self.context_processor = nn.Sequential(
            nn.Linear(input_dim + 32 + 32, context_dim),  # 256+32+32 -> 64
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(context_dim)
        )
        
    def forward(self, lstm_features: torch.Tensor, raw_features: torch.Tensor = None) -> Tuple[torch.Tensor, int, Dict]:
        """
        Processa features do LSTM e retorna contexto de mercado

        Args:
            lstm_features: Output do LSTM [batch, seq, 256]
            raw_features: Features brutas [batch, 7] extraídas diretamente da obs (OPCIONAL)

        Returns: (context_features, regime_id, info)
        """
        # Detectar regime
        regime_logits = self.regime_detector(lstm_features)

        # FIX: Handle batch dimension properly for regime detection - NO .item() CALLS!
        if len(regime_logits.shape) == 3:  # batch, seq, classes
            # Use last timestep for regime detection
            regime_logits_last = regime_logits[:, -1, :]  # batch, classes
            regime_id_tensor = torch.argmax(regime_logits_last[0], dim=-1)  # Keep tensor
        elif len(regime_logits.shape) == 2:  # batch, classes
            regime_id_tensor = torch.argmax(regime_logits[0], dim=-1)  # Keep tensor
        else:
            regime_id_tensor = torch.argmax(regime_logits, dim=-1)  # Keep tensor

        # Embedding do regime
        regime_emb = self.regime_embedding(regime_id_tensor)
        if len(lstm_features.shape) == 3:  # batch, seq, features
            batch_size, seq_len = lstm_features.shape[:2]
            regime_emb = regime_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, seq_len, -1)
        else:  # batch, features
            batch_size = lstm_features.shape[0]
            regime_emb = regime_emb.unsqueeze(0).expand(batch_size, -1)

        # ⚡ PROCESSAR RAW FEATURES (bypass do Transformer+LSTM)
        if raw_features is not None:
            raw_emb = self.raw_features_processor(raw_features)
            if len(lstm_features.shape) == 3:
                raw_emb = raw_emb.unsqueeze(1).expand(batch_size, seq_len, -1)

            # Combinar: LSTM (padrões temporais) + Regime + Raw (features específicas)
            combined = torch.cat([lstm_features, regime_emb, raw_emb], dim=-1)
        else:
            # Modo legado sem raw features
            combined = torch.cat([lstm_features, regime_emb], dim=-1)

        context_features = self.context_processor(combined)

        # Info para debug - SKIP FOR PERFORMANCE!
        info = {
            'regime_id': regime_id_tensor,  # Keep as tensor
        }  # Skip confidence and name calculation for performance

        return context_features, regime_id_tensor, info

class DaytradeEntryHead(nn.Module):
    """🎯 Entry Head ESPECÍFICO - Foca apenas em entry decision + confidence"""
    
    def __init__(self, input_dim: int = 320):  # LSTM(256) + context(64)
        super().__init__()
        
        self.input_dim = input_dim
        
        # Entry Decision Network (discrete: hold/long/short)
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
        
        # Entry Confidence Network (continuous: 0-1)
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
            nn.SiLU()  # SiLU for non-saturating output
        )
        
        # Quality Analysis Network (for gate_info compatibility)
        self.quality_analyzer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 6)  # 6 quality scores
        )
        
        # Inicialização adequada
        self._initialize_networks()
        
    def _initialize_networks(self):
        """🔧 Inicialização adequada das redes"""
        for module in [self.entry_decision_net, self.entry_confidence_net, self.quality_analyzer]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        # print("   ✅ DaytradeEntryHead inicializado com Xavier uniform")  # Commented for performance
    
    def forward(self, lstm_features: torch.Tensor, market_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward específico do Entry Head
        Returns: (entry_decision, entry_confidence, gate_info)
        """
        # Combinar LSTM features + market context
        combined_input = torch.cat([lstm_features, market_context], dim=-1)
        
        # Entry decision (raw logit)
        raw_entry = self.entry_decision_net(combined_input)
        
        # Entry confidence (0-1 range)
        raw_confidence = self.entry_confidence_net(combined_input)
        entry_confidence = (raw_confidence + 1.0) / 2.0  # [-1,1] → [0,1]
        
        # Quality analysis para compatibilidade (DISABLED FOR PERFORMANCE)
        # quality_scores = self.quality_analyzer(combined_input)
        
        # Gate info para compatibilidade com V7s (SIMPLIFIED)
        gate_info = {
            'entry_head_type': 'daytrade_specific',
            'raw_entry_logit': raw_entry,  # Keep on GPU as tensor for performance
            'confidence_score': entry_confidence,  # Keep on GPU as tensor for performance
            # 'quality_scores': disabled for performance
        }
        
        return raw_entry, entry_confidence, gate_info

class DaytradeManagementHead(nn.Module):
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
            nn.SiLU()  # SiLU for non-saturating output
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
            nn.SiLU()  # SiLU for non-saturating output
        )
        
    def forward(self, lstm_features: torch.Tensor, market_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processa entrada e retorna management para 2 posições
        Returns: (pos1_mgmt, pos2_mgmt)
        """
        # Combinar LSTM features + market context  
        combined_input = torch.cat([lstm_features, market_context], dim=-1)
        
        pos1_mgmt = self.pos1_mgmt_net(combined_input)
        pos2_mgmt = self.pos2_mgmt_net(combined_input)
        
        return pos1_mgmt, pos2_mgmt

class ElegantMemoryBank(nn.Module):
    """💾 Memory Bank Elegante - 512 trades, processamento direto"""
    
    def __init__(self, memory_size: int = 512, trade_dim: int = 8):
        super().__init__()
        
        self.memory_size = memory_size
        self.trade_dim = trade_dim
        
        # Banco de memória (menor que V7s)
        self.register_buffer('trade_memory', torch.zeros(memory_size, trade_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
        # Processador de memória (mais simples)
        self.memory_processor = nn.Sequential(
            nn.Linear(trade_dim, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(32),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(16, 8)
        )
        
    def add_trade(self, trade_data: torch.Tensor):
        """Adiciona trade à memória"""
        if not isinstance(trade_data, torch.Tensor):
            trade_data = torch.tensor(trade_data, dtype=torch.float32, device=self.trade_memory.device)
        
        ptr = int(self.memory_ptr.item())
        self.trade_memory[ptr] = trade_data[:self.trade_dim]  # Garantir dimensão correta
        self.memory_ptr[0] = (ptr + 1) % self.memory_size
    
    def get_memory_context(self, batch_size: int) -> torch.Tensor:
        """Retorna contexto processado da memória"""
        # Pegar amostra das últimas memórias
        recent_count = min(50, self.memory_size)
        recent_memories = self.trade_memory[-recent_count:]
        
        # Se memória vazia, gerar contexto base
        if recent_memories.sum() == 0:
            base_context = torch.randn(self.trade_dim, device=self.trade_memory.device) * 0.05
            processed = self.memory_processor(base_context)
            return processed.unsqueeze(0).expand(batch_size, -1)
        
        # Processar memórias recentes
        avg_memory = recent_memories.mean(dim=0)
        
        # 🔧 FIX 2: Ruído proporcional à magnitude dos dados
        memory_magnitude = torch.std(avg_memory) + 1e-8  # Evitar divisão por zero
        noise_scale = memory_magnitude * 0.02  # 2% da variação como ruído
        noisy_memory = avg_memory + torch.randn_like(avg_memory) * noise_scale
        
        processed = self.memory_processor(noisy_memory)
        return processed.unsqueeze(0).expand(batch_size, -1)

class TwoHeadV11Sigmoid(RecurrentActorCriticPolicy):
    """
    🚀 TwoHeadV8Elegance - "Simplicidade Focada no Daytrade"
    
    ARQUITETURA ELEGANTE:
    - UMA LSTM compartilhada (256D) - elegância
    - Entry Head específico (entry + confidence) - especialização
    - Management Head específico (SL/TP por posição) - especialização  
    - Memory Bank simplificado (512 trades) - eficiência
    - Market Context único (4 regimes) - simplicidade
    - 8D action space mantido - funcionalidade completa
    """
    
    def __init__(
        self,
        observation_space,
        action_space, 
        *args,
        # V8 ELEGANCE PARAMETERS
        v8_lstm_hidden: int = 256,
        v8_features_dim: int = 256,
        v8_context_dim: int = 64,
        v8_memory_size: int = 512,
        **kwargs
    ):
        
        # Performance optimized - prints commented out during training
        # print("🚀 TwoHeadV8Elegance inicializando - SIMPLICIDADE FOCADA!")
        # print(f"   Action Space: {action_space}")
        # print(f"   Expected Actions: 8D")
        # print(f"   LSTM Hidden: {v8_lstm_hidden}")
        # print(f"   Features Dim: {v8_features_dim}")
        # print(f"   Context Dim: {v8_context_dim}")
        # print(f"   Memory Size: {v8_memory_size}")
        
        # Validate action space
        if not hasattr(action_space, 'shape') or action_space.shape[0] != 4:
            raise ValueError(f"V8Elegance expects 4-dimensional action space, got {getattr(action_space, 'shape', 'Unknown')}")
        
        # Garantir features_extractor_class
        if 'features_extractor_class' not in kwargs:
            kwargs['features_extractor_class'] = TradingTransformerFeatureExtractor
        if 'features_extractor_kwargs' not in kwargs:
            kwargs['features_extractor_kwargs'] = {'features_dim': v8_features_dim}
        
        # Store parameters
        self.v8_lstm_hidden = v8_lstm_hidden
        self.v8_features_dim = v8_features_dim
        self.v8_context_dim = v8_context_dim
        self.v8_memory_size = v8_memory_size
        
        # 🚨 CRITIC FIX: Store critic learning rate from kwargs
        self.critic_learning_rate = kwargs.pop('critic_learning_rate', None)
        
        # Disable automatic LSTM critic
        kwargs['enable_critic_lstm'] = False
        
        # Initialize parent
        super().__init__(observation_space, action_space, *args, **kwargs)
        
        
        # 🚀 V11 HYBRID COMPONENTS - LSTM + GRU
        
        # 1. LSTM compartilhada (memória de longo prazo)
        self.v8_shared_lstm = nn.LSTM(
            input_size=self.v8_features_dim,
            hidden_size=self.v8_lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        
        # 1.5. GRU paralela (padrões recentes e reatividade)
        self.v11_shared_gru = nn.GRU(
            input_size=self.v8_features_dim,
            hidden_size=self.v8_lstm_hidden,  # Mesmo tamanho para fusão
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        
        # 1.6. Sistema de fusão LSTM+GRU
        self.hybrid_fusion = nn.Sequential(
            nn.Linear(self.v8_lstm_hidden * 2, self.v8_lstm_hidden),  # 512->256
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(self.v8_lstm_hidden),
            nn.Dropout(0.05)
        )
        
        # 2. Market Context Encoder (único)
        self.market_context = MarketContextEncoder(
            input_dim=self.v8_lstm_hidden,
            context_dim=self.v8_context_dim
        )
        
        # 3. Entry Head específico
        self.entry_head = DaytradeEntryHead(
            input_dim=self.v8_lstm_hidden + self.v8_context_dim
        )
        
        # 4. Management Head específico
        self.management_head = DaytradeManagementHead(
            input_dim=self.v8_lstm_hidden + self.v8_context_dim
        )
        
        # 5. Memory Bank elegante
        self.memory_bank = ElegantMemoryBank(
            memory_size=self.v8_memory_size,
            trade_dim=8
        )
        
        
        # 7. Critic direto (usa LSTM features) - 🚨 ENHANCED para prevenir overfitting
        self.v8_critic = nn.Sequential(
            nn.Linear(self.v8_lstm_hidden + self.v8_context_dim, 256),  # Expanded capacity
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(256),
            nn.Dropout(0.2),  # Increased dropout
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(128),
            nn.Dropout(0.2),  # More regularization
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        # 8. Estado interno
        self.current_regime = 2  # Start sideways
        self.training_step = 0
        self.last_context_info = {}
        
        # 9. Performance cache para market context
        self._cached_context_features = None
        self._cached_context_step = -1
        
        # 9. Inicialização adequada
        self._initialize_v8_components()
        
        # print("✅ TwoHeadV8Elegance PRONTA - Elegância com funcionalidade completa!")  # Commented for performance
        
    def _initialize_v8_components(self):
        """🔧 Inicialização específica dos componentes V11 Híbridos"""
        
        # 1. LSTM compartilhada
        for name, param in self.v8_shared_lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param, gain=1.0)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Forget gate bias = 1 para melhor gradient flow (sem in-place)
                if param.size(0) >= 4:
                    hidden_size = param.size(0) // 4
                    with torch.no_grad():
                        param.data[hidden_size:2*hidden_size].fill_(1.0)
        
        # 2. GRU paralela
        for name, param in self.v11_shared_gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param, gain=1.0)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # GRU reset gate bias = 0 (padrão), update gate bias = 0
                # Sem forget gate como LSTM
        
        # 3. Sistema de fusão híbrida
        for layer in self.hybrid_fusion:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # 4. Critic
        for layer in self.v8_critic:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # print("   ✅ V8 components inicializados com Xavier + orthogonal LSTM")  # Commented for performance
        
        # 🚨 CRITIC FIX: Setup separate optimizers if critic LR provided
        if self.critic_learning_rate is not None:
            self._setup_critic_optimizer()
    
    def forward_actor(self, features: torch.Tensor, lstm_states, episode_starts: torch.Tensor):
        """🎯 Forward Actor V8 - LSTM compartilhada + heads específicos + RAW FEATURES BYPASS"""

        self.training_step += 1  # Increment for cache

        # ⚡ EXTRAIR RAW FEATURES diretamente da observation original (BYPASS)
        # Features críticas estão nas posições [34-40] da última barra (barra 9)
        # Cada barra tem 45 features, última barra começa no índice 9*45 = 405
        # Features [34-40] da última barra = índices [405+34 : 405+41] = [439:446]
        batch_size = features.shape[0]
        raw_features = None

        if features.shape[1] >= 446:  # Verificar se observation tem tamanho correto (450D)
            raw_features = torch.zeros(batch_size, 7, device=features.device)
            raw_features[:, 0] = features[:, 439]  # volume_momentum
            raw_features[:, 1] = features[:, 440]  # price_position
            raw_features[:, 2] = features[:, 441]  # breakout_strength
            raw_features[:, 3] = features[:, 442]  # trend_consistency
            raw_features[:, 4] = features[:, 443]  # support_resistance
            raw_features[:, 5] = features[:, 444]  # volatility_regime
            raw_features[:, 6] = features[:, 445]  # market_structure

        # 1. Extract features first (450 → 256)
        extracted_features = self.extract_features(features)  # [batch, 256]

        # 2. Add sequence dimension for LSTM (single timestep)
        extracted_features = extracted_features.unsqueeze(1)  # [batch, 1, 256]

        # 3. Processar features através da arquitetura híbrida LSTM+GRU
        # LSTM para memória de longo prazo
        lstm_out, new_lstm_states = self.v8_shared_lstm(extracted_features, lstm_states)

        # GRU para padrões recentes (usa apenas hidden state inicial do LSTM)
        if lstm_states is not None:
            # Extrair hidden state do LSTM para inicializar GRU
            lstm_hidden = lstm_states[0]  # hidden state da LSTM
            gru_out, new_gru_states = self.v11_shared_gru(extracted_features, lstm_hidden)
        else:
            gru_out, new_gru_states = self.v11_shared_gru(extracted_features, None)

        # Fusão das saídas LSTM+GRU
        hybrid_input = torch.cat([lstm_out, gru_out], dim=-1)  # [batch, 1, 512]
        fused_features = self.hybrid_fusion(hybrid_input)  # [batch, 1, 256]

        # 4. Market context (usa features fusionadas LSTM+GRU + RAW FEATURES)
        current_step = self.training_step
        if current_step != self._cached_context_step:
            context_features, regime_id, context_info = self.market_context(
                fused_features,
                raw_features=raw_features  # ⚡ PASSAR RAW FEATURES
            )
            self.current_regime = regime_id
            self.last_context_info = context_info
            # Cache for critic
            self._cached_context_features = context_features
            self._cached_context_step = current_step
        else:
            context_features = self._cached_context_features
        
        # 5. Combinar features híbridas + context para heads
        combined_features = torch.cat([fused_features, context_features], dim=-1)
        
        # 6. Entry Head específico (entry + confidence) 
        # Squeeze para remover seq dimension se necessário
        hybrid_features_2d = fused_features.squeeze(1)  # [batch, seq, 256] → [batch, 256]
        context_features_2d = context_features.squeeze(1) if len(context_features.shape) == 3 else context_features
        
        raw_entry, entry_confidence, entry_gate_info = self.entry_head(hybrid_features_2d, context_features_2d)
        
        # 7. Management Head específico (SL/TP por posição) 
        pos1_mgmt, pos2_mgmt = self.management_head(hybrid_features_2d, context_features_2d)
        
        # 6. Combinar ações finais para 4D: [entry_decision, confidence, pos1_mgmt, pos2_mgmt]
        actions = torch.cat([
            raw_entry,        # [0] entry_decision
            entry_confidence, # [1] entry_confidence  
            pos1_mgmt,        # [2] pos1_mgmt
            pos2_mgmt         # [3] pos2_mgmt
        ], dim=-1)  # Total: 4D
        
        # 7. Gate info para compatibilidade  
        gate_info = {
            'v8_elegance_4d': True,
            'regime_info': context_info,
            'entry_info': entry_gate_info,
            'final_actions_shape': actions.shape,
            'lstm_norm': torch.norm(lstm_out)  # Keep as tensor for performance
        }
        
        return actions, new_lstm_states, gate_info
    
    def forward_critic(self, features: torch.Tensor, lstm_states, episode_starts: torch.Tensor):
        """💰 Forward Critic V8 - LSTM compartilhada + critic direto"""
        
        # 1. Extract features first (450 → 256)
        extracted_features = self.extract_features(features)  # [batch, 256]
        
        # 2. Add sequence dimension for LSTM (single timestep)  
        extracted_features = extracted_features.unsqueeze(1)  # [batch, 1, 256]
        
        # 3. Processar features através da arquitetura híbrida (mesmo que actor)
        # LSTM para memória de longo prazo
        lstm_out, new_lstm_states = self.v8_shared_lstm(extracted_features, lstm_states)
        
        # GRU para padrões recentes
        if lstm_states is not None:
            lstm_hidden = lstm_states[0]  # hidden state da LSTM
            gru_out, new_gru_states = self.v11_shared_gru(extracted_features, lstm_hidden)
        else:
            gru_out, new_gru_states = self.v11_shared_gru(extracted_features, None)
        
        # Fusão das saídas LSTM+GRU
        hybrid_input = torch.cat([lstm_out, gru_out], dim=-1)  # [batch, 1, 512]
        fused_features = self.hybrid_fusion(hybrid_input)  # [batch, 1, 256]
        
        # 4. Market context (USE CACHED from actor if available)
        current_step = self.training_step  
        if current_step == self._cached_context_step and self._cached_context_features is not None:
            context_features = self._cached_context_features
        else:
            context_features, regime_id, context_info = self.market_context(fused_features)
            self.current_regime = regime_id
            self.last_context_info = context_info
        
        # 5. Squeeze para remover seq dimension
        hybrid_features_2d = fused_features.squeeze(1)  # [batch, seq, 256] → [batch, 256]
        context_features_2d = context_features.squeeze(1) if len(context_features.shape) == 3 else context_features
        
        # 6. Combinar para critic
        critic_input = torch.cat([hybrid_features_2d, context_features_2d], dim=-1)
        
        # 5. Value prediction direto
        values = self.v8_critic(critic_input)
        
        # 5. Dummy states para compatibilidade
        dummy_states = new_lstm_states
        
        return values, dummy_states
    
    def predict_values(self, observations, lstm_states=None, episode_starts=None):
        """💰 Value prediction para treinamento"""
        if episode_starts is None:
            episode_starts = torch.zeros(observations.shape[0], dtype=torch.bool, device=observations.device)
        
        values, _ = self.forward_critic(observations, lstm_states, episode_starts)
        return values
    
    def get_v8_status(self) -> Dict[str, Any]:
        """📊 Status do sistema V8 Elegance"""
        return {
            'architecture': 'v8_elegance',
            'lstm_hidden': self.v8_lstm_hidden,
            'features_dim': self.v8_features_dim,
            'context_dim': self.v8_context_dim,
            'memory_size': self.v8_memory_size,
            'current_regime': self.current_regime,
            'regime_info': self.last_context_info,
            'training_step': self.training_step,
            'memory_usage': f"{int(self.memory_bank.memory_ptr.item())}/{self.v8_memory_size}"
        }
    
    def post_training_step(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """📈 Post-training processing"""
        self.training_step += 1
        
        # Add to memory if reward available
        if 'reward' in experience:
            trade_data = torch.tensor([
                experience.get('reward', 0.0),
                experience.get('duration', 1.0),
                float(self.current_regime),
                experience.get('confidence', 0.5),
                float(experience.get('done', False)),
                experience.get('pnl', 0.0),
                0.0,  # reserved
                0.0   # reserved
            ], dtype=torch.float32, device=self.memory_bank.trade_memory.device)
            
            self.memory_bank.add_trade(trade_data)
        
        return {
            'v8_regime': self.current_regime,
            'v8_training_step': self.training_step,
            'v8_context_info': self.last_context_info
        }
    
    def _setup_critic_optimizer(self):
        """🚨 CRITIC FIX: Setup separate optimizer for critic with different LR"""
        import torch.optim as optim
        
        # Get critic parameters
        critic_params = list(self.v8_critic.parameters())
        
        if critic_params and self.critic_learning_rate is not None:
            # Create separate optimizer for critic
            self.critic_optimizer = optim.Adam(
                critic_params, 
                lr=self.critic_learning_rate,
                weight_decay=1e-5  # Light regularization
            )
            print(f"✅ Critic optimizer created with LR: {self.critic_learning_rate}")
        else:
            print("⚠️ No critic parameters found or LR not set")

# 🛠️ UTILITY FUNCTIONS

def get_v11_sigmoid_kwargs():
    """Retorna kwargs para TwoHeadV11Sigmoid"""
    return {
        # V11 Sigmoid parameters
        'v8_lstm_hidden': 256,
        'v8_features_dim': 256,
        'v8_context_dim': 64,
        'v8_memory_size': 512,

        # Base policy parameters
        'features_extractor_class': TradingTransformerFeatureExtractor,
        'features_extractor_kwargs': {'features_dim': 256},
        'net_arch': [256, 128],
        'lstm_hidden_size': 256,
        'n_lstm_layers': 1,
        'activation_fn': torch.nn.LeakyReLU,
        'ortho_init': True,
        'log_std_init': -0.5,
        'full_std': True,
        'use_expln': False,
        'squash_output': False
    }

def get_v11_sigmoid_no_compression_kwargs():
    """
    🚀 Retorna kwargs para TwoHeadV11Sigmoid SEM COMPRESSÃO (450D → 450D)

    VANTAGENS:
    ✅ 100% fidelidade de informação (zero loss)
    ✅ Transformer preserva todas as features
    ✅ LSTM/GRU processam informação completa

    TRADE-OFFS:
    ⚠️ Mais parâmetros (450D vs 256D)
    ⚠️ Mais memória GPU (~1.7x)
    ⚠️ Treinamento ligeiramente mais lento (~1.3x)

    Usage:
        from trading_framework.policies.two_head_v11_sigmoid import get_v11_sigmoid_no_compression_kwargs

        policy_kwargs = get_v11_sigmoid_no_compression_kwargs()
        model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=policy_kwargs, ...)
    """
    from trading_framework.extractors.transformer_no_compression import TradingTransformerNoCompression

    return {
        # V11 Sigmoid NO-COMPRESSION parameters
        'v8_lstm_hidden': 512,  # AUMENTADO: 512D para processar 450D adequadamente
        'v8_features_dim': 450,  # 🚀 SEM COMPRESSÃO: 450D direto do transformer
        'v8_context_dim': 128,   # AUMENTADO: 128D para mais capacidade contextual
        'v8_memory_size': 512,

        # Base policy parameters - NO COMPRESSION
        'features_extractor_class': TradingTransformerNoCompression,
        'features_extractor_kwargs': {'features_dim': 450},  # 450D sem perda
        'net_arch': [512, 256],  # AUMENTADO: mais capacidade para processar 450D
        'lstm_hidden_size': 512,  # Match v8_lstm_hidden
        'n_lstm_layers': 1,
        'activation_fn': torch.nn.LeakyReLU,
        'ortho_init': True,
        'log_std_init': -0.5,
        'full_std': True,
        'use_expln': False,
        'squash_output': False
    }

def validate_v11_sigmoid_policy(policy):
    """Valida TwoHeadV11Sigmoid"""
    
    required_attrs = [
        'v8_shared_lstm', 'market_context', 'entry_head', 
        'management_head', 'memory_bank', 'v8_critic'
    ]
    
    for attr in required_attrs:
        if not hasattr(policy, attr):
            raise ValueError(f"V8Elegance missing required attribute: {attr}")
    
    # Verificar heads específicos
    if not isinstance(policy.entry_head, DaytradeEntryHead):
        raise ValueError("Entry head must be DaytradeEntryHead")
    
    if not isinstance(policy.management_head, DaytradeManagementHead):
        raise ValueError("Management head must be DaytradeManagementHead")
    
    print("✅ TwoHeadV8Elegance validada - Arquitetura elegante funcionando!")
    print(f"   🧠 Shared LSTM: {policy.v8_lstm_hidden}D")
    print(f"   🎯 Entry Head: DaytradeEntryHead específico")
    print(f"   💰 Management Head: DaytradeManagementHead específico")
    print(f"   💾 Memory: {policy.v8_memory_size} trades")
    print(f"   🌍 Context: {policy.v8_context_dim}D market context")
    
    return True

if __name__ == "__main__":
    print("🚀 TwoHeadV8Elegance - Simplicidade Focada no Daytrade!")
    print("   🧠 UMA LSTM compartilhada (elegância)")
    print("   🎯 Entry Head específico (entry + confidence)")
    print("   💰 Management Head específico (SL/TP por posição)")
    print("   💾 Memory Bank elegante (512 trades)")
    print("   🌍 Market Context único (4 regimes)")
    print("   ⚡ 4D actions otimizadas (trading eficiente)")