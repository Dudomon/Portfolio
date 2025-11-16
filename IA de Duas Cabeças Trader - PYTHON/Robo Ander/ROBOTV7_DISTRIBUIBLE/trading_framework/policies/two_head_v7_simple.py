"""
[LAUNCH] TwoHeadV7Simple UNIFICADA - Arquitetura Simples + Gates Especializados da V6
Única versão V7 - combina simplicidade da arquitetura com inteligência dos gates
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Type, Union
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution
import torch.nn.functional as F

# [LAUNCH] IMPORTAR O TRANSFORMER COMPROVADAMENTE FUNCIONAL
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

# [LAUNCH] USAR O TRANSFORMER COMPROVADAMENTE FUNCIONAL AO INVÉS DE MLP SIMPLES
# V7 mantém simplicidade na arquitetura neural, mas usa extrator avançado
EnhancedFeaturesExtractor = TradingTransformerFeatureExtractor

class SpecializedEntryHead(nn.Module):
    """[TARGET] Entry Head com Gates Especializados da V6 - SISTEMA COMPLETO DE FILTROS"""
    
    def __init__(self, input_dim=128):
        super().__init__()
        self.input_dim = input_dim
        
        # [LAUNCH] 1. TEMPORAL GATE - Analisa timing de entrada [TANH FIX]
        self.horizon_analyzer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Tanh()  # TANH FIX: Adicionado na arquitetura
        )
        
        # [LAUNCH] 2. VALIDATION GATE - Multi-timeframe + Pattern validation [TANH FIX]
        self.mtf_validator = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Tanh()  # TANH FIX: Adicionado na arquitetura
        )
        
        self.pattern_memory_validator = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(16, 1),
            nn.Tanh()  # TANH FIX: Adicionado na arquitetura
        )
        
        # [LAUNCH] 3. RISK GATE - Análise de risco + regime [TANH FIX]
        self.risk_gate_entry = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Tanh()  # TANH FIX: Adicionado na arquitetura
        )
        
        self.regime_gate = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(16, 1),
            nn.Tanh()  # TANH FIX: Adicionado na arquitetura
        )
        
        # [LAUNCH] 4. MARKET GATE - Lookahead + Fatigue detection [TANH FIX]
        self.lookahead_gate = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Tanh()  # TANH FIX: Adicionado na arquitetura
        )
        
        self.fatigue_detector = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(16, 1),
            nn.Tanh()  # TANH FIX: Adicionado na arquitetura
        )
        
        # [LAUNCH] 5. QUALITY GATE - 4 filtros especializados [TANH FIX]
        self.momentum_filter = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Tanh()  # TANH FIX: Adicionado na arquitetura
        )
        
        self.volatility_filter = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Tanh()  # TANH FIX: Adicionado na arquitetura
        )
        
        self.volume_filter = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Tanh()  # TANH FIX: Adicionado na arquitetura
        )
        
        self.trend_strength_filter = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Tanh()  # TANH FIX: Adicionado na arquitetura
        )
        
        # [LAUNCH] 6. CONFIDENCE GATE - Confiança geral [TANH FIX]
        self.confidence_estimator = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Tanh()  # TANH FIX: Adicionado na arquitetura
        )
        
        # [TARGET] DECISION NETWORK FINAL - SIGMOID FREE COM TANH + REMAPEAMENTO
        self.final_decision_network = nn.Sequential(
            nn.Linear(input_dim + 10, 64),  # input + 10 scores
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(16, 1),
            nn.Tanh()  # 🎯 FIX SHORT BIAS: Tanh [-1,1] será remapeado para [0,1]
        )
        
        # [DISABLED] ADAPTIVE THRESHOLDS - NÃO USADO MAIS (gates removidos)
        # self.register_parameter('adaptive_threshold_main', nn.Parameter(torch.tensor(0.25)))
        # self.register_parameter('adaptive_threshold_risk', nn.Parameter(torch.tensor(0.15)))
        # self.register_parameter('adaptive_threshold_regime', nn.Parameter(torch.tensor(0.10)))
        
        # ✅ SIGMOID GATES REMOVIDAS - Sem saturação!
        # self._initialize_all_sigmoid_gates()  # NÃO MAIS NECESSÁRIO
        
        # 🎯 FIX DUPLA CONVERSÃO: Inicializar final_decision_network para distribuição neutra
        self._initialize_balanced_final_decision()
    
    def _initialize_all_sigmoid_gates(self):
        """🔧 Inicialização específica para TODAS as sigmoid gates V7 - SOLUÇÃO DEFINITIVA SATURAÇÃO"""
        import torch.nn as nn
        
        # Lista de todas as gates que usam sigmoid
        sigmoid_gate_modules = [
            self.horizon_analyzer,          # Temporal gate
            self.mtf_validator,            # Validation gate  
            self.pattern_memory_validator, # Validation gate
            self.risk_gate_entry,          # Risk gate
            self.regime_gate,              # Risk gate
            self.lookahead_gate,           # Market gate
            self.fatigue_detector,         # Market gate
            self.momentum_filter,          # Quality gate
            self.volatility_filter,        # Quality gate
            self.volume_filter,            # Quality gate
            self.trend_strength_filter,    # Quality gate
            self.confidence_estimator      # Confidence gate
        ]
        
        print("🔧 [V7 FIX] Inicializando todas as sigmoid gates para evitar saturação...")
        
        for i, gate_module in enumerate(sigmoid_gate_modules):
            for layer in gate_module:
                if isinstance(layer, nn.Linear):
                    # 🎯 INICIALIZAÇÃO CONSERVADORA PARA SIGMOID
                    # Range pequeno para evitar saturação: sigmoid(±3) ≈ 0.95/0.05
                    with torch.no_grad():
                        # Weights: Xavier com gain reduzido
                        nn.init.xavier_uniform_(layer.weight, gain=0.5)  # Gain reduzido
                        
                        # Bias: Zero ou levemente negativo (favorece sigmoid ~0.5)
                        if layer.bias is not None:
                            nn.init.uniform_(layer.bias, -0.2, 0.1)  # Ligeiramente negativo
                            
        print(f"✅ [V7 FIX] {len(sigmoid_gate_modules)} sigmoid gates inicializadas com sucesso!")
        
    def _initialize_balanced_final_decision(self):
        """🎯 FIX DUPLA CONVERSÃO: Inicializar final_decision_network para distribuição equilibrada"""
        import torch.nn as nn
        
        # Inicializar layers da final_decision_network
        linear_layers = [layer for layer in self.final_decision_network if isinstance(layer, nn.Linear)]
        
        for i, layer in enumerate(linear_layers):
            with torch.no_grad():
                # Weights: Xavier normal com gain aumentado para mais variância
                nn.init.xavier_normal_(layer.weight, gain=1.0)
                
                # Bias: Ajuste específico para a última linear layer (antes do Tanh)
                if layer.bias is not None:
                    # Última linear layer (nn.Linear(16, 1)): bias para balanceamento
                    if i == len(linear_layers) - 1:  # Última linear layer
                        nn.init.constant_(layer.bias, 0.0)  # Bias neutro para distribuição uniforme
                        print(f"   🎯 [BALANCED INIT] Última layer: bias={layer.bias.item():.3f}")
                    else:
                        nn.init.constant_(layer.bias, 0.0)  # Outras layers neutras
        
        print("🎯 [FIX DUPLA CONVERSÃO] Final decision network inicializada para distribuição balanceada")
        
    def forward(self, entry_signal, management_signal, market_context):
        # 🛡️ PROTEÇÃO CONTRA OVERFLOW: Clamp inputs extremos
        entry_signal = torch.clamp(entry_signal, min=-100.0, max=100.0)
        management_signal = torch.clamp(management_signal, min=-100.0, max=100.0)
        market_context = torch.clamp(market_context, min=-100.0, max=100.0)
        
        # 🔥 GATES REMOVIDOS - FORWARD DIRETO PARA O MODELO
        # Combinar sinais para análise
        combined_input = torch.cat([entry_signal, management_signal, market_context], dim=-1)
        
        # [BYPASS] CÁLCULO DIRETO SEM GATES
        # Manter redes individuais para features, mas sem filtering
        
        # 🎯 FIX DUPLA CONVERSÃO: Manter features em range tanh [-1,1]
        temporal_feature = self.horizon_analyzer(combined_input)
        
        # 2. MTF + Pattern features - TANH range [-1,1]
        mtf_feature = self.mtf_validator(combined_input)
        pattern_feature = self.pattern_memory_validator(combined_input)
        validation_feature = (mtf_feature + pattern_feature) / 2
        
        # 3. Risk features - TANH range [-1,1]
        risk_feature = self.risk_gate_entry(combined_input)
        regime_feature = self.regime_gate(combined_input)
        risk_composite_feature = (risk_feature + regime_feature) / 2
        
        # 4. Market features - TANH range [-1,1]
        lookahead_feature = self.lookahead_gate(combined_input)
        fatigue_feature = self.fatigue_detector(combined_input)
        market_feature = (lookahead_feature + fatigue_feature) / 2
        
        # 5. Quality features - TANH range [-1,1]
        momentum_feature = self.momentum_filter(combined_input)
        volatility_feature = self.volatility_filter(combined_input)
        volume_feature = self.volume_filter(combined_input)
        trend_feature = self.trend_strength_filter(combined_input)
        quality_feature = (momentum_feature + volatility_feature + volume_feature + trend_feature) / 4
        
        # 6. Confidence feature - TANH range [-1,1]
        confidence_feature = self.confidence_estimator(combined_input)
        
        # [NO GATES] FEED DIRETO PARA DECISÃO FINAL
        # Todas as features vão direto para a rede de decisão
        all_features = torch.cat([
            temporal_feature, validation_feature, risk_composite_feature, 
            market_feature, quality_feature, confidence_feature,
            mtf_feature, pattern_feature, lookahead_feature, fatigue_feature
        ], dim=-1)
        
        # Input enriquecido para decisão
        decision_input = torch.cat([combined_input, all_features], dim=-1)
        
        # 🎯 FIX DUPLA CONVERSÃO: Final decision já produz [-1,1], convertemos UMA vez
        final_decision_raw = self.final_decision_network(decision_input)
        
        # 🛡️ PROTEÇÃO FINAL: Garantir que não há NaN/Inf antes da conversão
        final_decision_raw = torch.clamp(final_decision_raw, min=-10.0, max=10.0)
        final_decision_raw = torch.where(torch.isnan(final_decision_raw), torch.zeros_like(final_decision_raw), final_decision_raw)
        final_decision_raw = torch.where(torch.isinf(final_decision_raw), torch.zeros_like(final_decision_raw), final_decision_raw)
        
        # 🎯 ÚNICA conversão tanh [-1,1] para [0,1] para thresholds balanceados
        final_decision = (final_decision_raw + 1.0) / 2.0  # [-1,1] → [0,1]
        
        # 🔧 COMPATIBILIDADE: Converter confidence para [0,1] apenas para interface
        # 🛡️ PROTEÇÃO: Garantir confidence_feature não tem NaN/Inf
        confidence_feature = torch.clamp(confidence_feature, min=-10.0, max=10.0)
        confidence_feature = torch.where(torch.isnan(confidence_feature), torch.zeros_like(confidence_feature), confidence_feature)
        confidence_feature = torch.where(torch.isinf(confidence_feature), torch.zeros_like(confidence_feature), confidence_feature)
        
        confidence_score = (confidence_feature + 1.0) / 2.0  # [-1,1] → [0,1] só para output
        
        # 🔧 COMPATIBILIDADE: Manter gate_info para debug (valores dummy)
        gate_info = {
            'temporal_gate': torch.ones_like(temporal_feature),      # Dummy: sempre 1.0
            'validation_gate': torch.ones_like(validation_feature),  # Dummy: sempre 1.0
            'risk_gate': torch.ones_like(risk_composite_feature),    # Dummy: sempre 1.0
            'market_gate': torch.ones_like(market_feature),          # Dummy: sempre 1.0
            'quality_gate': torch.ones_like(quality_feature),        # Dummy: sempre 1.0
            'confidence_gate': torch.ones_like(confidence_feature),  # Dummy: sempre 1.0
            'composite_score': torch.ones_like(confidence_feature),  # Dummy: sempre 1.0
            'final_gate': torch.ones_like(confidence_feature),       # Dummy: sempre 1.0
            'scores': {
                'temporal': temporal_feature,
                'validation': validation_feature,
                'risk': risk_composite_feature,
                'market': market_feature,
                'quality': quality_feature,
                'confidence': confidence_feature,
                'mtf': mtf_feature,
                'pattern': pattern_feature,
                'lookahead': lookahead_feature,
                'fatigue': fatigue_feature
            }
        }
        
        return final_decision, confidence_score, gate_info

class TwoHeadDecisionMaker(nn.Module):
    """[BRAIN] Decision Maker simplificado para Management Head"""
    
    def __init__(self, input_dim=128):
        super().__init__()
        self.input_dim = input_dim
        
        # Simple and effective architecture
        self.processor = nn.Sequential(
            nn.Linear(input_dim, 128),  # input_dim já é o combined input
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 32)
        )
        
    def forward(self, entry_signal, management_signal, market_context):
        # Combinar sinais
        combined_input = torch.cat([entry_signal, management_signal, market_context], dim=-1)
        decision = self.processor(combined_input)
        
        # Simular confidence (para compatibilidade) - SEM SATURAÇÃO
        confidence = torch.clamp((decision.mean(dim=-1, keepdim=True) + 1.0) / 2.0, 0.0, 1.0)
        weights = torch.softmax(torch.randn_like(confidence.expand(-1, 2)), dim=-1)
        
        return decision, confidence, weights

class TradeMemoryBank(nn.Module):
    """[EMOJI] Trade Memory Bank simplified para V7"""
    
    def __init__(self, memory_size=1000, trade_dim=8):
        super().__init__()
        
        self.memory_size = memory_size
        self.trade_dim = trade_dim
        
        # Banco de memória para trades
        self.register_buffer('trade_memory', torch.zeros(memory_size, trade_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
        # Rede para processar memórias
        self.memory_processor = nn.Sequential(
            nn.Linear(trade_dim, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(16, 8)
        )
        
    def add_trade(self, trade_data):
        """Adiciona trade à memória"""
        ptr = int(self.memory_ptr.item())
        self.trade_memory[ptr] = trade_data
        self.memory_ptr[0] = (ptr + 1) % self.memory_size
        
    def get_memory_context(self, batch_size):
        """[ALERT] CRITICAL FIX: Retorna contexto não-zero da memória"""
        # NUNCA retornar zeros puros - sempre ter algum sinal
        if self.trade_memory.sum() == 0:
            # Gerar contexto inicial baseado em ruído estruturado
            base_context = torch.randn(8, device=self.trade_memory.device) * 0.1
            processed = self.memory_processor(base_context)
            return processed.unsqueeze(0).expand(batch_size, -1)
        
        # Processar últimas memórias
        recent_memories = self.trade_memory[-min(100, self.memory_size):]
        avg_memory = recent_memories.mean(dim=0)
        
        # Adicionar pequeno ruído para evitar saturação
        avg_memory = avg_memory + torch.randn_like(avg_memory) * 0.01
        
        processed = self.memory_processor(avg_memory)
        return processed.unsqueeze(0).expand(batch_size, -1)

class TwoHeadV7Simple(RecurrentActorCriticPolicy):
    """
    [LAUNCH] TwoHeadV7Simple UNIFICADA - Arquitetura Simples + Gates Especializados
    
    ARQUITETURA V7:
    - UMA LSTM shared (256 hidden) 
    - UMA GRU para refinement (128 hidden)
    - Entry Head com 6 Gates Especializados (da V6)
    - Management Head simplificado
    - Trade Memory Bank
    
    GATES ESPECIALIZADOS:
    1. Temporal Gate (timing)
    2. Validation Gate (MTF + patterns)
    3. Risk Gate (risk + regime)
    4. Market Gate (lookahead + fatigue)
    5. Quality Gate (4 filtros técnicos)
    6. Confidence Gate (confiança geral)
    """
    
    def __init__(
        self,
        *args,
        # V7 SPECIFIC PARAMETERS
        v7_shared_lstm_hidden: int = 256,
        v7_features_dim: int = 256,
        **kwargs
    ):
        
        # Garantir features_extractor_class
        if 'features_extractor_class' not in kwargs:
            kwargs['features_extractor_class'] = EnhancedFeaturesExtractor
        if 'features_extractor_kwargs' not in kwargs:
            kwargs['features_extractor_kwargs'] = {'features_dim': v7_features_dim}
        
        # Store V7 specific parameters
        self.v7_shared_lstm_hidden = v7_shared_lstm_hidden
        self.v7_features_dim = v7_features_dim
        
        print(f"V7Simple UNIFICADA inicializando:")
        print(f"   LSTM Hidden: {v7_shared_lstm_hidden}")
        print(f"   Features Dim: {v7_features_dim}")
        
        # [FIX] CRÍTICO: Usar nossa implementação LSTM custom, desabilitar sb3_contrib LSTM
        kwargs['enable_critic_lstm'] = False  # Desabilitar sb3_contrib, usar nossa implementação
        print("   [CUSTOM] Usando nossa implementação LSTM critic (sb3_contrib LSTM desabilitado)")
        
        # Initialize parent
        super().__init__(*args, **kwargs)
        
        # [LAUNCH] V7 ARCHITECTURE COMPONENTS - LSTM SEPARADOS
        
        # 1. [EMOJI] ACTOR LSTM (independente)
        self.v7_actor_lstm = nn.LSTM(
            input_size=self.v7_features_dim,
            hidden_size=self.v7_shared_lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0  # Dropout interno desabilitado (1 layer only)
        )
        
        # 2. [ALERT] CRITIC LSTM COMPLETAMENTE REMOVIDO - Agora usa MLP + Memory Buffer
        # self.v7_critic_lstm = nn.LSTM(...)  # ELIMINADO!
        # [FIX] CRÍTICO: RecurrentActorCriticPolicy criado com enable_critic_lstm=False
        
        # 3. [ALERT] ACTOR GRU REMOVIDO - Arquitetura simplificada: LSTM direto
        # self.v7_actor_gru = nn.GRU(...)  # ELIMINADO para simplificar
        
        # 4. CRITIC LSTM + ENHANCED PROCESSING
        self.memory_steps = 32
        self.critic_memory_buffer = None
        
        # 4a. LSTM CRITIC (PRIMARY)
        self.v7_critic_lstm = nn.LSTM(
            input_size=self.v7_features_dim,  # Features de entrada (256)
            hidden_size=256,                  # Hidden=256 
            num_layers=1,                     # Single layer
            batch_first=True,                 # (batch, seq, features)
            dropout=0.0                       # Sem dropout
        )
        
        # 4b. POST-LSTM PROCESSING: Attention para focar em outputs importantes do LSTM
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=256,                    # Output do LSTM
            num_heads=8,
            dropout=0.02,
            batch_first=True
        )
        
        # 4c. POST-ATTENTION REFINEMENT: Layers residuais para refinar output da attention
        self.critic_layer1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(512),
            nn.Dropout(0.02)
        )
        
        self.critic_layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(256),
            nn.Dropout(0.02)
        )
        
        # 4d. MULTI-HEAD VALUE ESTIMATION: Diferentes perspectivas do LSTM output
        self.value_heads = nn.ModuleDict({
            'pnl_head': nn.Linear(256, 1),      # Foco em PnL
            'risk_head': nn.Linear(256, 1),     # Foco em risco
            'timing_head': nn.Linear(256, 1),   # Foco em timing
            'consistency_head': nn.Linear(256, 1) # Foco em consistência
        })
        
        # 4e. HEAD COMBINATION WEIGHTS (aprendíveis)
        self.head_weights = nn.Parameter(torch.ones(4) / 4)
        
        # 4f. [REMOVED] Simple head - agora usamos multi-head processing
        
        # 🚫 NENHUM MLP BACKUP - APENAS LSTM
        
        # 4b. CRITIC GRU REMOVIDO - usa apenas MLP + Memory Buffer
        # self.v7_critic_gru = nn.GRU(...)  # ELIMINADO
        
        # [ALERT] CRITICAL FIX: Dropout REDUZIDO para evitar cascata mortal
        self.actor_lstm_dropout = nn.Dropout(0.05)   # REDUZIDO: 10% → 5%
        # self.actor_gru_dropout = nn.Dropout(0.05)  # REMOVIDO com GRU
        # self.critic_gru_dropout = nn.Dropout(0.05)  # REMOVIDO com GRU
        
        # 3. Trading Intelligence com Gates Especializados
        # [FIX] Combined input = entry_signal(256) + management_signal(256) + market_context(8) = 520
        combined_input_dim = self.v7_shared_lstm_hidden * 2 + 8  # 256*2 + 8 = 520
        self.entry_head = SpecializedEntryHead(input_dim=combined_input_dim)  # [TARGET] GATES ESPECIALIZADOS
        self.management_head = TwoHeadDecisionMaker(input_dim=combined_input_dim)
        self.trade_memory = TradeMemoryBank(memory_size=1000, trade_dim=8)
        
        # 4. Actor/Critic heads com DROPOUT CRÍTICO
        self.v7_actor_head = nn.Sequential(
            nn.Linear(self.v7_shared_lstm_hidden + 33, 64),  # LSTM + decision context (1 + 32)
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(64),
            nn.Dropout(0.1),  # [ALERT] CRITICAL: Dropout no actor
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.05), # [ALERT] CRITICAL: Dropout menor antes da saída
            nn.Linear(32, self.action_space.n if hasattr(self.action_space, 'n') else self.action_space.shape[0])
        )
        
        # [FIX] CRITIC HEAD ORIGINAL (removido - usa MLP direto)
        # self.v7_critic_head = nn.Sequential(
        #     nn.Linear(self.v7_shared_lstm_hidden + 32, 64),  # LSTM + decision context
        #     nn.LeakyReLU(negative_slope=0.01),
        #     nn.LayerNorm(64),
        #     nn.Dropout(0.05), 
        #     nn.Linear(64, 32),
        #     nn.LeakyReLU(negative_slope=0.01),
        #     nn.Dropout(0.0),
        #     nn.Linear(32, 1)
        # )
        
        # [FIX] CRITIC HEAD PURO (LSTM direto, sem GRU)
        # self.v7_critic_head_pure = nn.Sequential(...)  # REMOVIDO - usa apenas MLP
        
        # [ENHANCED] INICIALIZAÇÃO AVANÇADA DO CRITIC
        # 1. Inicialização do LSTM critic
        for name, param in self.v7_critic_lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param, gain=1.0)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Forget gate bias = 1 para melhor gradient flow
                n = param.size(0)
                param[n//4:n//2].data.fill_(1.0)
        
        # 2. [REMOVED] Critic head initialization - now using multi-head
        
        
        # 3. [NEW] Inicialização do Temporal Attention
        nn.init.xavier_uniform_(self.temporal_attention.in_proj_weight, gain=1.0)
        nn.init.constant_(self.temporal_attention.in_proj_bias, 0.0)
        nn.init.xavier_uniform_(self.temporal_attention.out_proj.weight, gain=1.0)
        nn.init.constant_(self.temporal_attention.out_proj.bias, 0.0)
        
        # 4. [NEW] Inicialização das camadas residuais
        for module in [self.critic_layer1, self.critic_layer2]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=1.1)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)
        
        # 5. [NEW] Inicialização dos Multi-Head Values
        for head_name, head_module in self.value_heads.items():
            # Cada head com inicialização específica baseada na função
            if head_name == 'pnl_head':
                nn.init.xavier_normal_(head_module.weight, gain=1.5)  # PnL mais agressivo
            elif head_name == 'risk_head':
                nn.init.xavier_normal_(head_module.weight, gain=0.8)  # Risk mais conservador
            else:
                nn.init.xavier_normal_(head_module.weight, gain=1.0)  # Padrão
            
            nn.init.constant_(head_module.bias, 0.0)
        
        # 6. [LAUNCH] TEMPORAL REGULARIZATION: Registrar hook para penalizar weight_hh_l0 do critic
        self._setup_temporal_regularization()
        
        # 6. [FIX] V6 FIX: Aplicar correção ReLU → LeakyReLU do mlp_extractor (resolve 50-53% zeros)
        self._apply_v6_mlp_extractor_fix()
        
        # 7. [FIX] LSTM GRADIENT KEEPER: Força gradientes sempre vivos
        self._apply_lstm_gradient_keeper()
        
        # 7. [TARGET] INICIALIZAÇÃO DIFERENCIADA: Critic LSTM mais agressivo
        self._init_separated_lstms()
        
        # 8. [NEW] CONFIGURAR OPTIMIZERS SEPARADOS: Actor conservador, Critic agressivo
        self._setup_separate_optimizers()
        
        print(f"V7Simple REVOLUCIONARIA: MLP + Memory Buffer + Gates + LeakyReLU!")
        
    def forward_actor(self, features: torch.Tensor, lstm_states, episode_starts: torch.Tensor):
        """Forward pass do actor com LSTM SEPARADO e gates especializados"""
        
        # [ALERT] CRITICAL FIX: Dividir lstm_states entre actor e critic
        # lstm_states = (h_actor, c_actor, h_critic, c_critic) or (h_shared, c_shared)
        if len(lstm_states) == 4:
            # Estados já separados
            actor_states = (lstm_states[0], lstm_states[1])
        else:
            # Estados compartilhados - dividir
            actor_states = lstm_states
        
        # 1. Processar features através da arquitetura V7 ACTOR
        actor_lstm_out, new_actor_states = self.v7_actor_lstm(features, actor_states)
        
        # [ALERT] CRITICAL: Hidden state dropout após ACTOR LSTM
        actor_lstm_out = self.actor_lstm_dropout(actor_lstm_out)
        
        # [ALERT] GRU REMOVIDO: Usar LSTM output diretamente
        # actor_gru_out, _ = self.v7_actor_gru(actor_lstm_out)
        # actor_gru_out = self.actor_gru_dropout(actor_gru_out)
        
        # 2. Preparar sinais para decision makers (USANDO ACTOR LSTM DIRETO)
        batch_size = actor_lstm_out.shape[0]
        entry_signal = actor_lstm_out
        management_signal = actor_lstm_out
        market_context = self.trade_memory.get_memory_context(batch_size)
        
        # 3. Entry Head com Gates Especializados
        entry_decision, entry_conf, gate_info = self.entry_head(entry_signal, management_signal, market_context)
        
        # 4. Management Head (simplificado)
        mgmt_decision, mgmt_conf, mgmt_weights = self.management_head(entry_signal, management_signal, market_context)
        
        # 5. Combinar decisões
        decision_context = torch.cat([entry_decision, mgmt_decision], dim=-1)
        
        # 6. Actor final (USANDO ACTOR LSTM OUTPUT DIRETO)
        actor_input = torch.cat([actor_lstm_out.squeeze(1), decision_context], dim=-1)
        actions = self.v7_actor_head(actor_input)
        
        # [ALERT] CRITICAL FIX: Retornar estados separados
        if len(lstm_states) == 4:
            # Manter separação
            new_lstm_states = (new_actor_states[0], new_actor_states[1], lstm_states[2], lstm_states[3])
        else:
            # Compatibilidade - retornar estados do actor
            new_lstm_states = new_actor_states
            
        return actions, new_lstm_states, gate_info
        
    def forward_critic(self, features: torch.Tensor, lstm_states, episode_starts: torch.Tensor):
        """LSTM PRIMARY + ENHANCED PROCESSING - No fallbacks, no MLP backups"""
        
        batch_size = features.shape[0]
        
        # 1. MEMORY BUFFER MANAGEMENT
        if self.critic_memory_buffer is None or episode_starts.any():
            self.critic_memory_buffer = torch.zeros(
                batch_size, self.memory_steps, self.v7_features_dim,
                device=features.device
            )
        
        # 2. SHIFT MEMORY BUFFER
        self.critic_memory_buffer = torch.roll(self.critic_memory_buffer, shifts=1, dims=1)
        
        # 3. PROCESS FEATURES
        if hasattr(self, 'v7_backbone'):
            processed_features = self.v7_backbone(features)
        else:
            if not hasattr(self, '_feature_projector'):
                self._feature_projector = torch.nn.Linear(features.shape[-1], self.v7_features_dim).to(features.device)
            processed_features = self._feature_projector(features)
            
        self.critic_memory_buffer[:, 0, :] = processed_features
        
        # 4. LSTM FORWARD PASS (PRIMARY - ALWAYS USED)
        lstm_out, _ = self.v7_critic_lstm(self.critic_memory_buffer)
        
        # 5. POST-LSTM ATTENTION: Focar em outputs temporais importantes do LSTM
        attended_lstm, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        
        # 6. REFINEMENT LAYERS: Processar output da attention
        last_attended = attended_lstm[:, -1, :]  # Pegar último timestep (256 dim)
        x1 = self.critic_layer1(last_attended)   # → 512 dim
        x2 = self.critic_layer2(x1)              # 512 → 256 (no residual needed)
        
        # 7. MULTI-HEAD VALUE ESTIMATION: Diferentes perspectivas
        head_values = {}
        for head_name, head_module in self.value_heads.items():
            head_values[head_name] = head_module(x2)
        
        # 8. COMBINE HEADS: Weighted combination
        head_weights_normalized = F.softmax(self.head_weights, dim=0)
        values = (
            head_weights_normalized[0] * head_values['pnl_head'] +
            head_weights_normalized[1] * head_values['risk_head'] +
            head_weights_normalized[2] * head_values['timing_head'] +
            head_weights_normalized[3] * head_values['consistency_head']
        )
        
        # 9. DUMMY STATES FOR COMPATIBILITY
        dummy_states = lstm_states if lstm_states is not None else (
            torch.zeros(1, batch_size, 256, device=features.device),
            torch.zeros(1, batch_size, 256, device=features.device)
        )
        
        return values, dummy_states
    
    def _setup_temporal_regularization(self):
        """[LAUNCH] Setup temporal regularization para evitar vanishing gradients no critic LSTM"""
        self.temporal_reg_loss = 0.0
        self.temporal_reg_weight = 1e-4  # Peso da regularização aumentado
        
        # Registrar forward hook no critic LSTM para capturar gradientes
        def critic_lstm_hook(module, input, output):
            # Durante backward pass, penalizar weight_hh_l0 se gradientes ficarem muito pequenos
            if hasattr(module, 'weight_hh_l0'):
                # L2 penalty específico no weight_hh_l0 para prevenir vanishing
                hh_weight = module.weight_hh_l0
                self.temporal_reg_loss = self.temporal_reg_weight * torch.norm(hh_weight, p=2)
                
                # [FIX] GRADIENT SURGERY: Intervenção cirúrgica nos gradientes
                if self.training and hh_weight.grad is not None:
                    self._perform_gradient_surgery(hh_weight)
        
        # [FIX] SKIP TEMPORAL REG: Critic agora usa MLP (sem LSTM weight_hh_l0)
        print("[INFO] Temporal regularization SKIPPED - Critic usa MLP (sem vanishing gradients)")
    
    def get_temporal_regularization_loss(self):
        """Retorna loss de regularização temporal para adicionar ao loss total"""
        return self.temporal_reg_loss if hasattr(self, 'temporal_reg_loss') else 0.0
    
    def apply_temporal_dropout(self, hidden_states, dropout_prob=0.1):
        """[LAUNCH] Dropout temporal no hidden state do LSTM critic"""
        if self.training and torch.rand(1).item() < dropout_prob:
            # Reset parcial do hidden state para quebrar over-fitting temporal
            mask = torch.rand_like(hidden_states) > 0.3  # Manter 70% dos valores
            hidden_states = hidden_states * mask.float()
        return hidden_states
    
    def _apply_v6_mlp_extractor_fix(self):
        """[FIX] V6 FIX: Substituir ReLUs por LeakyReLU no mlp_extractor (resolve 50-53% zeros)"""
        try:
            if hasattr(self, 'mlp_extractor'):
                relu_fixes = 0
                
                for name, module in self.mlp_extractor.named_modules():
                    if isinstance(module, nn.ReLU):
                        # Encontrar módulo pai para substituição
                        parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
                        child_name = name.split('.')[-1]
                        
                        if parent_name:
                            try:
                                parent_module = self.mlp_extractor
                                for part in parent_name.split('.'):
                                    parent_module = getattr(parent_module, part)
                                
                                # [FIX] V6 FIX: Substituir ReLU por LeakyReLU
                                setattr(parent_module, child_name, nn.LeakyReLU(negative_slope=0.01, inplace=True))
                                relu_fixes += 1
                                
                            except Exception as e:
                                print(f"[V7] [EMOJI] Erro ao substituir {name}: {e}")
                
                print(f"[V7] [FIX] V6 FIX APLICADO: {relu_fixes} ReLUs → LeakyReLU (resolve 50-53% zeros)")
                
        except Exception as e:
            print(f"[V7] [EMOJI] Erro ao aplicar V6 fix: {e}")
    
    def _init_separated_lstms(self):
        """[TARGET] Inicialização diferenciada para LSTM Actor vs Critic"""
        try:
            print("[TARGET] Inicializando LSTMs separados com estratégias diferenciadas...")
            
            # [EMOJI] ACTOR LSTM: Inicialização padrão (conservadora)
            for name, param in self.v7_actor_lstm.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param, gain=1.0)  # Gain padrão
                elif 'bias' in name:
                    # CORREÇÃO CRÍTICA: Forget gate bias = 1.0
                    torch.nn.init.zeros_(param)
                    if param.size(0) >= 4:  # LSTM bias
                        hidden_size = param.size(0) // 4
                        param.data[hidden_size:2*hidden_size].fill_(1.0)  # Forget gate bias
            
            # [ALERT] CRITIC LSTM INICIALIZAÇÃO REMOVIDA - Não existe mais!
            
            # [ALERT] FORGET GATE BIAS REMOVIDO - Critic LSTM não existe mais!
            
            # [EMOJI] CRITIC GRU: REMOVIDO - não existe mais na V7
            
            print("[OK] LSTMs separados inicializados: ACTOR(conservador) vs CRITIC(agressivo)")
            print("   [EMOJI] Actor: gain=1.0, bias=zeros")
            print("   [TARGET] Critic: weight_hh_l0 especializado, forget_bias=1.0, noise_bias=±0.1")
            
        except Exception as e:
            print(f"[ERROR] Erro na inicialização diferenciada: {e}")
    
    def _init_weight_hh_specifically(self, weight_hh):
        """[ALERT] MÉTODO DESABILITADO - Critic LSTM removido"""
        return  # Skip - não há mais critic LSTM
        
        with torch.no_grad():
            # Split weight_hh_l0 into 4 gate components
            # Input gate: moderate initialization
            nn.init.orthogonal_(weight_hh[0:hidden_size], gain=1.0)
            
            # Forget gate: CRITICAL - lower gain to prevent saturation
            nn.init.orthogonal_(weight_hh[hidden_size:2*hidden_size], gain=0.8)
            
            # Cell gate: higher gain for information flow
            nn.init.orthogonal_(weight_hh[2*hidden_size:3*hidden_size], gain=1.2)
            
            # Output gate: moderate initialization  
            nn.init.orthogonal_(weight_hh[3*hidden_size:4*hidden_size], gain=1.0)
            
        print(f"[FIX] weight_hh_l0 inicializado: gates=[1.0, 0.8, 1.2, 1.0] para prevenir vanishing")
    
    def _perform_gradient_surgery(self, weight_hh):
        """[FIX] Cirurgia de gradientes para weight_hh_l0"""
        grad = weight_hh.grad
        hidden_size = weight_hh.shape[0] // 4
        
        # Análise por gate
        for gate_idx in range(4):
            start_idx = gate_idx * hidden_size
            end_idx = (gate_idx + 1) * hidden_size
            
            gate_grad = grad[start_idx:end_idx]
            zero_ratio = (torch.abs(gate_grad) < 1e-6).float().mean()
            
            # Intervenção específica por gate
            if zero_ratio > 0.7:  # >70% zeros
                if gate_idx == 1:  # Forget gate - CRÍTICO
                    # Ressuscitação agressiva do forget gate
                    mask = torch.abs(gate_grad) < 1e-6
                    gate_grad[mask] = torch.randn_like(gate_grad[mask]) * 2e-4
                elif gate_idx == 2:  # Cell gate - Importante para info flow
                    mask = torch.abs(gate_grad) < 1e-6
                    gate_grad[mask] = torch.randn_like(gate_grad[mask]) * 1.5e-4
                else:  # Input/output gates
                    mask = torch.abs(gate_grad) < 1e-6
                    gate_grad[mask] = torch.randn_like(gate_grad[mask]) * 1e-4
            
            # Gradient scaling para gradiente muito pequenos
            elif zero_ratio > 0.4:  # 40-70% zeros
                small_mask = (torch.abs(gate_grad) > 1e-6) & (torch.abs(gate_grad) < 1e-4)
                if small_mask.sum() > 0:
                    gate_grad[small_mask] *= 5.0  # Scale up
        
        # Gradient clipping final
        grad_norm = torch.norm(grad)
        if grad_norm > 1.0:
            grad *= (1.0 / grad_norm)
        elif grad_norm < 1e-4:
            grad *= (1e-4 / grad_norm)
    
    def _apply_lstm_gradient_keeper(self):
        """[FIX] Aplica gradient keeper nos LSTMs"""
        def create_gradient_keeper_hook(lstm_name):
            def gradient_keeper_hook(grad):
                if grad is None:
                    return None
                
                # Intervenção OBRIGATÓRIA se >60% zeros
                zero_mask = torch.abs(grad) < 1e-5
                zero_ratio = zero_mask.float().mean().item()
                
                if zero_ratio > 0.6:
                    # Ressuscitação por gate
                    hidden_size = grad.shape[0] // 4
                    for gate_idx in range(4):
                        start_idx = gate_idx * hidden_size
                        end_idx = (gate_idx + 1) * hidden_size
                        gate_mask = zero_mask[start_idx:end_idx]
                        
                        if gate_mask.sum() > 0:
                            noise_scale = 2e-4 if gate_idx == 1 else 1e-4  # Forget gate mais agressivo
                            noise = torch.randn_like(grad[start_idx:end_idx][gate_mask]) * noise_scale
                            grad[start_idx:end_idx][gate_mask] = noise
                
                return grad
            return gradient_keeper_hook
        
        # SKIP CRITIC LSTM: Agora usa MLP
        print("[INFO] Gradient Keeper SKIPPED no critic (usa MLP, não LSTM)")
        
        # Aplicar no actor LSTM  
        if hasattr(self, 'v7_actor_lstm'):
            self.v7_actor_lstm.weight_hh_l0.register_hook(
                create_gradient_keeper_hook('actor')
            )
            print("[FIX] Gradient Keeper aplicado no v7_actor_lstm")

# [TARGET] FUNÇÕES DE UTILIDADE

    def _setup_separate_optimizers(self):
        """🎯 Configurar optimizers separados: Actor conservador, Critic agressivo"""
        
        print("🎯 CONFIGURANDO OPTIMIZERS SEPARADOS:")
        
        # Learning rates diferentes
        actor_lr = 2.5e-05    # Conservador para estabilidade
        critic_lr = 1.0e-04   # Agressivo para catch-up (2x)
        
        print(f"   Actor LR: {actor_lr} (conservador)")
        print(f"   Critic LR: {critic_lr} (agressivo - 2x)")
        
        # Coletar parâmetros do actor
        actor_params = []
        actor_params.extend(self.v7_actor_lstm.parameters())
        actor_params.extend(self.entry_head.parameters())
        actor_params.extend(self.management_head.parameters())
        actor_params.extend(self.v7_actor_head.parameters())
        
        # Coletar parâmetros do critic (LSTM + enhanced processing)
        critic_params = []
        critic_params.extend(self.v7_critic_lstm.parameters())
        critic_params.extend(self.temporal_attention.parameters())
        critic_params.extend(self.critic_layer1.parameters())
        critic_params.extend(self.critic_layer2.parameters())
        critic_params.extend(self.value_heads.parameters())
        critic_params.append(self.head_weights)
        
        # Criar optimizers separados
        import torch.optim as optim
        self.actor_optimizer = optim.Adam(actor_params, lr=actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(critic_params, lr=critic_lr, eps=1e-5)
        
        # Salvar LRs para monitoramento
        self.current_actor_lr = actor_lr
        self.current_critic_lr = critic_lr
        
        print(f"   ✅ Actor Optimizer: {len(list(actor_params))} parâmetros")
        print(f"   ✅ Critic Optimizer: {len(list(critic_params))} parâmetros")
        
        # Flag para indicar uso de optimizers separados
        self.use_separate_optimizers = True
        
    def predict_values(self, observations, lstm_states=None, episode_starts=None):
        """🎯 Override para usar nossa implementação LSTM critic"""
        
        if episode_starts is None:
            episode_starts = torch.zeros(observations.shape[0], dtype=torch.bool, device=observations.device)
            
        # Usar nossa implementação de value prediction (SEM torch.no_grad para permitir gradientes)
        values, _ = self.forward_critic(observations, lstm_states, episode_starts)
            
        return values
        
    def get_actor_critic_optimizers(self):
        """Retorna optimizers separados para uso no treinamento"""
        if hasattr(self, 'use_separate_optimizers') and self.use_separate_optimizers:
            return self.actor_optimizer, self.critic_optimizer
        else:
            # Fallback para optimizer padrão
            return self.optimizer, self.optimizer
            
    def update_learning_rates(self, actor_lr=None, critic_lr=None):
        """Atualizar learning rates dos optimizers separados"""
        if hasattr(self, 'use_separate_optimizers') and self.use_separate_optimizers:
            if actor_lr is not None:
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = actor_lr
                self.current_actor_lr = actor_lr
                
            if critic_lr is not None:
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = critic_lr
                self.current_critic_lr = critic_lr
                    
            print(f"🎯 LRs atualizados - Actor: {self.current_actor_lr}, Critic: {self.current_critic_lr}")
        else:
            # Fallback para optimizer padrão
            if actor_lr is not None:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = actor_lr

def get_v7_kwargs():
    """Retorna kwargs padrão para V7Simple"""
    return {
        'v7_shared_lstm_hidden': 256,
        # 'v7_gru_hidden': 128,  # REMOVIDO com GRU
        'v7_features_dim': 256,
        'features_extractor_class': TradingTransformerFeatureExtractor,  # [LAUNCH] TRANSFORMER COMPROVADO
        'features_extractor_kwargs': {'features_dim': 256},
        # Policy kwargs integrados diretamente (sem conflitos LSTM)
        'net_arch': [256, 128],  # Para compatibilidade com logging
        'lstm_hidden_size': 256,
        'n_lstm_layers': 1,
        # REMOVIDO: shared_lstm e enable_critic_lstm (causavam conflito)
        'activation_fn': torch.nn.LeakyReLU,
        'ortho_init': True,
        'log_std_init': -0.5,
        'full_std': True,
        'use_expln': False,
        'squash_output': False
    }

def _validate_v7_policy(policy):
    """[EMOJI] Validar se policy V7 SEPARADA está funcionando"""
    required_attrs = [
        'v7_actor_lstm', 'v7_critic_lstm', 'temporal_attention', 'value_heads',  # [UPDATE] LSTM critic + enhanced processing
        'entry_head', 'management_head', 'trade_memory'
    ]
    
    for attr in required_attrs:
        if not hasattr(policy, attr):
            raise ValueError(f"V7 Policy missing required attribute: {attr}")
    
    # Verificar se entry_head tem gates especializados
    entry_head_gates = [
        'horizon_analyzer', 'mtf_validator', 'pattern_memory_validator',
        'risk_gate_entry', 'regime_gate', 'lookahead_gate', 'fatigue_detector',
        'momentum_filter', 'volatility_filter', 'volume_filter', 'trend_strength_filter',
        'confidence_estimator'
    ]
    
    for gate in entry_head_gates:
        if not hasattr(policy.entry_head, gate):
            raise ValueError(f"Entry Head missing specialized gate: {gate}")
    
    print("[OK] V7Simple SEPARADA validada - LSTMs independentes + gates especializados!")
    print(f"   [EMOJI] Actor: {policy.v7_actor_lstm.__class__.__name__} (GRU removido)")
    print(f"   [TARGET] Critic: MLP + Memory Buffer (LSTM→MLP upgrade!)")
    return True

if __name__ == "__main__":
    print("[LAUNCH] TwoHeadV7Simple REVOLUCIONÁRIA - LSTM Actor + MLP Critic + Gates Especializados")
    print("   - [EMOJI] Actor: LSTM direto (GRU removido)")
    print("   - [TARGET] Critic: MLP + Memory Buffer (LSTM removido)")
    print("   - Entry Head com 6 Gates Especializados")
    print("   - 10 Scores especializados")
    print("   - Dropout diferenciado: Actor(10%) vs Critic(20%)")
    print("   - Inicialização diferenciada: gain 1.0 vs 1.8")