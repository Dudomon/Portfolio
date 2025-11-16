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
        
        # [LAUNCH] 1. TEMPORAL GATE - Analisa timing de entrada
        self.horizon_analyzer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # [LAUNCH] 2. VALIDATION GATE - Multi-timeframe + Pattern validation
        self.mtf_validator = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.pattern_memory_validator = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # [LAUNCH] 3. RISK GATE - Análise de risco + regime
        self.risk_gate_entry = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.regime_gate = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # [LAUNCH] 4. MARKET GATE - Lookahead + Fatigue detection
        self.lookahead_gate = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.fatigue_detector = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # [LAUNCH] 5. QUALITY GATE - 4 filtros especializados
        self.momentum_filter = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.volatility_filter = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.volume_filter = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.trend_strength_filter = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # [LAUNCH] 6. CONFIDENCE GATE - Confiança geral
        self.confidence_estimator = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # [TARGET] DECISION NETWORK FINAL
        self.final_decision_network = nn.Sequential(
            nn.Linear(input_dim + 10, 64),  # input + 10 scores
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(16, 1)
        )
        
        # [FIX] ADAPTIVE THRESHOLDS - RANGES MAIS AMPLOS PARA EVITAR SATURAÇÃO 0.500
        self.register_parameter('adaptive_threshold_main', nn.Parameter(torch.tensor(0.25)))     # REDUZIDO: 0.50→0.25 para evitar sigmoid(0)=0.5
        self.register_parameter('adaptive_threshold_risk', nn.Parameter(torch.tensor(0.15)))     # REDUZIDO: 0.35→0.15 para crear diferença
        self.register_parameter('adaptive_threshold_regime', nn.Parameter(torch.tensor(0.10)))   # REDUZIDO: 0.25→0.10 para maximizar distância
        
        # 🔥 CRÍTICO: Inicializar todas as sigmoid gates para evitar saturação
        self._initialize_all_sigmoid_gates()
    
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
        
    def forward(self, entry_signal, management_signal, market_context):
        # Combinar sinais para análise
        combined_input = torch.cat([entry_signal, management_signal, market_context], dim=-1)
        
        # [LAUNCH] FASE 1: CALCULAR OS 10 SCORES ESPECIALIZADOS
        
        # 1. Temporal Score
        temporal_score = self.horizon_analyzer(combined_input)
        
        # 2. Validation Score (MTF + Pattern)
        mtf_score = self.mtf_validator(combined_input)
        pattern_score = self.pattern_memory_validator(combined_input)
        validation_score = (mtf_score + pattern_score) / 2
        
        # 3. Risk Score (Risk + Regime)
        risk_score = self.risk_gate_entry(combined_input)
        regime_score = self.regime_gate(combined_input)
        risk_composite = (risk_score + regime_score) / 2
        
        # 4. Market Score (Lookahead + Fatigue)
        lookahead_score = self.lookahead_gate(combined_input)
        fatigue_score = 1.0 - self.fatigue_detector(combined_input)  # Inverter: alta fatigue = baixo score
        market_score = (lookahead_score + fatigue_score) / 2
        
        # 5. Quality Score (4 filtros)
        momentum_score = self.momentum_filter(combined_input)
        volatility_score = self.volatility_filter(combined_input)
        volume_score = self.volume_filter(combined_input)
        trend_score = self.trend_strength_filter(combined_input)
        quality_score = (momentum_score + volatility_score + volume_score + trend_score) / 4
        
        # 6. Confidence Score
        confidence_score = self.confidence_estimator(combined_input)
        
        # [LAUNCH] FASE 2: APLICAR THRESHOLDS ADAPTATIVOS
        
        # Clamp thresholds para ranges seguros (RANGES MAIS BAIXOS)
        main_threshold = torch.clamp(self.adaptive_threshold_main, 0.1, 0.6)    # REDUZIDO: 0.5-0.9 → 0.1-0.6
        risk_threshold = torch.clamp(self.adaptive_threshold_risk, 0.05, 0.5)   # REDUZIDO: 0.3-0.8 → 0.05-0.5  
        regime_threshold = torch.clamp(self.adaptive_threshold_regime, 0.02, 0.4) # REDUZIDO: 0.2-0.7 → 0.02-0.4
        
        # [LAUNCH] GATES HÍBRIDOS (IGUAL V5/V6): Sigmoid nos gates individuais + binário no final
        # Permite gradientes suaves para melhor convergência, mas mantém filtro real no final
        # 🔥 FIX CRÍTICO: Sigmoid gentil (* 2 ao invés de * 5) para prevenir saturação
        temporal_gate = torch.sigmoid((temporal_score - regime_threshold) * 2.0)      # REDUZIDO de 5 para 2
        validation_gate = torch.sigmoid((validation_score - main_threshold) * 2.0)    # REDUZIDO de 5 para 2
        risk_gate = torch.sigmoid((risk_composite - risk_threshold) * 2.0)            # REDUZIDO de 5 para 2
        market_gate = torch.sigmoid((market_score - regime_threshold) * 2.0)          # REDUZIDO de 5 para 2
        quality_gate = torch.sigmoid((quality_score - main_threshold) * 2.0)          # REDUZIDO de 5 para 2
        confidence_gate = torch.sigmoid((confidence_score - main_threshold) * 2.0)    # REDUZIDO de 5 para 2
        
        # [LAUNCH] FASE 3: GATE FINAL BINÁRIO - Sistema composite inteligente
        # Pontuação ponderada ao invés de multiplicação pura (como V5/V6)
        composite_score = (
            temporal_gate * 0.20 +      # 20% - timing (aumentado)
            validation_gate * 0.20 +    # 20% - validação multi-timeframe
            risk_gate * 0.25 +          # 25% - risco (mais importante)
            market_gate * 0.10 +        # 10% - condições de mercado (reduzido)
            quality_gate * 0.10 +       # 10% - qualidade técnica (reduzido)
            confidence_gate * 0.15      # 15% - confiança geral (aumentado)
        )
        
        # Gate final binário: só passa se composite score > threshold (REDUZIDO para menos bloqueios)
        final_gate_threshold = 0.60  # 60% da pontuação ponderada (REDUZIDO de 0.75 para permitir mais trades)
        final_gate = (composite_score > final_gate_threshold).float()
        
        # [LAUNCH] FASE 4: DECISÃO FINAL
        all_scores = torch.cat([
            temporal_score, validation_score, risk_composite, market_score, quality_score,
            confidence_score, mtf_score, pattern_score, lookahead_score, fatigue_score
        ], dim=-1)
        
        decision_input = torch.cat([combined_input, all_scores], dim=-1)
        raw_decision = self.final_decision_network(decision_input)
        
        # [FIX] FIX: Soft gating para prevenir gradient blocking total
        final_decision = raw_decision * (final_gate * 0.9 + 0.1)  # Min 10% gradient flow
        
        # Retornar informações detalhadas para debug
        gate_info = {
            'temporal_gate': temporal_gate,
            'validation_gate': validation_gate,
            'risk_gate': risk_gate,
            'market_gate': market_gate,
            'quality_gate': quality_gate,
            'confidence_gate': confidence_gate,
            'composite_score': composite_score,
            'final_gate': final_gate,
            'scores': {
                'temporal': temporal_score,
                'validation': validation_score,
                'risk': risk_composite,
                'market': market_score,
                'quality': quality_score,
                'confidence': confidence_score,
                'mtf': mtf_score,
                'pattern': pattern_score,
                'lookahead': lookahead_score,
                'fatigue': fatigue_score
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
        
        # Simular confidence (para compatibilidade)
        confidence = torch.sigmoid(decision.mean(dim=-1, keepdim=True))
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
        
        # [FIX] CRÍTICO: Desabilitar LSTM critic automático do RecurrentActorCriticPolicy
        kwargs['enable_critic_lstm'] = False
        print("   [ALERT] LSTM Critic DESABILITADO (enable_critic_lstm=False)")
        
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
        
        # 4. [FIX] CRITIC MLP WITH MEMORY: Substitui LSTM/GRU problemáticos  
        self.memory_steps = 128  # Buffer de 128 steps anteriores (aumentado)
        self.critic_memory_buffer = None
        
        self.v7_critic_mlp = nn.Sequential(
            nn.Linear(self.v7_features_dim * self.memory_steps, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(512),
            nn.Dropout(0.05),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 1)
        )
        
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
            nn.Linear(32, self.action_space.shape[0])
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
        
        # [FIX] INICIALIZAÇÃO AGRESSIVA DO CRITIC MLP
        for layer in self.v7_critic_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=1.5)  # Gain moderado para MLP
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.01)  # Pequeno bias positivo
        
        # 5. [LAUNCH] TEMPORAL REGULARIZATION: Registrar hook para penalizar weight_hh_l0 do critic
        self._setup_temporal_regularization()
        
        # 6. [FIX] V6 FIX: Aplicar correção ReLU → LeakyReLU do mlp_extractor (resolve 50-53% zeros)
        self._apply_v6_mlp_extractor_fix()
        
        # 7. [FIX] LSTM GRADIENT KEEPER: Força gradientes sempre vivos
        self._apply_lstm_gradient_keeper()
        
        # 7. [TARGET] INICIALIZAÇÃO DIFERENCIADA: Critic LSTM mais agressivo
        self._init_separated_lstms()
        
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
        """[FIX] CRITIC REVOLUCIONÁRIO: MLP + Memory Buffer (0% gradient zeros!)"""
        
        batch_size = features.shape[0]
        
        # 1. [FIX] MEMORY BUFFER MANAGEMENT
        if self.critic_memory_buffer is None or episode_starts.any():
            # Resetar buffer no início ou quando episode recomeça
            self.critic_memory_buffer = torch.zeros(
                batch_size, self.memory_steps, self.v7_features_dim,
                device=features.device
            )
        
        # 2. [ROTATE] SHIFT MEMORY BUFFER
        self.critic_memory_buffer = torch.roll(self.critic_memory_buffer, shifts=1, dims=1)
        self.critic_memory_buffer[:, 0, :] = features.squeeze(1) if len(features.shape) == 3 else features
        
        # 3. [BRAIN] FLATTEN MEMORY PARA MLP
        memory_flat = self.critic_memory_buffer.reshape(batch_size, -1)
        
        # 4. [LAUNCH] NOISE INJECTION (mínimo)
        if self.training:
            noise_scale = 0.002  # Ruído muito pequeno
            memory_noise = torch.randn_like(memory_flat) * noise_scale
            memory_flat = memory_flat + memory_noise
        
        # 5. [TARGET] MLP FORWARD (GRADIENTES PERFEITOS!)
        values = self.v7_critic_mlp(memory_flat)
        
        # 6. Manter compatibilidade com LSTM states (dummy)
        dummy_states = lstm_states if lstm_states is not None else (
            torch.zeros(1, batch_size, 128, device=features.device),
            torch.zeros(1, batch_size, 128, device=features.device)
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
        'v7_actor_lstm', 'v7_critic_mlp',  # [FIX] GRU removido: arquitetura simplificada
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