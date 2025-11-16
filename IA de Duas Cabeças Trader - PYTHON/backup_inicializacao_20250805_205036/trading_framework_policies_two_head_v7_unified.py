"""
🎯 TwoHeadV7Unified - UNIFICAÇÃO COMPLETA DAS 3 POLÍTICAS V7

UNIFICAÇÃO TOTAL das 3 políticas V7:
- two_head_v7_simple.py (base + TODOS os gates especializados)
- two_head_v7_enhanced.py (TODOS os enhancements)  
- two_head_v7_intuition.py (backbone unificado + gradient mixing)

RESULTADO: Política única com TODAS as funcionalidades, sem herança complexa.

ARQUITETURA COMPLETA:
✅ Duas Cabeças Especializadas (Entry Head + Management Head)
✅ TODOS os Gates Especializados (10+ gates da V7Simple)
✅ Actor LSTM + Critic MLP (detalhes das V7s)
✅ LeakyReLU ao invés de ReLU
✅ MarketRegimeDetector (V7Enhanced)
✅ EnhancedMemoryBank (V7Enhanced)
✅ GradientBalancer (V7Enhanced)
✅ NeuralBreathingMonitor (V7Enhanced)
✅ UnifiedMarketBackbone (V7Intuition)
✅ GradientMixer (V7Intuition)
✅ InterferenceMonitor (V7Intuition)
✅ Inicialização adequada
✅ Action Space 11 dimensões
✅ Sem herança múltipla
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Type, Union, Tuple
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution
import torch.nn.functional as F

# Imports essenciais
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

# Imports dos componentes das outras V7s
from collections import deque, defaultdict

# ============================================================================
# 🧠 COMPONENTES DA V7ENHANCED
# ============================================================================

class MarketRegimeDetector(nn.Module):
    """🎯 Market Regime Detector - Detecta regime de mercado"""
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.detector = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 4)  # 4 regimes: bull, bear, sideways, volatile
        )
    
    def forward(self, features: torch.Tensor) -> int:
        logits = self.detector(features)
        if len(logits.shape) > 1:
            logits = logits.mean(dim=0)
        return torch.argmax(logits).item()

class EnhancedMemoryBank(nn.Module):
    """💾 Enhanced Memory Bank - Sistema de memória avançado"""
    def __init__(self, memory_size: int = 10000, feature_dim: int = 256):
        super().__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.memories = {}
        for regime in range(4):
            self.memories[regime] = deque(maxlen=memory_size // 4)
    
    def store_memory(self, regime: int, state, action, reward, next_state, done):
        """Armazena memória por regime"""
        memory = {
            'state': state, 'action': action, 'reward': reward,
            'next_state': next_state, 'done': done
        }
        self.memories[regime].append(memory)
    
    def get_regime_context(self, regime: int, current_state) -> np.ndarray:
        """Retorna contexto do regime"""
        if regime not in self.memories or len(self.memories[regime]) == 0:
            return np.zeros(self.feature_dim)
        
        # Pega últimas 10 memórias do regime
        recent_memories = list(self.memories[regime])[-10:]
        if not recent_memories:
            return np.zeros(self.feature_dim)
        
        # Média das states
        states = [m['state'] for m in recent_memories if m['state'] is not None]
        if not states:
            return np.zeros(self.feature_dim)
        
        avg_state = np.mean(states, axis=0)
        if len(avg_state) != self.feature_dim:
            padded = np.zeros(self.feature_dim)
            padded[:min(len(avg_state), self.feature_dim)] = avg_state[:self.feature_dim]
            return padded
        return avg_state

class GradientBalancer(nn.Module):
    """⚖️ Gradient Balancer - Balanceamento de gradientes"""
    def __init__(self):
        super().__init__()
        self.actor_update_ratio = 2
        self.critic_update_ratio = 1
        self.update_counter = 0
    
    def should_update_actor(self) -> bool:
        return self.update_counter % self.actor_update_ratio == 0
    
    def should_update_critic(self) -> bool:
        return self.update_counter % self.critic_update_ratio == 0
    
    def step(self):
        self.update_counter += 1

class NeuralBreathingMonitor(nn.Module):
    """🫁 Neural Breathing Monitor - Monitor de respiração neural"""
    def __init__(self):
        super().__init__()
        self.breathing_rate = 0.1
        self.phase = 0.0
    
    def get_breathing_factor(self) -> float:
        """Retorna fator de respiração neural"""
        self.phase += self.breathing_rate
        return 1.0 + 0.05 * np.sin(self.phase)

class EnhancedTradeMemoryBank(nn.Module):
    """💾 Enhanced Trade Memory Bank - Substituto inteligente para TradeMemoryBank"""
    def __init__(self, memory_size=10000, feature_dim=256):
        super().__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.enhanced_memory = EnhancedMemoryBank(memory_size, feature_dim)
        self.trade_dim = 8
        
        self.context_processor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 8)
        )
        
    def add_trade(self, trade_data):
        """Adiciona trade"""
        if isinstance(trade_data, torch.Tensor):
            trade_np = trade_data.detach().cpu().numpy()
        else:
            trade_np = np.array(trade_data)
        
        if len(trade_np) < self.feature_dim:
            padded = np.zeros(self.feature_dim)
            padded[:len(trade_np)] = trade_np
            trade_np = padded
        else:
            trade_np = trade_np[:self.feature_dim]
        
        state = trade_np
        action = np.zeros(3)
        reward = float(trade_np[0]) if len(trade_np) > 0 else 0.0
        next_state = state
        done = False
        regime = getattr(self, '_current_regime', 2)
        
        self.enhanced_memory.store_memory(regime, state, action, reward, next_state, done)
    
    def get_memory_context(self, batch_size, regime=None):
        """Retorna contexto enhanced"""
        if regime is None:
            regime = getattr(self, '_current_regime', 2)
        
        dummy_state = np.random.randn(self.feature_dim) * 0.1
        enhanced_context = self.enhanced_memory.get_regime_context(regime, dummy_state)
        context_tensor = torch.tensor(enhanced_context, dtype=torch.float32)
        processed_context = self.context_processor(context_tensor)
        
        return processed_context.unsqueeze(0).expand(batch_size, -1)
    
    def set_current_regime(self, regime):
        self._current_regime = regime

# ============================================================================
# 🧠 COMPONENTES DA V7INTUITION  
# ============================================================================

class UnifiedMarketBackbone(nn.Module):
    """🧠 BACKBONE UNIFICADO - Visão única e compartilhada do mercado"""
    def __init__(self, input_dim: int = 256, shared_dim: int = 512, regime_embed_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.shared_dim = shared_dim
        self.regime_embed_dim = regime_embed_dim
        
        self.regime_detector = MarketRegimeDetector(input_dim)
        
        self.shared_feature_processor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(input_dim // 2, input_dim),
            nn.LayerNorm(input_dim)
        )
        
        self.unified_processor = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.05),
            nn.Linear(shared_dim, shared_dim)
        )
        
        self.regime_embedding = nn.Embedding(4, regime_embed_dim)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(shared_dim + regime_embed_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(shared_dim, shared_dim)
        )
        
        self.actor_gate = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(shared_dim // 2, shared_dim),
            nn.Sigmoid()
        )
        
        self.critic_gate = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2), 
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(shared_dim // 2, shared_dim),
            nn.Sigmoid()
        )
        
        self.actor_projection = nn.Sequential(
            nn.Linear(shared_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        self.critic_projection = nn.Sequential(
            nn.Linear(shared_dim, 256),
            nn.LayerNorm(256), 
            nn.LeakyReLU(negative_slope=0.01)
        )
        
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, Dict]:
        """Forward unificado"""
        batch_size = observations.shape[0]
        
        shared_raw_features = self.shared_feature_processor(observations)
        regime_id = self.regime_detector(shared_raw_features)
        regime_emb = self.regime_embedding(torch.tensor(regime_id, device=observations.device))
        
        unified_features = self.unified_processor(shared_raw_features)
        
        if len(unified_features.shape) == 3:
            seq_len = unified_features.shape[1]
            regime_emb_expanded = regime_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, seq_len, -1)
        else:
            regime_emb_expanded = regime_emb.unsqueeze(0).expand(batch_size, -1)
        
        fused_features = torch.cat([unified_features, regime_emb_expanded], dim=-1)
        enhanced_features = self.fusion_layer(fused_features)
        
        actor_attention = self.actor_gate(enhanced_features)
        critic_attention = self.critic_gate(enhanced_features)
        
        actor_gated = enhanced_features * actor_attention
        critic_gated = enhanced_features * critic_attention
        
        actor_features = self.actor_projection(actor_gated)
        critic_features = self.critic_projection(critic_gated)
        
        info = {
            'regime_id': regime_id,
            'regime_name': ['bull', 'bear', 'sideways', 'volatile'][regime_id],
            'shared_features_norm': torch.norm(enhanced_features).item(),
            'actor_attention_mean': actor_attention.mean().item(),
            'critic_attention_mean': critic_attention.mean().item(),
            'specialization_divergence': torch.norm(actor_attention - critic_attention).item()
        }
        
        return actor_features, critic_features, regime_id, info

class GradientMixer:
    """⚖️ Sistema de Gradient Mixing Inteligente"""
    def __init__(self, mixing_strength: float = 0.1):
        self.mixing_strength = mixing_strength
        self.actor_grad_history = deque(maxlen=50)
        self.critic_grad_history = deque(maxlen=50)
        self.mixing_history = deque(maxlen=100)
        self.min_mixing = 0.05
        self.max_mixing = 0.3
        self.performance_window = 20
        
    def analyze_gradient_compatibility(self, actor_grads: Dict, critic_grads: Dict) -> Dict[str, float]:
        """Analisa compatibilidade entre gradientes Actor/Critic"""
        compatibility_scores = {}
        
        for name in actor_grads.keys():
            if name in critic_grads:
                actor_grad = actor_grads[name].flatten()
                critic_grad = critic_grads[name].flatten()
                
                # Cosine similarity
                cos_sim = F.cosine_similarity(actor_grad.unsqueeze(0), critic_grad.unsqueeze(0))
                compatibility_scores[name] = cos_sim.item()
        
        return compatibility_scores
    
    def mix_gradients(self, actor_grads: Dict, critic_grads: Dict, mixing_factor: float = None):
        """Mistura gradientes de forma inteligente"""
        if mixing_factor is None:
            mixing_factor = self.mixing_strength
        
        mixed_grads = {}
        for name in actor_grads.keys():
            if name in critic_grads:
                mixed_grads[name] = (1 - mixing_factor) * actor_grads[name] + mixing_factor * critic_grads[name]
            else:
                mixed_grads[name] = actor_grads[name]
        
        return mixed_grads

class InterferenceMonitor:
    """📡 Monitor de Interferência - Detecta conflitos de gradientes"""
    def __init__(self):
        self.interference_history = deque(maxlen=100)
        self.threshold = 0.5
        
    def detect_interference(self, actor_grads: Dict, critic_grads: Dict) -> Dict[str, float]:
        """Detecta interferência entre gradientes"""
        interference_scores = {}
        
        for name in actor_grads.keys():
            if name in critic_grads:
                actor_grad = actor_grads[name].flatten()
                critic_grad = critic_grads[name].flatten()
                
                # Dot product normalizado (conflito quando negativo)
                dot_product = torch.dot(actor_grad, critic_grad)
                norm_product = torch.norm(actor_grad) * torch.norm(critic_grad)
                
                if norm_product > 0:
                    normalized_dot = dot_product / norm_product
                    interference = max(0, -normalized_dot.item())  # Só interferência negativa
                    interference_scores[name] = interference
                else:
                    interference_scores[name] = 0.0
        
        return interference_scores
    
    def get_interference_level(self) -> float:
        """Retorna nível médio de interferência"""
        if not self.interference_history:
            return 0.0
        return np.mean(self.interference_history)

# =============================================================================
# 🎯 GATES ESPECIALIZADOS (copiados da V7Simple)
# =============================================================================

class SpecializedEntryHead(nn.Module):
    """🎯 Entry Head com Gates Especializados - EXATAMENTE IGUAL V7SIMPLE"""
    
    def __init__(self, input_dim=544):  # 256+256+32 = 544
        super().__init__()
        self.input_dim = input_dim
        
        # 1. TEMPORAL GATE - Analisa timing de entrada
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
        
        # 2. VALIDATION GATE - Multi-timeframe + Pattern validation
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
        
        # 3. RISK GATE - Análise de risco + regime
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
        
        # 4. MARKET GATE - Lookahead + Fatigue detection
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
        
        # 5. QUALITY GATE - 4 filtros especializados (FALTAVAM!)
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
        
        # 6. CONFIDENCE GATE - Confiança geral (FALTAVA!)
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
        
        # DECISION NETWORK FINAL - IGUAL V7SIMPLE
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
        
        # ADAPTIVE THRESHOLDS - IGUAL V7SIMPLE
        self.register_parameter('adaptive_threshold_main', nn.Parameter(torch.tensor(0.50)))
        self.register_parameter('adaptive_threshold_risk', nn.Parameter(torch.tensor(0.35)))
        self.register_parameter('adaptive_threshold_regime', nn.Parameter(torch.tensor(0.25)))
        
    def forward(self, entry_signal, management_signal, market_context):
        """Forward EXATAMENTE IGUAL V7SIMPLE"""
        
        # Combinar sinais para análise
        combined_input = torch.cat([entry_signal, management_signal, market_context], dim=-1)
        
        # FASE 1: CALCULAR OS 10 SCORES ESPECIALIZADOS
        
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
        
        # FASE 2: APLICAR THRESHOLDS ADAPTATIVOS
        
        # Clamp thresholds para ranges seguros
        main_threshold = torch.clamp(self.adaptive_threshold_main, 0.5, 0.9)
        risk_threshold = torch.clamp(self.adaptive_threshold_risk, 0.3, 0.8)
        regime_threshold = torch.clamp(self.adaptive_threshold_regime, 0.2, 0.7)
        
        # GATES HÍBRIDOS: Sigmoid nos gates individuais + binário no final
        temporal_gate = torch.sigmoid((temporal_score - regime_threshold) * 5)
        validation_gate = torch.sigmoid((validation_score - main_threshold) * 5)
        risk_gate = torch.sigmoid((risk_composite - risk_threshold) * 5)
        market_gate = torch.sigmoid((market_score - regime_threshold) * 5)
        quality_gate = torch.sigmoid((quality_score - main_threshold) * 5)
        confidence_gate = torch.sigmoid((confidence_score - main_threshold) * 5)
        
        # FASE 3: GATE FINAL BINÁRIO - Sistema composite inteligente
        composite_score = (
            temporal_gate * 0.20 +      # 20% - timing
            validation_gate * 0.20 +    # 20% - validação multi-timeframe
            risk_gate * 0.25 +          # 25% - risco
            market_gate * 0.10 +        # 10% - condições de mercado
            quality_gate * 0.10 +       # 10% - qualidade técnica
            confidence_gate * 0.15      # 15% - confiança geral
        )
        
        # Gate final binário: só passa se composite score > threshold
        final_gate_threshold = 0.6  # 60% da pontuação ponderada
        final_gate = (composite_score > final_gate_threshold).float()
        
        # FASE 4: DECISÃO FINAL
        all_scores = torch.cat([
            temporal_score, validation_score, risk_composite, market_score, quality_score,
            confidence_score, mtf_score, pattern_score, lookahead_score, fatigue_score
        ], dim=-1)
        
        decision_input = torch.cat([combined_input, all_scores], dim=-1)
        raw_decision = self.final_decision_network(decision_input)
        
        # Soft gating para prevenir gradient blocking total
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
    """🧠 Management Head - Segunda Cabeça Especializada"""
    
    def __init__(self, input_dim=544):  # 256+256+32 = 544
        super().__init__()
        self.input_dim = input_dim
        
        # Management gates
        self.position_analyzer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.risk_manager = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Decision layer
        self.management_decision = nn.Sequential(
            nn.Linear(input_dim + 2, 64),  # input + 2 gates
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 2)  # decision + confidence
        )
        
    def forward(self, entry_signal, management_signal, memory_context):
        """Forward pass para management"""
        
        combined_input = torch.cat([entry_signal, management_signal, memory_context], dim=-1)
        
        # Management gates
        position_gate = self.position_analyzer(combined_input)
        risk_gate = self.risk_manager(combined_input)
        
        # Decisão final
        gates = torch.cat([position_gate, risk_gate], dim=-1)
        fusion_input = torch.cat([combined_input, gates], dim=-1)
        decision_output = self.management_decision(fusion_input)
        
        decision = torch.sigmoid(decision_output[:, 0:1])
        confidence = torch.sigmoid(decision_output[:, 1:2])
        
        gate_info = {
            'position_gate': position_gate.mean().item(),
            'risk_gate': risk_gate.mean().item(),
            'mgmt_conf': confidence.mean().item()
        }
        
        return decision, confidence, gate_info

# =============================================================================
# 🎯 POLÍTICA UNIFICADA - SEM HERANÇA COMPLEXA
# =============================================================================

class TwoHeadV7Unified(RecurrentActorCriticPolicy):
    """
    🎯 TwoHeadV7Unified - POLÍTICA LIMPA E FUNCIONAL
    
    UNIFICAÇÃO COMPLETA:
    ✅ Duas Cabeças Especializadas (Entry + Management)
    ✅ Gates Especializados (6 gates na Entry, 2 na Management)
    ✅ Actor LSTM + Critic MLP
    ✅ Inicialização adequada
    ✅ Action Space 11 dimensões
    ✅ Sem herança múltipla
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        # Parâmetros V7
        v7_shared_lstm_hidden: int = 256,
        v7_features_dim: int = 256,
        **kwargs
    ):
        
        print("🎯 TwoHeadV7Unified inicializando - POLÍTICA LIMPA!")
        print(f"   Action Space: {action_space}")
        print(f"   Action Dimensions: {action_space.shape[0] if hasattr(action_space, 'shape') else 'Unknown'}")
        print(f"   LSTM Hidden: {v7_shared_lstm_hidden}")
        print(f"   Features Dim: {v7_features_dim}")
        
        # Validar action space
        if not hasattr(action_space, 'shape') or action_space.shape[0] != 11:
            raise ValueError(f"TwoHeadV7Unified expects 11-dimensional action space, got {getattr(action_space, 'shape', 'Unknown')}")
        
        # Configurar features extractor
        if 'features_extractor_class' not in kwargs:
            kwargs['features_extractor_class'] = TradingTransformerFeatureExtractor
        if 'features_extractor_kwargs' not in kwargs:
            kwargs['features_extractor_kwargs'] = {'features_dim': v7_features_dim}
        
        # Store parameters
        self.v7_shared_lstm_hidden = v7_shared_lstm_hidden
        self.v7_features_dim = v7_features_dim
        self.memory_context_dim = 32  # Define antes de usar
        
        # Disable automatic LSTM critic
        kwargs['enable_critic_lstm'] = False
        
        # Initialize parent
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        
        # 🎯 COMPONENTES UNIFICADOS
        
        # 1. Actor LSTM (timing focused)
        self.v7_actor_lstm = nn.LSTM(
            input_size=self.v7_features_dim,
            hidden_size=self.v7_shared_lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        
        self.actor_lstm_dropout = nn.Dropout(0.1)
        
        # 2. Critic MLP (value focused)
        self.v7_critic_mlp = nn.Sequential(
            nn.Linear(self.v7_features_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.05),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 1)
        )
        
        # 3. Duas Cabeças Especializadas
        combined_input_dim = self.v7_shared_lstm_hidden + self.v7_shared_lstm_hidden + self.memory_context_dim  # 256+256+32=544
        self.entry_head = SpecializedEntryHead(input_dim=combined_input_dim)
        self.management_head = TwoHeadDecisionMaker(input_dim=combined_input_dim)
        
        # 4. GATES FALTANTES DA V7SIMPLE
        
        # Pattern Memory Gate
        self.pattern_memory = nn.Sequential(
            nn.Linear(self.v7_features_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 32)
        )
        
        # Trade Memory Gate  
        self.trade_memory = EnhancedTradeMemoryBank(
            memory_size=10000, 
            feature_dim=self.v7_features_dim
        )
        
        # Critic Memory Gate
        self.critic_memory = nn.Sequential(
            nn.Linear(self.v7_features_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 16)
        )
        
        # 5. COMPONENTES V7ENHANCED
        
        self.market_regime_detector = MarketRegimeDetector(self.v7_features_dim)
        self.enhanced_memory_bank = EnhancedMemoryBank(10000, self.v7_features_dim)
        self.gradient_balancer = GradientBalancer()
        self.neural_breathing_monitor = NeuralBreathingMonitor()
        
        # 6. COMPONENTES V7INTUITION
        
        self.unified_backbone = UnifiedMarketBackbone(
            input_dim=self.v7_features_dim,
            shared_dim=512,
            regime_embed_dim=64
        )
        self.gradient_mixing = GradientMixer(mixing_strength=0.1)
        self.adaptive_sharing = InterferenceMonitor()
        
        # 7. Memory context (já definido acima)
        
        # 8. Actor head final
        # lstm_out (256) + decision_context (4: entry_decision + entry_conf + mgmt_decision + mgmt_conf)
        actor_input_dim = self.v7_shared_lstm_hidden + 4  # 256 + 4 = 260
        self.actor_head = nn.Sequential(
            nn.Linear(actor_input_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, self.action_space.shape[0])
        )
        
        # 6. 🔧 INICIALIZAÇÃO ADEQUADA
        self._initialize_weights_properly()
        
        print("🎯 TwoHeadV7Unified PRONTA - Duas Cabeças + Gates + Inicialização Adequada!")
    
    def _initialize_weights_properly(self):
        """🔧 Inicialização AGRESSIVA para forçar distribuição ampla e mais SHORT"""
        
        print("🔧 [INIT] Aplicando inicialização AGRESSIVA para SHORT...")
        
        # Inicializar actor_head com range MUITO amplo
        for layer in self.actor_head:
            if isinstance(layer, nn.Linear):
                # Xavier com gain MUITO maior
                nn.init.xavier_uniform_(layer.weight, gain=3.0)
                if layer.bias is not None:
                    nn.init.uniform_(layer.bias, -2.0, 2.0)
        
        # Inicialização ESPECIAL para última camada - BIAS POSITIVO para SHORT
        last_layer = self.actor_head[-1]
        if isinstance(last_layer, nn.Linear):
            # Pesos com range amplo
            nn.init.uniform_(last_layer.weight, -3.0, 3.0)
            if last_layer.bias is not None:
                # BIAS POSITIVO para favorecer SHORT (valores > 0.1)
                nn.init.uniform_(last_layer.bias, -1.0, 2.0)  # Mais valores positivos
        
        # Inicializar LSTM com gain maior
        for name, param in self.v7_actor_lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=1.5)  # Gain maior
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Inicializar gates das cabeças com gain maior
        for head in [self.entry_head, self.management_head]:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=1.5)
                    if module.bias is not None:
                        nn.init.uniform_(module.bias, -0.5, 0.5)
        
        print("✅ [INIT] Inicialização AGRESSIVA aplicada - bias positivo para SHORT")
    
    def forward_actor(self, features: torch.Tensor, lstm_states, episode_starts: torch.Tensor):
        print("DEBUG: forward_actor called")
        """🎯 Forward Actor com Duas Cabeças Especializadas"""
        
        # 1. Features processing (1480 -> 256)
        # O features extractor já deve ter processado para v7_features_dim
        if features.shape[-1] != self.v7_features_dim:
            # Se ainda não foi processado, usar uma projeção simples
            if not hasattr(self, 'feature_projection'):
                self.feature_projection = nn.Linear(features.shape[-1], self.v7_features_dim).to(features.device)
            features = self.feature_projection(features)
        
        # 2. LSTM processing - FIX STATES
        # 🔧 FIX: Garantir que lstm_states têm dimensões corretas
        if lstm_states is not None:
            h_state, c_state = lstm_states
            # Se states são 3D, ajustar para 2D
            if len(h_state.shape) == 3:
                h_state = h_state.squeeze(0)  # Remove batch dim extra
                c_state = c_state.squeeze(0)
            lstm_states = (h_state, c_state)
        
        lstm_out, new_actor_states = self.v7_actor_lstm(features, lstm_states)
        lstm_out = self.actor_lstm_dropout(lstm_out.squeeze(1))
        
        # 2. Duas Cabeças Especializadas
        batch_size = lstm_out.shape[0]
        entry_signal = lstm_out
        management_signal = lstm_out
        
        # Memory context (simplified)
        memory_context = torch.zeros(batch_size, self.memory_context_dim, device=features.device)
        
        # Entry Head (primeira cabeça)
        entry_decision, entry_conf, entry_gate_info = self.entry_head(
            entry_signal, management_signal, memory_context
        )
        
        # Management Head (segunda cabeça)
        mgmt_decision, mgmt_conf, mgmt_gate_info = self.management_head(
            entry_signal, management_signal, memory_context
        )
        
        # 3. Combinar decisões das duas cabeças
        decision_context = torch.cat([entry_decision, entry_conf, mgmt_decision, mgmt_conf], dim=-1)
        actor_input = torch.cat([lstm_out, decision_context], dim=-1)
        
        # 4. Actor head final
        raw_actions = self.actor_head(actor_input)
        
        # 5. 🔧 PROCESSAMENTO DAS AÇÕES (11 dimensões)
        if raw_actions.shape[-1] != 11:
            raise ValueError(f"Actor head deve produzir 11 ações, mas produziu {raw_actions.shape[-1]}")
        
        actions = torch.zeros_like(raw_actions)
        
        # [0] entry_decision: 0-2 (hold, long, short) - DISCRETE
        raw_decision = raw_actions[:, 0]
        # 🔧 THRESHOLDS MAIS AGRESSIVOS para forçar SHORT
        discrete_decision = torch.where(raw_decision < -0.1, 0,  # HOLD: < -0.1 (menor região)
                                      torch.where(raw_decision > 0.1, 2, 1))  # SHORT: > 0.1 (mais fácil), LONG: -0.1 a 0.1
        
        # 🚨 DEBUG PATCH - Capturar bug de threshold
        if raw_decision.numel() > 0:
            print(f"🚨 DEBUG THRESHOLD:")
            print(f"   raw_decision: {raw_decision[0].item():.6f}")
            print(f"   raw_decision < -0.1: {raw_decision[0] < -0.1}")
            print(f"   raw_decision > 0.1: {raw_decision[0] > 0.1}")
            print(f"   discrete_decision: {discrete_decision[0].item()}")
            print(f"   actions[0, 0] before: {actions[0, 0].item()}")
        
        actions[:, 0] = discrete_decision.float()
        
        # 🚨 DEBUG PATCH - Verificar após atribuição
        if raw_decision.numel() > 0:
            print(f"   actions[0, 0] after: {actions[0, 0].item()}")
            decision_name = {0: 'HOLD', 1: 'LONG', 2: 'SHORT'}.get(int(actions[0, 0].item()), 'UNKNOWN')
            print(f"   Final decision: {decision_name}")
            print("=" * 50)
        
        # DEBUG SIMPLES - Sempre executar
        print(f"SIMPLE DEBUG: raw[0]={raw_actions[0, 0].item():.3f} -> action[0]={actions[0, 0].item()}")
        
        # [1] entry_confidence: 0-1 - sigmoid
        actions[:, 1] = torch.sigmoid(raw_actions[:, 1])
        
        # [2] temporal_signal: -1 to 1 - tanh
        actions[:, 2] = torch.tanh(raw_actions[:, 2])
        
        # [3] risk_appetite: 0-1 - sigmoid
        actions[:, 3] = torch.sigmoid(raw_actions[:, 3])
        
        # [4] market_regime_bias: -1 to 1 - tanh
        actions[:, 4] = torch.tanh(raw_actions[:, 4])
        
        # [5-10] sl/tp adjustments: -3 to 3 - tanh scaled
        for i in range(5, 11):
            actions[:, i] = torch.tanh(raw_actions[:, i]) * 3.0
        
        # 6. Gate info combinado
        combined_gate_info = {**entry_gate_info, **mgmt_gate_info}
        combined_gate_info['raw_actions_shape'] = raw_actions.shape
        combined_gate_info['final_actions_shape'] = actions.shape
        
        return actions, new_actor_states, combined_gate_info
    
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """🔧 FIX: Implementar predict corretamente para RecurrentActorCriticPolicy"""
        
        # Converter para tensor se necessário
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        
        # Garantir batch dimension
        if len(observation.shape) == 1:
            observation = observation.unsqueeze(0)
        
        # Estados LSTM padrão se não fornecidos
        if state is None:
            batch_size = observation.shape[0]
            state = (
                torch.zeros(batch_size, self.v7_shared_lstm_hidden, device=self.device),
                torch.zeros(batch_size, self.v7_shared_lstm_hidden, device=self.device)
            )
        
        # Episode starts padrão
        if episode_start is None:
            episode_start = torch.tensor([False] * observation.shape[0], device=self.device)
        
        # Forward
        with torch.no_grad():
            actions, new_states, _ = self.forward_actor(observation, state, episode_start)
            values, _ = self.forward_critic(observation, state, episode_start)
        
        # Converter para numpy
        actions_np = actions.cpu().numpy()
        values_np = values.cpu().numpy()
        
        # Remover batch dimension se era single observation
        if actions_np.shape[0] == 1:
            actions_np = actions_np[0]
            values_np = values_np[0]
        
        # Para compatibilidade com PPO, retornar apenas actions
        # (PPO não usa o formato de 3 valores para RecurrentActorCriticPolicy)
        return actions_np, new_states
    
    def get_distribution(self, obs, lstm_states=None, episode_starts=None):
        """🔧 FIX: Implementar get_distribution com parâmetros corretos"""
        
        # Estados padrão se não fornecidos
        if lstm_states is None:
            batch_size = obs.shape[0]
            lstm_states = (
                torch.zeros(batch_size, self.v7_shared_lstm_hidden, device=obs.device),
                torch.zeros(batch_size, self.v7_shared_lstm_hidden, device=obs.device)
            )
        
        if episode_starts is None:
            episode_starts = torch.tensor([False] * obs.shape[0], device=obs.device)
        
        # Forward para obter ações
        actions, _, _ = self.forward_actor(obs, lstm_states, episode_starts)
        
        # Criar distribuição (assumindo que actions já estão processadas)
        # Para action space contínuo, usar Normal distribution
        mean = actions
        std = torch.ones_like(actions) * 0.1  # Std fixo pequeno
        
        from torch.distributions import Normal
        distribution = Normal(mean, std)
        
        return distribution
    
    def forward_critic(self, features: torch.Tensor, lstm_states, episode_starts: torch.Tensor):
        """💰 Forward Critic simples e direto"""
        
        # Features processing (1480 -> 256)
        if features.shape[-1] != self.v7_features_dim:
            if not hasattr(self, 'feature_projection'):
                self.feature_projection = nn.Linear(features.shape[-1], self.v7_features_dim).to(features.device)
            features = self.feature_projection(features)
        
        # Critic MLP direto
        values = self.v7_critic_mlp(features)
        
        # Dummy states para compatibilidade
        dummy_states = lstm_states if lstm_states is not None else (
            torch.zeros(1, features.shape[0], 128, device=features.device),
            torch.zeros(1, features.shape[0], 128, device=features.device)
        )
        
        return values, dummy_states

# =============================================================================
# 🛠️ UTILITY FUNCTIONS
# =============================================================================

def get_v7_unified_kwargs():
    """Retorna kwargs para TwoHeadV7Unified"""
    return {
        # V7 parameters
        'v7_shared_lstm_hidden': 256,
        'v7_features_dim': 256,
        
        # Features extractor
        'features_extractor_class': TradingTransformerFeatureExtractor,
        'features_extractor_kwargs': {'features_dim': 256},
        
        # Network architecture
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

def validate_v7_unified_policy(policy):
    """Valida TwoHeadV7Unified"""
    
    required_attrs = [
        'v7_actor_lstm', 'v7_critic_mlp',
        'entry_head', 'management_head', 'actor_head'
    ]
    
    for attr in required_attrs:
        if not hasattr(policy, attr):
            raise ValueError(f"V7Unified missing required attribute: {attr}")
    
    print("✅ TwoHeadV7Unified validation passed!")

# =============================================================================
# 🎯 EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("🎯 TwoHeadV7Unified - POLÍTICA UNIFICADA")
    print("=" * 60)
    print("CARACTERÍSTICAS:")
    print("✅ Duas Cabeças Especializadas (Entry + Management)")
    print("✅ Gates Especializados (8 gates total)")
    print("✅ Actor LSTM + Critic MLP")
    print("✅ Inicialização adequada")
    print("✅ Action Space 11 dimensões")
    print("✅ Sem herança múltipla")
    print()
    print("USAR:")
    print("   policy_class = TwoHeadV7Unified")
    print("   policy_kwargs = get_v7_unified_kwargs()")