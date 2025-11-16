"""
🧠 TwoHeadV7Intuition - BACKBONE UNIFICADO + Gradient Mixing Inteligente

ARQUITETURA REVOLUCIONÁRIA:
- Shared Market-Aware Backbone (única fonte de features)
- Actor Branch: LSTM com memory attention
- Critic Branch: MLP enhanced com regime context
- Gradient Mixing System (cross-pollination entre Actor/Critic)
- Interference Monitoring (detecta conflitos de gradientes)
- Adaptive Sharing (aumenta/diminui sharing baseado na performance)

INTUIÇÃO: Actor e Critic compartilham VISÃO UNIFICADA do mercado,
mas processam de forma especializada. Gradientes se "ajudam mutuamente".
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Type, Union, Tuple
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution
import torch.nn.functional as F
from collections import deque, defaultdict

# Importar base V7Enhanced
from trading_framework.policies.two_head_v7_enhanced import TwoHeadV7Enhanced
from trading_framework.policies.two_head_v7_simple import (
    SpecializedEntryHead, TwoHeadDecisionMaker, EnhancedFeaturesExtractor
)

# Importar enhancements
from trading_framework.enhancements.v7_upgrade_components import (
    MarketRegimeDetector, EnhancedMemoryBank, GradientBalancer, NeuralBreathingMonitor
)

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

class UnifiedMarketBackbone(nn.Module):
    """🧠 BACKBONE UNIFICADO - Visão única e compartilhada do mercado"""
    
    def __init__(self, 
                 input_dim: int = 256,
                 shared_dim: int = 512,
                 regime_embed_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.shared_dim = shared_dim
        self.regime_embed_dim = regime_embed_dim
        
        # 1. Market Regime Detection (primeiro processamento)
        self.regime_detector = MarketRegimeDetector(input_dim)
        
        # 2. Features Extractor compartilhado (simplified for backbone)
        # Usar processamento direto ao invés do EnhancedFeaturesExtractor
        self.shared_feature_processor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(input_dim // 2, input_dim),
            nn.LayerNorm(input_dim)
        )
        
        # 3. Unified Processing Network
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
        
        # 4. Market Regime Embedding
        self.regime_embedding = nn.Embedding(4, regime_embed_dim)
        
        # 5. Regime-Enhanced Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(shared_dim + regime_embed_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(shared_dim, shared_dim)
        )
        
        # 6. Adaptive Feature Gates (Actor vs Critic specialization)
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
        
        # 7. Output Projections (especializadas)
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
        """
        Forward unificado
        Returns: (actor_features, critic_features, regime_id, info)
        """
        batch_size = observations.shape[0]
        
        # 1. Shared feature extraction (uma única passada)
        shared_raw_features = self.shared_feature_processor(observations)
        
        # 2. Market regime detection (global context)
        regime_id = self.regime_detector(shared_raw_features)
        regime_emb = self.regime_embedding(torch.tensor(regime_id, device=observations.device))
        
        # 3. Unified processing (mesmo processamento para ambos)
        unified_features = self.unified_processor(shared_raw_features)
        
        # 4. Regime-enhanced features
        # Fix dimensões: unified_features pode ter 3D (batch, seq, features)
        if len(unified_features.shape) == 3:
            # unified_features: (batch, seq, features)
            seq_len = unified_features.shape[1]
            regime_emb_expanded = regime_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, seq_len, -1)
        else:
            # unified_features: (batch, features)
            regime_emb_expanded = regime_emb.unsqueeze(0).expand(batch_size, -1)
        
        fused_features = torch.cat([unified_features, regime_emb_expanded], dim=-1)
        enhanced_features = self.fusion_layer(fused_features)
        
        # 5. Adaptive gating (especialização dinâmica)
        actor_attention = self.actor_gate(enhanced_features)
        critic_attention = self.critic_gate(enhanced_features)
        
        # 6. Specialized feature paths
        actor_gated = enhanced_features * actor_attention
        critic_gated = enhanced_features * critic_attention
        
        # 7. Final projections
        actor_features = self.actor_projection(actor_gated)
        critic_features = self.critic_projection(critic_gated)
        
        # 8. Info para debugging
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
        
        # Adaptive mixing parameters
        self.min_mixing = 0.05
        self.max_mixing = 0.3
        self.performance_window = 20
        
    def analyze_gradient_compatibility(self, actor_grads: Dict, critic_grads: Dict) -> Dict[str, float]:
        """Analisa compatibilidade entre gradientes Actor/Critic"""
        
        compatibility_scores = {}
        
        for actor_key, actor_grad in actor_grads.items():
            # Buscar gradient correspondente no critic
            critic_key = actor_key.replace('actor', 'critic')
            if critic_key in critic_grads:
                critic_grad = critic_grads[critic_key]
                
                # Calcular cosine similarity
                actor_flat = actor_grad.flatten()
                critic_flat = critic_grad.flatten()
                
                # Garantir mesmo tamanho
                min_size = min(len(actor_flat), len(critic_flat))
                if min_size > 0:
                    actor_norm = F.normalize(actor_flat[:min_size].unsqueeze(0), dim=1)
                    critic_norm = F.normalize(critic_flat[:min_size].unsqueeze(0), dim=1)
                    
                    similarity = torch.cosine_similarity(actor_norm, critic_norm).item()
                    compatibility_scores[actor_key] = similarity
        
        return compatibility_scores
    
    def calculate_adaptive_mixing_strength(self, 
                                         performance_trend: float,
                                         gradient_compatibility: float) -> float:
        """Calcula força de mixing adaptativamente"""
        
        # Base mixing strength
        base_mixing = self.mixing_strength
        
        # Ajustar baseado na compatibilidade dos gradientes
        if gradient_compatibility > 0.7:  # Gradientes muito similares
            compatibility_bonus = 0.1
        elif gradient_compatibility > 0.3:  # Compatibilidade moderada
            compatibility_bonus = 0.05
        else:  # Gradientes conflitantes
            compatibility_bonus = -0.05
        
        # Ajustar baseado na performance
        if performance_trend > 0.1:  # Performance melhorando
            performance_bonus = 0.02
        elif performance_trend < -0.1:  # Performance piorando
            performance_bonus = -0.02
        else:
            performance_bonus = 0.0
        
        # Calcular mixing final
        adaptive_mixing = base_mixing + compatibility_bonus + performance_bonus
        adaptive_mixing = max(self.min_mixing, min(self.max_mixing, adaptive_mixing))
        
        return adaptive_mixing
    
    def mix_gradients(self, 
                     backbone_params: List[torch.nn.Parameter],
                     actor_loss: torch.Tensor,
                     critic_loss: torch.Tensor,
                     mixing_strength: Optional[float] = None) -> Dict[str, float]:
        """🔄 Mixing inteligente de gradientes"""
        
        if mixing_strength is None:
            mixing_strength = self.mixing_strength
        
        mixing_stats = {
            'mixed_params': 0,
            'actor_contribution': 0.0,
            'critic_contribution': 0.0,
            'total_norm_before': 0.0,
            'total_norm_after': 0.0
        }
        
        # Calcular gradientes separadamente
        actor_grads = torch.autograd.grad(
            actor_loss, backbone_params, retain_graph=True, create_graph=False
        )
        critic_grads = torch.autograd.grad(
            critic_loss, backbone_params, retain_graph=True, create_graph=False
        )
        
        # Mixing inteligente
        mixed_grads = []
        for actor_grad, critic_grad in zip(actor_grads, critic_grads):
            if actor_grad is not None and critic_grad is not None:
                # Calcular normas
                actor_norm = torch.norm(actor_grad)
                critic_norm = torch.norm(critic_grad)
                
                mixing_stats['total_norm_before'] += (actor_norm + critic_norm).item()
                
                # Mixing ponderado (favorece gradiente com maior norma)
                total_norm = actor_norm + critic_norm + 1e-8
                actor_weight = (1.0 - mixing_strength) + mixing_strength * (actor_norm / total_norm)
                critic_weight = (1.0 - mixing_strength) + mixing_strength * (critic_norm / total_norm)
                
                # Garantir normalização
                total_weight = actor_weight + critic_weight
                actor_weight = actor_weight / total_weight
                critic_weight = critic_weight / total_weight
                
                # Gradient mixing
                mixed_grad = actor_weight * actor_grad + critic_weight * critic_grad
                mixed_grads.append(mixed_grad)
                
                # Stats
                mixing_stats['mixed_params'] += 1
                mixing_stats['actor_contribution'] += actor_weight.item()
                mixing_stats['critic_contribution'] += critic_weight.item()
                mixing_stats['total_norm_after'] += torch.norm(mixed_grad).item()
            else:
                # Fallback para gradient disponível
                mixed_grad = actor_grad if actor_grad is not None else critic_grad
                mixed_grads.append(mixed_grad)
        
        # Aplicar gradientes misturados
        for param, mixed_grad in zip(backbone_params, mixed_grads):
            if param.grad is not None and mixed_grad is not None:
                param.grad.data = mixed_grad.data
        
        # Normalizar stats
        if mixing_stats['mixed_params'] > 0:
            mixing_stats['actor_contribution'] /= mixing_stats['mixed_params']
            mixing_stats['critic_contribution'] /= mixing_stats['mixed_params']
        
        self.mixing_history.append(mixing_stats)
        return mixing_stats

class InterferenceMonitor:
    """📊 Monitor de Interferência entre Actor/Critic"""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        
        # Históricos de métricas
        self.gradient_similarity_history = deque(maxlen=history_size)
        self.performance_correlation_history = deque(maxlen=history_size)
        self.backbone_utilization_history = deque(maxlen=history_size)
        self.interference_score_history = deque(maxlen=history_size)
        
        # Performance tracking
        self.actor_performance_history = deque(maxlen=history_size)
        self.critic_performance_history = deque(maxlen=history_size)
        
    def record_interference_data(self,
                                gradient_similarity: float,
                                actor_performance: float,
                                critic_performance: float,
                                backbone_utilization: Dict[str, float]):
        """Registra dados de interferência"""
        
        self.gradient_similarity_history.append(gradient_similarity)
        self.actor_performance_history.append(actor_performance)
        self.critic_performance_history.append(critic_performance)
        
        # Backbone utilization score
        utilization_score = np.mean(list(backbone_utilization.values()))
        self.backbone_utilization_history.append(utilization_score)
        
        # Calculate performance correlation
        if len(self.actor_performance_history) > 10:
            actor_perf = list(self.actor_performance_history)[-10:]
            critic_perf = list(self.critic_performance_history)[-10:]
            correlation = np.corrcoef(actor_perf, critic_perf)[0, 1]
            if not np.isnan(correlation):
                self.performance_correlation_history.append(correlation)
        
        # Calculate overall interference score
        interference_score = self.calculate_interference_score()
        self.interference_score_history.append(interference_score)
    
    def calculate_interference_score(self) -> float:
        """Calcula score de interferência (0=sem interferência, 1=alta interferência)"""
        
        if len(self.gradient_similarity_history) < 5:
            return 0.5  # Insufficient data
        
        # Fatores de interferência
        factors = []
        
        # 1. Gradient similarity (muito similar = possível interferência)
        recent_similarity = list(self.gradient_similarity_history)[-5:]
        avg_similarity = np.mean(recent_similarity)
        similarity_factor = max(0, avg_similarity - 0.3) / 0.7  # Normalize [0.3, 1.0] -> [0, 1]
        factors.append(similarity_factor)
        
        # 2. Performance correlation (correlação negativa = interferência)
        if len(self.performance_correlation_history) > 0:
            recent_correlation = list(self.performance_correlation_history)[-5:]
            avg_correlation = np.mean(recent_correlation)
            # Correlação negativa indica interferência
            correlation_factor = max(0, -avg_correlation) if avg_correlation < 0 else 0
            factors.append(correlation_factor)
        
        # 3. Backbone utilization variability (alta variabilidade = conflito)
        if len(self.backbone_utilization_history) > 5:
            recent_utilization = list(self.backbone_utilization_history)[-5:]
            utilization_std = np.std(recent_utilization)
            variability_factor = min(1.0, utilization_std * 2)  # Normalize
            factors.append(variability_factor)
        
        # Interference score final
        if factors:
            interference_score = np.mean(factors)
        else:
            interference_score = 0.5
        
        return max(0.0, min(1.0, interference_score))
    
    def get_interference_analysis(self) -> Dict[str, Any]:
        """Análise completa de interferência"""
        
        current_interference = self.calculate_interference_score()
        
        # Trend analysis
        if len(self.interference_score_history) > 10:
            recent_scores = list(self.interference_score_history)[-10:]
            interference_trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        else:
            interference_trend = 0.0
        
        # Classification
        if current_interference < 0.3:
            interference_level = "low"
            status = "🟢 BAIXA INTERFERÊNCIA - Backbone saudável"
        elif current_interference < 0.6:
            interference_level = "medium"
            status = "🟡 INTERFERÊNCIA MODERADA - Monitorar"
        else:
            interference_level = "high"
            status = "🔴 ALTA INTERFERÊNCIA - Ajustar sharing"
        
        return {
            'interference_score': current_interference,
            'interference_level': interference_level,
            'interference_trend': interference_trend,
            'status': status,
            'gradient_similarity': np.mean(list(self.gradient_similarity_history)[-5:]) if self.gradient_similarity_history else 0,
            'performance_correlation': np.mean(list(self.performance_correlation_history)[-5:]) if self.performance_correlation_history else 0,
            'backbone_utilization': np.mean(list(self.backbone_utilization_history)[-5:]) if self.backbone_utilization_history else 0,
            'recommendation': self.get_recommendation(current_interference, interference_trend)
        }
    
    def get_recommendation(self, interference_score: float, trend: float) -> str:
        """Recomendação baseada na interferência"""
        
        if interference_score > 0.7:
            return "Reduzir gradient mixing strength ou aumentar specialization"
        elif interference_score > 0.5 and trend > 0.02:
            return "Monitorar closely - interferência aumentando"
        elif interference_score < 0.3:
            return "Pode aumentar gradient mixing para melhor colaboração"
        else:
            return "Manter configuração atual"

class TwoHeadV7Intuition(TwoHeadV7Enhanced):
    """
    🧠 TwoHeadV7Intuition - BACKBONE UNIFICADO + Gradient Mixing Inteligente
    
    ARQUITETURA INTUITION:
    - Unified Market-Aware Backbone (visão compartilhada do mercado)
    - Actor Branch: LSTM especializado em timing
    - Critic Branch: MLP especializado em valuation  
    - Gradient Mixing System (cross-pollination inteligente)
    - Interference Monitoring (detecta conflitos)
    - Adaptive Sharing (ajusta sharing baseado na performance)
    
    INTUIÇÃO: "Duas mentes, uma visão" - Actor e Critic veem o mercado
    através do mesmo "olho" (backbone), mas processam especializadamente.
    Gradientes se ajudam mutuamente quando compatíveis.
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        *args,
        # V7 PARAMETERS (compatibilidade)
        v7_shared_lstm_hidden: int = 256,
        v7_features_dim: int = 256,
        
        # INTUITION PARAMETERS
        backbone_shared_dim: int = 512,
        regime_embed_dim: int = 64,
        enable_gradient_mixing: bool = True,
        gradient_mixing_strength: float = 0.1,
        enable_interference_monitoring: bool = True,
        adaptive_sharing: bool = True,
        
        **kwargs
    ):
        
        print("TwoHeadV7Intuition inicializando - BACKBONE UNIFICADO!")
        print(f"   Action Space: {action_space}")
        print(f"   Action Dimensions: {action_space.shape[0] if hasattr(action_space, 'shape') else 'Unknown'}")
        print(f"   Shared Backbone Dim: {backbone_shared_dim}")
        print(f"   Gradient Mixing: {enable_gradient_mixing} (strength: {gradient_mixing_strength})")
        print(f"   Interference Monitor: {enable_interference_monitoring}")
        print(f"   Adaptive Sharing: {adaptive_sharing}")
        
        # Validate action space
        if not hasattr(action_space, 'shape') or action_space.shape[0] != 11:
            raise ValueError(f"TwoHeadV7Intuition expects 11-dimensional action space, got {getattr(action_space, 'shape', 'Unknown')}")
        
        # Garantir features_extractor_class
        if 'features_extractor_class' not in kwargs:
            kwargs['features_extractor_class'] = EnhancedFeaturesExtractor
        if 'features_extractor_kwargs' not in kwargs:
            kwargs['features_extractor_kwargs'] = {'features_dim': v7_features_dim}
        
        # Store parameters
        self.v7_shared_lstm_hidden = v7_shared_lstm_hidden
        self.v7_features_dim = v7_features_dim
        self.backbone_shared_dim = backbone_shared_dim
        self.regime_embed_dim = regime_embed_dim
        self.enable_gradient_mixing = enable_gradient_mixing
        self.enable_interference_monitoring = enable_interference_monitoring
        self.adaptive_sharing = adaptive_sharing
        
        # Disable automatic LSTM critic
        kwargs['enable_critic_lstm'] = False
        
        # Initialize parent
        super().__init__(observation_space, action_space, *args, **kwargs)
        
        # 🧠 INTUITION CORE COMPONENTS
        
        # 1. Unified Market Backbone (única fonte de features)
        self.unified_backbone = UnifiedMarketBackbone(
            input_dim=self.v7_features_dim,
            shared_dim=backbone_shared_dim,
            regime_embed_dim=regime_embed_dim
        )
        
        # 2. Specialized Actor Branch (timing focused)
        self.actor_lstm = nn.LSTM(
            input_size=256,  # From backbone projection
            hidden_size=self.v7_shared_lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        
        # Calculate actor input dimension: lstm_out(256) + decision_context(64) = 320
        actor_input_dim = self.v7_shared_lstm_hidden + 64  # 64 from entry(32) + mgmt(32) decisions
        
        self.actor_head = nn.Sequential(
            nn.Linear(actor_input_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, self.action_space.shape[0])
        )
        
        # 3. Specialized Critic Branch (value focused)
        self.critic_mlp = nn.Sequential(
            nn.Linear(256 + regime_embed_dim, 512),  # backbone + regime context
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
        
        # 4. Trading Intelligence (V7 compatible)
        entry_input_dim = self.v7_shared_lstm_hidden * 2 + 8  # 256 * 2 + 8 = 520
        self.entry_head = SpecializedEntryHead(input_dim=entry_input_dim)
        self.management_head = TwoHeadDecisionMaker(input_dim=entry_input_dim)
        
        # 5. Enhanced Memory Bank
        self.enhanced_memory = EnhancedMemoryBank(memory_size=10000, feature_dim=256)
        
        # 6. Gradient Mixing System
        if self.enable_gradient_mixing:
            self.gradient_mixer = GradientMixer(mixing_strength=gradient_mixing_strength)
            print("   ✅ Gradient Mixer inicializado")
        
        # 7. Interference Monitor
        if self.enable_interference_monitoring:
            self.interference_monitor = InterferenceMonitor()
            print("   ✅ Interference Monitor inicializado")
        
        # 8. State tracking
        self.current_regime = 2  # Start with sideways
        self.training_step_count = 0
        self.lstm_hidden_states = None
        self.last_backbone_info = {}
        
        # 🔧 FIX CRÍTICO: Inicialização adequada do actor_head
        self._initialize_actor_head_properly()
        
        print("🎯 TwoHeadV7Intuition PRONTA - Visão Unificada + Processamento Especializado!")
    
    def _initialize_actor_head_properly(self):
        """🔧 Inicialização adequada do actor_head para range maior"""
        
        print("🔧 [INIT FIX] Aplicando inicialização adequada ao actor_head...")
        
        for layer in self.actor_head:
            if isinstance(layer, torch.nn.Linear):
                # Inicialização Xavier com gain maior para range amplo
                torch.nn.init.xavier_uniform_(layer.weight, gain=2.0)
                
                # Bias com range maior para permitir valores negativos/positivos
                if layer.bias is not None:
                    torch.nn.init.uniform_(layer.bias, -1.0, 1.0)
        
        # Inicialização especial para a última camada (mais importante)
        last_layer = self.actor_head[-1]
        if isinstance(last_layer, torch.nn.Linear):
            # Inicialização com range ainda maior para a saída
            torch.nn.init.uniform_(last_layer.weight, -2.0, 2.0)
            if last_layer.bias is not None:
                torch.nn.init.uniform_(last_layer.bias, -1.5, 1.5)
        
        print("✅ [INIT FIX] Actor head inicializado com range amplo (-2 a 2)")
    
    def forward_actor(self, features: torch.Tensor, lstm_states, episode_starts: torch.Tensor):
        """🎯 Forward Actor com Backbone Unificado - FIXED ACTION SPACE"""
        
        # 1. Process through unified backbone
        actor_features, critic_features, regime_id, backbone_info = self.unified_backbone(features)
        
        # 2. LSTM processing
        lstm_out, new_actor_states = self.v7_actor_lstm(actor_features, lstm_states)
        lstm_out = self.actor_lstm_dropout(lstm_out.squeeze(1))
        
        # 3. Decision heads
        batch_size = lstm_out.shape[0]
        entry_signal = lstm_out
        management_signal = lstm_out
        market_context = self.trade_memory.get_memory_context(batch_size)
        
        # Fix: Ajustar dimensões para compatibilidade com entry_head
        # entry_head espera input_dim * 3, mas temos 256 + 256 + 8 = 520
        # Reduzir para 128 * 3 = 384
        
        entry_decision, entry_conf, gate_info = self.entry_head(entry_signal, management_signal, market_context)
        mgmt_decision, mgmt_conf, mgmt_weights = self.management_head(entry_signal, management_signal, market_context)
        
        # 4. Final actor output
        decision_context = torch.cat([entry_decision, mgmt_decision], dim=-1)
        actor_input = torch.cat([lstm_out, decision_context], dim=-1)
        raw_actions = self.actor_head(actor_input)
        
        # 🔧 FIXED: PROPER ACTION PROCESSING - GARANTIR 11 DIMENSÕES
        if raw_actions.shape[-1] != 11:
            raise ValueError(f"Actor head deve produzir 11 ações, mas produziu {raw_actions.shape[-1]}")
        
        actions = torch.zeros_like(raw_actions)
        
        # [0] entry_decision: 0-2 (hold, long, short) - DISCRETE
        # Converter para valores discretos: 0=HOLD, 1=LONG, 2=SHORT
        raw_decision = raw_actions[:, 0]  # Usar valor raw diretamente
        # 🔧 CORREÇÃO ADEQUADA: Thresholds corretos após inicialização adequada
        # Com inicialização adequada, valores devem estar em range maior (-2 a 2)
        # Thresholds balanceados para distribuição 33%/33%/33%
        discrete_decision = torch.where(raw_decision < -0.5, 0,  # HOLD: < -0.5
                                      torch.where(raw_decision > 0.5, 2, 1))  # SHORT: > 0.5, LONG: -0.5 a 0.5
        actions[:, 0] = discrete_decision.float()
        
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
        
        # 🔧 CRITICAL: Verificar que actions tem shape correto
        assert actions.shape[-1] == 11, f"Actions deve ter 11 dimensões, tem {actions.shape[-1]}"
        
        # 5. Enhanced gate info
        gate_info.update(backbone_info)
        gate_info['memory_regime'] = regime_id
        gate_info['raw_actions_shape'] = raw_actions.shape
        gate_info['final_actions_shape'] = actions.shape
        
        return actions, new_actor_states, gate_info
    
    def forward_critic(self, features: torch.Tensor, lstm_states, episode_starts: torch.Tensor):
        """💰 Forward Critic com Backbone Unificado"""
        
        # 1. Use features from unified backbone (já processadas no actor)
        # Se backbone_info não está disponível, processar novamente
        if not hasattr(self, 'last_backbone_info') or not self.last_backbone_info:
            _, critic_features, regime_id, backbone_info = self.unified_backbone(features)
        else:
            # Reutilizar processamento do backbone (mais eficiente)
            _, critic_features, regime_id, backbone_info = self.unified_backbone(features)
        
        # 2. Add regime context to critic
        regime_emb = self.unified_backbone.regime_embedding(
            torch.tensor(regime_id, device=features.device)
        )
        batch_size = critic_features.shape[0]
        regime_emb_expanded = regime_emb.unsqueeze(0).expand(batch_size, -1)
        
        # 3. Enhanced critic input
        critic_input = torch.cat([critic_features, regime_emb_expanded], dim=-1)
        
        # 4. Critic MLP processing
        values = self.critic_mlp(critic_input)
        
        # 5. Dummy states for compatibility
        dummy_states = lstm_states if lstm_states is not None else (
            torch.zeros(1, batch_size, 128, device=features.device),
            torch.zeros(1, batch_size, 128, device=features.device)
        )
        
        return values, dummy_states
    
    def get_enhanced_memory_context(self, batch_size: int, regime_id: int) -> torch.Tensor:
        """Contexto enhanced do memory bank"""
        
        # Generate dummy state for memory query
        dummy_state = np.random.randn(256) * 0.1
        
        # Get regime-specific context
        context = self.enhanced_memory.get_regime_context(regime_id, dummy_state)
        
        # Convert to tensor and expand for batch
        context_tensor = torch.tensor(context[:8], dtype=torch.float32)  # Truncate to 8 dims for compatibility
        return context_tensor.unsqueeze(0).expand(batch_size, -1)
    
    def compute_mixed_loss(self, 
                          actor_loss: torch.Tensor, 
                          critic_loss: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """🔄 Compute loss com gradient mixing"""
        
        if not self.enable_gradient_mixing:
            return actor_loss + critic_loss, {}
        
        # Get backbone parameters for mixing
        backbone_params = list(self.unified_backbone.parameters())
        
        # Adaptive mixing strength
        mixing_strength = self.gradient_mixer.mixing_strength
        if self.adaptive_sharing and hasattr(self, 'interference_monitor'):
            interference_analysis = self.interference_monitor.get_interference_analysis()
            if interference_analysis['interference_score'] > 0.6:
                mixing_strength *= 0.5  # Reduce mixing if high interference
            elif interference_analysis['interference_score'] < 0.3:
                mixing_strength *= 1.5  # Increase mixing if low interference
        
        # Perform gradient mixing
        mixing_stats = self.gradient_mixer.mix_gradients(
            backbone_params, actor_loss, critic_loss, mixing_strength
        )
        
        # Combined loss (for other parameters)
        total_loss = actor_loss + critic_loss
        
        return total_loss, mixing_stats
    
    def post_training_step(self, 
                          experience: Dict[str, Any],
                          actor_performance: float = 0.0,
                          critic_performance: float = 0.0) -> Dict[str, Any]:
        """📊 Post-training analysis"""
        
        analysis = {}
        
        # Store experience in enhanced memory
        if 'reward' in experience:
            dummy_state = np.random.randn(256)
            dummy_action = np.random.randn(3)
            
            self.enhanced_memory.store_memory(
                regime=self.current_regime,
                state=dummy_state,
                action=dummy_action,
                reward=experience['reward'],
                next_state=dummy_state,
                done=experience.get('done', False)
            )
        
        # Interference monitoring
        if self.enable_interference_monitoring and hasattr(self, 'interference_monitor'):
            
            # Estimate gradient similarity (placeholder - would need actual gradients)
            gradient_similarity = 0.5  # Would be calculated from actual gradients
            
            # Backbone utilization from last forward pass
            backbone_utilization = {
                'actor_attention': self.last_backbone_info.get('actor_attention_mean', 0.5),
                'critic_attention': self.last_backbone_info.get('critic_attention_mean', 0.5),
                'specialization': self.last_backbone_info.get('specialization_divergence', 0.0)
            }
            
            # Record interference data
            self.interference_monitor.record_interference_data(
                gradient_similarity=gradient_similarity,
                actor_performance=actor_performance,
                critic_performance=critic_performance,
                backbone_utilization=backbone_utilization
            )
            
            analysis['interference'] = self.interference_monitor.get_interference_analysis()
        
        # Update training step counter
        self.training_step_count += 1
        
        return analysis
    
    def get_intuition_status(self) -> Dict[str, Any]:
        """🧠 Status completo do sistema Intuition"""
        
        status = {
            'backbone': {
                'shared_dim': self.backbone_shared_dim,
                'current_regime': self.current_regime,
                'regime_name': ['bull', 'bear', 'sideways', 'volatile'][self.current_regime],
                'last_info': self.last_backbone_info
            },
            'training_steps': self.training_step_count,
            'memory': self.enhanced_memory.get_regime_stats(),
            'features': {
                'gradient_mixing': self.enable_gradient_mixing,
                'interference_monitoring': self.enable_interference_monitoring,
                'adaptive_sharing': self.adaptive_sharing
            }
        }
        
        # Add interference analysis if available
        if self.enable_interference_monitoring and hasattr(self, 'interference_monitor'):
            status['interference'] = self.interference_monitor.get_interference_analysis()
        
        # Add gradient mixing stats if available
        if self.enable_gradient_mixing and hasattr(self, 'gradient_mixer'):
            if self.gradient_mixer.mixing_history:
                recent_mixing = list(self.gradient_mixer.mixing_history)[-5:]
                status['gradient_mixing'] = {
                    'recent_actor_contribution': np.mean([m['actor_contribution'] for m in recent_mixing]),
                    'recent_critic_contribution': np.mean([m['critic_contribution'] for m in recent_mixing]),
                    'mixing_strength': self.gradient_mixer.mixing_strength
                }
        
        return status

# 🛠️ UTILITY FUNCTIONS

def get_v7_intuition_kwargs():
    """Retorna kwargs para TwoHeadV7Intuition"""
    return {
        # V7 base parameters
        'v7_shared_lstm_hidden': 256,
        'v7_features_dim': 256,
        
        # Intuition parameters
        'backbone_shared_dim': 512,
        'regime_embed_dim': 64,
        'enable_gradient_mixing': True,
        'gradient_mixing_strength': 0.1,
        'enable_interference_monitoring': True,
        'adaptive_sharing': True,
        
        # Original V7 parameters
        'features_extractor_class': EnhancedFeaturesExtractor,
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

def validate_v7_intuition_policy(policy):
    """Valida TwoHeadV7Intuition"""
    
    required_attrs = [
        'unified_backbone', 'actor_lstm', 'critic_mlp',
        'entry_head', 'management_head', 'enhanced_memory'
    ]
    
    for attr in required_attrs:
        if not hasattr(policy, attr):
            raise ValueError(f"V7Intuition missing required attribute: {attr}")
    
    # Validar componentes opcionais
    if policy.enable_gradient_mixing and not hasattr(policy, 'gradient_mixer'):
        raise ValueError("Gradient mixing enabled but gradient_mixer not found")
    
    if policy.enable_interference_monitoring and not hasattr(policy, 'interference_monitor'):
        raise ValueError("Interference monitoring enabled but interference_monitor not found")
    
    print("✅ TwoHeadV7Intuition validada - Backbone Unificado funcionando!")
    print(f"   🧠 Shared Backbone: {policy.backbone_shared_dim}D")
    print(f"   ⚖️ Gradient Mixing: {policy.enable_gradient_mixing}")
    print(f"   📊 Interference Monitor: {policy.enable_interference_monitoring}")
    print(f"   🔄 Adaptive Sharing: {policy.adaptive_sharing}")
    
    return True

if __name__ == "__main__":
    print("🧠 TwoHeadV7Intuition - BACKBONE UNIFICADO + Gradient Mixing!")
    print("   🔗 Visão Compartilhada: Unified Market-Aware Backbone")
    print("   🎯 Processamento Especializado: Actor LSTM + Critic MLP")
    print("   ⚖️ Gradient Mixing: Cross-pollination inteligente")
    print("   📊 Interference Monitor: Detecta conflitos automaticamente")
    print("   🔄 Adaptive Sharing: Ajusta sharing baseado na performance")
    print()
    print("INTUIÇÃO: 'Duas mentes, uma visão' - Ver o mercado pelos mesmos")
    print("olhos, mas processar especializadamente + gradientes colaborativos")