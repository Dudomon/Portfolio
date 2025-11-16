"""
🚀 TwoHeadV6Intelligent48h - Policy LIMPA e FUNCIONAL para Convergência
Criada do ZERO com foco em SIMPLICIDADE e EFETIVIDADE

PRINCIPIOS V6:
✅ Gates que REALMENTE filtram (não apenas informativos)
✅ Logic simples e clara (sem over-engineering)
✅ Thresholds que funcionam (nem muito altos, nem muito baixos)
✅ Composite score que realmente varia
✅ Toda inteligência da V5 mantida mas simplificada
✅ Arquitetura limpa focada em CONVERGÊNCIA PPO
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Any, Dict, List, Optional, Type, Union
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.utils.checkpoint import checkpoint

# Fallback para PyTorchObs
try:
    from stable_baselines3.common.type_aliases import PyTorchObs
except ImportError:
    import torch
    PyTorchObs = torch.Tensor

# Imports corretos para RecurrentPPO
try:
    from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
    from sb3_contrib.common.recurrent.type_aliases import RNNStates
except ImportError:
    from stable_baselines3.common.policies import RecurrentActorCriticPolicy
    from stable_baselines3.common.recurrent_policies import RNNStates


class CustomRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    """
    🚀 CUSTOM RECURRENT POLICY - Override para controle total das LSTMs
    
    Melhorias implementadas:
    - Dropout configurável nas LSTMs principais
    - Layer normalization nas LSTMs
    - Gradient clipping específico para LSTMs
    - Inicialização melhorada dos bias (anti-zeros)
    """
    
    def __init__(
        self, 
        *args,
        # Novos parâmetros customizados
        lstm_dropout: float = 0.1,
        lstm_layer_norm: bool = True,
        lstm_gradient_clipping: float = 0.5,
        gradient_checkpointing: bool = False,  # DESABILITADO: Debugger corrompido
        enable_dense_connection: bool = False,  # DESABILITADO (teste rollback)
        **kwargs
    ):
        # Salvar configurações customizadas
        self.lstm_dropout = lstm_dropout
        self.lstm_layer_norm = lstm_layer_norm
        self.lstm_gradient_clipping = lstm_gradient_clipping
        self.gradient_checkpointing = gradient_checkpointing
        self.enable_dense_connection = enable_dense_connection
        
        # Inicializar classe pai NORMAL
        super().__init__(*args, **kwargs)
        
        # APÓS inicialização pai - customizar LSTMs
        self._customize_lstms()
        
    def _customize_lstms(self):
        """Customiza LSTMs após inicialização da classe pai - SOLUÇÃO INTELIGENTE"""
        
        print(f"🎯 LSTM SMART CUSTOMIZATION - Modifica SEM quebrar shapes")
        
        # 1. MODIFICAR DROPOUT das LSTMs existentes (sem substituir)
        if hasattr(self, 'lstm_actor') and self.lstm_dropout > 0:
            self._inject_lstm_dropout(self.lstm_actor, "actor")
            
        if hasattr(self, 'lstm_critic') and self.lstm_dropout > 0:
            self._inject_lstm_dropout(self.lstm_critic, "critic")
        
        # 2. Aplicar melhor inicialização nos bias existentes
        if hasattr(self, 'lstm_actor'):
            self._init_lstm_bias(self.lstm_actor)
            print(f"   ✅ LSTM actor: bias re-inicializado (anti-zeros)")
            
        if hasattr(self, 'lstm_critic'):
            self._init_lstm_bias(self.lstm_critic)
            print(f"   ✅ LSTM critic: bias re-inicializado (anti-zeros)")
        
        # 3. Registrar hooks para gradient clipping
        self._register_gradient_hooks()
        
        # 4. Aplicar gradient checkpointing nos LSTMs reais (SB3)
        if self.gradient_checkpointing:
            self._apply_checkpointing_to_real_lstms()
        
        print(f"✅ LSTMs customizadas INTELIGENTEMENTE: dropout={self.lstm_dropout}")
        print(f"⚡ Gradient Checkpointing: {'ATIVADO' if self.gradient_checkpointing else 'DESATIVADO'}")
        if self.gradient_checkpointing:
            print(f"   🎯 TARGET: Apenas LSTMs reais com 80%+ zeros (actor/critic)")
            print(f"   💾 V6 Components: Mantidos normais (não têm problema de zeros)")
        print(f"🔗 Dense Connection: {'ATIVADO' if self.enable_dense_connection else 'DESATIVADO'}")
        if self.enable_dense_connection:
            print(f"   ⚡ Skip connection: 30% raw features bypass para combater vanishing gradients")
    
    def _inject_lstm_dropout(self, lstm, name):
        """Injeta dropout na LSTM existente SEM alterar estrutura"""
        
        # Verificar se LSTM já tem dropout
        if hasattr(lstm, 'dropout') and lstm.dropout > 0:
            print(f"   📍 LSTM {name} já tem dropout nativo: {lstm.dropout}")
            return
            
        # Se LSTM tem múltiplas camadas, modificar dropout nativo
        if lstm.num_layers > 1:
            lstm.dropout = self.lstm_dropout
            print(f"   🎯 LSTM {name}: dropout nativo ativado: {self.lstm_dropout}")
        else:
            # Para 1 camada, adicionar hook de dropout manual
            dropout_layer = nn.Dropout(self.lstm_dropout)
            
            def lstm_dropout_hook(module, input, output):
                if self.training:
                    # Aplicar dropout apenas na saída (output[0]), preservar hidden states (output[1])
                    return (dropout_layer(output[0]), output[1])
                return output
            
            lstm.register_forward_hook(lstm_dropout_hook)
            print(f"   🎯 LSTM {name}: dropout manual via hook: {self.lstm_dropout}")
    
    def _create_enhanced_lstm(self, input_size, hidden_size, num_layers, name):
        """Cria LSTM melhorada com dropout e outras otimizações"""
        
        # SEMPRE usar LSTM nativo do PyTorch com dropout
        # Isso garante compatibilidade total com stable-baselines3
        lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.lstm_dropout if num_layers > 1 else 0.0,  # 🎯 DROPOUT (só funciona com >1 layer)
            batch_first=True,
            bias=True
        )
        
        # Inicialização melhorada dos bias
        self._init_lstm_bias(lstm)
        
        # Se temos apenas 1 layer, adicionar dropout manualmente via hook
        if num_layers == 1 and self.lstm_dropout > 0:
            self._add_manual_dropout(lstm, name)
            
        return lstm
    
    def _add_manual_dropout(self, lstm, name):
        """Adiciona dropout manual para LSTMs de 1 layer"""
        dropout_layer = nn.Dropout(self.lstm_dropout)
        
        def dropout_hook(module, input, output):
            # output[0] é a sequência de saídas, output[1] são os hidden states
            if self.training:
                # Aplicar dropout apenas na saída, não nos hidden states
                dropout_output = dropout_layer(output[0])
                return (dropout_output, output[1])
            return output
        
        lstm.register_forward_hook(dropout_hook)
        print(f"   🎯 Dropout manual ({self.lstm_dropout}) adicionado à LSTM {name}")
    
    def _init_lstm_bias(self, lstm):
        """Inicialização melhorada dos bias para evitar zeros extremos"""
        for name, param in lstm.named_parameters():
            if 'bias' in name:
                n = param.size(0)
                # Inicializar gates com valores não-zero
                nn.init.uniform_(param.data[:n//4], -0.05, 0.05)      # Input gate
                nn.init.uniform_(param.data[n//4:n//2], 0.8, 1.2)     # Forget gate
                nn.init.uniform_(param.data[n//2:3*n//4], -0.05, 0.05) # Cell gate  
                nn.init.uniform_(param.data[3*n//4:], -0.05, 0.05)    # Output gate
    
    def _transfer_lstm_weights(self, old_lstm, new_lstm):
        """Transfere pesos da LSTM antiga para nova"""
        try:
            # Tentar transferir pesos compatíveis
            if hasattr(old_lstm, 'state_dict') and hasattr(new_lstm, 'load_state_dict'):
                old_state = old_lstm.state_dict()
                new_state = new_lstm.state_dict()
                
                # Transferir apenas chaves compatíveis
                compatible_state = {}
                for key in new_state.keys():
                    if key in old_state and old_state[key].shape == new_state[key].shape:
                        compatible_state[key] = old_state[key]
                        
                if compatible_state:
                    new_lstm.load_state_dict(compatible_state, strict=False)
                    print(f"   🔄 {len(compatible_state)} pesos transferidos")
                    
        except Exception as e:
            print(f"   ⚠️ Não foi possível transferir pesos: {e}")
            # Não é crítico - nova LSTM será treinada do zero
    
    def _register_gradient_hooks(self):
        """Registra hooks para monitoramento e clipping de gradientes"""
        
        def lstm_gradient_hook(module):
            """Hook para clippar gradientes das LSTMs"""
            def hook_fn(grad):
                if grad is not None:
                    # Clip gradiente da LSTM
                    return torch.clamp(grad, -self.lstm_gradient_clipping, self.lstm_gradient_clipping)
                return grad
            return hook_fn
                
        # Registrar hooks nos parâmetros das LSTMs
        if hasattr(self, 'lstm_actor'):
            for param in self.lstm_actor.parameters():
                if param.requires_grad:
                    param.register_hook(lstm_gradient_hook(self.lstm_actor))
                    
        if hasattr(self, 'lstm_critic'): 
            for param in self.lstm_critic.parameters():
                if param.requires_grad:
                    param.register_hook(lstm_gradient_hook(self.lstm_critic))
    
    def _apply_checkpointing_to_real_lstms(self):
        """Aplica gradient checkpointing nos LSTMs REAIS do stable-baselines3"""
        try:
            print(f"🔍 Procurando LSTMs reais para checkpointing...")
            
            # Procurar por todos os módulos LSTM na policy inteira
            lstm_modules = []
            for name, module in self.named_modules():
                if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                    lstm_modules.append((name, module))
                    print(f"   📍 LSTM encontrado: {name} ({type(module).__name__})")
            
            if not lstm_modules:
                print(f"   ⚠️  Nenhum LSTM real encontrado para checkpointing")
                return
            
            # Aplicar checkpointing wrapper
            for name, lstm_module in lstm_modules:
                # Substituir forward por versão com checkpointing
                original_forward = lstm_module.forward
                
                def checkpointed_forward(input, hx=None):
                    if self.training and self.gradient_checkpointing:
                        # Usar checkpoint apenas durante treino
                        def lstm_func(inp, h):
                            return original_forward(inp, h)
                        return checkpoint(lstm_func, input, hx)
                    else:
                        return original_forward(input, hx)
                
                lstm_module.forward = checkpointed_forward
                print(f"   ✅ Checkpointing aplicado: {name}")
            
            # Também aplicar no features_extractor se existe
            if hasattr(self, 'features_extractor'):
                self._apply_checkpointing_to_features_extractor()
                
            print(f"🚀 GRADIENT CHECKPOINTING aplicado em {len(lstm_modules)} LSTMs reais")
            
        except Exception as e:
            print(f"❌ ERRO ao aplicar checkpointing real: {e}")
            import traceback
            traceback.print_exc()
    
    def _apply_checkpointing_to_features_extractor(self):
        """Aplica checkpointing no features_extractor que tem os zeros extremos"""
        try:
            print(f"🔍 SKIP: Features extractor checkpointing temporariamente desabilitado")
            print(f"   🎯 Focando apenas nos LSTMs reais que têm 80% zeros")
            # TODO: Implementar checkpointing correto para features_extractor depois
            # Por agora, focar nos LSTMs que têm os piores zeros extremos
                
        except Exception as e:
            print(f"❌ ERRO ao aplicar checkpointing no features_extractor: {e}")


# EnhancedLSTMCell removida - usar apenas LSTMs nativas do PyTorch
# para garantir compatibilidade total com stable-baselines3

from stable_baselines3.common.utils import obs_as_tensor
from trading_framework.extractors.transformer_extractor import TradingTransformerFeatureExtractor


class ConflictResolutionCore(nn.Module):
    """
    🧠 CONFLICT RESOLUTION CORE - Resolve conflitos entre Entry e Management heads
    
    Funcionalidades:
    - Analisa sinais conflitantes das duas heads
    - Calcula pesos adaptativos para cada head
    - Gera resolução baseada em contexto de mercado
    """
    
    def __init__(self, input_dim=96):
        super().__init__()
        
        self.input_dim = input_dim
        self.dynamic_input = None
        
        # Placeholder - será criado dinamicamente
        self.conflict_analyzer = None
        
        # Placeholder - serão criados dinamicamente
        self.weight_calculator = None
        self.confidence_estimator = None
        
    def _create_networks(self, input_dim):
        """Cria redes dinamicamente baseado na dimensão real de entrada"""
        if self.dynamic_input == input_dim:
            return  # Já criado
            
        self.dynamic_input = input_dim
        
        # Processador de conflitos
        self.conflict_analyzer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU()
        ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')
        
        # Calculador de pesos
        self.weight_calculator = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # [entry_weight, management_weight]
            nn.Softmax(dim=-1)
        ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')
        
        # Estimador de confiança
        self.confidence_estimator = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')
        
    def forward(self, entry_signal, management_signal, market_context):
        # Combinar sinais
        combined_input = torch.cat([entry_signal, management_signal, market_context], dim=-1)
        
        # Criar redes dinamicamente se necessário
        self._create_networks(combined_input.shape[-1])
        
        # Analisar conflitos
        conflict_features = self.conflict_analyzer(combined_input)
        
        # Calcular pesos
        weights = self.weight_calculator(conflict_features)
        entry_weight = weights[:, 0:1]
        management_weight = weights[:, 1:2]
        
        # Calcular confiança
        confidence = self.confidence_estimator(conflict_features)
        
        return {
            'entry_weight': entry_weight,
            'management_weight': management_weight,
            'confidence': confidence,
            'conflict_features': conflict_features
        }


class TemporalCoordinationCore(nn.Module):
    """
    🕐 TEMPORAL COORDINATION CORE - Coordenação temporal inteligente
    
    Funcionalidades:
    - Analisa timing ideal para trades
    - Considera fatores de risco temporal
    - Memória de decisões anteriores
    """
    
    def __init__(self, input_dim=64, memory_size=50):
        super().__init__()
        
        self.input_dim = input_dim
        self.memory_size = memory_size
        self.dynamic_input = None
        
        # Placeholder - será criado dinamicamente
        self.temporal_analyzer = None
        
        # Placeholder - serão criados dinamicamente
        self.timing_predictor = None
        self.risk_calculator = None
        
        # Memória temporal
        self.register_buffer('timing_memory', torch.zeros(memory_size, 8))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
    def _create_networks(self, input_dim):
        """Cria redes dinamicamente baseado na dimensão real de entrada"""
        if self.dynamic_input == input_dim:
            return  # Já criado
            
        self.dynamic_input = input_dim
        
        # Analisador temporal
        self.temporal_analyzer = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
            nn.LayerNorm(48),
            nn.Dropout(0.1),
            nn.Linear(48, 32),
            nn.ReLU()
        ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')
        
        # Preditor de timing
        self.timing_predictor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),  # [immediate, delayed, wait]
            nn.Softmax(dim=-1)
        ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')
        
        # Calculador de fator de risco temporal
        self.risk_calculator = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')
        
    def forward(self, conflict_resolution, market_context):
        # Combinar inputs
        combined_input = torch.cat([
            conflict_resolution['conflict_features'], 
            market_context
        ], dim=-1)
        
        # Criar redes dinamicamente se necessário
        self._create_networks(combined_input.shape[-1])
        
        # Análise temporal
        temporal_features = self.temporal_analyzer(combined_input)
        
        # Predição de timing
        timing_probs = self.timing_predictor(temporal_features)
        
        # Fator de risco temporal
        risk_time_factor = self.risk_calculator(temporal_features)
        
        return {
            'timing_probs': timing_probs,
            'risk_time_factor': risk_time_factor,
            'temporal_features': temporal_features
        }
    
    def update_memory(self, timing_decision, market_state, outcome):
        """Atualiza memória temporal"""
        ptr = int(self.memory_ptr.item())
        
        memory_entry = torch.cat([
            timing_decision.flatten()[:3],
            market_state.flatten()[:2], 
            outcome.flatten()[:3]
        ])
        
        self.timing_memory[ptr] = memory_entry
        self.memory_ptr[0] = (ptr + 1) % self.memory_size


class AdaptiveLearningCore(nn.Module):
    """
    📚 ADAPTIVE LEARNING CORE - Aprendizado adaptativo baseado em performance
    
    Funcionalidades:
    - Monitora performance das decisões
    - Adapta estratégias baseado em resultados
    - Melhora reliability das heads
    """
    
    def __init__(self, memory_size=100):
        super().__init__()
        
        self.memory_size = memory_size
        
        # Analisador de contexto
        self.context_analyzer = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.LayerNorm(24),
            nn.Dropout(0.1),
            nn.Linear(24, 16),
            nn.ReLU()
        )
        
        # Gerador de adaptações
        self.adaptation_generator = nn.Sequential(
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, 4),  # [risk_adjust, timing_adjust, confidence_adjust, aggression_adjust]
            nn.Tanh()
        )
        
        # Estimador de reliability
        self.reliability_estimator = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),  # [entry_reliability, mgmt_reliability]
            nn.Sigmoid()
        )
        
        # Memórias de aprendizado
        self.register_buffer('performance_memory', torch.zeros(memory_size, 6))  # Performance data
        self.register_buffer('decision_memory', torch.zeros(memory_size, 8))     # Decision data  
        self.register_buffer('context_memory', torch.zeros(memory_size, 4))      # Context data
        self.register_buffer('perf_ptr', torch.zeros(1, dtype=torch.long))
        
    def forward(self, current_context):
        # Análise do contexto atual
        context_features = self.context_analyzer(current_context)
        
        # Gerar adaptações
        adaptations = self.adaptation_generator(context_features)
        
        # Estimar reliability
        reliability = self.reliability_estimator(context_features)
        entry_reliability = reliability[:, 0:1]
        mgmt_reliability = reliability[:, 1:2]
        
        return {
            'adaptations': adaptations,
            'entry_reliability': entry_reliability,
            'mgmt_reliability': mgmt_reliability,
            'context_features': context_features
        }
    
    def update_learning_memory(self, decision_data, performance_data, context_data):
        """Atualiza memórias de aprendizado"""
        ptr = int(self.perf_ptr.item())
        
        self.decision_memory[ptr] = decision_data[:8]
        self.performance_memory[ptr] = performance_data[:6]
        self.context_memory[ptr] = context_data[:4]
        
        self.perf_ptr[0] = (ptr + 1) % self.memory_size


class GPUOptimizedGatesFusion(nn.Module):
    """
    🚀 GPU OPTIMIZED GATES FUSION - Otimizações GPU para gates
    """
    
    def __init__(self, input_dim=68):
        super().__init__()
        
        # Fusão otimizada de gates
        self.gates_fusion = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),  # [temporal, risk, quality, confidence]
            nn.Sigmoid()
        )
        
        # Raw gates calculator
        self.raw_calculator = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 4),
            nn.Sigmoid()
        )
    
    def forward(self, features, market_analysis, thresholds_dict):
        # Combinar inputs
        combined = torch.cat([features, market_analysis], dim=-1)
        
        # Calcular gates
        gates = self.gates_fusion(combined)
        raw_gates = self.raw_calculator(combined)
        
        return gates, raw_gates


class MultiHeadAttentionOptimized(nn.Module):
    """
    🎯 MULTI-HEAD ATTENTION OPTIMIZED - Attention otimizada
    """
    
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        
        # Residual connection + normalization
        output = self.norm(x + attn_output)
        
        return output


class StrategicFusionLayer(nn.Module):
    """
    🎪 STRATEGIC FUSION LAYER - Coordenação estratégica inteligente entre heads
    
    Integra todos os componentes para coordenação cognitiva:
    - ConflictResolutionCore: Resolve conflitos entre heads
    - TemporalCoordinationCore: Timing inteligente
    - AdaptiveLearningCore: Aprendizado baseado em performance
    """
    
    def __init__(self, entry_dim=64, management_dim=32, market_dim=4):
        super().__init__()
        
        self.entry_dim = entry_dim
        self.management_dim = management_dim
        self.market_dim = market_dim
        
        # Componentes principais - dimensões serão ajustadas dinamicamente
        self.conflict_resolver = ConflictResolutionCore()
        self.temporal_coordinator = TemporalCoordinationCore()
        
        self.adaptive_learner = AdaptiveLearningCore()
        
        # Processador de contexto de mercado
        self.market_processor = nn.Sequential(
            nn.Linear(market_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        
        # Fusion final - será criado dinamicamente
        self.final_fusion = None
        self.dynamic_fusion = None
        
        # Calculadores de saída - serão criados dinamicamente
        self.action_predictor = None
        self.confidence_calculator = None
        self.position_sizer = None
        
    def _create_fusion_networks(self, input_dim):
        """Cria redes de fusion dinamicamente"""
        if self.dynamic_fusion == input_dim:
            return  # Já criado
            
        self.dynamic_fusion = input_dim
        
        # Fusion final
        self.final_fusion = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
            nn.LayerNorm(48),
            nn.Dropout(0.1),
            nn.Linear(48, 32),
            nn.ReLU()
        ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')
        
        # Calculadores de saída
        self.action_predictor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),  # [hold, long, short]
            nn.Softmax(dim=-1)
        ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')
        
        self.confidence_calculator = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')
        
        self.position_sizer = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')
        
    def forward(self, entry_output, management_output, market_context):
        batch_size = entry_output.shape[0]
        
        # Processar contexto de mercado
        market_processed = self.market_processor(market_context)
        market_expanded = market_processed.repeat(1, 8)  # Expandir para temporal coordinator
        
        # 1. Resolução de conflitos
        conflict_resolution = self.conflict_resolver(
            entry_output, management_output, 
            torch.cat([market_context, market_processed], dim=-1)
        )
        
        # 2. Coordenação temporal
        timing_decision = self.temporal_coordinator(
            conflict_resolution,
            market_expanded
        )
        
        # 3. Aprendizado adaptativo
        adaptive_learning = self.adaptive_learner(market_processed)
        
        # 4. Fusion reasoning
        fusion_reasoning = torch.cat([
            conflict_resolution['conflict_features'],
            timing_decision['temporal_features'],
            adaptive_learning['context_features']
        ], dim=-1)
        
        # 5. Fusion final - criar rede dinamicamente
        fusion_input = torch.cat([entry_output, management_output, fusion_reasoning], dim=-1)
        self._create_fusion_networks(fusion_input.shape[-1])
        fused_features = self.final_fusion(fusion_input)
        
        # 6. Saídas finais
        action_probs = self.action_predictor(fused_features)
        confidence = self.confidence_calculator(fused_features)
        position_size = self.position_sizer(fused_features)
        
        return {
            'action_probs': action_probs,
            'confidence': confidence,
            'position_size': position_size,
            'entry_weight': conflict_resolution['entry_weight'],
            'mgmt_weight': conflict_resolution['management_weight'],
            'timing_decision': timing_decision['timing_probs'],
            'adaptations': adaptive_learning['adaptations'],
            'fusion_reasoning': fusion_reasoning
        }
    
    def update_learning(self, decision_outcome, performance_metrics, market_state):
        """Atualiza aprendizado baseado em resultados"""
        # Preparar dados para learning update
        decision_data = torch.tensor([
            decision_outcome['action'],
            decision_outcome['confidence'], 
            decision_outcome['position_size'],
            decision_outcome['pnl'],
            decision_outcome['duration'],
            decision_outcome['success'],
            decision_outcome['drawdown'],
            decision_outcome['profit_factor']
        ], dtype=torch.float32)
        
        performance_data = torch.tensor([
            performance_metrics['win_rate'],
            performance_metrics['profit_factor'],
            performance_metrics['sharpe_ratio'],
            performance_metrics['max_drawdown'],
            performance_metrics['total_trades'],
            performance_metrics['avg_trade_duration']
        ], dtype=torch.float32)
        
        context_data = torch.tensor([
            market_state['volatility'],
            market_state['trend_strength'],
            market_state['momentum'],
            market_state['market_regime']
        ], dtype=torch.float32)
        
        # Atualizar memórias
        self.adaptive_learner.update_learning_memory(
            decision_data, performance_data, context_data
        )
        
        # Atualizar temporal coordination se disponível
        if 'timing_success' in decision_outcome:
            timing_decision = torch.tensor([
                decision_outcome['timing_success'],
                decision_outcome['timing_profit'], 
                0.0
            ], dtype=torch.float32)
            
            market_state_tensor = torch.tensor([
                market_state['volatility'],
                market_state['trend_strength']
            ], dtype=torch.float32)
            
            outcome_tensor = torch.tensor([
                decision_outcome['pnl'],
                decision_outcome['success'],
                decision_outcome['duration']
            ], dtype=torch.float32)
            
            self.temporal_coordinator.update_memory(
                timing_decision, market_state_tensor, outcome_tensor
            )


class CleanEntryHeadV6(nn.Module):
    """
    🎯 ENTRY HEAD V6 - LIMPA E FUNCIONAL
    
    PRINCIPIOS:
    - Gates que REALMENTE filtram
    - Lógica simples e clara
    - Thresholds funcionais
    - Composite score que varia
    """
    
    def __init__(self, features_dim=128):
        super().__init__()
        
        self.features_dim = features_dim
        
        # 🎯 1. FEATURE PROCESSOR - Simples e efetivo
        self.feature_processor = nn.Sequential(
            nn.Linear(features_dim, 96),
            nn.ReLU(),
            nn.LayerNorm(96),
            nn.Dropout(0.1),
            nn.Linear(96, 64),
            nn.ReLU()
        )
        
        # 🎯 2. MARKET ANALYZER - Analisa contexto de mercado
        self.market_analyzer = nn.Sequential(
            nn.Linear(features_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # [volatility, trend, momentum, quality]
            nn.Sigmoid()
        )
        
        # 🎯 3. GATES CALCULATOR - Calcula os 4 gates principais
        self.gates_calculator = nn.Sequential(
            nn.Linear(64 + 4, 48),  # features + market_analysis
            nn.ReLU(),
            nn.LayerNorm(48),
            nn.Dropout(0.1),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # [temporal, risk, quality, confidence]
            nn.Sigmoid()
        )
        
        # 🎯 4. FINAL DECISION NETWORK
        self.final_decision = nn.Sequential(
            nn.Linear(64 + 4 + 4, 48),  # features + market + gates
            nn.ReLU(),
            nn.LayerNorm(48),
            nn.Dropout(0.1),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 64)  # Output final
        )
        
        # 🎯 5. THRESHOLDS BALANCEADOS - Buscar ~30-40% aprovação (ajuste fino)
        self.temporal_threshold = nn.Parameter(torch.tensor(0.46), requires_grad=True)    # 46%
        self.risk_threshold = nn.Parameter(torch.tensor(0.42), requires_grad=True)        # 42%
        self.quality_threshold = nn.Parameter(torch.tensor(0.50), requires_grad=True)     # 50%
        self.confidence_threshold = nn.Parameter(torch.tensor(0.56), requires_grad=True)  # 56%
        
        # 🎯 6. COMPOSITE THRESHOLD PARA HARD THRESHOLD - OTIMIZADO para day trading GOLD
        self.composite_base = nn.Parameter(torch.tensor(0.55), requires_grad=True)        # 55% (seletivo mas realista para GOLD)
        
        # 🎯 7. MEMORY BUFFER - Simples e efetivo
        self.register_buffer('trade_memory', torch.zeros(20, 6))  # 20 trades, 6 features
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
    
    def update_memory(self, pnl, duration, volatility, confidence, success, market_regime):
        """Atualiza memória de trades"""
        ptr = int(self.memory_ptr.item())
        
        self.trade_memory[ptr] = torch.tensor([
            pnl, duration, volatility, confidence, success, market_regime
        ], device=self.trade_memory.device)
        
        self.memory_ptr[0] = (ptr + 1) % 20
    
    def get_market_context(self, features):
        """Extrai contexto de mercado das features"""
        batch_size = features.shape[0]
        
        # Análise de mercado
        market_analysis = self.market_analyzer(features)
        volatility = market_analysis[:, 0:1]
        trend = market_analysis[:, 1:2] 
        momentum = market_analysis[:, 2:3]
        quality = market_analysis[:, 3:4]
        
        # 🎯 CONTEXTO SIMPLES MAS EFETIVO
        market_context = {
            'volatility': volatility,
            'trend': trend,
            'momentum': momentum,
            'quality': quality,
            'is_high_vol': volatility > 0.7,
            'is_strong_trend': torch.abs(trend - 0.5) > 0.3,
            'is_high_momentum': momentum > 0.6,
            'is_high_quality': quality > 0.6
        }
        
        return market_context, market_analysis
    
    def calculate_adaptive_thresholds(self, market_context):
        """Calcula thresholds adaptativos - SIMPLES E FUNCIONAL"""
        
        volatility = market_context['volatility']
        quality = market_context['quality']
        
        # 🎯 ADAPTAÇÃO MAIS DINÂMICA
        # Alta volatilidade = thresholds menores (mais trades)
        # Baixa qualidade = thresholds maiores (menos trades)
        
        vol_factor = 1.0 - (volatility - 0.5) * 0.8  # 0.6 a 1.4
        quality_factor = 0.6 + quality * 0.8          # 0.6 a 1.4
        
        adapted_temporal = torch.clamp(self.temporal_threshold * vol_factor, 0.1, 0.6)
        adapted_risk = torch.clamp(self.risk_threshold * quality_factor, 0.1, 0.5)
        adapted_quality = torch.clamp(self.quality_threshold * quality_factor, 0.15, 0.7)
        adapted_confidence = torch.clamp(self.confidence_threshold * vol_factor, 0.2, 0.8)
        
        return {
            'temporal': adapted_temporal,
            'risk': adapted_risk,
            'quality': adapted_quality,
            'confidence': adapted_confidence
        }
    
    def calculate_gates(self, processed_features, market_analysis, thresholds):
        """Calcula os 4 gates principais"""
        
        # Input para gates
        gates_input = torch.cat([processed_features, market_analysis], dim=-1)
        
        # Calcular gates raw
        gates_raw = self.gates_calculator(gates_input)
        temporal_raw = gates_raw[:, 0:1]
        risk_raw = gates_raw[:, 1:2]
        quality_raw = gates_raw[:, 2:3]
        confidence_raw = gates_raw[:, 3:4]
        
        # 🎯 GATES QUE REALMENTE FILTRAM
        # Usar step function balanceada para filtrar adequadamente
        
        temporal_gate = torch.sigmoid((temporal_raw - thresholds['temporal']) * 10)
        risk_gate = torch.sigmoid((risk_raw - thresholds['risk']) * 10)
        quality_gate = torch.sigmoid((quality_raw - thresholds['quality']) * 10)
        confidence_gate = torch.sigmoid((confidence_raw - thresholds['confidence']) * 10)
        
        gates = {
            'temporal': temporal_gate,
            'risk': risk_gate,
            'quality': quality_gate,
            'confidence': confidence_gate
        }
        
        return gates, gates_raw
    
    def calculate_composite_score(self, gates, market_context):
        """Calcula composite score - SIMPLES E FUNCIONAL"""
        
        # 🎯 PESOS SIMPLES E CLAROS
        # Baseado no contexto de mercado
        
        if market_context['is_high_vol'].any():
            # Mercado volátil - priorizar risk e quality
            weights = torch.tensor([0.20, 0.35, 0.30, 0.15], device=gates['temporal'].device)
        elif market_context['is_strong_trend'].any():
            # Mercado em tendência - priorizar temporal e confidence
            weights = torch.tensor([0.35, 0.15, 0.25, 0.25], device=gates['temporal'].device)
        else:
            # Mercado normal - pesos balanceados
            weights = torch.tensor([0.25, 0.25, 0.25, 0.25], device=gates['temporal'].device)
        
        # Composite score ponderado
        composite_score = (
            gates['temporal'] * weights[0] +
            gates['risk'] * weights[1] +
            gates['quality'] * weights[2] +
            gates['confidence'] * weights[3]
        )
        
        return composite_score
    
    def apply_final_gate(self, entry_decision, composite_score, market_context):
        """Aplica gate final - REALMENTE FILTRA"""
        
        # 🎯 THRESHOLD ADAPTATIVO MAIS PERMISSIVO
        base_threshold = self.composite_base
        
        # Ajustar threshold baseado no contexto - MAIS CONSERVADOR
        if market_context['is_high_quality'].any():
            threshold = base_threshold * 0.85  # Levemente mais permissivo com alta qualidade
        elif market_context['is_high_vol'].any():
            threshold = base_threshold * 0.92  # Muito pouco mais permissivo com alta vol
        elif market_context['is_strong_trend'].any():
            threshold = base_threshold * 0.95  # Quase padrão com tendência forte
        else:
            threshold = base_threshold          # Threshold padrão em contexto normal
        
        # 🎯 FILTRO REAL - ALTERNATIVAS AO SIGMOID MERDA
        diff = composite_score - threshold
        
        # OPÇÃO 1: Hard threshold (limpo e direto)
        final_gate = (composite_score > threshold).float()
        
        # OPÇÃO 2: Clamp suave (muito melhor que sigmoid)
        # final_gate = torch.clamp((diff + 0.2) / 0.4, 0, 1)  # Transição suave entre threshold±0.2
        
        # OPÇÃO 3: Tanh normalizado (não satura tanto)
        # final_gate = (torch.tanh(diff * 8) + 1) / 2
        
        # OPÇÃO 4: ReLU6 escalado (controle total)
        # final_gate = torch.clamp(diff * 5 + 0.5, 0, 1)  # Slope controlado, não satura
        
        # OPÇÃO 5: Swish suave (boa alternativa)
        # final_gate = diff * torch.sigmoid(diff * 3) + 0.5
        
        # REALMENTE APLICAR O FILTRO
        filtered_decision = entry_decision * final_gate
        
        return filtered_decision, final_gate, threshold
    
    def forward(self, features):
        """Forward pass limpo e funcional"""
        
        # 1. Processar features
        processed_features = self.feature_processor(features)
        
        # 2. Análise de mercado
        market_context, market_analysis = self.get_market_context(features)
        
        # 3. Calcular thresholds adaptativos
        thresholds = self.calculate_adaptive_thresholds(market_context)
        
        # 4. Calcular gates
        gates, gates_raw = self.calculate_gates(processed_features, market_analysis, thresholds)
        
        # 5. Calcular composite score
        composite_score = self.calculate_composite_score(gates, market_context)
        
        # 6. Decisão inicial
        decision_input = torch.cat([processed_features, market_analysis, 
                                   torch.cat(list(gates.values()), dim=-1)], dim=-1)
        initial_decision = self.final_decision(decision_input)
        
        # 7. 🎯 APLICAR GATE FINAL - REALMENTE FILTRA
        final_decision, final_gate, threshold = self.apply_final_gate(
            initial_decision, composite_score, market_context
        )
        
        return {
            'entry_decision': final_decision,
            'gates': gates,
            'gates_raw': gates_raw,
            'composite_score': composite_score,
            'final_gate': final_gate,
            'market_context': market_context,
            'market_analysis': market_analysis,
            'thresholds': thresholds,
            'threshold_used': threshold
        }


class TwoHeadV6Intelligent48h(CustomRecurrentActorCriticPolicy):
    """
    🚀 TWOHEAD V6 INTELLIGENT 48H - LIMPA E FUNCIONAL
    
    ARQUITETURA LIMPA:
    - 2 LSTM Layers (comprovado)
    - 1 GRU Stabilizer (comprovado)
    - 4 Attention Heads (suficiente)
    - Pattern Recognition (simples)
    - Entry Head V6 (limpa e funcional)
    - Management Head (simples)
    
    FOCO EM CONVERGÊNCIA:
    - Gates que realmente filtram
    - Lógica simples e clara
    - Thresholds funcionais
    - Composite score efetivo
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = TradingTransformerFeatureExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        # Parâmetros principais
        lstm_hidden_size: int = 128,
        n_lstm_layers: int = 2,
        attention_heads: int = 4,
        # 🎯 NOVOS PARÂMETROS LSTM CUSTOMIZADOS
        lstm_dropout: float = 0.1,
        lstm_layer_norm: bool = True,
        lstm_gradient_clipping: float = 0.5,
        # 🛡️ PARÂMETROS AVANÇADOS LSTM
        lstm_bias_init: float = 1.0,
        lstm_weight_init: str = 'xavier',
        lstm_hidden_init: str = 'orthogonal',
        enable_lstm_smart_init: bool = True,
        lstm_recurrent_dropout: float = 0.0,
        lstm_bidirectional: bool = False,
        # 🔥 PARÂMETROS DE OPTIMIZAÇÃO
        enable_gradient_health_monitor: bool = True,
        gradient_check_frequency: int = 500,
        auto_gradient_fix: bool = True,
        gradient_alert_threshold: float = 0.3,
        # 🛡️ ESTRATÉGIAS ANTI-ZEROS AVANÇADAS
        enable_differential_weight_decay: bool = True,
        lstm_weight_decay: float = 1e-6,              # Weight decay reduzido para LSTMs
        attention_weight_decay: float = 5e-6,         # Weight decay reduzido para Attention
        standard_weight_decay: float = 1e-4,          # Weight decay padrão para outros
        enable_dead_neuron_activation: bool = True,
        dead_neuron_threshold: float = 1e-5,          # Threshold para detectar neurônios dormentes
        activation_noise_scale: float = 1e-4,         # Escala do ruído para reativação
        **kwargs
    ):
        """TwoHeadV6Intelligent48h - Limpa e Funcional"""
        
        # Configurações padrão
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {
                'features_dim': 128,
                'seq_len': 10
            }
        
        if net_arch is None:
            net_arch = [dict(pi=[128, 64], vf=[128, 64])]
        
        # Parâmetros principais
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.attention_heads = attention_heads
        
        # Parâmetros LSTM customizados
        self.lstm_dropout = lstm_dropout
        self.lstm_layer_norm = lstm_layer_norm
        self.lstm_gradient_clipping = lstm_gradient_clipping
        
        # Parâmetros avançados LSTM
        self.lstm_bias_init = lstm_bias_init
        self.lstm_weight_init = lstm_weight_init
        self.lstm_hidden_init = lstm_hidden_init
        self.enable_lstm_smart_init = enable_lstm_smart_init
        self.lstm_recurrent_dropout = lstm_recurrent_dropout
        self.lstm_bidirectional = lstm_bidirectional
        
        # Parâmetros de otimização
        self.enable_gradient_health_monitor = enable_gradient_health_monitor
        self.gradient_check_frequency = gradient_check_frequency
        self.auto_gradient_fix = auto_gradient_fix
        self.gradient_alert_threshold = gradient_alert_threshold
        
        # Estratégias anti-zeros avançadas
        self.enable_differential_weight_decay = enable_differential_weight_decay
        self.lstm_weight_decay = lstm_weight_decay
        self.attention_weight_decay = attention_weight_decay
        self.standard_weight_decay = standard_weight_decay
        self.enable_dead_neuron_activation = enable_dead_neuron_activation
        self.dead_neuron_threshold = dead_neuron_threshold
        self.activation_noise_scale = activation_noise_scale
        
        # Filtrar kwargs customizados que não são suportados pela classe pai
        custom_lstm_params = {
            'lstm_bias_init', 'lstm_weight_init', 'lstm_hidden_init',
            'enable_lstm_smart_init', 'lstm_recurrent_dropout', 'lstm_bidirectional',
            'enable_gradient_health_monitor', 'gradient_check_frequency',
            'auto_gradient_fix', 'gradient_alert_threshold',
            'enable_differential_weight_decay', 'lstm_weight_decay',
            'attention_weight_decay', 'standard_weight_decay',
            'enable_dead_neuron_activation', 'dead_neuron_threshold',
            'activation_noise_scale'
        }
        
        # Separar kwargs suportados dos customizados
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in custom_lstm_params}
        
        # Inicializar classe pai apenas com parâmetros suportados
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            lstm_hidden_size=self.lstm_hidden_size,
            n_lstm_layers=self.n_lstm_layers,
            # 🎯 APENAS PARÂMETROS LSTM SUPORTADOS PELA CLASSE PAI
            lstm_dropout=self.lstm_dropout,
            lstm_layer_norm=self.lstm_layer_norm,
            lstm_gradient_clipping=self.lstm_gradient_clipping,
            **filtered_kwargs
        )
        
        # Inicializar componentes
        self._init_components()
        
        # 🔥 CORREÇÃO CRÍTICA: Desabilitar mlp_extractor padrão do PPO
        # O mlp_extractor consome gradientes sem contribuir para nossa arquitetura transformer
        self._disable_default_mlp_extractor()
        
        # 🔥 CORREÇÃO FINAL: Re-inicializar features_extractor após super().__init__()
        # O super().__init__() cria features_extractor mas com inicialização padrão
        if hasattr(self, 'features_extractor'):
            self.features_extractor._initialize_temporal_weights()
            print("[V6] 🔥 Features extractor RE-INICIALIZADO após super().__init__()")
        
        # 🔥 CORREÇÃO CRÍTICA: Re-inicializar action_net e value_net após super().__init__()
        # Essas camadas são criadas pelo stable-baselines3 com inicialização padrão problemática
        self._force_reinitialize_action_value_nets()
        
        # 🛡️ APLICAR INICIALIZAÇÃO LSTM CUSTOMIZADA
        if self.enable_lstm_smart_init:
            self._apply_custom_lstm_initialization()
        
        self._init_weights()
        
        # Logs
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[V6] TwoHeadV6Intelligent48h: {total_params:,} parametros")
        print(f"[V6] Arquitetura LIMPA: 2-LSTM + 1-GRU + 4-Head Attention")
        print(f"[V6] Entry Head V6: Gates que REALMENTE filtram")
        print(f"[V6] Foco: CONVERGENCIA PPO")
    
    def _apply_custom_lstm_initialization(self):
        """🛡️ Aplicar inicialização LSTM customizada com parâmetros avançados"""
        try:
            import torch.nn as nn
            
            print("🛡️ Aplicando inicialização LSTM customizada...")
            lstm_count = 0
            
            for name, module in self.named_modules():
                if isinstance(module, (nn.LSTM, nn.LSTMCell)):
                    lstm_count += 1
                    
                    # Aplicar inicialização customizada baseada nos parâmetros
                    for param_name, param in module.named_parameters():
                        if 'weight_ih' in param_name:
                            # Input-hidden weights
                            if self.lstm_weight_init == 'xavier':
                                nn.init.xavier_uniform_(param)
                            elif self.lstm_weight_init == 'kaiming':
                                nn.init.kaiming_uniform_(param)
                                
                        elif 'weight_hh' in param_name:
                            # Hidden-hidden weights
                            if self.lstm_hidden_init == 'orthogonal':
                                nn.init.orthogonal_(param)
                            elif self.lstm_hidden_init == 'xavier':
                                nn.init.xavier_uniform_(param)
                                
                        elif 'bias' in param_name:
                            # Bias initialization
                            nn.init.zeros_(param)
                            # Set forget gate bias to configured value
                            if 'bias_ih' in param_name or 'bias_hh' in param_name:
                                hidden_size = param.size(0) // 4
                                param.data[hidden_size:2*hidden_size].fill_(self.lstm_bias_init)
            
            print(f"✅ Inicialização LSTM customizada aplicada em {lstm_count} módulos LSTM")
            print(f"   🎯 Weight Init: {self.lstm_weight_init}")
            print(f"   🎯 Hidden Init: {self.lstm_hidden_init}")
            print(f"   🎯 Bias Init: {self.lstm_bias_init}")
            
        except Exception as e:
            print(f"⚠️ Erro na inicialização LSTM customizada: {e}")
            import traceback
            traceback.print_exc()
        
        # 🚀 REGISTRAR HOOKS PARA GRADIENT MONITORING
        self._register_gradient_hooks()
    
    def _register_gradient_hooks(self):
        """🚀 HOOKS ULTRA-LEVES: Monitoramento sem impacto na velocidade"""
        try:
            # Importar apenas quando necessário (lazy import)
            from lightweight_gradient_monitor import setup_lightweight_monitoring
            
            # Setup monitor com frequência baixa para manter velocidade
            self.gradient_monitor = setup_lightweight_monitoring(
                model=self, 
                check_frequency=500  # Menos frequente = mais rápido
            )
            print("🔧 Hooks de ativação registrados na policy")
            
        except ImportError:
            print("⚠️ LightweightGradientMonitor não disponível - hooks desabilitados")
    
    def _gradient_hook_callback(self):
        """🚀 Callback ultra-rápido executado após backward pass"""
        if self.gradient_monitor is not None:
            self.gradient_step_counter += 1
            self.gradient_monitor.quick_health_check(self.gradient_step_counter)
    
    def _disable_default_mlp_extractor(self):
        """🔥 CORREÇÃO DEFINITIVA: Substituir ReLUs por LeakyReLU no mlp_extractor"""
        try:
            if hasattr(self, 'mlp_extractor'):
                # 🎯 CORREÇÃO DOS 50-53% ZEROS: Substituir ReLUs por LeakyReLU
                relu_fixes = 0
                
                for name, module in self.mlp_extractor.named_modules():
                    if isinstance(module, nn.ReLU):
                        # Encontrar módulo pai para substituição
                        parts = name.split('.')
                        if len(parts) > 1:
                            parent_name = '.'.join(parts[:-1])
                            child_name = parts[-1]
                            
                            try:
                                parent_module = self.mlp_extractor
                                for part in parent_name.split('.'):
                                    parent_module = getattr(parent_module, part)
                                
                                # Substituir ReLU por LeakyReLU
                                setattr(parent_module, child_name, nn.LeakyReLU(negative_slope=0.01, inplace=True))
                                relu_fixes += 1
                                
                            except Exception as e:
                                print(f"[V6] ⚠️ Erro ao substituir {name}: {e}")
                
                print(f"[V6] 🔥 mlp_extractor CORRIGIDO: {relu_fixes} ReLUs → LeakyReLU (resolve 50-53% zeros)")
                
        except Exception as e:
            print(f"[V6] ⚠️ Erro ao corrigir mlp_extractor: {e}")
    
    def _init_components(self):
        """Inicializa componentes principais"""
        features_dim = self.features_extractor.features_dim
        
        # 🎯 1. GRU STABILIZER
        self.gru_stabilizer = nn.GRU(
            input_size=self.lstm_hidden_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        self.gru_norm = nn.LayerNorm(self.lstm_hidden_size)
        
        # 🎯 2. ATTENTION
        self.attention = nn.MultiheadAttention(
            embed_dim=self.lstm_hidden_size,
            num_heads=self.attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 🎯 3. PATTERN RECOGNITION (SIMPLES)
        self.pattern_detector = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.Sigmoid()
        )
        
        # 🎯 4. FEATURE FUSION
        fusion_input = self.lstm_hidden_size * 2 + 32  # lstm + gru + patterns
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        # 🎯 5. ENTRY HEAD V6 (LIMPA E FUNCIONAL)
        self.entry_head = CleanEntryHeadV6(features_dim=128)
        
        # 🎯 6. MANAGEMENT HEAD (SIMPLES)
        self.management_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 🎯 7. STRATEGIC FUSION LAYER - FORÇAR ATIVAÇÃO
        # Strategic Fusion Layer será inicializada e ATIVADA por padrão
        self.strategic_fusion = None
        self.strategic_fusion_enabled = True  # ✅ FORÇAR ATIVAÇÃO para modelos existentes
        self._fusion_initialization_attempted = False
        
        # 🎯 8. FINAL COMBINATION - OTIMIZADO para reduzir sparsity
        self.final_projector = nn.Sequential(
            nn.Linear(64 + 32, 256),  # 🔥 OTIMIZADO: Mais capacidade para reduzir zeros
            nn.GELU(),                # 🔥 OTIMIZADO: Ativação mais suave que ReLU
            nn.LayerNorm(256),
            nn.Dropout(0.02),         # 🔥 OTIMIZADO: Menos dropout (era implícito 0.1)
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128)
        )
        
        # 🎯 9. DEBUG COUNTER
        self.debug_counter = 0
        
        # 🚀 10. GRADIENT MONITOR ULTRA-LEVE
        self.gradient_monitor = None
        self.gradient_step_counter = 0
    
    def _force_reinitialize_action_value_nets(self):
        """🔥 CORREÇÃO CRÍTICA: Re-inicializar action_net e value_net para eliminar zeros extremos"""
        
        # Re-inicializar action_net (camada de saída de ações)
        if hasattr(self, 'action_net') and self.action_net is not None:
            for module in self.action_net.modules():
                if isinstance(module, nn.Linear):
                    # Inicialização Xavier para camada de saída
                    nn.init.xavier_uniform_(module.weight, gain=0.01)  # Gain baixo para estabilidade
                    if module.bias is not None:
                        nn.init.uniform_(module.bias, -0.001, 0.001)  # Bias muito pequeno
            print("[V6] 🔥 action_net RE-INICIALIZADO com Xavier (gain=0.01)")
        
        # Re-inicializar value_net (camada de saída de valor)
        if hasattr(self, 'value_net') and self.value_net is not None:
            for module in self.value_net.modules():
                if isinstance(module, nn.Linear):
                    # Inicialização Xavier para camada de valor
                    nn.init.xavier_uniform_(module.weight, gain=0.01)  # Gain baixo para estabilidade
                    if module.bias is not None:
                        nn.init.uniform_(module.bias, -0.001, 0.001)  # Bias muito pequeno
            print("[V6] 🔥 value_net RE-INICIALIZADO com Xavier (gain=0.01)")
    
    def _init_weights(self):
        """🔧 CORREÇÃO COMPLETA: Inicialização robusta para eliminar zeros extremos"""
        print("🔧 Aplicando inicialização avançada de pesos...")
        
        # Contador para monitorar aplicação
        modules_initialized = {
            'linear': 0,
            'attention': 0,
            'lstm_gru': 0,
            'layernorm': 0,
            'other': 0
        }
        
        for name, module in self.named_modules():
            # 🎯 1. LINEAR LAYERS - Inicialização He para ReLU
            if isinstance(module, nn.Linear):
                # He initialization para ativações ReLU
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                
                if module.bias is not None:
                    # Bias adaptativo baseado no tamanho da camada
                    fan_in = module.weight.size(1)
                    bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0.01
                    
                    # Bias ligeiramente positivo para ativação inicial
                    nn.init.uniform_(module.bias, -bound/2, bound)
                    
                    # Especial para camadas finais (action, value)
                    if 'action_net' in name or 'value_net' in name:
                        nn.init.uniform_(module.bias, -0.001, 0.001)
                        
                modules_initialized['linear'] += 1
                
            # 🎯 2. ATTENTION LAYERS - Inicialização especializada
            elif isinstance(module, nn.MultiheadAttention):
                for param_name, param in module.named_parameters():
                    if 'in_proj_weight' in param_name:
                        # Projection weights - Xavier com gain ajustado
                        nn.init.xavier_uniform_(param, gain=1.0 / np.sqrt(self.attention_heads))
                    elif 'in_proj_bias' in param_name:
                        # CRITICAL FIX: Attention bias com small random values
                        nn.init.normal_(param, mean=0.0, std=0.02)
                    elif 'out_proj.weight' in param_name:
                        nn.init.xavier_uniform_(param, gain=0.5)
                    elif 'out_proj.bias' in param_name:
                        nn.init.uniform_(param, -0.01, 0.01)
                        
                modules_initialized['attention'] += 1
                
            # 🎯 3. LSTM/GRU - Inicialização ortogonal
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                for param_name, param in module.named_parameters():
                    if 'weight_ih' in param_name:
                        # Input-to-hidden weights - orthogonal
                        nn.init.orthogonal_(param.data)
                    elif 'weight_hh' in param_name:
                        # Hidden-to-hidden weights - orthogonal
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in param_name:
                        # 🔥 CORREÇÃO CRÍTICA: LSTM/GRU bias initialization SEM ZEROS
                        # NÃO usar fill_(0) que causa 80%+ zeros!
                        
                        if isinstance(module, nn.LSTM):
                            n = param.size(0)
                            # Inicializar TODOS os gates com valores não-zero
                            nn.init.uniform_(param.data[:n//4], -0.05, 0.05)      # Input gate
                            nn.init.uniform_(param.data[n//4:n//2], 0.8, 1.2)     # Forget gate (1.0 ± 0.2)
                            nn.init.uniform_(param.data[n//2:3*n//4], -0.05, 0.05) # Cell gate  
                            nn.init.uniform_(param.data[3*n//4:], -0.05, 0.05)    # Output gate
                        elif isinstance(module, nn.GRU):
                            n = param.size(0)
                            # Inicializar TODOS os gates com valores não-zero
                            nn.init.uniform_(param.data[:n//3], -0.05, 0.05)      # Reset gate
                            nn.init.uniform_(param.data[n//3:2*n//3], 0.7, 1.0)   # Update gate
                            nn.init.uniform_(param.data[2*n//3:], -0.05, 0.05)    # New gate
                        else:
                            # Fallback para RNN simples
                            nn.init.uniform_(param.data, -0.05, 0.05)
                            
                modules_initialized['lstm_gru'] += 1
                
            # 🎯 4. LAYER NORMALIZATION
            elif isinstance(module, nn.LayerNorm):
                # Weight próximo de 1 com pequena variação
                nn.init.normal_(module.weight, mean=1.0, std=0.02)
                # Bias pequeno mas não zero
                nn.init.normal_(module.bias, mean=0.0, std=0.01)
                modules_initialized['layernorm'] += 1
                
            # 🎯 5. BATCH NORMALIZATION
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                modules_initialized['other'] += 1
        
        # 🎯 6. INICIALIZAÇÃO ESPECIAL PARA COMPONENTES CUSTOMIZADOS
        self._init_custom_components()
        
        # Log de inicialização
        total_modules = sum(modules_initialized.values())
        print(f"✅ Inicialização completa: {total_modules} módulos")
        print(f"   Linear: {modules_initialized['linear']}, Attention: {modules_initialized['attention']}")
        print(f"   LSTM/GRU: {modules_initialized['lstm_gru']}, LayerNorm: {modules_initialized['layernorm']}")
        
    def _init_custom_components(self):
        """Inicialização especializada para componentes customizados"""
        
        # Conflict Resolution Core
        if hasattr(self, 'conflict_resolution'):
            for module in self.conflict_resolution.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.uniform_(module.bias, -0.01, 0.01)
        
        # Entry Head V6
        if hasattr(self, 'entry_head'):
            for module in self.entry_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.uniform_(module.bias, -0.005, 0.005)
        
        # Management Head
        if hasattr(self, 'management_head'):
            for module in self.management_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.uniform_(module.bias, -0.005, 0.005)
        
        # Temporal Processing
        if hasattr(self, 'temporal_processor'):
            for module in self.temporal_processor.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.normal_(module.bias, 0.0, 0.01)
        
        print("✅ Componentes customizados inicializados")
    
    # 🔥 CORREÇÃO: Remover forward() customizado - usar implementação padrão do RecurrentPPO
    # O forward() padrão funciona corretamente com nosso _get_latent() customizado
        
    def _get_latent(self, obs):
        """🔥 SOBRESCREVER: Usar nossa arquitetura ao invés de mlp_extractor"""
        # Extract features usando nosso transformer
        raw_features = self.extract_features(obs)
        
        # Apply V6 processing - nossa arquitetura completa
        processed_features = self._apply_v6_processing(raw_features)
        
        # 🔗 DENSE CONNECTION: Conectar features originais direto ao final
        # Isso bypassa os LSTMs problemáticos do stable-baselines3
        if hasattr(self, 'enable_dense_connection') and self.enable_dense_connection:
            # Garantir mesma dimensão para concatenação
            if raw_features.shape[-1] != processed_features.shape[-1]:
                if not hasattr(self, 'raw_features_projector'):
                    self.raw_features_projector = nn.Linear(
                        raw_features.shape[-1], 
                        processed_features.shape[-1]
                    ).to(raw_features.device)
                projected_raw = self.raw_features_projector(raw_features)
            else:
                projected_raw = raw_features
                
            # 🔗 DENSE CONNECTION: processed + raw (residual)
            dense_features = processed_features + 0.3 * projected_raw  # 30% skip connection
            print(f"🔗 Dense connection ativa: {processed_features.shape} + 0.3*{projected_raw.shape}")
        else:
            dense_features = processed_features
        
        # 🔥 GARANTIR SHAPE CORRETO: [batch, features_dim]
        if dense_features.dim() != 2:
            dense_features = dense_features.view(dense_features.size(0), -1)
        
        # Return latents for both policy and value
        return dense_features, dense_features
        
    def _get_latent_pi(self, features):
        """Policy latent usando nossa arquitetura"""
        return features
        
    def _get_latent_vf(self, features):
        """Value latent usando nossa arquitetura"""  
        return features
    
    def _apply_v6_processing(self, input_features):
        """Processamento V6 - LIMPO E FUNCIONAL"""
        
        # 1. Projeção para dimensão correta
        if input_features.shape[-1] != self.lstm_hidden_size:
            if not hasattr(self, 'input_projector'):
                self.input_projector = nn.Linear(
                    input_features.shape[-1], 
                    self.lstm_hidden_size
                ).to(input_features.device)
            projected_features = self.input_projector(input_features)
        else:
            projected_features = input_features
        
        # 2. GRU Stabilizer (sem checkpointing V6 - V6 não tem problema de zeros)
        gru_input = projected_features.unsqueeze(1)
        gru_out, _ = self.gru_stabilizer(gru_input)
        gru_out = self.gru_norm(gru_out.squeeze(1))
        
        # 3. Attention (sem checkpointing V6 - V6 não tem problema de zeros)
        attn_input = projected_features.unsqueeze(1)
        attn_out, _ = self.attention(attn_input, attn_input, attn_input)
        attn_out = attn_out.squeeze(1)
        
        # 4. Pattern Recognition (sem checkpointing V6 - V6 não tem problema de zeros)
        patterns = self.pattern_detector(attn_out)
        
        # 5. Feature Fusion (sem checkpointing V6 - V6 não tem problema de zeros)
        fused_input = torch.cat([projected_features, gru_out, patterns], dim=-1)
        fused_features = self.feature_fusion(fused_input)
        
        # 6. Entry Head V6 (sem checkpointing V6 - V6 não tem problema de zeros) 
        entry_output = self.entry_head(fused_features)
        entry_decision = entry_output['entry_decision']
        
        # 7. Management Head
        management_decision = self.management_head(fused_features)
        
        # 8. Strategic Fusion Layer - Coordenação inteligente ATIVA (se disponível)
        if (self.strategic_fusion_enabled and 
            hasattr(self, 'strategic_fusion') and 
            self.strategic_fusion is not None):
            market_context = entry_output['market_analysis']  # 4-dim market context
            fusion_output = self.strategic_fusion(
                entry_decision, management_decision, market_context
            )
            
            # 9. Combinar com fusion
            combined = torch.cat([entry_decision, management_decision], dim=-1)
            final_output = self.final_projector(combined)
            
            # 10. Aplicar coordenação da fusion layer
            # Usar pesos da fusion para balancear entry vs management
            entry_weight = fusion_output['entry_weight']
            mgmt_weight = fusion_output['mgmt_weight']
            
            # Aplicar pesos estratégicos ao output final
            final_output = final_output * (entry_weight + mgmt_weight) * fusion_output['confidence']
        else:
            # Fallback sem fusion (compatibilidade)
            combined = torch.cat([entry_decision, management_decision], dim=-1)
            final_output = self.final_projector(combined)
        
        # 9. Debug a cada 1000 steps
        self.debug_counter += 1
        if self.debug_counter % 1000 == 0:
            self._debug_v6_output(entry_output)
        
        # Inicializar Strategic Fusion Layer na primeira execução (após carregamento)
        if not self._fusion_initialization_attempted:
            self._initialize_strategic_fusion()
        
        return final_output
    
    def _initialize_strategic_fusion(self):
        """Inicializa Strategic Fusion Layer após carregamento do modelo"""
        self._fusion_initialization_attempted = True
        
        try:
            # ✅ SEMPRE CRIAR Strategic Fusion Layer para modelos V6
            fusion_layer = StrategicFusionLayer(
                entry_dim=64,
                management_dim=32,
                market_dim=4
            )
            
            # Mover para device correto
            device = next(self.parameters()).device if list(self.parameters()) else 'cpu'
            fusion_layer = fusion_layer.to(device)
            
            # ✅ ATIVAR SEMPRE
            self.strategic_fusion = fusion_layer
            self.strategic_fusion_enabled = True
            print("[V6 INFO] Strategic Fusion Layer ATIVADA FORCADAMENTE - Legion V1 com fusion!")
            
        except Exception as e:
            # Fallback: mesmo com erro, tentar criar layer simples
            print(f"[V6 WARNING] Erro ao criar Strategic Fusion Layer: {str(e)[:100]}")
            print("[V6 INFO] Tentando criar layer simples...")
            
            try:
                # Layer simples como fallback
                self.strategic_fusion = nn.Sequential(
                    nn.Linear(100, 32),  # entry(64) + management(32) + market(4) = 100
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.Sigmoid()
                )
                device = next(self.parameters()).device if list(self.parameters()) else 'cpu'
                self.strategic_fusion = self.strategic_fusion.to(device)
                self.strategic_fusion_enabled = True
                print("[V6 INFO] Strategic Fusion Layer SIMPLES ativada com sucesso!")
                
            except Exception as e2:
                self.strategic_fusion = None
                self.strategic_fusion_enabled = False
                print(f"[V6 ERROR] Falha total na criacao de Strategic Fusion Layer: {str(e2)[:100]}")
    
    def _debug_v6_output(self, entry_output):
        """Debug V6 - Verificar se gates estão funcionando"""
        gates = entry_output['gates']
        composite_score = entry_output['composite_score']
        final_gate = entry_output['final_gate']
        threshold = entry_output['threshold_used']
        
        print(f"\n{'='*60}")
        print(f"DEBUG V6 - STEP {self.debug_counter}")
        print(f"{'='*60}")
        
        for gate_name, gate_value in gates.items():
            print(f"Gate {gate_name}: {gate_value.mean().item():.3f}")
        
        print(f"Composite Score: {composite_score.mean().item():.3f}")
        print(f"Final Gate: {final_gate.mean().item():.3f}")
        print(f"Threshold: {threshold.mean().item():.3f}")
        
        # Verificar se gates estão realmente filtrando
        filtering_rate = (final_gate < 0.5).float().mean().item()
        print(f"Filtering Rate: {filtering_rate:.1%} (trades being blocked)")
        
        print(f"{'='*60}\n")
    
    def _get_latent_from_obs(
        self,
        obs: PyTorchObs,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor
    ) -> torch.Tensor:
        """Método interno compatível com RecurrentPPO"""
        features = self.extract_features(obs)
        return self._apply_v6_processing(features)
    
    def update_entry_memory(self, pnl, duration, volatility, confidence, success, market_regime):
        """Atualiza memória da entry head"""
        if hasattr(self.entry_head, 'update_memory'):
            self.entry_head.update_memory(pnl, duration, volatility, confidence, success, market_regime)
    
    def setup_differential_weight_decay(self, optimizer):
        """🛡️ ESTRATÉGIA 4: Weight Decay Diferenciado por Componente"""
        if not self.enable_differential_weight_decay:
            return optimizer
            
        try:
            print("🛡️ Configurando Weight Decay Diferenciado...")
            
            # Coletar parâmetros por categoria
            lstm_params = []
            attention_params = []
            other_params = []
            
            for name, param in self.named_parameters():
                if 'lstm' in name.lower():
                    lstm_params.append(param)
                elif 'attn' in name.lower() or 'attention' in name.lower():
                    attention_params.append(param)
                else:
                    other_params.append(param)
            
            # Criar grupos de parâmetros com weight decay diferenciado
            param_groups = []
            
            if lstm_params:
                param_groups.append({
                    'params': lstm_params,
                    'weight_decay': self.lstm_weight_decay,
                    'name': 'lstm_components'
                })
                print(f"   🎯 LSTM params: {len(lstm_params)} com weight_decay={self.lstm_weight_decay}")
            
            if attention_params:
                param_groups.append({
                    'params': attention_params,
                    'weight_decay': self.attention_weight_decay,
                    'name': 'attention_components'
                })
                print(f"   🎯 Attention params: {len(attention_params)} com weight_decay={self.attention_weight_decay}")
            
            if other_params:
                param_groups.append({
                    'params': other_params,
                    'weight_decay': self.standard_weight_decay,
                    'name': 'other_components'
                })
                print(f"   🎯 Other params: {len(other_params)} com weight_decay={self.standard_weight_decay}")
            
            # Recriar optimizer com grupos diferenciados
            if param_groups:
                new_optimizer = torch.optim.Adam(param_groups, lr=optimizer.param_groups[0]['lr'])
                print(f"✅ Weight Decay Diferenciado ativado com {len(param_groups)} grupos")
                return new_optimizer
            else:
                print("⚠️ Nenhum parâmetro encontrado para weight decay diferenciado")
                return optimizer
                
        except Exception as e:
            print(f"❌ Erro ao configurar weight decay diferenciado: {e}")
            return optimizer
    
    def activate_dead_neurons(self):
        """🛡️ ESTRATÉGIA 5: Ativação de Neurônios Dormentes - OTIMIZADA"""
        if not self.enable_dead_neuron_activation:
            return
            
        try:
            activated_count = 0
            total_params = 0
            
            for name, param in self.named_parameters():
                if param.grad is not None:
                    total_params += param.numel()
                    
                    # 🚀 PROTOCOLO DE SALVAMENTO DE NEURÔNIOS - ULTRA-ESPECÍFICO
                    if 'lstm_critic.weight_hh_l0' in name:
                        # 🚨 UTI INTENSIVA: Componente persistentemente crítico (80%+ problemas)
                        threshold = self.dead_neuron_threshold * 6.0  # 6e-5 MÁXIMO agressivo
                        noise_scale = self.activation_noise_scale * 6.0  # 6e-4 MÁXIMO forte
                        rescue_mode = "🚨 UTI INTENSIVA"
                    elif 'weight_hh_l0' in name:
                        # 🔴 CRÍTICO: Outros weight_hh_l0
                        threshold = self.dead_neuron_threshold * 3.0  # 3e-5 muito agressivo
                        noise_scale = self.activation_noise_scale * 3.0  # 3e-4 muito forte
                        rescue_mode = "🔴 CRÍTICO"
                    elif 'weight_ih_l0' in name or 'bias' in name:
                        # 🟡 MODERADO: Componentes moderadamente críticos
                        threshold = self.dead_neuron_threshold * 1.5  # 1.5e-5
                        noise_scale = self.activation_noise_scale * 1.5  # 1.5e-4
                        rescue_mode = "🟡 MODERADO"
                    else:
                        # 🟢 NORMAL: Componentes menos críticos
                        threshold = self.dead_neuron_threshold
                        noise_scale = self.activation_noise_scale
                        rescue_mode = "🟢 NORMAL"
                    
                    # Detectar gradientes muito pequenos (neurônios dormentes)
                    dead_mask = torch.abs(param.grad) < threshold
                    dead_count = dead_mask.sum().item()
                    
                    if dead_count > 0:
                        # 🧬 PROTOCOLO DE RESSURREIÇÃO: Verificar se foi operado recentemente
                        post_surgery_boost = getattr(self, '_post_surgery_lr_boost', {})
                        if name in post_surgery_boost and post_surgery_boost[name] > 0:
                            # Pós-cirurgia: ressurreição agressiva
                            resurrection_scale = noise_scale * post_surgery_boost[name]
                            noise = torch.randn_like(param.grad[dead_mask]) * resurrection_scale
                            param.grad[dead_mask] += noise
                            
                            # Decrementar contador de boost
                            post_surgery_boost[name] -= 0.005  # Decay gradual
                            if post_surgery_boost[name] <= 0:
                                del post_surgery_boost[name]
                            self._post_surgery_lr_boost = post_surgery_boost
                            
                            activated_count += dead_count
                            print(f"   🧬 RESSURREIÇÃO PÓS-CIRÚRGICA: {name} boost {post_surgery_boost.get(name, 0):.2f}x")
                        else:
                            # Protocolo normal: ruído adaptativo padrão
                            noise = torch.randn_like(param.grad[dead_mask]) * noise_scale
                            param.grad[dead_mask] += noise
                            activated_count += dead_count
                        
                        # 🛡️ PROTOCOLO DE SALVAMENTO: Log detalhado para componentes críticos
                        if 'lstm' in name.lower() and dead_count > param.numel() * 0.1:  # >10% dead
                            dead_pct = dead_count/param.numel()*100
                            threshold_str = f"{threshold:.1e}"
                            noise_str = f"{noise_scale:.1e}"
                            
                            # Log com código de emergência baseado na criticidade
                            if dead_pct > 80:
                                status = "🆘 COMA NEURONAL"
                            elif dead_pct > 60:
                                status = "🔴 UTI"
                            elif dead_pct > 40:
                                status = "🟡 ENFERMARIA"
                            else:
                                status = "🟢 CONSULTÓRIO"
                            
                            # 🚨 ALERTA ESPECIAL para lstm_critic.weight_hh_l0
                            if 'lstm_critic.weight_hh_l0' in name and dead_pct > 75:
                                print(f"🚨 ALERTA CRÍTICO: {name} em estado vegetativo há múltiplos ciclos!")
                                print(f"   📊 Histórico: Persistente >75% dormentes - considerar intervenção cirúrgica")
                                
                                # 🔪 CIRURGIA NEURONAL EXPERIMENTAL: Transplante de camada
                                if dead_pct > 95:  # >95% = morte cerebral
                                    print(f"🔪 INICIANDO CIRURGIA NEURONAL EXPERIMENTAL em {name}")
                                    self._perform_neural_surgery(name, param, dead_mask, dead_pct)
                                
                            print(f"🧠 {name}: {rescue_mode} salvou {dead_count}/{param.numel()} ({dead_pct:.1f}%) {status} [T:{threshold_str}, N:{noise_str}]")
            
            # 🚨 PROTOCOLO DE EMERGÊNCIA TOTAL
            if activated_count > 0:
                activation_rate = activated_count/total_params*100
                
                if activation_rate > 50:
                    status = "🆘 EMERGÊNCIA NEURONAL CRÍTICA"
                elif activation_rate > 30:
                    status = "🔴 ESTADO GRAVE"
                elif activation_rate > 15:
                    status = "🟡 ESTADO MODERADO"
                elif activation_rate > 5:
                    status = "🟢 CUIDADOS BÁSICOS"
                else:
                    status = "💚 NEURÔNIOS SAUDÁVEIS"
                
                print(f"🧠💊 HOSPITAL DOS NEURÔNIOS: {activated_count}/{total_params} salvos ({activation_rate:.2f}%) - {status}")
                
                # 🚨 Alerta especial para situações críticas
                if activation_rate > 40:
                    print(f"⚡ DESFIBRILADOR NEURONAL ATIVADO! Aplicando choque de {activation_rate:.1f}% nos gradientes!")
                
        except Exception as e:
            print(f"🆘 ERRO NO HOSPITAL DOS NEURÔNIOS: {e}")
            import traceback
            traceback.print_exc()
    
    def _perform_neural_surgery(self, param_name, param, dead_mask, dead_pct):
        """🔪 CIRURGIA NEURONAL EXPERIMENTAL - Transplante de camadas em coma vegetativo"""
        try:
            surgery_count = getattr(self, '_surgery_count', {})
            if param_name not in surgery_count:
                surgery_count[param_name] = 0
            surgery_count[param_name] += 1
            self._surgery_count = surgery_count
            
            # 🏥 LIMITE DE CIRURGIAS: Máximo 3 por componente para evitar instabilidade
            if surgery_count[param_name] > 3:
                print(f"   ⚠️ LIMITE DE CIRURGIAS ATINGIDO para {param_name} (3/3) - paciente terminal")
                return
            
            print(f"   🔪 CIRURGIA #{surgery_count[param_name]}/3 em {param_name}")
            print(f"   📊 Diagnóstico: {dead_pct:.1f}% neurônios em morte cerebral")
            
            # 🧬 ESTRATÉGIA 1: RESSURREIÇÃO COMPLETA - Reset Xavier/He para neurônios mortos
            if 'weight' in param_name:
                if 'lstm' in param_name:
                    # LSTM weights - usar inicialização específica
                    with torch.no_grad():
                        # Calcular fan_in e fan_out para inicialização adequada
                        fan_in = param.size(-1) if param.dim() > 1 else param.size(0)
                        fan_out = param.size(0) if param.dim() > 1 else param.size(0)
                        
                        # Xavier/Glorot para LSTM
                        std = math.sqrt(2.0 / (fan_in + fan_out))
                        
                        # Aplicar ressurreição apenas nos neurônios mortos
                        resurrection_values = torch.normal(0, std, size=param[dead_mask].shape, device=param.device)
                        param.data[dead_mask] = resurrection_values
                        
                        # Se gradiente também está morto, ressuscitar com pequeno valor
                        if param.grad is not None:
                            param.grad[dead_mask] = torch.normal(0, std * 0.1, size=param.grad[dead_mask].shape, device=param.device)
                        
                        resurrected = dead_mask.sum().item()
                        print(f"   ⚡ RESSURREIÇÃO: {resurrected} neurônios transplantados com Xavier LSTM")
            
            # 🧬 ESTRATÉGIA 2: HIBRIDIZAÇÃO - Copiar de componente saudável similar
            try:
                # Procurar componente saudável similar para doação
                donor_param = None
                donor_name = None
                
                for name, other_param in self.named_parameters():
                    if (name != param_name and 
                        param.shape == other_param.shape and
                        'lstm' in name.lower() and
                        other_param.grad is not None):
                        
                        # Verificar se é saudável (< 30% mortos)
                        other_dead_mask = torch.abs(other_param.grad) < self.dead_neuron_threshold
                        other_dead_pct = other_dead_mask.sum().item() / other_param.numel() * 100
                        
                        if other_dead_pct < 30:  # Componente saudável encontrado
                            donor_param = other_param
                            donor_name = name
                            break
                
                if donor_param is not None:
                    with torch.no_grad():
                        # Transplante híbrido: 70% donor + 30% ressurreição
                        hybrid_mask = dead_mask.clone()
                        
                        # Selecionar aleatoriamente 70% dos mortos para transplante
                        dead_indices = torch.where(dead_mask.flatten())[0]
                        if len(dead_indices) > 0:
                            transplant_count = int(len(dead_indices) * 0.7)
                            transplant_indices = dead_indices[torch.randperm(len(dead_indices))[:transplant_count]]
                            
                            # Criar máscara de transplante
                            transplant_mask = torch.zeros_like(dead_mask.flatten(), dtype=torch.bool)
                            transplant_mask[transplant_indices] = True
                            transplant_mask = transplant_mask.reshape(dead_mask.shape)
                            
                            # Realizar transplante
                            param.data[transplant_mask] = donor_param.data[transplant_mask].clone()
                            
                            print(f"   🧬 TRANSPLANTE: {transplant_count} neurônios doados de {donor_name}")
            
            except Exception as transplant_error:
                print(f"   ⚠️ Falha no transplante: {transplant_error} - prosseguindo apenas com ressurreição")
            
            # 🔧 ESTRATÉGIA 3: LEARNING RATE BOOST - Acelerar aprendizado pós-cirurgia
            post_surgery_lr_boost = getattr(self, '_post_surgery_lr_boost', {})
            post_surgery_lr_boost[param_name] = 5.0  # 5x learning rate por 1000 steps
            self._post_surgery_lr_boost = post_surgery_lr_boost
            
            print(f"   💉 PÓS-CIRURGIA: Learning rate boost 5x por 1000 steps")
            print(f"   ✅ CIRURGIA CONCLUÍDA: {param_name} operado com sucesso")
            
        except Exception as e:
            print(f"   ❌ FALHA CIRÚRGICA em {param_name}: {e}")
            print(f"   🚨 Paciente permanece em coma vegetativo")
    
    def apply_emergency_learning_rate_adaptation(self, optimizer):
        """💉 LEARNING RATE ADAPTATIVO DE EMERGÊNCIA - Resposta dinâmica a crises neurais"""
        try:
            # Calcular severidade geral do sistema neural
            total_params = 0
            critical_params = 0
            vegetative_params = 0
            
            for name, param in self.named_parameters():
                if param.grad is not None:
                    total_params += param.numel()
                    
                    # Detectar neurônios críticos e vegetativos
                    dead_mask = torch.abs(param.grad) < self.dead_neuron_threshold * 3.0
                    dead_count = dead_mask.sum().item()
                    dead_pct = dead_count / param.numel() * 100
                    
                    if dead_pct > 95:  # Vegetativo
                        vegetative_params += dead_count
                    elif dead_pct > 75:  # Crítico
                        critical_params += dead_count
            
            # Calcular índices de emergência
            vegetative_rate = vegetative_params / total_params * 100 if total_params > 0 else 0
            critical_rate = critical_params / total_params * 100 if total_params > 0 else 0
            emergency_index = vegetative_rate + (critical_rate * 0.5)
            
            # 🚨 PROTOCOLOS DE EMERGÊNCIA ESCALONADOS
            if emergency_index > 40:  # EMERGÊNCIA MÁXIMA
                lr_multiplier = 8.0
                emergency_level = "🆘 EMERGÊNCIA MÁXIMA"
            elif emergency_index > 25:  # EMERGÊNCIA ALTA
                lr_multiplier = 5.0
                emergency_level = "🚨 EMERGÊNCIA ALTA"
            elif emergency_index > 15:  # EMERGÊNCIA MODERADA
                lr_multiplier = 3.0
                emergency_level = "⚠️ EMERGÊNCIA MODERADA"
            elif emergency_index > 8:  # ALERTA
                lr_multiplier = 2.0
                emergency_level = "🟡 ALERTA NEURAL"
            else:  # ESTÁVEL
                lr_multiplier = 1.0
                emergency_level = "🟢 ESTÁVEL"
            
            # Aplicar adaptação ao optimizer
            if lr_multiplier > 1.0:
                for group in optimizer.param_groups:
                    # Salvar LR original se não existir
                    if 'original_lr' not in group:
                        group['original_lr'] = group['lr']
                    
                    # Aplicar multiplicador adaptativo
                    group['lr'] = group['original_lr'] * lr_multiplier
                
                print(f"💉 ADAPTAÇÃO LR EMERGENCIAL: {emergency_level}")
                print(f"   📊 Índice Emergência: {emergency_index:.1f}% (V:{vegetative_rate:.1f}% + C:{critical_rate:.1f}%)")
                print(f"   ⚡ Learning Rate Boost: {lr_multiplier}x")
                
                # Registrar intervenção para decaimento gradual
                emergency_decay = getattr(self, '_emergency_lr_decay', {})
                emergency_decay['multiplier'] = lr_multiplier
                emergency_decay['steps_remaining'] = 2000  # Duração da emergência
                self._emergency_lr_decay = emergency_decay
            
            # 📉 DECAIMENTO GRADUAL PÓS-EMERGÊNCIA
            else:
                emergency_decay = getattr(self, '_emergency_lr_decay', {})
                if 'steps_remaining' in emergency_decay and emergency_decay['steps_remaining'] > 0:
                    # Decaimento linear
                    steps_remaining = emergency_decay['steps_remaining']
                    original_mult = emergency_decay['multiplier']
                    current_mult = 1.0 + (original_mult - 1.0) * (steps_remaining / 2000)
                    
                    for group in optimizer.param_groups:
                        if 'original_lr' in group:
                            group['lr'] = group['original_lr'] * current_mult
                    
                    emergency_decay['steps_remaining'] -= 1
                    if emergency_decay['steps_remaining'] <= 0:
                        # Restaurar LR original
                        for group in optimizer.param_groups:
                            if 'original_lr' in group:
                                group['lr'] = group['original_lr']
                        print(f"🟢 EMERGÊNCIA NEURAL RESOLVIDA: Learning rate restaurado")
                        del self._emergency_lr_decay
                    
                    self._emergency_lr_decay = emergency_decay
                    
        except Exception as e:
            print(f"❌ Erro na adaptação de learning rate: {e}")
    
    def perform_component_replacement_therapy(self):
        """🧬 TERAPIA DE SUBSTITUIÇÃO DE COMPONENTES - Último recurso para casos terminais"""
        try:
            replacement_count = getattr(self, '_replacement_count', {})
            replacements_performed = 0
            
            for name, param in self.named_parameters():
                if param.grad is not None:
                    # Detectar componentes terminais (99%+ mortos por múltiplos ciclos)
                    dead_mask = torch.abs(param.grad) < self.dead_neuron_threshold * 10
                    dead_pct = dead_mask.sum().item() / param.numel() * 100
                    
                    # Critérios para substituição total:
                    # 1. >99% neurônios mortos
                    # 2. Já passou por 3 cirurgias sem sucesso
                    # 3. Componente LSTM crítico
                    surgery_count = getattr(self, '_surgery_count', {}).get(name, 0)
                    
                    if (dead_pct > 99 and 
                        surgery_count >= 3 and 
                        'lstm' in name.lower() and
                        name not in replacement_count):
                        
                        print(f"🧬 INICIANDO TERAPIA DE SUBSTITUIÇÃO para {name}")
                        print(f"   📊 Componente terminal: {dead_pct:.1f}% morto após {surgery_count} cirurgias")
                        
                        # 🔄 SUBSTITUIÇÃO COMPLETA: Recriar componente do zero
                        with torch.no_grad():
                            if 'weight' in name:
                                # Reinicialização completa com método moderno (Kaiming He)
                                if param.dim() >= 2:
                                    torch.nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                                else:
                                    torch.nn.init.normal_(param, mean=0, std=0.01)
                                
                                # Gradiente inicial suave
                                if param.grad is not None:
                                    param.grad.zero_()
                                    param.grad.add_(torch.randn_like(param.grad) * 1e-6)
                                
                            elif 'bias' in name:
                                # Bias: inicialização zero + pequeno ruído
                                torch.nn.init.zeros_(param)
                                param.add_(torch.randn_like(param) * 1e-7)
                                
                                if param.grad is not None:
                                    param.grad.zero_()
                                    param.grad.add_(torch.randn_like(param.grad) * 1e-7)
                        
                        # 🏥 TERAPIA PÓS-SUBSTITUIÇÃO
                        # Learning rate boost extensivo (10x por 5000 steps)
                        post_surgery_boost = getattr(self, '_post_surgery_lr_boost', {})
                        post_surgery_boost[name] = 10.0
                        self._post_surgery_lr_boost = post_surgery_boost
                        
                        # Registrar substituição
                        replacement_count[name] = 1
                        self._replacement_count = replacement_count
                        replacements_performed += 1
                        
                        print(f"   🧬 SUBSTITUIÇÃO COMPLETA: Componente recriado com Kaiming He")
                        print(f"   💉 Terapia intensiva: LR boost 10x por 5000 steps")
                        print(f"   ✅ NOVA VIDA: {name} reiniciado biologicamente")
            
            if replacements_performed > 0:
                print(f"🧬💊 TERAPIA DE SUBSTITUIÇÃO: {replacements_performed} componentes substituídos")
                print(f"   🔬 Status: Componentes regenerados aguardando adaptação neural")
                
        except Exception as e:
            print(f"❌ Erro na terapia de substituição: {e}")
    
    def get_v6_diagnostics(self):
        """Retorna diagnósticos V6"""
        if hasattr(self.entry_head, 'trade_memory'):
            memory_stats = {
                'total_trades': int(self.entry_head.memory_ptr.item()),
                'avg_pnl': self.entry_head.trade_memory[:, 0].mean().item(),
                'avg_duration': self.entry_head.trade_memory[:, 1].mean().item(),
                'success_rate': self.entry_head.trade_memory[:, 4].mean().item()
            }
            return memory_stats
        return {}
    
    def update_fusion_learning(self, decision_outcome, performance_metrics, market_state):
        """Atualiza aprendizado da Strategic Fusion Layer"""
        if (hasattr(self, 'strategic_fusion') and 
            self.strategic_fusion_enabled and 
            self.strategic_fusion is not None):
            self.strategic_fusion.update_learning(decision_outcome, performance_metrics, market_state)
    
    def get_fusion_diagnostics(self):
        """Retorna diagnósticos da Strategic Fusion Layer"""
        if (hasattr(self, 'strategic_fusion') and 
            self.strategic_fusion_enabled and 
            self.strategic_fusion is not None):
            return {
                'fusion_status': 'active',
                'conflict_resolution_stats': self.strategic_fusion.conflict_resolver.get_stats(),
                'temporal_coordination_stats': self.strategic_fusion.temporal_coordinator.get_stats(),
                'adaptive_learning_stats': self.strategic_fusion.adaptive_learner.get_stats()
            }
        return {'fusion_status': 'disabled', 'reason': 'model_without_fusion_layer'}
    
    def reset_v6_state(self):
        """Reset estado V6"""
        if hasattr(self.entry_head, 'trade_memory'):
            self.entry_head.trade_memory.zero_()
            self.entry_head.memory_ptr.zero_()
        
        # Strategic Fusion Layer desativada - sem reset necessário
        
        self.debug_counter = 0
        print("[V6] Estado resetado (Strategic Fusion Layer desativada)")
    
    def get_initial_state(self, batch_size: int = 1) -> RNNStates:
        """
        🎯 CORREÇÃO CRÍTICA: Método get_initial_state faltando!
        Necessário para compatibilidade com RecurrentPPO
        """
        device = next(self.parameters()).device
        
        # Criar estados LSTM com estrutura correta para RecurrentPPO
        from collections import namedtuple
        
        # Definir estrutura de estados
        LSTMStates = namedtuple('LSTMStates', ['pi', 'vf'])
        
        # Estados para policy (actor) e value function (critic)
        pi_states = (
            torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_size, device=device),
            torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        )
        
        vf_states = (
            torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_size, device=device),
            torch.zeros(self.n_lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        )
        
        return LSTMStates(pi=pi_states, vf=vf_states)
    
    def setup_gradient_monitoring(self, check_frequency: int = 500, log_dir: str = "gradient_logs"):
        """
        🔍 CONFIGURAR MONITORAMENTO DE GRADIENTES
        Sistema automático para garantir qualidade dos gradientes
        """
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
            from gradient_health_monitor import GradientHealthMonitor
            
            self.gradient_monitor = GradientHealthMonitor(
                model=self,
                log_dir=log_dir,
                check_frequency=check_frequency,
                gradient_clip_value=1.0,
                min_gradient_norm=1e-8,
                max_gradient_norm=10.0
            )
            
            print(f"✅ Gradient Health Monitor ativado (check a cada {check_frequency} steps)")
            return True
            
        except ImportError as e:
            print(f"⚠️ Gradient Health Monitor não disponível: {e}")
            return False
    
    def check_and_fix_gradients(self, step: int) -> Dict:
        """
        🔧 VERIFICAR E CORRIGIR GRADIENTES
        Chamado automaticamente durante treinamento
        """
        if not hasattr(self, 'gradient_monitor'):
            return {}
        
        # Verificar saúde dos gradientes
        health_report = self.gradient_monitor.check_gradient_health(step)
        
        if health_report and health_report.get('health_score', 1.0) < 0.5:
            # Aplicar correções automáticas
            corrections = self.gradient_monitor.apply_gradient_corrections()
            
            if corrections > 0:
                print(f"🔧 Step {step}: {corrections} correções de gradiente aplicadas")
                print(f"   Saúde: {health_report['health_score']:.3f}")
            
            # Alertar sobre problemas críticos
            if health_report['health_score'] < 0.3:
                print(f"⚠️ Step {step}: Gradientes problemáticos!")
                for rec in health_report.get('recommendations', [])[:2]:
                    print(f"   💡 {rec}")
        
        return health_report
    
    def get_gradient_health_summary(self) -> Dict:
        """📊 Obter resumo da saúde dos gradientes"""
        if hasattr(self, 'gradient_monitor'):
            return self.gradient_monitor.get_health_summary()
        return {"status": "monitoring_disabled"}
    
    def save_gradient_report(self) -> Optional[str]:
        """💾 Salvar relatório detalhado dos gradientes"""
        if hasattr(self, 'gradient_monitor'):
            return self.gradient_monitor.save_detailed_report()
        return None


def get_v6_kwargs() -> Dict[str, Any]:
    """Retorna kwargs para TwoHeadV6Intelligent48h com configurações LSTM customizadas"""
    return {
        'features_extractor_class': TradingTransformerFeatureExtractor,
        'features_extractor_kwargs': {'features_dim': 128, 'seq_len': 10},
        'net_arch': [dict(pi=[128, 64], vf=[128, 64])],
        
        # 🎯 CONFIGURAÇÕES LSTM CORE
        'lstm_hidden_size': 128,
        'n_lstm_layers': 2,
        'attention_heads': 4,
        
        # 🎯 CONFIGURAÇÕES LSTM CUSTOMIZADAS - ANTI-ZEROS
        'lstm_dropout': 0.1,               # 10% dropout nas LSTMs principais
        'lstm_layer_norm': True,           # Layer normalization nas LSTMs
        'lstm_gradient_clipping': 0.5,     # Gradient clipping específico para LSTMs
        
        # 🛡️ CONFIGURAÇÕES AVANÇADAS LSTM
        'lstm_bias_init': 1.0,             # Inicialização bias forget gate
        'lstm_weight_init': 'xavier',      # Inicialização Xavier para weights
        'lstm_hidden_init': 'orthogonal',  # Inicialização ortogonal para hidden weights
        'enable_lstm_smart_init': True,    # Ativar inicialização inteligente
        'lstm_recurrent_dropout': 0.0,     # Dropout recorrente (mantido 0 para estabilidade)
        'lstm_bidirectional': False,       # Manter unidirecional para trading sequencial
        
        # 🔥 CONFIGURAÇÕES DE OPTIMIZAÇÃO
        'enable_gradient_health_monitor': True,  # Monitor de saúde dos gradientes
        'gradient_check_frequency': 500,         # Verificar a cada 500 steps
        'auto_gradient_fix': True,               # Auto-correção de gradientes
        'gradient_alert_threshold': 0.3,         # Alertar se >30% zeros
        
        # 🛡️ ESTRATÉGIAS ANTI-ZEROS AVANÇADAS - OTIMIZADAS
        'enable_differential_weight_decay': True,  # Weight decay diferenciado
        'lstm_weight_decay': 5e-7,                 # 🚀 REDUZIDO: 5e-7 (era 1e-6) para LSTMs críticos
        'attention_weight_decay': 2e-6,            # 🚀 REDUZIDO: 2e-6 (era 5e-6) para Attention
        'standard_weight_decay': 1e-4,             # Weight decay padrão para outros
        'enable_dead_neuron_activation': True,     # Ativação de neurônios dormentes
        'dead_neuron_threshold': 1e-5,             # Base threshold (será adaptado por componente)
        'activation_noise_scale': 1e-4             # Base noise scale (será adaptado por componente)
    }