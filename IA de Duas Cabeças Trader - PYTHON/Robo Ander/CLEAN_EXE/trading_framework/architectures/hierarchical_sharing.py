"""
🏗️ HIERARCHICAL SHARING ARCHITECTURE + MEMORY BANK
Implementação completa da arquitetura top sugerida

Componentes:
- Market-Aware Shared Backbone
- LSTM Actor Branch (temporal decisions)  
- MLP Critic Branch (value estimation) + Memory Bank
- Gradient Balancing System
- Market Regime Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import pickle
import os

class MarketRegimeDetector(nn.Module):
    """Detecta regime de mercado em tempo real"""
    
    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.volatility_threshold = 0.02
        self.trend_window = 20
        
        # Network para classificação de regime
        self.regime_classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # bull, bear, sideways, volatile
        )
        
        # Buffers para cálculos de regime
        self.register_buffer('price_history', torch.zeros(self.trend_window))
        self.register_buffer('returns_history', torch.zeros(self.trend_window))
        
    def update_price_history(self, price: torch.Tensor):
        """Atualiza histórico de preços"""
        self.price_history = torch.roll(self.price_history, -1)
        self.price_history[-1] = price
        
        if len(self.price_history) > 1:
            ret = (price - self.price_history[-2]) / self.price_history[-2]
            self.returns_history = torch.roll(self.returns_history, -1)
            self.returns_history[-1] = ret
    
    def calculate_volatility(self) -> float:
        """Calcula volatilidade dos retornos"""
        if torch.all(self.returns_history == 0):
            return 0.0
        return torch.std(self.returns_history).item()
    
    def calculate_trend(self) -> float:
        """Calcula tendência dos preços"""
        if torch.all(self.price_history == 0):
            return 0.0
        return (self.price_history[-1] - self.price_history[0]) / self.price_history[0]
    
    def detect_regime_rules(self) -> int:
        """Detecção de regime baseada em regras"""
        volatility = self.calculate_volatility()
        trend = self.calculate_trend()
        
        if volatility > self.volatility_threshold:
            return 3  # volatile
        elif trend > 0.01:
            return 0  # bull
        elif trend < -0.01:
            return 1  # bear
        else:
            return 2  # sideways
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Forward pass
        Returns: (regime_logits, regime_id)
        """
        regime_logits = self.regime_classifier(features)
        regime_neural = torch.argmax(regime_logits, dim=-1).item()
        regime_rules = self.detect_regime_rules()
        
        # Combina detecção neural + regras (weighted average)
        final_regime = int(0.7 * regime_neural + 0.3 * regime_rules)
        final_regime = max(0, min(3, final_regime))  # clamp [0,3]
        
        return regime_logits, final_regime

class MemoryBank:
    """Memory Bank para armazenar padrões por regime de mercado"""
    
    def __init__(self, memory_size: int = 10000, feature_dim: int = 256):
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        
        # Memórias separadas por regime
        self.regime_memories = {
            0: deque(maxlen=memory_size),  # bull
            1: deque(maxlen=memory_size),  # bear
            2: deque(maxlen=memory_size),  # sideways
            3: deque(maxlen=memory_size),  # volatile
        }
        
        # Metadados de cada memória
        self.memory_metadata = {
            0: {'success_rate': 0.0, 'avg_reward': 0.0, 'count': 0},
            1: {'success_rate': 0.0, 'avg_reward': 0.0, 'count': 0},
            2: {'success_rate': 0.0, 'avg_reward': 0.0, 'count': 0},
            3: {'success_rate': 0.0, 'avg_reward': 0.0, 'count': 0},
        }
        
        self.regime_names = {0: 'bull', 1: 'bear', 2: 'sideways', 3: 'volatile'}
    
    def store_memory(self, regime: int, state: np.ndarray, reward: float, success: bool):
        """Armazena uma memória"""
        memory_entry = {
            'state': state.copy(),
            'reward': reward,
            'success': success,
            'timestamp': len(self.regime_memories[regime])
        }
        
        self.regime_memories[regime].append(memory_entry)
        
        # Atualiza metadados
        meta = self.memory_metadata[regime]
        meta['count'] += 1
        meta['avg_reward'] = (meta['avg_reward'] * (meta['count'] - 1) + reward) / meta['count']
        meta['success_rate'] = sum(1 for m in self.regime_memories[regime] if m['success']) / len(self.regime_memories[regime])
    
    def query_similar_memories(self, regime: int, current_state: np.ndarray, k: int = 5) -> np.ndarray:
        """Recupera memórias similares"""
        if not self.regime_memories[regime]:
            return np.zeros((k, self.feature_dim))
        
        memories = list(self.regime_memories[regime])
        
        # Calcula similaridade (cosine similarity)
        similarities = []
        for memory in memories:
            memory_state = memory['state']
            if len(memory_state) == len(current_state):
                similarity = np.dot(current_state, memory_state) / (
                    np.linalg.norm(current_state) * np.linalg.norm(memory_state) + 1e-8
                )
                similarities.append((similarity, memory['state']))
        
        if not similarities:
            return np.zeros((k, self.feature_dim))
        
        # Ordena por similaridade e retorna top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_memories = [sim[1] for sim in similarities[:k]]
        
        # Pad com zeros se necessário
        while len(top_memories) < k:
            top_memories.append(np.zeros(self.feature_dim))
        
        return np.array(top_memories)
    
    def get_regime_stats(self) -> Dict:
        """Retorna estatísticas dos regimes"""
        stats = {}
        for regime, name in self.regime_names.items():
            meta = self.memory_metadata[regime]
            stats[name] = {
                'count': meta['count'],
                'success_rate': meta['success_rate'],
                'avg_reward': meta['avg_reward'],
                'memory_size': len(self.regime_memories[regime])
            }
        return stats
    
    def save(self, filepath: str):
        """Salva memory bank"""
        save_data = {
            'regime_memories': {k: list(v) for k, v in self.regime_memories.items()},
            'memory_metadata': self.memory_metadata,
            'memory_size': self.memory_size,
            'feature_dim': self.feature_dim
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load(self, filepath: str):
        """Carrega memory bank"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.regime_memories = {k: deque(v, maxlen=self.memory_size) for k, v in data['regime_memories'].items()}
            self.memory_metadata = data['memory_metadata']
            self.memory_size = data['memory_size']
            self.feature_dim = data['feature_dim']
            return True
        return False

class MarketAwareBackbone(nn.Module):
    """Backbone compartilhado com consciência de mercado"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Market regime detector
        self.regime_detector = MarketRegimeDetector(input_dim)
        
        # Shared feature extractor (Transformer-based)
        self.feature_extractor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Regime-specific embeddings
        self.regime_embedding = nn.Embedding(4, hidden_dim)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Forward pass
        Returns: (enhanced_features, regime_id, regime_logits)
        """
        # Extract features
        if len(x.shape) == 2:  # (batch_size, features)
            x = x.unsqueeze(1)  # (batch_size, 1, features)
        
        features = self.feature_extractor(x)
        features = features.squeeze(1)  # (batch_size, features)
        
        # Detect market regime
        regime_logits, regime_id = self.regime_detector(features)
        
        # Get regime embedding
        regime_emb = self.regime_embedding(torch.tensor(regime_id, device=features.device))
        
        # Combine features with regime
        enhanced_features = torch.cat([features, regime_emb.unsqueeze(0).expand(features.size(0), -1)], dim=-1)
        enhanced_features = self.output_projection(enhanced_features)
        
        return enhanced_features, regime_id, regime_logits

class LSTMActorBranch(nn.Module):
    """Branch do Actor com LSTM + Memory Attention"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, action_dim: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        # LSTM core
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        
        # Memory attention
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Hidden state management
        self.hidden_state = None
        
    def reset_hidden(self):
        """Reset LSTM hidden state"""
        self.hidden_state = None
    
    def forward(self, shared_features: torch.Tensor, regime_memories: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        Args:
            shared_features: Features from shared backbone
            regime_memories: Memory context from current regime
        """
        if len(shared_features.shape) == 2:
            shared_features = shared_features.unsqueeze(1)  # Add sequence dimension
        
        # LSTM forward
        lstm_out, self.hidden_state = self.lstm(shared_features, self.hidden_state)
        
        # Memory attention if available
        if regime_memories is not None and regime_memories.numel() > 0:
            try:
                attended_out, _ = self.memory_attention(
                    lstm_out, regime_memories, regime_memories
                )
                lstm_out = lstm_out + attended_out  # Residual connection
            except Exception:
                pass  # Fallback to pure LSTM if attention fails
        
        # Action prediction
        if len(lstm_out.shape) == 3:
            lstm_out = lstm_out.squeeze(1)  # Remove sequence dimension
        
        actions = self.actor_head(lstm_out)
        return actions

class MLPCriticBranch(nn.Module):
    """Branch do Critic com MLP + Memory Bank Integration"""
    
    def __init__(self, input_dim: int, memory_bank: MemoryBank):
        super().__init__()
        self.input_dim = input_dim
        self.memory_bank = memory_bank
        
        # Memory context dimension
        self.memory_context_dim = 64
        
        # Memory context processor
        self.memory_processor = nn.Sequential(
            nn.Linear(memory_bank.feature_dim * 5, self.memory_context_dim),  # 5 similar memories
            nn.ReLU(),
            nn.Linear(self.memory_context_dim, self.memory_context_dim)
        )
        
        # Main MLP critic
        self.critic_mlp = nn.Sequential(
            nn.Linear(input_dim + self.memory_context_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, shared_features: torch.Tensor, regime_id: int) -> torch.Tensor:
        """
        Forward pass
        Args:
            shared_features: Features from shared backbone
            regime_id: Current market regime
        """
        batch_size = shared_features.size(0)
        
        # Query memory bank for similar states
        memory_context_list = []
        for i in range(batch_size):
            current_state = shared_features[i].detach().cpu().numpy()
            similar_memories = self.memory_bank.query_similar_memories(regime_id, current_state, k=5)
            memory_context_list.append(similar_memories.flatten())
        
        # Process memory context
        memory_context = torch.tensor(
            np.array(memory_context_list), 
            dtype=torch.float32, 
            device=shared_features.device
        )
        memory_context = self.memory_processor(memory_context)
        
        # Combine shared features with memory context
        enhanced_features = torch.cat([shared_features, memory_context], dim=-1)
        
        # Value prediction
        value = self.critic_mlp(enhanced_features)
        return value

class GradientBalancer:
    """Sistema de balanceamento de gradientes"""
    
    def __init__(self):
        self.actor_lr = 3e-4
        self.critic_lr = 1e-3
        self.backbone_lr = 1e-4
        self.zero_threshold = 0.3
        
        # Histórico de zeros
        self.lstm_zero_history = deque(maxlen=100)
        self.attention_zero_history = deque(maxlen=100)
        
    def count_zeros_in_gradients(self, model: nn.Module, prefix: str = "") -> Dict[str, float]:
        """Conta zeros nos gradientes"""
        zero_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_flat = param.grad.flatten()
                zero_percentage = (torch.abs(grad_flat) < 1e-8).float().mean().item()
                zero_stats[f"{prefix}{name}"] = zero_percentage
        
        return zero_stats
    
    def analyze_lstm_health(self, actor_branch: LSTMActorBranch) -> Dict[str, float]:
        """Analisa saúde do LSTM"""
        lstm_zeros = self.count_zeros_in_gradients(actor_branch.lstm, "lstm_")
        
        # Calcula média de zeros no LSTM
        if lstm_zeros:
            avg_lstm_zeros = np.mean(list(lstm_zeros.values()))
            self.lstm_zero_history.append(avg_lstm_zeros)
        else:
            avg_lstm_zeros = 0.0
        
        return {
            'current_lstm_zeros': avg_lstm_zeros,
            'lstm_trend': np.mean(self.lstm_zero_history) if self.lstm_zero_history else 0.0,
            'lstm_stable': avg_lstm_zeros < self.zero_threshold
        }
    
    def calculate_gradient_weights(self, lstm_health: Dict[str, float]) -> Tuple[float, float, float]:
        """Calcula pesos dos gradientes baseado na saúde do LSTM"""
        lstm_zeros = lstm_health['current_lstm_zeros']
        
        if lstm_zeros > self.zero_threshold:
            # LSTM em problemas - reduz influência do critic
            actor_weight = 0.7
            critic_weight = 0.3
            backbone_weight = 0.5
        elif lstm_zeros < 0.1:
            # LSTM saudável - balanceamento normal
            actor_weight = 0.5
            critic_weight = 0.5
            backbone_weight = 1.0
        else:
            # Zona intermediária - transição suave
            factor = (lstm_zeros - 0.1) / (self.zero_threshold - 0.1)
            actor_weight = 0.5 + 0.2 * factor
            critic_weight = 0.5 - 0.2 * factor
            backbone_weight = 1.0 - 0.5 * factor
        
        return actor_weight, critic_weight, backbone_weight
    
    def get_adaptive_learning_rates(self, lstm_health: Dict[str, float]) -> Dict[str, float]:
        """Retorna learning rates adaptativos"""
        lstm_zeros = lstm_health['current_lstm_zeros']
        
        if lstm_zeros > self.zero_threshold:
            # Aumenta LR do actor para reativar LSTM
            actor_lr_mult = 1.5
            critic_lr_mult = 0.8
            backbone_lr_mult = 0.8
        elif lstm_zeros < 0.05:
            # LSTM muito ativo - reduz LR para estabilizar
            actor_lr_mult = 0.8
            critic_lr_mult = 1.2
            backbone_lr_mult = 1.0
        else:
            # Normal
            actor_lr_mult = 1.0
            critic_lr_mult = 1.0
            backbone_lr_mult = 1.0
        
        return {
            'actor_lr': self.actor_lr * actor_lr_mult,
            'critic_lr': self.critic_lr * critic_lr_mult,
            'backbone_lr': self.backbone_lr * backbone_lr_mult
        }

class HierarchicalSharingPPO(nn.Module):
    """
    🏆 ARQUITETURA COMPLETA: Hierarchical Sharing + Memory Bank + Gradient Balancing
    """
    
    def __init__(self, 
                 observation_dim: int,
                 action_dim: int = 3,
                 hidden_dim: int = 256,
                 memory_bank_size: int = 10000):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Memory Bank
        self.memory_bank = MemoryBank(memory_bank_size, hidden_dim)
        
        # Shared backbone
        self.backbone = MarketAwareBackbone(observation_dim, hidden_dim)
        
        # Actor branch (LSTM)
        self.actor_branch = LSTMActorBranch(hidden_dim, hidden_dim // 2, action_dim)
        
        # Critic branch (MLP + Memory Bank)
        self.critic_branch = MLPCriticBranch(hidden_dim, self.memory_bank)
        
        # Gradient balancer
        self.gradient_balancer = GradientBalancer()
        
        # Training metrics
        self.training_step = 0
        self.regime_stats = defaultdict(int)
        
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass completo
        Returns: (actions, values)
        """
        # Shared backbone processing
        shared_features, regime_id, regime_logits = self.backbone(observations)
        
        # Update regime stats
        self.regime_stats[regime_id] += 1
        
        # Actor branch (actions)
        # Query memory for regime-specific context
        if self.training and self.memory_bank.regime_memories[regime_id]:
            current_state = shared_features[0].detach().cpu().numpy()
            regime_memories = self.memory_bank.query_similar_memories(regime_id, current_state, k=3)
            regime_memories = torch.tensor(regime_memories, dtype=torch.float32, device=shared_features.device)
            regime_memories = regime_memories.unsqueeze(0)  # Add batch dimension
        else:
            regime_memories = None
        
        actions = self.actor_branch(shared_features, regime_memories)
        
        # Critic branch (values)
        values = self.critic_branch(shared_features, regime_id)
        
        return actions, values
    
    def store_experience(self, observation: np.ndarray, reward: float, done: bool):
        """Armazena experiência no memory bank"""
        if hasattr(self, '_last_regime_id') and hasattr(self, '_last_features'):
            success = reward > 0  # Simple success definition
            self.memory_bank.store_memory(
                self._last_regime_id,
                self._last_features,
                reward,
                success
            )
    
    def analyze_gradient_health(self) -> Dict[str, any]:
        """Analisa saúde dos gradientes"""
        lstm_health = self.gradient_balancer.analyze_lstm_health(self.actor_branch)
        
        # Contabiliza zeros em diferentes componentes
        backbone_zeros = self.gradient_balancer.count_zeros_in_gradients(self.backbone, "backbone_")
        actor_zeros = self.gradient_balancer.count_zeros_in_gradients(self.actor_branch, "actor_")
        critic_zeros = self.gradient_balancer.count_zeros_in_gradients(self.critic_branch, "critic_")
        
        return {
            'lstm_health': lstm_health,
            'backbone_zeros': backbone_zeros,
            'actor_zeros': actor_zeros,
            'critic_zeros': critic_zeros,
            'regime_stats': dict(self.regime_stats),
            'memory_stats': self.memory_bank.get_regime_stats()
        }
    
    def get_adaptive_learning_rates(self) -> Dict[str, float]:
        """Retorna learning rates adaptativos"""
        lstm_health = self.gradient_balancer.analyze_lstm_health(self.actor_branch)
        return self.gradient_balancer.get_adaptive_learning_rates(lstm_health)
    
    def save_memory_bank(self, filepath: str):
        """Salva memory bank"""
        self.memory_bank.save(filepath)
    
    def load_memory_bank(self, filepath: str) -> bool:
        """Carrega memory bank"""
        return self.memory_bank.load(filepath)
    
    def reset_episode(self):
        """Reset para novo episódio"""
        self.actor_branch.reset_hidden()
    
    def get_breathing_status(self) -> Dict[str, any]:
        """Retorna status da 'respiração neural'"""
        lstm_health = self.gradient_balancer.analyze_lstm_health(self.actor_branch)
        
        current_zeros = lstm_health['current_lstm_zeros']
        if current_zeros < 0.05:
            status = "🟢 INSPIRANDO - LSTM muito ativo"
        elif current_zeros > 0.4:
            status = "🔴 EXPIRANDO - LSTM descansando"
        else:
            status = "🟡 RESPIRAÇÃO NORMAL - LSTM equilibrado"
        
        return {
            'status': status,
            'current_zeros': current_zeros,
            'zero_history': list(self.gradient_balancer.lstm_zero_history),
            'is_healthy': lstm_health['lstm_stable']
        }