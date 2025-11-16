"""
🚀 V7 UPGRADE COMPONENTS - Sistema Completo de Melhorias

Componentes para upgrade da TwoHeadV7Simple:
1. Market Regime Detection (mínimo compartilhado)
2. Enhanced Memory Bank (regime-aware)
3. Gradient Balancing System
4. Update Frequency Controller
5. Neural Breathing Monitor
6. Adaptive Learning Rates

COMPATIBILIDADE: Mantém arquitetura V7 existente
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Deque
from collections import defaultdict, deque
import pickle
import os
import time

class MarketRegimeDetector(nn.Module):
    """🎯 Market Regime Detection MÍNIMO (compartilhado entre Actor/Critic)"""
    
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.volatility_threshold = 0.02
        self.trend_window = 20
        
        # Network LEVE para classificação
        self.regime_classifier = nn.Sequential(
            nn.Linear(input_dim, 32),  # Bem pequeno
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 4)  # bull, bear, sideways, volatile
        )
        
        # Buffers para cálculos (pequenos)
        self.register_buffer('price_history', torch.zeros(self.trend_window))
        self.register_buffer('returns_history', torch.zeros(self.trend_window))
        
    def update_price_history(self, current_price: float):
        """Atualiza histórico com preço atual"""
        self.price_history = torch.roll(self.price_history, -1)
        self.price_history[-1] = current_price
        
        if len(self.price_history) > 1:
            ret = (current_price - self.price_history[-2]) / (self.price_history[-2] + 1e-8)
            self.returns_history = torch.roll(self.returns_history, -1)
            self.returns_history[-1] = ret
    
    def calculate_volatility(self) -> float:
        """Calcula volatilidade realizadas"""
        if torch.all(self.returns_history == 0):
            return 0.0
        return torch.std(self.returns_history).item()
    
    def calculate_trend(self) -> float:
        """Calcula força da tendência"""
        if torch.all(self.price_history == 0):
            return 0.0
        return ((self.price_history[-1] - self.price_history[0]) / (self.price_history[0] + 1e-8)).item()
    
    def detect_regime_rules(self) -> int:
        """Detecção baseada em regras simples"""
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
    
    def forward(self, features: torch.Tensor) -> int:
        """
        Forward pass LEVE
        Returns: regime_id (0=bull, 1=bear, 2=sideways, 3=volatile)
        """
        with torch.no_grad():  # Sem gradientes para economizar
            # Ensure features has batch dimension
            if features.dim() == 1:
                features = features.unsqueeze(0)
            elif features.dim() > 2:
                features = features.mean(dim=list(range(1, features.dim()-1)))  # Flatten to 2D
            
            # Take mean over batch if needed
            if features.shape[0] > 1:
                features = features.mean(dim=0, keepdim=True)
                
            regime_logits = self.regime_classifier(features)
            regime_neural = torch.argmax(regime_logits).item()
            regime_rules = self.detect_regime_rules()
            
            # Combinação: 70% neural + 30% rules
            final_regime = int(0.7 * regime_neural + 0.3 * regime_rules)
            final_regime = max(0, min(3, final_regime))  # clamp [0,3]
            
        return final_regime

class EnhancedMemoryBank:
    """💾 Enhanced Memory Bank - 10x mais inteligente que TradeMemoryBank"""
    
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
        
        # Metadados inteligentes
        self.memory_metadata = {
            0: {'success_rate': 0.0, 'avg_reward': 0.0, 'count': 0, 'best_reward': -float('inf')},
            1: {'success_rate': 0.0, 'avg_reward': 0.0, 'count': 0, 'best_reward': -float('inf')},
            2: {'success_rate': 0.0, 'avg_reward': 0.0, 'count': 0, 'best_reward': -float('inf')},
            3: {'success_rate': 0.0, 'avg_reward': 0.0, 'count': 0, 'best_reward': -float('inf')},
        }
        
        self.regime_names = {0: 'bull', 1: 'bear', 2: 'sideways', 3: 'volatile'}
        
    def store_memory(self, regime: int, state: np.ndarray, action: np.ndarray, 
                    reward: float, next_state: np.ndarray, done: bool):
        """Armazena memória completa"""
        success = reward > 0.0
        
        memory_entry = {
            'state': state.copy() if isinstance(state, np.ndarray) else np.array(state),
            'action': action.copy() if isinstance(action, np.ndarray) else np.array(action),
            'reward': float(reward),
            'next_state': next_state.copy() if isinstance(next_state, np.ndarray) else np.array(next_state),
            'done': bool(done),
            'success': success,
            'timestamp': time.time(),
            'episode_step': len(self.regime_memories[regime])
        }
        
        self.regime_memories[regime].append(memory_entry)
        
        # Atualiza metadados
        meta = self.memory_metadata[regime]
        meta['count'] += 1
        
        # Running average do reward
        alpha = 0.01  # Learning rate para média móvel
        meta['avg_reward'] = (1 - alpha) * meta['avg_reward'] + alpha * reward
        
        # Best reward tracking
        meta['best_reward'] = max(meta['best_reward'], reward)
        
        # Success rate (últimas 100 memórias)
        recent_memories = list(self.regime_memories[regime])[-100:]
        if recent_memories:
            meta['success_rate'] = sum(1 for m in recent_memories if m['success']) / len(recent_memories)
    
    def query_similar_memories(self, regime: int, current_state: np.ndarray, k: int = 5) -> np.ndarray:
        """Busca memórias similares usando cosine similarity"""
        if not self.regime_memories[regime]:
            return np.zeros((k, self.feature_dim))
        
        memories = list(self.regime_memories[regime])
        current_state = np.array(current_state).flatten()
        
        # Calcula similaridades
        similarities = []
        for memory in memories:
            memory_state = np.array(memory['state']).flatten()
            
            # Garante mesmo tamanho
            min_len = min(len(current_state), len(memory_state))
            current_truncated = current_state[:min_len]
            memory_truncated = memory_state[:min_len]
            
            if min_len > 0:
                # Cosine similarity
                norm_current = np.linalg.norm(current_truncated) + 1e-8
                norm_memory = np.linalg.norm(memory_truncated) + 1e-8
                similarity = np.dot(current_truncated, memory_truncated) / (norm_current * norm_memory)
                
                # Bonus para memórias bem-sucedidas
                success_bonus = 0.1 if memory['success'] else 0.0
                final_similarity = similarity + success_bonus
                
                similarities.append((final_similarity, memory['state']))
        
        if not similarities:
            return np.zeros((k, self.feature_dim))
        
        # Ordena e pega top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_memories = [sim[1] for sim in similarities[:k]]
        
        # Pad com zeros se necessário
        result = []
        for i in range(k):
            if i < len(top_memories):
                state = np.array(top_memories[i]).flatten()
                if len(state) >= self.feature_dim:
                    result.append(state[:self.feature_dim])
                else:
                    padded = np.zeros(self.feature_dim)
                    padded[:len(state)] = state
                    result.append(padded)
            else:
                result.append(np.zeros(self.feature_dim))
        
        return np.array(result)
    
    def get_regime_context(self, regime: int, current_state: np.ndarray) -> np.ndarray:
        """Retorna contexto específico do regime"""
        if not self.regime_memories[regime]:
            # Contexto inicial estruturado
            return np.random.randn(self.feature_dim) * 0.05
        
        # Busca memórias similares
        similar_memories = self.query_similar_memories(regime, current_state, k=3)
        
        # Pondera por success rate do regime
        success_weight = max(0.1, self.memory_metadata[regime]['success_rate'])
        context = similar_memories.mean(axis=0) * success_weight
        
        # Adiciona ruído pequeno para evitar saturação
        noise = np.random.randn(self.feature_dim) * 0.01
        context += noise
        
        return context
    
    def get_regime_stats(self) -> Dict:
        """Estatísticas completas dos regimes"""
        stats = {}
        for regime, name in self.regime_names.items():
            meta = self.memory_metadata[regime]
            stats[name] = {
                'count': meta['count'],
                'success_rate': meta['success_rate'],
                'avg_reward': meta['avg_reward'],
                'best_reward': meta['best_reward'],
                'memory_size': len(self.regime_memories[regime]),
                'recent_activity': len([m for m in self.regime_memories[regime] 
                                     if time.time() - m['timestamp'] < 3600])  # Última hora
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
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load(self, filepath: str) -> bool:
        """Carrega memory bank"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                self.regime_memories = {k: deque(v, maxlen=self.memory_size) 
                                      for k, v in data['regime_memories'].items()}
                self.memory_metadata = data['memory_metadata']
                self.memory_size = data['memory_size']
                self.feature_dim = data['feature_dim']
                return True
            except Exception as e:
                print(f"Erro ao carregar memory bank: {e}")
                return False
        return False

class GradientBalancer:
    """⚖️ Sistema completo de balanceamento de gradientes"""
    
    def __init__(self):
        # Learning rates base
        self.base_actor_lr = 3e-4
        self.base_critic_lr = 1e-3
        
        # Thresholds para intervenção
        self.zero_threshold = 0.3
        self.extreme_zero_threshold = 0.5
        
        # Histórico de zeros (para breathing monitoring)
        self.lstm_zero_history = deque(maxlen=200)
        self.attention_zero_history = deque(maxlen=100)
        self.bias_zero_history = deque(maxlen=200)
        
        # Update frequency control
        self.actor_update_ratio = 2  # Actor updates 2x mais que critic
        self.critic_update_ratio = 1
        self.update_counter = 0
        
    def count_zeros_in_gradients(self, model: nn.Module, prefix: str = "") -> Dict[str, float]:
        """Conta zeros nos gradientes de forma eficiente"""
        zero_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_flat = param.grad.flatten()
                zero_percentage = (torch.abs(grad_flat) < 1e-8).float().mean().item()
                zero_stats[f"{prefix}{name}"] = zero_percentage
                
                # Detecta componentes críticos
                if 'lstm' in name.lower() and 'bias' in name.lower():
                    self.bias_zero_history.append(zero_percentage)
                elif 'lstm' in name.lower() and 'weight_hh' in name.lower():
                    self.lstm_zero_history.append(zero_percentage)
        
        return zero_stats
    
    def analyze_breathing_pattern(self) -> Dict[str, any]:
        """🫁 Analisa padrão de 'respiração neural'"""
        if len(self.lstm_zero_history) < 10:
            return {'status': 'insufficient_data', 'pattern': 'unknown'}
        
        recent_zeros = list(self.lstm_zero_history)[-20:]
        current_zeros = recent_zeros[-1] if recent_zeros else 0.0
        trend = np.polyfit(range(len(recent_zeros)), recent_zeros, 1)[0] if len(recent_zeros) > 5 else 0.0
        
        # Detecta fase da respiração
        if current_zeros < 0.05:
            phase = "inspirando"  # LSTM muito ativo
            status = "🟢 INSPIRAÇÃO - LSTM hiperativo"
        elif current_zeros > 0.4:
            phase = "expirando"   # LSTM descansando
            status = "🔴 EXPIRAÇÃO - LSTM descansando"
        else:
            phase = "normal"      # Equilibrado
            status = "🟡 RESPIRAÇÃO NORMAL - LSTM equilibrado"
        
        # Calcula variabilidade (respiração saudável tem oscilação)
        variability = np.std(recent_zeros) if len(recent_zeros) > 5 else 0.0
        is_breathing = variability > 0.05  # Oscilação > 5%
        
        return {
            'status': status,
            'phase': phase,
            'current_zeros': current_zeros,
            'trend': trend,
            'variability': variability,
            'is_breathing_healthy': is_breathing,
            'history': recent_zeros
        }
    
    def should_update_actor(self) -> bool:
        """Determina se deve atualizar actor neste step"""
        return (self.update_counter % (self.actor_update_ratio + self.critic_update_ratio)) < self.actor_update_ratio
    
    def should_update_critic(self) -> bool:
        """Determina se deve atualizar critic neste step"""
        return not self.should_update_actor()
    
    def get_adaptive_learning_rates(self, lstm_health: Dict[str, float]) -> Dict[str, float]:
        """📈 Learning rates adaptativos baseados na saúde do LSTM"""
        breathing = self.analyze_breathing_pattern()
        current_zeros = breathing['current_zeros']
        
        # Adaptação baseada na fase da respiração
        if breathing['phase'] == 'inspirando':
            # LSTM hiperativo - acalmar um pouco
            actor_lr_mult = 0.8
            critic_lr_mult = 1.2
        elif breathing['phase'] == 'expirando':
            # LSTM dormindo - acordar
            actor_lr_mult = 1.5
            critic_lr_mult = 0.8
        else:
            # Normal - manter base
            actor_lr_mult = 1.0
            critic_lr_mult = 1.0
        
        # Ajuste fino baseado em tendência
        if breathing['trend'] > 0.02:  # Zeros aumentando muito
            actor_lr_mult *= 1.2  # Boost actor
        elif breathing['trend'] < -0.02:  # Zeros diminuindo muito
            actor_lr_mult *= 0.9  # Desacelerar
        
        return {
            'actor_lr': self.base_actor_lr * actor_lr_mult,
            'critic_lr': self.base_critic_lr * critic_lr_mult,
            'breathing_status': breathing['status'],
            'lr_multipliers': {'actor': actor_lr_mult, 'critic': critic_lr_mult}
        }
    
    def next_update_step(self):
        """Avança contador de updates"""
        self.update_counter += 1
    
    def get_update_statistics(self) -> Dict[str, any]:
        """Estatísticas de update frequency"""
        total_updates = self.update_counter
        actor_updates = sum(1 for i in range(total_updates) 
                          if i % (self.actor_update_ratio + self.critic_update_ratio) < self.actor_update_ratio)
        critic_updates = total_updates - actor_updates
        
        return {
            'total_updates': total_updates,
            'actor_updates': actor_updates,
            'critic_updates': critic_updates,
            'actor_ratio': actor_updates / max(1, total_updates),
            'critic_ratio': critic_updates / max(1, total_updates),
            'breathing_pattern': self.analyze_breathing_pattern()
        }

class NeuralBreathingMonitor:
    """🫁 Monitor dedicado para 'respiração neural'"""
    
    def __init__(self, history_size: int = 500):
        self.history_size = history_size
        self.zero_history = deque(maxlen=history_size)
        self.gradient_norm_history = deque(maxlen=history_size)
        self.timestamp_history = deque(maxlen=history_size)
        
    def record_breathing_data(self, lstm_zeros: float, gradient_norm: float):
        """Registra dados de respiração"""
        self.zero_history.append(lstm_zeros)
        self.gradient_norm_history.append(gradient_norm)
        self.timestamp_history.append(time.time())
    
    def analyze_breathing_cycle(self) -> Dict[str, any]:
        """Analisa ciclo completo de respiração"""
        if len(self.zero_history) < 50:
            return {'status': 'learning', 'cycle_detected': False}
        
        zeros = np.array(list(self.zero_history))
        
        # Detecta picos e vales (inspiração/expiração)
        from scipy.signal import find_peaks
        
        peaks, _ = find_peaks(zeros, height=0.3, distance=10)
        valleys, _ = find_peaks(-zeros, height=-0.1, distance=10)
        
        # Calcula período médio
        if len(peaks) > 2:
            peak_intervals = np.diff(peaks)
            avg_period = np.mean(peak_intervals)
            period_std = np.std(peak_intervals)
            regularity = 1.0 / (1.0 + period_std)  # Mais regular = respiração mais saudável
        else:
            avg_period = 0
            regularity = 0
        
        # Estado atual
        current_zeros = zeros[-1]
        recent_trend = np.polyfit(range(20), zeros[-20:], 1)[0] if len(zeros) >= 20 else 0
        
        if current_zeros < 0.05:
            current_phase = "inspiração_profunda"
        elif current_zeros > 0.4:
            current_phase = "expiração_profunda"
        elif recent_trend > 0.02:
            current_phase = "inspirando"
        elif recent_trend < -0.02:
            current_phase = "expirando"
        else:
            current_phase = "pausa_respiratória"
        
        return {
            'status': 'breathing_detected' if len(peaks) > 1 else 'irregular',
            'cycle_detected': len(peaks) > 1,
            'current_phase': current_phase,
            'current_zeros': current_zeros,
            'avg_period': avg_period,
            'regularity_score': regularity,
            'peak_count': len(peaks),
            'valley_count': len(valleys),
            'breathing_amplitude': np.max(zeros) - np.min(zeros) if len(zeros) > 0 else 0,
            'health_score': min(1.0, regularity * (1.0 - abs(current_zeros - 0.25)) * 2)
        }
    
    def get_breathing_visualization_data(self) -> Dict[str, any]:
        """Dados para visualizar respiração"""
        return {
            'zeros_history': list(self.zero_history),
            'gradient_norms': list(self.gradient_norm_history),
            'timestamps': list(self.timestamp_history),
            'cycle_analysis': self.analyze_breathing_cycle()
        }

class V7UpgradeManager:
    """🚀 Manager principal para upgrade da V7"""
    
    def __init__(self, v7_policy, feature_dim: int = 256):
        self.v7_policy = v7_policy
        self.feature_dim = feature_dim
        
        # Componentes do upgrade
        self.regime_detector = MarketRegimeDetector(feature_dim)
        self.enhanced_memory = EnhancedMemoryBank(memory_size=10000, feature_dim=feature_dim)
        self.gradient_balancer = GradientBalancer()
        self.breathing_monitor = NeuralBreathingMonitor()
        
        # Estado interno
        self.current_regime = 2  # Start with sideways
        self.last_features = None
        self.total_steps = 0
        
        print("🚀 V7 UPGRADE MANAGER inicializado:")
        print("   📊 Market Regime Detection")
        print("   💾 Enhanced Memory Bank (10k memories)")
        print("   ⚖️ Gradient Balancing System")
        print("   🫁 Neural Breathing Monitor")
        print("   🔄 Update Frequency Controller")
    
    def pre_training_step(self, features: torch.Tensor) -> Dict[str, any]:
        """Executado ANTES do training step"""
        # Detectar regime de mercado
        self.current_regime = self.regime_detector(features)
        self.last_features = features.detach().cpu().numpy()
        
        # Verificar se deve atualizar actor/critic
        should_update_actor = self.gradient_balancer.should_update_actor()
        should_update_critic = self.gradient_balancer.should_update_critic()
        
        # Learning rates adaptativos
        lstm_health = self.gradient_balancer.analyze_breathing_pattern()
        adaptive_lrs = self.gradient_balancer.get_adaptive_learning_rates(lstm_health)
        
        return {
            'regime': self.current_regime,
            'regime_name': self.enhanced_memory.regime_names[self.current_regime],
            'should_update_actor': should_update_actor,
            'should_update_critic': should_update_critic,
            'adaptive_lrs': adaptive_lrs,
            'breathing_status': lstm_health
        }
    
    def post_training_step(self, experience: Dict[str, any]) -> Dict[str, any]:
        """Executado APÓS o training step"""
        # Armazenar experiência no memory bank
        if self.last_features is not None and 'reward' in experience:
            self.enhanced_memory.store_memory(
                regime=self.current_regime,
                state=self.last_features,
                action=experience.get('action', np.zeros(3)),
                reward=experience['reward'],
                next_state=experience.get('next_state', self.last_features),
                done=experience.get('done', False)
            )
        
        # Analisar gradientes
        actor_zeros = self.gradient_balancer.count_zeros_in_gradients(
            self.v7_policy.v7_actor_lstm, "actor_"
        )
        critic_zeros = self.gradient_balancer.count_zeros_in_gradients(
            self.v7_policy.v7_critic_mlp, "critic_"
        )
        
        # Monitor respiração neural
        if actor_zeros:
            avg_lstm_zeros = np.mean([v for k, v in actor_zeros.items() if 'lstm' in k.lower()])
            grad_norm = sum(p.grad.norm().item() for p in self.v7_policy.v7_actor_lstm.parameters() 
                          if p.grad is not None)
            self.breathing_monitor.record_breathing_data(avg_lstm_zeros, grad_norm)
        
        # Avançar contador
        self.gradient_balancer.next_update_step()
        self.total_steps += 1
        
        return {
            'actor_zeros': actor_zeros,
            'critic_zeros': critic_zeros,
            'breathing_cycle': self.breathing_monitor.analyze_breathing_cycle(),
            'memory_stats': self.enhanced_memory.get_regime_stats(),
            'update_stats': self.gradient_balancer.get_update_statistics()
        }
    
    def get_enhanced_context(self, current_state: np.ndarray) -> np.ndarray:
        """Retorna contexto enhanced do memory bank"""
        return self.enhanced_memory.get_regime_context(self.current_regime, current_state)
    
    def get_comprehensive_status(self) -> Dict[str, any]:
        """Status completo do sistema"""
        return {
            'regime': {
                'current': self.current_regime,
                'name': self.enhanced_memory.regime_names[self.current_regime],
            },
            'memory': self.enhanced_memory.get_regime_stats(),
            'breathing': self.breathing_monitor.analyze_breathing_cycle(),
            'gradients': self.gradient_balancer.get_update_statistics(),
            'total_steps': self.total_steps
        }
    
    def save_state(self, base_path: str):
        """Salva estado completo"""
        os.makedirs(base_path, exist_ok=True)
        
        # Memory bank
        self.enhanced_memory.save(f"{base_path}/enhanced_memory_bank.pkl")
        
        # Regime detector
        torch.save(self.regime_detector.state_dict(), f"{base_path}/regime_detector.pth")
        
        # Breathing data
        breathing_data = self.breathing_monitor.get_breathing_visualization_data()
        with open(f"{base_path}/breathing_data.pkl", 'wb') as f:
            pickle.dump(breathing_data, f)
        
        print(f"✅ V7 Upgrade state salvo em: {base_path}")
    
    def load_state(self, base_path: str) -> bool:
        """Carrega estado completo"""
        try:
            # Memory bank
            if not self.enhanced_memory.load(f"{base_path}/enhanced_memory_bank.pkl"):
                print("⚠️ Memory bank não encontrado, iniciando vazio")
            
            # Regime detector
            detector_path = f"{base_path}/regime_detector.pth"
            if os.path.exists(detector_path):
                self.regime_detector.load_state_dict(torch.load(detector_path))
                print("✅ Regime detector carregado")
            
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar state: {e}")
            return False