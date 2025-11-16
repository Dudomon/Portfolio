"""
🔍 SIMPLE PATTERN DETECTOR - OPUS FASE 2
ML pattern detection simplificado baseado em momentum
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from collections import deque
from dataclasses import dataclass

@dataclass
class PatternSignal:
    """Sinal de pattern detectado"""
    pattern_type: str
    strength: float      # 0.0 - 1.0
    confidence: float    # 0.0 - 1.0
    bonus_multiplier: float
    description: str

class SimplePatternDetector:
    """Detector de padrões baseado em momentum e correlações simples"""
    
    def __init__(self, history_size: int = 50):
        self.history_size = history_size
        
        # 📊 HISTÓRICO DE DADOS
        self.trade_history = deque(maxlen=history_size)
        self.price_history = deque(maxlen=history_size)
        self.volume_history = deque(maxlen=history_size)
        
        # 🎯 PATTERN WEIGHTS
        self.pattern_weights = {
            "momentum_alignment": 1.5,      # Entry alinhado com momentum
            "volume_confirmation": 1.2,     # Volume confirma movimento
            "time_pattern": 0.8,           # Padrões de horário
            "win_streak_momentum": 1.3,    # Momentum de win streak
            "mean_reversion": 1.0          # Padrões de reversão
        }
        
        # 📈 TRACKING DE PERFORMANCE POR PATTERN
        self.pattern_performance = {pattern: [] for pattern in self.pattern_weights.keys()}
        
    def add_trade_data(self, trade_data: Dict[str, Any], market_data: Dict[str, Any]):
        """Adicionar dados de trade e mercado para análise"""
        # Trade data
        trade_entry = {
            'pnl': trade_data.get('pnl_usd', 0.0),
            'duration': trade_data.get('duration_steps', 0),
            'entry_price': trade_data.get('entry_price', 0.0),
            'exit_price': trade_data.get('exit_price', 0.0),
            'trade_type': trade_data.get('type', 'unknown'),
            'timestamp': trade_data.get('timestamp', 0),
            'sl_points': trade_data.get('sl_points', 0.0),
            'tp_points': trade_data.get('tp_points', 0.0)
        }
        self.trade_history.append(trade_entry)
        
        # Market data
        if 'price' in market_data:
            self.price_history.append(market_data['price'])
        if 'volume' in market_data:
            self.volume_history.append(market_data['volume'])
    
    def detect_patterns(self, current_context: Dict[str, Any]) -> List[PatternSignal]:
        """Detectar padrões e retornar sinais"""
        if len(self.trade_history) < 5:
            return []
        
        patterns = []
        
        # 1. MOMENTUM ALIGNMENT PATTERN
        momentum_signal = self._detect_momentum_alignment(current_context)
        if momentum_signal:
            patterns.append(momentum_signal)
        
        # 2. VOLUME CONFIRMATION PATTERN
        volume_signal = self._detect_volume_confirmation(current_context)
        if volume_signal:
            patterns.append(volume_signal)
        
        # 3. WIN STREAK MOMENTUM PATTERN
        streak_signal = self._detect_win_streak_momentum()
        if streak_signal:
            patterns.append(streak_signal)
        
        # 4. TIME-BASED PATTERNS
        time_signal = self._detect_time_patterns(current_context)
        if time_signal:
            patterns.append(time_signal)
        
        # 5. MEAN REVERSION PATTERN
        reversion_signal = self._detect_mean_reversion()
        if reversion_signal:
            patterns.append(reversion_signal)
        
        return patterns
    
    def _detect_momentum_alignment(self, context: Dict[str, Any]) -> PatternSignal:
        """Detectar alinhamento entre entry e momentum de preço"""
        if len(self.price_history) < 10 or len(self.trade_history) < 3:
            return None
        
        # Calcular momentum de preços (últimos 10 períodos)
        recent_prices = list(self.price_history)[-10:]
        price_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Analisar últimos 3 trades
        recent_trades = list(self.trade_history)[-3:]
        aligned_trades = 0
        
        for trade in recent_trades:
            trade_direction = 1 if trade['trade_type'] == 'long' else -1
            momentum_direction = 1 if price_momentum > 0 else -1
            
            if trade_direction == momentum_direction and trade['pnl'] > 0:
                aligned_trades += 1
        
        # Calcular strength baseado no alinhamento
        alignment_ratio = aligned_trades / len(recent_trades)
        momentum_strength = min(1.0, abs(price_momentum) * 100)  # Converter para %
        
        if alignment_ratio >= 0.67 and momentum_strength > 0.5:  # 2/3 aligned + momentum > 0.5%
            strength = alignment_ratio * momentum_strength
            confidence = min(0.9, strength + 0.3)
            
            return PatternSignal(
                pattern_type="momentum_alignment",
                strength=strength,
                confidence=confidence,
                bonus_multiplier=self.pattern_weights["momentum_alignment"],
                description=f"Entry aligned with {momentum_strength:.1f}% momentum"
            )
        
        return None
    
    def _detect_volume_confirmation(self, context: Dict[str, Any]) -> PatternSignal:
        """Detectar confirmação por volume"""
        if len(self.volume_history) < 5 or len(self.trade_history) < 2:
            return None
        
        # Volume médio vs atual
        recent_volumes = list(self.volume_history)[-5:]
        avg_volume = np.mean(recent_volumes[:-1])
        current_volume = recent_volumes[-1]
        
        volume_ratio = current_volume / (avg_volume + 1e-8)
        
        # Último trade teve sucesso com volume alto?
        last_trade = list(self.trade_history)[-1]
        
        if volume_ratio > 1.3 and last_trade['pnl'] > 0:  # Volume 30% acima + trade positivo
            strength = min(1.0, (volume_ratio - 1.0) * 2)  # Max strength quando volume 1.5x
            confidence = min(0.8, strength + 0.2)
            
            return PatternSignal(
                pattern_type="volume_confirmation",
                strength=strength,
                confidence=confidence,
                bonus_multiplier=self.pattern_weights["volume_confirmation"],
                description=f"Volume {volume_ratio:.1f}x confirms move"
            )
        
        return None
    
    def _detect_win_streak_momentum(self) -> PatternSignal:
        """Detectar momentum de sequência de wins"""
        if len(self.trade_history) < 5:
            return None
        
        # Analisar últimos 5 trades
        recent_trades = list(self.trade_history)[-5:]
        wins = [1 if trade['pnl'] > 0 else 0 for trade in recent_trades]
        
        # Detectar streak atual
        current_streak = 0
        for win in reversed(wins):
            if win == 1:
                current_streak += 1
            else:
                break
        
        if current_streak >= 3:  # Streak de 3+ wins
            # Força baseada no tamanho do streak e qualidade dos wins
            avg_win_size = np.mean([t['pnl'] for t in recent_trades[-current_streak:]])
            
            strength = min(1.0, current_streak / 5 + avg_win_size / 20)  # Normalizar
            confidence = min(0.85, current_streak * 0.2 + 0.25)
            
            return PatternSignal(
                pattern_type="win_streak_momentum",
                strength=strength,
                confidence=confidence,
                bonus_multiplier=self.pattern_weights["win_streak_momentum"],
                description=f"Win streak of {current_streak} trades"
            )
        
        return None
    
    def _detect_time_patterns(self, context: Dict[str, Any]) -> PatternSignal:
        """Detectar padrões baseados em horário"""
        if len(self.trade_history) < 10:
            return None
        
        current_hour = context.get('current_hour', 12)  # Default meio-dia
        
        # Analisar performance por faixas horárias
        hourly_performance = {}
        for trade in self.trade_history:
            trade_hour = trade.get('timestamp', 12) % 24  # Assumir timestamp em horas
            if trade_hour not in hourly_performance:
                hourly_performance[trade_hour] = []
            hourly_performance[trade_hour].append(trade['pnl'])
        
        # Verificar se hora atual tem boa performance histórica
        current_hour_key = current_hour
        if current_hour_key in hourly_performance and len(hourly_performance[current_hour_key]) >= 3:
            hour_performance = hourly_performance[current_hour_key]
            avg_pnl = np.mean(hour_performance)
            win_rate = len([p for p in hour_performance if p > 0]) / len(hour_performance)
            
            if avg_pnl > 2.0 and win_rate > 0.6:  # Boa performance histórica neste horário
                strength = min(1.0, (avg_pnl / 10) + win_rate)
                confidence = min(0.7, len(hour_performance) / 10)  # Mais dados = mais confiança
                
                return PatternSignal(
                    pattern_type="time_pattern",
                    strength=strength,
                    confidence=confidence,
                    bonus_multiplier=self.pattern_weights["time_pattern"],
                    description=f"Good historical performance at hour {current_hour}"
                )
        
        return None
    
    def _detect_mean_reversion(self) -> PatternSignal:
        """Detectar padrões de reversão à média"""
        if len(self.trade_history) < 8:
            return None
        
        # Analisar sequência de PnLs para detectar reversão
        recent_pnls = [trade['pnl'] for trade in list(self.trade_history)[-8:]]
        
        # Identificar se há alternância entre ganhos/perdas (mean reversion)
        alternations = 0
        for i in range(1, len(recent_pnls)):
            if (recent_pnls[i] > 0) != (recent_pnls[i-1] > 0):  # Mudança de sinal
                alternations += 1
        
        alternation_ratio = alternations / (len(recent_pnls) - 1)
        
        # Mean reversion detectado se alternância > 50% e resultado geral positivo
        total_pnl = sum(recent_pnls)
        
        if alternation_ratio > 0.5 and total_pnl > 0:
            strength = alternation_ratio * min(1.0, total_pnl / 20)
            confidence = min(0.75, alternation_ratio + 0.25)
            
            return PatternSignal(
                pattern_type="mean_reversion",
                strength=strength,
                confidence=confidence,
                bonus_multiplier=self.pattern_weights["mean_reversion"],
                description=f"Mean reversion pattern detected ({alternation_ratio:.1%} alternation)"
            )
        
        return None
    
    def calculate_pattern_bonus(self, patterns: List[PatternSignal]) -> Tuple[float, Dict[str, Any]]:
        """Calcular bônus total baseado nos padrões detectados"""
        if not patterns:
            return 0.0, {"patterns_detected": 0}
        
        total_bonus = 0.0
        pattern_info = {
            "patterns_detected": len(patterns),
            "patterns": []
        }
        
        for pattern in patterns:
            # Bônus = strength × confidence × weight
            pattern_bonus = pattern.strength * pattern.confidence * pattern.bonus_multiplier
            total_bonus += pattern_bonus
            
            pattern_info["patterns"].append({
                "type": pattern.pattern_type,
                "strength": pattern.strength,
                "confidence": pattern.confidence,
                "bonus": pattern_bonus,
                "description": pattern.description
            })
            
            # Tracking performance do pattern
            self.pattern_performance[pattern.pattern_type].append(pattern_bonus)
        
        # Cap total bonus em +8.0
        total_bonus = min(8.0, total_bonus)
        pattern_info["total_bonus"] = total_bonus
        
        return total_bonus, pattern_info
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Estatísticas de performance dos patterns"""
        stats = {}
        
        for pattern_type, performance_history in self.pattern_performance.items():
            if performance_history:
                stats[pattern_type] = {
                    "activations": len(performance_history),
                    "avg_bonus": np.mean(performance_history),
                    "max_bonus": np.max(performance_history),
                    "success_rate": len([b for b in performance_history if b > 0.5]) / len(performance_history)
                }
            else:
                stats[pattern_type] = {
                    "activations": 0,
                    "avg_bonus": 0.0,
                    "max_bonus": 0.0,
                    "success_rate": 0.0
                }
        
        return stats

def create_simple_pattern_detector(history_size: int = 50) -> SimplePatternDetector:
    """Factory function"""
    return SimplePatternDetector(history_size)