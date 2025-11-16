"""
🧠 PROGRESSIVE REWARD SHAPING - OPUS FASE 2
Curriculum learning embutido para acelerar convergência
"""

import numpy as np
from typing import Dict, Any, Tuple
from enum import Enum

class TrainingPhase(Enum):
    """Fases de treinamento com características específicas"""
    EXPLORATION = "exploration"      # 0-100k steps: Foco em descoberta
    REFINEMENT = "refinement"        # 100k-500k steps: Foco em consistência  
    MASTERY = "mastery"             # 500k+ steps: Foco em performance

class ProgressiveRewardShaper:
    """Sistema de reward shaping baseado em curriculum learning"""
    
    def __init__(self):
        self.total_steps = 0
        self.current_phase = TrainingPhase.EXPLORATION
        self.phase_history = []
        
        # 🎯 THRESHOLDS DAS FASES
        self.exploration_threshold = 100_000
        self.refinement_threshold = 500_000
        
        # 🔥 MULTIPLICADORES POR FASE
        self.phase_multipliers = {
            TrainingPhase.EXPLORATION: {
                "base_reward": 0.7,           # Reward reduzido para incentivar exploração
                "exploration_bonus": 1.5,     # Bônus por variety de ações
                "consistency_penalty": 0.3,   # Penalidade baixa por inconsistência
                "risk_tolerance": 1.5          # Maior tolerância a riscos
            },
            TrainingPhase.REFINEMENT: {
                "base_reward": 0.9,           # Reward quase normal
                "exploration_bonus": 0.8,     # Menos foco em exploração
                "consistency_penalty": 1.0,   # Penalidade normal
                "risk_tolerance": 1.0         # Tolerância normal
            },
            TrainingPhase.MASTERY: {
                "base_reward": 1.0,           # Reward completo
                "exploration_bonus": 0.3,     # Exploração mínima
                "consistency_penalty": 1.5,   # Penalidade alta por inconsistência
                "risk_tolerance": 0.8         # Menor tolerância a riscos
            }
        }
        
        # 📊 TRACKING DE MÉTRICAS POR FASE
        self.phase_metrics = {
            "exploration": {"trades_count": 0, "variety_score": 0.0, "discovery_bonus": 0.0},
            "refinement": {"consistency_score": 0.0, "win_rate_trend": 0.0, "stability_bonus": 0.0},
            "mastery": {"performance_score": 0.0, "efficiency_ratio": 0.0, "mastery_bonus": 0.0}
        }
    
    def update_step_count(self, step_count: int):
        """Atualizar contagem de steps e fase atual"""
        self.total_steps = step_count
        old_phase = self.current_phase
        
        # Determinar fase atual
        if step_count < self.exploration_threshold:
            self.current_phase = TrainingPhase.EXPLORATION
        elif step_count < self.refinement_threshold:
            self.current_phase = TrainingPhase.REFINEMENT
        else:
            self.current_phase = TrainingPhase.MASTERY
        
        # Log mudança de fase
        if old_phase != self.current_phase:
            self.phase_history.append({
                "step": step_count,
                "from_phase": old_phase.value,
                "to_phase": self.current_phase.value
            })
            print(f"[CURRICULUM] Phase transition at step {step_count}: {old_phase.value} -> {self.current_phase.value}")
    
    def shape_reward(self, base_reward: float, context: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Aplicar shaping baseado na fase atual"""
        phase_config = self.phase_multipliers[self.current_phase]
        shaping_info = {
            "phase": self.current_phase.value,
            "base_reward": base_reward,
            "phase_multipliers": {}
        }
        
        # 1. APLICAR MULTIPLICADOR BASE
        shaped_reward = base_reward * phase_config["base_reward"]
        shaping_info["phase_multipliers"]["base"] = phase_config["base_reward"]
        
        # 2. BÔNUS ESPECÍFICO DA FASE
        phase_bonus = 0.0
        
        if self.current_phase == TrainingPhase.EXPLORATION:
            phase_bonus = self._calculate_exploration_bonus(context, phase_config)
            shaping_info["exploration_bonus"] = phase_bonus
            
        elif self.current_phase == TrainingPhase.REFINEMENT:
            phase_bonus = self._calculate_refinement_bonus(context, phase_config)
            shaping_info["refinement_bonus"] = phase_bonus
            
        elif self.current_phase == TrainingPhase.MASTERY:
            phase_bonus = self._calculate_mastery_bonus(context, phase_config)
            shaping_info["mastery_bonus"] = phase_bonus
        
        # 3. AJUSTES DE TOLERÂNCIA A RISCO
        risk_adjustment = self._calculate_risk_adjustment(context, phase_config)
        shaping_info["risk_adjustment"] = risk_adjustment
        
        final_reward = shaped_reward + phase_bonus + risk_adjustment
        shaping_info["final_reward"] = final_reward
        shaping_info["total_bonus"] = phase_bonus + risk_adjustment
        
        return final_reward, shaping_info
    
    def _calculate_exploration_bonus(self, context: Dict[str, Any], config: Dict[str, float]) -> float:
        """Bônus para incentivar exploração na fase inicial"""
        exploration_bonus = 0.0
        
        # Bônus por variety de duração de trades
        trade_durations = context.get('recent_trade_durations', [])
        if len(trade_durations) >= 3:
            duration_variety = np.std(trade_durations) / (np.mean(trade_durations) + 1e-8)
            exploration_bonus += min(2.0, duration_variety * config["exploration_bonus"])
        
        # Bônus por experimentar diferentes ranges SL/TP
        sl_variety = context.get('sl_variety_score', 0.0)
        tp_variety = context.get('tp_variety_score', 0.0)
        exploration_bonus += (sl_variety + tp_variety) * 0.5 * config["exploration_bonus"]
        
        # Bônus por trading em diferentes sessões
        session_variety = context.get('session_variety_score', 0.0)
        exploration_bonus += session_variety * config["exploration_bonus"]
        
        return min(5.0, exploration_bonus)  # Cap em +5
    
    def _calculate_refinement_bonus(self, context: Dict[str, Any], config: Dict[str, float]) -> float:
        """Bônus para consistência na fase de refinamento"""
        refinement_bonus = 0.0
        
        # Bônus por win rate estável
        recent_wins = context.get('recent_win_sequence', [])
        if len(recent_wins) >= 10:
            win_rate = np.mean(recent_wins)
            win_stability = 1.0 - np.std(recent_wins)  # Menos volatilidade = mais bônus
            if win_rate > 0.5:  # Só bonificar se win rate > 50%
                refinement_bonus += win_rate * win_stability * 3.0
        
        # Bônus por drawdown controlado
        max_drawdown = context.get('recent_max_drawdown', 100.0)
        if max_drawdown < 10.0:  # Drawdown < 10%
            refinement_bonus += (10.0 - max_drawdown) * 0.2
        
        # Bônus por consistency em profit/loss ratio
        avg_profit = context.get('avg_profit_per_trade', 0.0)
        avg_loss = context.get('avg_loss_per_trade', 0.0)
        if avg_loss < 0 and avg_profit > 0:
            profit_loss_ratio = avg_profit / abs(avg_loss)
            if 1.2 <= profit_loss_ratio <= 2.5:  # Range ótimo
                refinement_bonus += profit_loss_ratio * 0.8
        
        return min(4.0, refinement_bonus)  # Cap em +4
    
    def _calculate_mastery_bonus(self, context: Dict[str, Any], config: Dict[str, float]) -> float:
        """Bônus para performance de elite na fase de maestria"""
        mastery_bonus = 0.0
        
        # Bônus por Sharpe ratio alto
        sharpe_ratio = context.get('sharpe_ratio', 0.0)
        if sharpe_ratio > 1.5:
            mastery_bonus += min(3.0, (sharpe_ratio - 1.5) * 2.0)
        
        # Bônus por efficiency (profit per trade / time)
        profit_per_minute = context.get('profit_per_minute', 0.0)
        if profit_per_minute > 0.1:  # $0.1 por minuto
            mastery_bonus += min(2.0, profit_per_minute * 10)
        
        # Bônus por consistency em high-performance
        recent_performance = context.get('recent_performance_scores', [])
        if len(recent_performance) >= 5:
            avg_performance = np.mean(recent_performance)
            performance_stability = 1.0 - (np.std(recent_performance) / (avg_performance + 1e-8))
            if avg_performance > 0.8:  # Performance > 80%
                mastery_bonus += avg_performance * performance_stability * 2.0
        
        return min(6.0, mastery_bonus)  # Cap em +6
    
    def _calculate_risk_adjustment(self, context: Dict[str, Any], config: Dict[str, float]) -> float:
        """Ajuste baseado na tolerância a risco da fase"""
        risk_adjustment = 0.0
        
        # Penalidade por trades muito arriscados (baseado na fase)
        risk_level = context.get('current_risk_level', 1.0)  # 1.0 = normal
        risk_tolerance = config["risk_tolerance"]
        
        if risk_level > risk_tolerance:
            # Penalidade crescente por excesso de risco
            excess_risk = risk_level - risk_tolerance
            risk_adjustment -= excess_risk * 2.0
        elif risk_level < risk_tolerance * 0.5:
            # Penalidade por risco muito baixo (pode indicar stagnação)
            risk_adjustment -= (risk_tolerance * 0.5 - risk_level) * 1.0
        
        return max(-3.0, risk_adjustment)  # Cap em -3
    
    def get_phase_info(self) -> Dict[str, Any]:
        """Informações da fase atual para debugging"""
        return {
            "current_phase": self.current_phase.value,
            "total_steps": self.total_steps,
            "progress_to_next_phase": self._calculate_phase_progress(),
            "phase_config": self.phase_multipliers[self.current_phase],
            "phase_transitions": len(self.phase_history),
            "last_transition": self.phase_history[-1] if self.phase_history else None
        }
    
    def _calculate_phase_progress(self) -> float:
        """Progresso atual dentro da fase (0.0 - 1.0)"""
        if self.current_phase == TrainingPhase.EXPLORATION:
            return min(1.0, self.total_steps / self.exploration_threshold)
        elif self.current_phase == TrainingPhase.REFINEMENT:
            steps_in_phase = self.total_steps - self.exploration_threshold
            phase_duration = self.refinement_threshold - self.exploration_threshold
            return min(1.0, steps_in_phase / phase_duration)
        else:  # MASTERY
            return 1.0  # Fase final, sempre 100%

def create_progressive_shaper() -> ProgressiveRewardShaper:
    """Factory function"""
    return ProgressiveRewardShaper()