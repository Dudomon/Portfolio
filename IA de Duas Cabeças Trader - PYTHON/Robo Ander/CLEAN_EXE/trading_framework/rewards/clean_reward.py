"""
🧹 SISTEMA DE RECOMPENSAS LIMPO E MELHORADO
Sistema ultra-simplificado focado apenas no essencial + elementos críticos

🎯 PRINCÍPIOS MANTIDOS:
1. PnL direto como recompensa principal
2. Penalidades mínimas e pontuais
3. Incentivos simples para atividade
4. Zero complexidade desnecessária

🚀 MELHORIAS INTEGRADAS:
5. Context awareness básico (regime de mercado simples)
6. Position management simplificado
7. SL/TP awareness básico
8. Drawdown protection simples
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Configuração padrão para compatibilidade
CLEAN_REWARD_CONFIG = {
    "target_trades_per_day": 16,
    "min_trade_duration": 3,
    "max_drawdown_threshold": 0.15,
    "trend_lookback": 20,
    "max_positions": 3,
    "weights": {
        "pnl_direct": 5.0,
        "win_bonus": 2.0,
        "loss_penalty": -1.0,
        "trade_action": 0.2,
        "daily_activity": 0.3,
        "target_zone_bonus": 1.0,
        "flip_flop_penalty": -8.0,
        "micro_trade_penalty": -5.0,
        "hold_penalty": -0.1,
        "context_bonus": 0.5,
        "sltp_quality": 0.3,
        "drawdown_protection": -2.0,
        "position_management": 0.2,
    }
}

class CleanRewardCalculator:
    """
    🧹 SISTEMA DE RECOMPENSAS ULTRA-LIMPO V2
    Focado no essencial + elementos críticos que faltavam
    """
    
    def __init__(self, initial_balance: float = 500.0):
        self.initial_balance = initial_balance
        self.step_count = 0
        
        # Pesos balanceados (sem alterar filosofia)
        self.weights = {
            # 💰 CORE - PnL direto (65% do peso - mantido como principal)
            "pnl_direct": 5.0,           # Recompensa = PnL real * 5
            "win_bonus": 2.0,            # +2.0 bônus por trade vencedor
            "loss_penalty": -1.0,        # -1.0 penalidade por trade perdedor
            
            # 🎯 ATIVIDADE BÁSICA (20% do peso)
            "trade_action": 0.0,         # 🚨 DESABILITADO: Sem bônus por fazer trade
            "daily_activity": 0.0,       # 🚨 DESABILITADO: Sem bônus por atividade diária
            "target_zone_bonus": 0.0,    # 🚨 DESABILITADO: Sem bônus por zona target
            
            # 🛡️ DISCIPLINA BÁSICA (10% do peso)
            "flip_flop_penalty": -8.0,   # -8.0 por flip-flop
            "micro_trade_penalty": -5.0, # -5.0 por micro-trade
            "hold_penalty": 0.0,        # 🚨 DESABILITADO: Sem penalidade por hold
            
            # 🚀 MELHORIAS CRÍTICAS (5% do peso - sem desbalancear)
            "context_bonus": 0.5,        # +0.5 por trade no contexto certo
            "sltp_quality": 0.3,         # +0.3 por SL/TP adequados
            "drawdown_protection": -2.0, # -2.0 por trading em drawdown alto
            "position_management": 0.2,   # +0.2 por gestão adequada de posições
        }
        
        # Configurações
        self.target_trades_per_day = 16
        self.min_trade_duration = 3  # steps
        self.flip_flop_memory = 6    # steps para detectar flip-flop
        
        # 🚀 CONFIGURAÇÕES SIMPLES PARA MELHORIAS
        self.max_drawdown_threshold = 0.15  # 15% drawdown = cuidado
        self.trend_lookback = 20             # 20 steps para detectar tendência simples
        self.max_positions = 3               # Máximo de posições simultâneas
        
        # Limites
        self.max_reward = 15.0
        self.min_reward = -20.0
        
        # 🚀 CACHE SIMPLES PARA PERFORMANCE
        self._last_trend_cache = {"step": -1, "trend": 0}
        
    def reset(self):
        """Reset para novo episódio"""
        self.step_count = 0
        self._last_trend_cache = {"step": -1, "trend": 0}
        
    def calculate_reward_and_info(self, env, action: np.ndarray, old_state: Dict) -> Tuple[float, Dict, bool]:
        """
        🧹 SISTEMA ULTRA-LIMPO V2 DE RECOMPENSAS
        
        FÓRMULA:
        Reward = PnL_real * 5 + pequenos_incentivos + disciplina_básica + melhorias_críticas
        """
        self.step_count += 1
        
        reward = 0.0
        info = {"components": {}, "clean_system_v2": True}
        
        # Processar ações
        entry_decision = int(action[0]) if len(action) > 0 else 0
        
        # 💰 COMPONENTE PRINCIPAL: PnL DIRETO (65%)
        pnl_reward = self._calculate_pnl_reward(env, old_state)
        reward += pnl_reward
        info["components"]["pnl_direct"] = pnl_reward
        
        # 🎯 ATIVIDADE BÁSICA (20%)
        activity_reward = self._calculate_activity_reward(env, entry_decision)
        reward += activity_reward
        info["components"]["activity"] = activity_reward
        
        # 🛡️ DISCIPLINA BÁSICA (10%)
        discipline_penalty = self._calculate_discipline_penalties(env, action, entry_decision)
        reward += discipline_penalty
        info["components"]["discipline"] = discipline_penalty
        
        # 🚀 MELHORIAS CRÍTICAS (5% - sem desbalancear)
        improvements_reward = self._calculate_critical_improvements(env, action, entry_decision)
        reward += improvements_reward
        info["components"]["improvements"] = improvements_reward
        
        # Aplicar limites
        reward = np.clip(reward, self.min_reward, self.max_reward)
        
        # Info adicional
        info.update({
            "total_reward": reward,
            "trades_today": self._get_trades_today(env),
            "system_type": "clean_reward_v2"
        })
        
        return reward, info, False
    
    def _calculate_pnl_reward(self, env, old_state: Dict) -> float:
        """💰 Recompensa baseada no PnL real dos trades"""
        reward = 0.0
        
        # Verificar se houve novos trades
        old_trades_count = old_state.get('trades_count', 0)
        current_trades_count = len(env.trades) if hasattr(env, 'trades') else 0
        
        if current_trades_count > old_trades_count:
            # Processar novos trades
            new_trades = env.trades[old_trades_count:]
            
            for trade in new_trades:
                pnl = trade.get('pnl_usd', 0.0)
                
                # Recompensa direta do PnL
                reward += pnl * self.weights["pnl_direct"]
                
                # Bônus/penalidade adicional
                if pnl > 0:
                    reward += self.weights["win_bonus"]
                else:
                    reward += self.weights["loss_penalty"]
        
        return reward
    
    def _calculate_activity_reward(self, env, entry_decision: int) -> float:
        """🎯 Recompensa por atividade adequada"""
        reward = 0.0
        
        # Bônus por fazer trade
        if entry_decision != 0:
            reward += self.weights["trade_action"]
        
        # Verificar zona alvo de trades por dia
        trades_today = self._get_trades_today(env)
        
        if 10 <= trades_today <= 18:  # Zona alvo
            reward += self.weights["target_zone_bonus"]
        elif trades_today > 0:  # Pelo menos alguma atividade
            reward += self.weights["daily_activity"]
        
        return reward
    
    def _calculate_discipline_penalties(self, env, action: np.ndarray, entry_decision: int) -> float:
        """🛡️ Penalidades por falta de disciplina"""
        penalty = 0.0
        
        # Penalidade por flip-flop (mudança frequente de direção)
        if hasattr(env, 'action_history') and len(env.action_history) >= self.flip_flop_memory:
            recent_actions = env.action_history[-self.flip_flop_memory:]
            if self._detect_flip_flop(recent_actions):
                penalty += self.weights["flip_flop_penalty"]
        
        # Penalidade por micro-trades
        if hasattr(env, 'trades') and env.trades:
            last_trade = env.trades[-1]
            duration = last_trade.get('duration_steps', 0)
            if duration < self.min_trade_duration:
                penalty += self.weights["micro_trade_penalty"]
        
        # Penalidade por inatividade excessiva
        if entry_decision == 0:  # Hold
            penalty += self.weights["hold_penalty"]
        
        return penalty
    
    def _calculate_critical_improvements(self, env, action: np.ndarray, entry_decision: int) -> float:
        """🚀 MELHORIAS CRÍTICAS - Elementos essenciais que faltavam"""
        reward = 0.0
        
        # 1. 🧠 CONTEXT AWARENESS SIMPLES
        if entry_decision != 0:  # Se está fazendo trade
            trend_alignment = self._check_simple_trend_alignment(env, entry_decision)
            if trend_alignment:
                reward += self.weights["context_bonus"]
        
        # 2. 🎯 SL/TP QUALITY SIMPLES
        if len(action) >= 6:  # Action space com SL/TP
            sltp_quality = self._check_simple_sltp_quality(action)
            reward += sltp_quality * self.weights["sltp_quality"]
        
        # 3. 🛡️ DRAWDOWN PROTECTION SIMPLES
        if hasattr(env, 'current_drawdown'):
            if env.current_drawdown > self.max_drawdown_threshold:
                if entry_decision != 0:  # Fazendo trade em drawdown alto
                    reward += self.weights["drawdown_protection"]
        
        # 4. 📊 POSITION MANAGEMENT SIMPLES
        position_count = len(env.positions) if hasattr(env, 'positions') else 0
        if position_count <= self.max_positions:
            reward += self.weights["position_management"]
        
        return reward
    
    def _check_simple_trend_alignment(self, env, entry_decision: int) -> bool:
        """🧠 Verificação simples de alinhamento com tendência"""
        try:
            # Cache para performance
            if self._last_trend_cache["step"] == env.current_step:
                trend = self._last_trend_cache["trend"]
            else:
                # Calcular tendência simples (apenas uma vez por step)
                if hasattr(env, 'df') and env.current_step >= self.trend_lookback:
                    start_idx = max(0, env.current_step - self.trend_lookback)
                    end_idx = min(env.current_step, len(env.df) - 1)
                    
                    start_price = env.df['close_5m'].iloc[start_idx]
                    end_price = env.df['close_5m'].iloc[end_idx]
                    
                    trend = 1 if end_price > start_price else -1
                else:
                    trend = 0
                
                # Atualizar cache
                self._last_trend_cache = {"step": env.current_step, "trend": trend}
            
            # Verificar alinhamento
            if trend == 1 and entry_decision == 1:  # Uptrend + Long
                return True
            elif trend == -1 and entry_decision == 2:  # Downtrend + Short
                return True
            
        except Exception:
            pass
        
        return False
    
    def _check_simple_sltp_quality(self, action: np.ndarray) -> float:
        """🎯 Verificação simples da qualidade do SL/TP"""
        try:
            # Assumindo action space: [estratégica, táticas, sltp...]
            if len(action) >= 6:
                sl_values = action[4:7]  # Primeiros 3 SL
                tp_values = action[7:10] if len(action) >= 10 else action[4:6]  # TP
                
                # Qualidade simples: SL e TP não extremos
                sl_quality = 1.0 - np.mean(np.abs(sl_values)) / 3.0  # Normalizar [-3,3] -> [0,1]
                tp_quality = 1.0 - np.mean(np.abs(tp_values)) / 3.0
                
                return (sl_quality + tp_quality) / 2.0
        except Exception:
            pass
        
        return 0.0
    
    def _detect_flip_flop(self, actions: List[int]) -> bool:
        """Detectar padrão de flip-flop nas ações"""
        if len(actions) < 4:
            return False
        
        # Contar mudanças de direção
        direction_changes = 0
        for i in range(1, len(actions)):
            if actions[i] != 0 and actions[i-1] != 0 and actions[i] != actions[i-1]:
                direction_changes += 1
        
        # Flip-flop se mais de 2 mudanças em 6 steps
        return direction_changes >= 2
    
    def _get_trades_today(self, env) -> int:
        """Calcular trades do dia atual"""
        if not hasattr(env, 'trades') or not env.trades:
            return 0
        
        # 288 steps = 1 dia em timeframe de 5min
        steps_per_day = 288
        current_day = env.current_step // steps_per_day
        
        trades_today = 0
        for trade in env.trades:
            trade_day = trade.get('entry_step', 0) // steps_per_day
            if trade_day == current_day:
                trades_today += 1
        
        return trades_today

def create_simple_reward_system(initial_balance: float = 500.0):
    """Criar sistema de recompensas limpo melhorado"""
    return CleanRewardCalculator(initial_balance)

# Manter compatibilidade com código existente
def create_clean_reward_system(initial_balance: float = 500.0):
    """Criar sistema de recompensas limpo melhorado (alias)"""
    return CleanRewardCalculator(initial_balance) 