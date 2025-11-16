import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
import logging
from datetime import datetime
from dataclasses import dataclass

@dataclass
class RewardConfig:
    """🎯 CONFIGURAÇÃO COMPLETA DO SISTEMA DE RECOMPENSAS"""
    
    # 🧠 1. DETECÇÃO DE REGIME DE MERCADO
    regime_alignment_bonus: float = 2.0
    regime_counter_penalty: float = -1.5
    regime_confidence_multiplier: float = 1.5
    
    # 🔥 2. FILTRO DE QUALIDADE
    excellent_trade_bonus: float = 3.0     # >90% score
    quality_trade_bonus: float = 1.5       # >80% score
    poor_trade_penalty: float = -2.0       # blocked by filter
    
    # 💰 3. PnL DIRETO (COMPONENTE PRINCIPAL)
    pnl_multiplier: float = 5.0            # Multiplicador principal do PnL
    win_bonus: float = 2.0                 # Bônus extra para wins  
    loss_penalty: float = -0.5             # Penalidade suave para losses
    
    # 🎯 4. QUALIDADE SOBRE QUANTIDADE
    target_zone_bonus: float = 1.0         # 10-18 trades/dia
    quality_volume_bonus: float = 2.0      # Win rate + volume adequados
    
    # ⭐ 5. QUALIDADE ESCALADA
    mastery_trade_bonus: float = 5.0       # PnL muito alto
    excellent_pnl_bonus: float = 3.0       # PnL alto
    quality_pnl_bonus: float = 1.5         # PnL bom
    
    # 🔄 6. SEQUÊNCIA DE WINS
    consecutive_wins_multiplier: float = 1.2
    streak_bonus_base: float = 1.0
    
    # ⏱️ 7. TIMING PERFEITO
    perfect_timing_bonus: float = 2.5      # 5-50 steps + PnL >2.0
    timing_mastery_bonus: float = 1.5
    
    # 🧠 8. EXPERT SL/TP
    sltp_smart_bonus: float = 1.5          # Ranges inteligentes
    risk_reward_bonus: float = 2.0         # Ratio ótimo
    flexible_duration_bonus: float = 1.0
    
    # ⌛ 9. HOLD INTELIGENTE
    patience_base_bonus: float = 0.3
    perfect_hold_bonus: float = 0.8
    discipline_hold_bonus: float = 0.5
    
    # 📊 10. ATIVIDADE GUIADA
    extreme_overtrading_penalty: float = -5.0   # >60 trades/dia
    high_overtrading_penalty: float = -3.0      # >40 trades/dia
    moderate_overtrading_penalty: float = -1.5  # >30 trades/dia
    ideal_zone_bonus: float = 0.0               # 🚨 DESABILITADO: Sem bônus por zona ideal
    acceptable_zone_bonus: float = 0.0          # 🚨 DESABILITADO: Sem bônus por zona aceitável
    inactivity_penalty: float = 0.0            # 🚨 DESABILITADO: Sem penalidade por inatividade
    mastery_zone_bonus: float = 0.0             # 🚨 DESABILITADO: Sem bônus por zona mastery
    
    # 🧠 11. ANÁLISE COMPORTAMENTAL
    consistency_bonus: float = 1.5         # Win rate >=60% últimos 5 trades
    
    # ⚠️ 12. PENALIDADES DISCIPLINARES
    flip_flop_penalty: float = -2.0        # <10 steps
    micro_trade_penalty: float = -1.5      # <3 steps
    extreme_sltp_penalty: float = -2.5     # SL/TP absurdos
    
    # 🎯 13. GESTÃO DE RISCO
    high_drawdown_penalty: float = -2.0    # >8% DD
    excellent_risk_bonus: float = 2.5      # <3% DD + atividade
    good_risk_bonus: float = 1.5           # <5% DD
    
    # 🚨 14. CONTROLE DE EMERGÊNCIA
    cooldown_violation_penalty: float = -3.0
    quota_exceeded_penalty: float = -5.0
    severe_violation_penalty: float = -10.0
    
    # 📈 15-18. MÉTRICAS E PROTEÇÕES
    reward_min_limit: float = -20.0
    reward_max_limit: float = 30.0

class ComprehensiveRewardSystem:
    """
    🎯 SISTEMA COMPLETO DE RECOMPENSAS - 18 CATEGORIAS INTEGRADAS
    Sistema balanceado que aborda todos os aspectos do trading
    VERSÃO OTIMIZADA PARA ALTA PERFORMANCE
    """
    
    def __init__(self, initial_balance: float = 500.0):
        self.config = RewardConfig()
        
        # 💰 SISTEMA DE EXECUÇÃO (PRESERVADO)
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {}  # position_id -> position_data
        self.completed_trades = []
        self.daily_trade_count = 0
        self.last_trade_day = None
        
        # 🧠 DETECÇÃO DE REGIME (OTIMIZADA)
        self.regime_detector = MarketRegimeDetector()
        
        # 📊 MÉTRICAS AVANÇADAS (CACHE)
        self.consecutive_wins = 0
        self.last_actions = []  # Para flip-flop detection
        self.hold_streak = 0
        self.peak_balance = initial_balance
        self.current_drawdown = 0.0
        
        # 📈 HISTÓRICO DE PERFORMANCE (LIMITADO)
        self.recent_trades = []  # Últimos 5 trades para análise (reduzido de 10)
        self.daily_pnl = 0.0
        self.session_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_consecutive_wins': 0
        }
        
        # 🎯 CONTROLES DE QUALIDADE
        self.last_trade_step = 0
        self.adjustment_cooldown = {}  # position_id -> last_adjustment_step
        
        # ⚡ OTIMIZAÇÕES DE PERFORMANCE
        self.calculation_cache = {}  # Cache para cálculos pesados
        self.last_calculation_step = -1
        self.skip_heavy_calculations = False  # Flag para pular cálculos pesados
        
        self.logger = logging.getLogger(__name__)

    def calculate_comprehensive_reward(self, env, action: np.ndarray, current_price: float, 
                                     timestamp: Any) -> Tuple[float, Dict[str, Any], bool]:
        """
        🎯 CÁLCULO PRINCIPAL DE RECOMPENSA INTEGRADA - VERSÃO OTIMIZADA
        Processa as categorias mais importantes com cache inteligente
        """
        try:
            # ⚡ OTIMIZAÇÃO: Evitar recálculos desnecessários
            if timestamp == self.last_calculation_step:
                cached_result = self.calculation_cache.get('last_reward')
                if cached_result:
                    return cached_result
            
            # 🔄 RESET DIÁRIO DE CONTADORES (OTIMIZADO)
            self._update_daily_counters_fast(timestamp)
            
            # 🎯 PROCESSAR AÇÃO (SIMPLIFICADO)
            entry_decision, entry_confidence, position_size, mgmt_action, sl_adjust, tp_adjust = self._parse_action_fast(action)
            
            # 💰 EXECUTAR ORDENS (LÓGICA PRESERVADA)
            trade_executed = False
            if entry_decision > 0:  # Buy ou Sell
                trade_executed = self._execute_entry_order_fast(
                    entry_decision, entry_confidence, position_size, 
                    current_price, timestamp, env
                )
            
            # 🔧 GESTÃO DE POSIÇÕES (SIMPLIFICADA)
            mgmt_reward = self._execute_management_fast(
                mgmt_action, sl_adjust, tp_adjust, current_price, timestamp
            )
            
            # 🔍 VERIFICAR FECHAMENTOS AUTOMÁTICOS (OTIMIZADO)
            auto_close_reward = self._check_automatic_closes_fast(current_price, timestamp)
            
            # 📊 ATUALIZAR MÉTRICAS (MÍNIMO NECESSÁRIO)
            self._update_positions_pnl_fast(current_price)
            self._update_metrics_fast(timestamp)
            
            # 🎯 CÁLCULO INTEGRADO DE RECOMPENSAS (OTIMIZADO - APENAS PRINCIPAIS)
            reward_breakdown = self._calculate_core_rewards_fast(
                env, action, current_price, timestamp, 
                trade_executed, entry_decision, mgmt_action
            )
            
            # 📈 RECOMPENSA FINAL
            total_reward = sum(reward_breakdown.values()) + mgmt_reward + auto_close_reward
            
            # 🛡️ APLICAR LIMITES DE SEGURANÇA
            total_reward = np.clip(total_reward, 
                                  self.config.reward_min_limit, 
                                  self.config.reward_max_limit)
            
            # 📊 INFO SIMPLIFICADO
            info = self._get_fast_info(
                entry_decision, mgmt_action, trade_executed, 
                reward_breakdown
            )
            
            # ⚡ CACHE DO RESULTADO
            result = (total_reward, info, trade_executed)
            self.calculation_cache['last_reward'] = result
            self.last_calculation_step = timestamp
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de recompensa: {e}")
            return 0.0, {"error": str(e)}, False
    
    def _parse_action_fast(self, action: np.ndarray) -> Tuple[int, float, float, int, float, float]:
        """⚡ PARSE RÁPIDO DA AÇÃO"""
        if len(action) >= 6:
            return (int(action[0]), float(action[1]), float(action[2]), 
                   int(action[3]), float(action[4]), float(action[5]))
        elif len(action) >= 3:
            return (int(action[0]), 
                   float(action[1]) if len(action) > 1 else 0.5,
                   float(action[2]) if len(action) > 2 else 0.1,
                   0, 0.0, 0.0)
        else:
            return (int(action[0]) if len(action) > 0 else 0, 0.5, 0.1, 0, 0.0, 0.0)
    
    def _calculate_core_rewards_fast(self, env, action, current_price, timestamp, 
                                   trade_executed, entry_decision, mgmt_action) -> Dict[str, float]:
        """🎯 CÁLCULO OTIMIZADO - APENAS CATEGORIAS PRINCIPAIS"""
        
        rewards = {}
        
        # 💰 3. PnL DIRETO (PRINCIPAL - SEMPRE ATIVO)
        rewards['pnl_direct'] = self._calculate_pnl_reward_fast()
        
        # 📊 10. ATIVIDADE GUIADA (CRÍTICO)
        rewards['guided_activity'] = self._calculate_activity_guidance_fast(entry_decision)
        
        # ⌛ 9. HOLD INTELIGENTE (LEVE)
        rewards['intelligent_hold'] = self._calculate_hold_reward_fast(entry_decision)
        
        # ⚠️ 12. PENALIDADES DISCIPLINARES (IMPORTANTES)
        rewards['discipline'] = self._calculate_discipline_fast(action, entry_decision)
        
        # 🎯 CÁLCULOS PESADOS APENAS A CADA 10 STEPS
        if timestamp % 10 == 0:
            # 🔥 2. FILTRO DE QUALIDADE
            rewards['quality_filter'] = self._calculate_quality_filter_fast(env, entry_decision)
            
            # ⭐ 5. QUALIDADE ESCALADA
            rewards['quality_scaled'] = self._calculate_scaled_quality_fast()
            
            # 🔄 6. SEQUÊNCIA DE WINS
            rewards['win_sequence'] = self._calculate_win_sequence_fast()
            
            # 🎯 13. GESTÃO DE RISCO
            rewards['risk_management'] = self._calculate_risk_management_fast()
        else:
            # Usar valores cached ou zeros
            rewards['quality_filter'] = 0.0
            rewards['quality_scaled'] = 0.0
            rewards['win_sequence'] = 0.0
            rewards['risk_management'] = 0.0
        
        # 🧠 CÁLCULOS MUITO PESADOS APENAS A CADA 50 STEPS
        if timestamp % 50 == 0:
            # 🧠 1. REGIME DE MERCADO
            regime_info = self.regime_detector.detect_market_regime(env)
            rewards['regime'] = self._calculate_regime_reward_fast(regime_info, entry_decision)
            
            # 🧠 8. EXPERT SL/TP
            rewards['expert_sltp'] = self._calculate_expert_sltp_fast(action)
        else:
            rewards['regime'] = 0.0
            rewards['expert_sltp'] = 0.0
        
        return rewards
    
    def _calculate_pnl_reward_fast(self) -> float:
        """💰 3. PnL DIRETO - VERSÃO OTIMIZADA"""
        if not self.completed_trades:
            return 0.0
        
        # Apenas último trade
        last_trade = self.completed_trades[-1]
        pnl = last_trade.get('pnl_usd', 0.0)
        
        # Cálculo direto
        base_reward = pnl * self.config.pnl_multiplier
        if pnl > 0:
            base_reward += self.config.win_bonus
        elif pnl < 0:
            base_reward += self.config.loss_penalty
        
        return base_reward
    
    def _calculate_activity_guidance_fast(self, entry_decision: int) -> float:
        """📊 10. ATIVIDADE GUIADA - VERSÃO OTIMIZADA"""
        trades_today = self.daily_trade_count
        
        # Lookup table para performance
        if trades_today > 60:
            return self.config.extreme_overtrading_penalty
        elif trades_today > 40:
            return self.config.high_overtrading_penalty
        elif trades_today > 30:
            return self.config.moderate_overtrading_penalty
        elif 15 <= trades_today <= 25:
            return self.config.mastery_zone_bonus
        elif 12 <= trades_today <= 25:
            return self.config.ideal_zone_bonus
        elif 8 <= trades_today <= 35:
            return self.config.acceptable_zone_bonus
        elif trades_today < 5:
            return self.config.inactivity_penalty
        else:
            return 0.0
    
    def _calculate_hold_reward_fast(self, entry_decision: int) -> float:
        """⌛ 9. HOLD INTELIGENTE - VERSÃO OTIMIZADA"""
        if entry_decision != 0:
            self.hold_streak = 0
            return 0.0
        
        self.hold_streak += 1
        
        # Cálculo direto
        reward = self.config.patience_base_bonus
        if self.daily_trade_count > 20:
            reward += self.config.perfect_hold_bonus
        if self.hold_streak > 10:
            reward += self.config.discipline_hold_bonus
        
        return reward
    
    def _calculate_discipline_fast(self, action, entry_decision: int) -> float:
        """⚠️ 12. PENALIDADES DISCIPLINARES - VERSÃO OTIMIZADA"""
        penalty = 0.0
        
        # Flip-flop detection simplificado
        self.last_actions.append(entry_decision)
        if len(self.last_actions) > 5:  # Reduzido de 10 para 5
            self.last_actions.pop(0)
        
        # Verificação simplificada
        if len(self.last_actions) >= 3:
            recent = self.last_actions[-3:]
            if recent[0] != 0 and recent[2] != 0 and recent[0] != recent[2]:
                penalty += self.config.flip_flop_penalty
        
        # Micro-trading (apenas se há trades)
        if self.completed_trades:
            last_trade = self.completed_trades[-1]
            if last_trade.get('duration_steps', 10) < 3:
                penalty += self.config.micro_trade_penalty
        
        return penalty
    
    def _calculate_quality_filter_fast(self, env, entry_decision: int) -> float:
        """🔥 2. FILTRO DE QUALIDADE - VERSÃO OTIMIZADA"""
        if entry_decision == 0:
            return 0.0
        
        try:
            current_idx = min(env.current_step, len(env.df) - 1)
            quality_score = 0.0
            
            # Verificações rápidas
            if 'rsi_14' in env.df.columns:
                rsi = env.df['rsi_14'].iloc[current_idx]
                if 20 <= rsi <= 80:
                    quality_score += 30
                elif rsi < 20 or rsi > 80:
                    quality_score += 40
            
            # Retorno direto baseado no score
            if quality_score >= 40:
                return self.config.excellent_trade_bonus
            elif quality_score >= 30:
                return self.config.quality_trade_bonus
            else:
                return self.config.poor_trade_penalty
                
        except Exception:
            return 0.0
    
    def _calculate_scaled_quality_fast(self) -> float:
        """⭐ 5. QUALIDADE ESCALADA - VERSÃO OTIMIZADA"""
        if not self.completed_trades:
            return 0.0
        
        pnl = self.completed_trades[-1].get('pnl_usd', 0.0)
        
        # Lookup direto
        if pnl > 10.0:
            return self.config.mastery_trade_bonus
        elif pnl > 5.0:
            return self.config.excellent_pnl_bonus
        elif pnl > 2.0:
            return self.config.quality_pnl_bonus
        else:
            return 0.0
    
    def _calculate_win_sequence_fast(self) -> float:
        """🔄 6. SEQUÊNCIA DE WINS - VERSÃO OTIMIZADA"""
        if self.consecutive_wins <= 1:
            return 0.0
        
        # Cálculo direto
        multiplier = min(self.consecutive_wins * self.config.consecutive_wins_multiplier, 5.0)
        return self.config.streak_bonus_base * multiplier
    
    def _calculate_risk_management_fast(self) -> float:
        """🎯 13. GESTÃO DE RISCO - VERSÃO OTIMIZADA"""
        if self.current_drawdown > 0.08:
            return self.config.high_drawdown_penalty
        elif self.current_drawdown < 0.03 and self.daily_trade_count > 5:
            return self.config.excellent_risk_bonus
        elif self.current_drawdown < 0.05:
            return self.config.good_risk_bonus
        else:
            return 0.0
    
    def _calculate_regime_reward_fast(self, regime_info: Dict, entry_decision: int) -> float:
        """🧠 1. REGIME DE MERCADO - VERSÃO OTIMIZADA"""
        if entry_decision == 0:
            return 0.0
        
        regime = regime_info.get('regime', 'SIDEWAYS')
        confidence = regime_info.get('confidence', 0.5)
        
        # Verificação direta
        aligned = ((regime == 'UPTREND' and entry_decision == 1) or
                  (regime == 'DOWNTREND' and entry_decision == 2) or
                  (regime == 'SIDEWAYS'))
        
        if aligned:
            return self.config.regime_alignment_bonus * confidence * self.config.regime_confidence_multiplier
        else:
            return self.config.regime_counter_penalty * confidence
    
    def _calculate_expert_sltp_fast(self, action) -> float:
        """🧠 8. EXPERT SL/TP - VERSÃO OTIMIZADA"""
        sl_adjust = float(action[4]) if len(action) > 4 else 0.0
        tp_adjust = float(action[5]) if len(action) > 5 else 0.0
        
        # Cálculo direto
        sl_points = abs(sl_adjust) * 15 + 10
        tp_points = abs(tp_adjust) * 20 + 15
        
        reward = 0.0
        if 10 <= sl_points <= 50 and 15 <= tp_points <= 80:
            reward += self.config.sltp_smart_bonus
        
        if sl_points > 0:
            rr_ratio = tp_points / sl_points
            if 1.4 <= rr_ratio <= 2.5:
                reward += self.config.risk_reward_bonus
        
        return reward
    
    # === MÉTODOS DE EXECUÇÃO OTIMIZADOS ===
    
    def _execute_entry_order_fast(self, decision: int, confidence: float, size: float, 
                                 price: float, timestamp: Any, env=None) -> bool:
        """💰 EXECUTAR ORDEM DE ENTRADA - VERSÃO OTIMIZADA"""
        try:
            position_id = f"pos_{timestamp}_{decision}"
            
            # Cálculo direto do tamanho
            size_usd = min(abs(size) * 100, self.current_balance * 0.1)  # Max 10% do balance
            
            position = {
                'id': position_id,
                'type': 'BUY' if decision == 1 else 'SELL',
                'entry_price': price,
                'size_lots': abs(size),
                'size_usd': size_usd,
                'entry_time': timestamp,
                'entry_step': timestamp,
                'confidence': confidence,
                'sl_price': None,
                'tp_price': None,
                'unrealized_pnl': 0.0
            }
            
            self.positions[position_id] = position
            self.daily_trade_count += 1
            self.last_trade_step = timestamp
            
            return True
            
        except Exception:
            return False
    
    def _execute_management_fast(self, action: int, sl_adjust: float, tp_adjust: float, 
                               current_price: float, timestamp: Any) -> float:
        """🔧 GESTÃO OTIMIZADA"""
        if action == 0 or not self.positions:
            return 0.0
        
        reward = 0.0
        
        if action == 1:  # Close profitable
            for pos_id, pos in list(self.positions.items()):
                pnl = self._calculate_position_pnl_usd_fast(pos, current_price)
                if pnl > 1.0:
                    reward += self._close_position_fast(pos_id, current_price, timestamp, "PROFITABLE")
        
        elif action == 2:  # Close all
            for pos_id in list(self.positions.keys()):
                reward += self._close_position_fast(pos_id, current_price, timestamp, "CLOSE_ALL")
        
        return reward
    
    def _close_position_fast(self, position_id: str, exit_price: float, timestamp: Any, reason: str) -> float:
        """💰 FECHAR POSIÇÃO - VERSÃO OTIMIZADA"""
        if position_id not in self.positions:
            return 0.0
        
        position = self.positions[position_id]
        pnl_usd = self._calculate_position_pnl_usd_fast(position, exit_price)
        
        # Atualizar balance
        self.current_balance += pnl_usd
        
        # Registro simplificado
        trade_record = {
            'id': position_id,
            'type': position['type'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl_usd': pnl_usd,
            'duration_steps': timestamp - position.get('entry_step', 0),
            'reason': reason
        }
        
        self.completed_trades.append(trade_record)
        
        # Manter apenas últimos 50 trades para economia de memória
        if len(self.completed_trades) > 50:
            self.completed_trades = self.completed_trades[-50:]
        
        # Atualizar consecutive wins
        if pnl_usd > 0:
            self.consecutive_wins += 1
        else:
            self.consecutive_wins = 0
        
        del self.positions[position_id]
        return pnl_usd * 0.1
    
    def _calculate_position_pnl_usd_fast(self, position: Dict, current_price: float) -> float:
        """💰 CÁLCULO RÁPIDO DE PnL"""
        try:
            entry_price = position['entry_price']
            size_usd = position['size_usd']
            pos_type = position['type']
            
            if pos_type == 'BUY':
                price_diff = current_price - entry_price
            else:
                price_diff = entry_price - current_price
            
            return (price_diff / entry_price) * size_usd if entry_price > 0 else 0.0
        except Exception:
            return 0.0
    
    def _check_automatic_closes_fast(self, current_price: float, timestamp: Any) -> float:
        """🔍 VERIFICAÇÃO RÁPIDA DE SL/TP"""
        reward = 0.0
        positions_to_close = []
        
        for pos_id, position in self.positions.items():
            sl_price = position.get('sl_price')
            tp_price = position.get('tp_price')
            pos_type = position['type']
            
            should_close = False
            reason = ""
            
            if pos_type == 'BUY':
                if sl_price and current_price <= sl_price:
                    should_close, reason = True, "STOP_LOSS"
                elif tp_price and current_price >= tp_price:
                    should_close, reason = True, "TAKE_PROFIT"
            else:  # SELL
                if sl_price and current_price >= sl_price:
                    should_close, reason = True, "STOP_LOSS"
                elif tp_price and current_price <= tp_price:
                    should_close, reason = True, "TAKE_PROFIT"
            
            if should_close:
                positions_to_close.append((pos_id, reason))
        
        for pos_id, reason in positions_to_close:
            reward += self._close_position_fast(pos_id, current_price, timestamp, reason)
        
        return reward
    
    def _update_daily_counters_fast(self, timestamp: Any):
        """📅 ATUALIZAÇÃO RÁPIDA DE CONTADORES"""
        try:
            current_day = int(timestamp // 288)
            if self.last_trade_day != current_day:
                self.daily_trade_count = 0
                self.last_trade_day = current_day
        except Exception:
            pass
    
    def _update_positions_pnl_fast(self, current_price: float):
        """📊 ATUALIZAÇÃO RÁPIDA DE PnL"""
        for position in self.positions.values():
            position['unrealized_pnl'] = self._calculate_position_pnl_usd_fast(position, current_price)
    
    def _update_metrics_fast(self, timestamp: Any):
        """📊 ATUALIZAÇÃO MÍNIMA DE MÉTRICAS"""
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        self.current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance if self.peak_balance > 0 else 0.0
    
    def _get_fast_info(self, entry_decision: int, mgmt_action: int, 
                      trade_executed: bool, reward_breakdown: Dict) -> Dict[str, Any]:
        """📊 INFO SIMPLIFICADO PARA PERFORMANCE"""
        unrealized_pnl = sum(pos.get('unrealized_pnl', 0) for pos in self.positions.values())
        
        return {
            'entry_decision': entry_decision,
            'mgmt_action': mgmt_action,
            'trade_executed': trade_executed,
            'current_balance': self.current_balance,
            'unrealized_pnl': unrealized_pnl,
            'total_equity': self.current_balance + unrealized_pnl,
            'daily_trades': self.daily_trade_count,
            'open_positions': len(self.positions),
            'total_trades': len(self.completed_trades),
            'consecutive_wins': self.consecutive_wins,
            'reward_breakdown': reward_breakdown,
            'current_drawdown': self.current_drawdown
        }

    def reset(self):
        """🔄 RESET COMPLETO DO SISTEMA"""
        self.current_balance = self.initial_balance
        self.positions.clear()
        self.completed_trades.clear()
        self.daily_trade_count = 0
        self.last_trade_day = None
        self.consecutive_wins = 0
        self.last_actions.clear()
        self.hold_streak = 0
        self.peak_balance = self.initial_balance
        self.current_drawdown = 0.0
        self.recent_trades.clear()
        self.daily_pnl = 0.0
        self.session_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_consecutive_wins': 0
        }
        self.last_trade_step = 0
        self.adjustment_cooldown.clear()
    
    def get_portfolio_value(self) -> float:
        """💰 VALOR TOTAL DO PORTFÓLIO"""
        unrealized_pnl = sum(pos.get('unrealized_pnl', 0) for pos in self.positions.values())
        return self.current_balance + unrealized_pnl

def create_comprehensive_reward_system(initial_balance: float = 500.0) -> ComprehensiveRewardSystem:
    """🎯 FACTORY FUNCTION PARA CRIAR O SISTEMA COMPLETO"""
    return ComprehensiveRewardSystem(initial_balance)

class MarketRegimeDetector:
    """🧠 DETECTOR INTELIGENTE DE REGIME DE MERCADO"""
    
    def __init__(self):
        self.current_regime = "SIDEWAYS"
        self.regime_strength = 0.5
        self.regime_confidence = 0.5
    
    def detect_market_regime(self, env) -> Dict[str, Any]:
        """🔍 Detecção simplificada mas eficaz"""
        try:
            if not hasattr(env, 'df') or env.current_step < 50:
                return {
                    'regime': self.current_regime,
                    'strength': self.regime_strength,
                    'confidence': self.regime_confidence
                }
            
            # Análise básica de tendência
            lookback = 30
            start_idx = max(0, env.current_step - lookback)
            prices = env.df['close_5m'].iloc[start_idx:env.current_step+1]
            
            if len(prices) > 10:
                slope = np.polyfit(range(len(prices)), prices.values, 1)[0]
                price_change = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
                
                if slope > 0.5 and price_change > 0.01:
                    self.current_regime = "UPTREND"
                    self.regime_strength = min(abs(price_change) * 10, 1.0)
                elif slope < -0.5 and price_change < -0.01:
                    self.current_regime = "DOWNTREND"
                    self.regime_strength = min(abs(price_change) * 10, 1.0)
                else:
                    self.current_regime = "SIDEWAYS"
                    self.regime_strength = 0.5
                
                self.regime_confidence = min(abs(slope) * 0.1, 1.0)
            
            return {
                'regime': self.current_regime,
                'strength': self.regime_strength,
                'confidence': self.regime_confidence
            }
            
        except Exception:
            return {
                'regime': 'SIDEWAYS',
                'strength': 0.5,
                'confidence': 0.5
            } 