"""
💰 SISTEMA DE RECOMPENSAS ESPECÍFICO PARA DAY TRADING - V1.0
Focado em scalping, trades rápidos e precisão intraday

🎯 CARACTERÍSTICAS DAY TRADING:
- Timeframe: 5min (288 steps = 1 dia)
- SL Range: 2-8 pontos (muito agressivo)
- TP Range: 3-15 pontos (scalping)
- Target: 20-50 trades/dia (alta frequência)
- Duração ideal: 5-60 steps (25min-5h)
- Risk/Reward: 1.2-2.0 (mais flexível que swing)

🔥 DIFERENÇAS DO SWING TRADE:
1. Recompensas por velocidade de execução
2. Penalidades por holds longos (>4h)
3. Bônus por múltiplos scalps no dia
4. Análise técnica intraday (suporte/resistência de curto prazo)
5. Gestão de risco adaptada para alta frequência

🧠 FILOSOFIA:
- 70% PnL direto (foco no resultado)
- 15% velocidade/timing (essencial para day trade)
- 10% gestão de risco intraday
- 5% consistência e padrões
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class DayTradingSession(Enum):
    """Sessões de day trading com características diferentes"""
    OPENING = "opening"      # Primeiras 2h (alta volatilidade)
    MIDDAY = "midday"       # Meio dia (consolidação)
    CLOSING = "closing"     # Últimas 2h (movimento final)
    OVERNIGHT = "overnight" # Fora do horário

@dataclass
class ScalpingMetrics:
    """Métricas específicas de scalping"""
    avg_trade_duration: float
    quick_trades_ratio: float
    precision_entries: int
    rapid_exits: int
    intraday_consistency: float

class DayTradingRewardCalculator:
    """Sistema de recompensas otimizado para day trading"""
    
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        self.step_count = 0
        self.episode_count = 0
        self.current_volatility = 1.0  # Fator de volatilidade para scaling dinâmico
        
        # 🎯 TARGETS ESPECÍFICOS DAY TRADING
        self.target_trades_per_day = 35          # 35 trades/dia (mais ativo que swing)
        self.target_zone_min = 20                # Mínimo: 20 trades/dia
        self.target_zone_max = 50                # Máximo: 50 trades/dia
        self.steps_per_day = 288                 # 288 steps = 24h em 5min
        
        # 🎯 RANGES SL/TP DAY TRADING
        self.sl_range_min = 2.0
        self.sl_range_max = 8.0
        self.tp_range_min = 3.0
        self.tp_range_max = 15.0
        self.optimal_risk_reward_min = 1.2       # Mais flexível que swing (1.2 vs 1.5)
        self.optimal_risk_reward_max = 2.0       # Mais flexível que swing (2.0 vs 1.8)
        
        # 🎯 DURAÇÃO IDEAL TRADES DAY TRADING
        self.quick_scalp_max = 20                # Scalp rápido: <20 steps (100min)
        self.optimal_trade_max = 60              # Trade ótimo: <60 steps (5h)
        self.long_hold_penalty = 96              # Hold longo: >96 steps (8h)
        
        # 🔥 PESOS DINÂMICOS PARA DAY TRADING (OPUS FASE 1) - AJUSTADO
        self.base_weights = {
            # 💰 PnL DOMINANTE (70% do peso total) - SCALING REDUZIDO
            "pnl_direct": 4.0,                   # Reduzido: $4.0 por cada $1 de PnL (era 8.0)
            "win_bonus": 3.0,                    # Reduzido: +$3.0 por trade vencedor (era 5.0)  
            "loss_penalty": -2.0,                # Mantido: -$2.0 por trade perdedor
            
            # ⚡ VELOCIDADE & TIMING (15% do peso total) - OTIMIZADO
            "quick_scalp_bonus": 2.0,            # Aumentado: +2.0 por scalp <20 steps
            "rapid_entry_bonus": 1.5,            # Aumentado: +1.5 por entrada precisa
            "swift_exit_bonus": 1.2,             # Aumentado: +1.2 por saída rápida
            "speed_execution_bonus": 0.8,        # Aumentado: +0.8 por execução veloz
            
            # 🛡️ GESTÃO DE RISCO INTRADAY (10% do peso total) - REDUZIDO 2x
            "optimal_ratio_bonus": 1.0,          # +1.0 por ratio 1.2-2.0 (era 2.0)
            "risk_management_bonus": 0.5,        # +0.5 por gestão ativa (era 1.0)
            
            # 📊 CONSISTÊNCIA & PADRÕES (5% do peso total) - REDUZIDO 2x
            "multiple_scalps_bonus": 0.5,        # +0.5 por múltiplos scalps (era 1.0)
            "session_consistency": 0.4,          # +0.4 por consistência na sessão (era 0.8)
            "pattern_recognition": 0.3,          # +0.3 por reconhecer padrões (era 0.6)
            
            # 🚫 PENALIDADES ESPECÍFICAS DAY TRADING - REDUZIDO 2x
            "long_hold_penalty": -1.0,           # -1.0 por hold >8h (era -2.0)
            "overnight_penalty": -1.5,           # -1.5 por posição overnight (era -3.0)
            "overtrading_penalty": -0.5,         # -0.5 por trade >60/dia (era -1.0)
            "session_fatigue_penalty": -0.25,    # -0.25 por fadiga de sessão (era -0.5)
            
            # 🎯 BÔNUS QUALIDADE SCALPING - OPUS OTIMIZADO
            "perfect_scalp_bonus": 4.0,          # DOBRADO: +4.0 por scalp perfeito (Opus sugestão)
            "precision_entry_streak": 2.5,      # DOBRADO: +2.5 por sequência de entradas precisas
            "rapid_profit_taking": 2.0,          # DOBRADO: +2.0 por take profit rápido
            "momentum_capture": 1.8,             # DOBRADO: +1.8 por capturar momentum
            
            # 📈 ANÁLISE TÉCNICA INTRADAY - REDUZIDO 2x
            "support_resistance_entry": 0.75,    # +0.75 por entrada em S/R (era 1.5)
            "breakout_timing": 1.25,             # +1.25 por timing de breakout (era 2.5)
            "volume_confirmation": 0.5,          # +0.5 por confirmação de volume (era 1.0)
            "trend_alignment_intraday": 0.6,     # +0.6 por alinhamento intraday (era 1.2)
        }
        
        # 🚀 OPUS FASE 1: Propriedades dinâmicas de peso
        self.weights = self.base_weights.copy()
        self.win_rate_momentum = 0.0  # Momentum baseado em win rate
        self.volatility_history = []  # Histórico de volatilidade
        
        # 🧠 OPUS FASE 2: Sistemas avançados
        try:
            from .progressive_reward_shaper import create_progressive_shaper
            from .simple_pattern_detector import create_simple_pattern_detector
            from .anti_gaming_system import create_anti_gaming_system
            
            self.progressive_shaper = create_progressive_shaper()
            self.pattern_detector = create_simple_pattern_detector()
            self.anti_gaming = create_anti_gaming_system()
            self.phase2_enabled = True
            
        except ImportError as e:
            print(f"[WARNING] OPUS Phase 2 systems not available: {e}")
            self.phase2_enabled = False
        
        # 📊 TRACKING ESPECÍFICO DAY TRADING
        self.scalping_metrics = ScalpingMetrics(
            avg_trade_duration=0.0,
            quick_trades_ratio=0.0,
            precision_entries=0,
            rapid_exits=0,
            intraday_consistency=0.0
        )
        
        # 🕐 TRACKING DE SESSÃO
        self.session_trades = {"opening": 0, "midday": 0, "closing": 0}
        self.daily_scalps = 0
        self.consecutive_quick_trades = 0
        self.last_trade_step = 0
        
    def reset(self):
        """Reset para novo episódio de day trading"""
        self.step_count = 0
        self.episode_count += 1
        self.session_trades = {"opening": 0, "midday": 0, "closing": 0}
        self.daily_scalps = 0
        self.consecutive_quick_trades = 0
        self.last_trade_step = 0
        
        # 🚀 OPUS FASE 1: Update pesos dinâmicos
        self._update_dynamic_weights()
        
        # 🧠 OPUS FASE 2: Update progressive shaper
        if self.phase2_enabled:
            # Update step count for progressive shaping
            total_steps = self.episode_count * 10000  # Estimate total steps
            self.progressive_shaper.update_step_count(total_steps)
        
        self.scalping_metrics = ScalpingMetrics(
            avg_trade_duration=0.0,
            quick_trades_ratio=0.0,
            precision_entries=0,
            rapid_exits=0,
            intraday_consistency=0.0
        )
    
    def calculate_reward_and_info(self, env, action: np.ndarray, old_state: Dict) -> Tuple[float, Dict, bool]:
        """
        🎯 SISTEMA DE RECOMPENSAS DAY TRADING V1.0 - OTIMIZADO
        Focado em scalping rápido e preciso com early exits para performance
        """
        self.step_count += 1
        
        # 🚀 OPTIMIZATION: Early exit se não há ação relevante
        old_trades_count = old_state.get('trades_count', 0)
        current_trades_count = len(env.trades)
        
        # 🚀 FAST PATH: Se não há trades novos, retorno simples
        if current_trades_count == old_trades_count and self.step_count % 20 != 0:  # Reduzir cálculos
            return 0.0, {
                "components": {"activity_reward": 0.0},
                "trades_today": old_trades_count,
                "total_reward": 0.0,
                "step_count": self.step_count,
                "system_info": {"system_type": "daytrading_specialized", "version": "1.0_optimized"}
            }, False
        
        # 🔥 FULL CALCULATION: Quando há trades ou a cada 20 steps
        reward = 0.0
        info = {"components": {}, "daytrading_analysis": {}, "scalping_metrics": {}}
        done = False  # 🚀 FIX: Inicializar done aqui também
        
        if current_trades_count > old_trades_count:
            # Trade fechado - análise completa de day trading
            last_trade = env.trades[-1]
            
            # 🚀 OPUS FASE 1: Validação robusta
            if not self._validate_trade_data(last_trade):
                # Fallback para dados básicos se validação falhar
                last_trade = {
                    'pnl_usd': last_trade.get('pnl_usd', 0.0),
                    'duration_steps': max(1, last_trade.get('duration_steps', 1)),
                    'entry_price': max(0.01, last_trade.get('entry_price', 1.0))
                }
            
            pnl = last_trade.get('pnl_usd', 0.0)
            duration = last_trade.get('duration_steps', 0)
            
            # 🚀 OPUS FASE 1: Atualizar volatilidade do mercado
            self.current_volatility = self._calculate_market_volatility(env)
            
            # 🔥 PnL DIRETO - COMPONENTE DOMINANTE (COM CLIPPING)
            pnl_reward = pnl * self.weights["pnl_direct"]
            # Clipping para evitar valores extremos
            pnl_reward = np.clip(pnl_reward, -50.0, 50.0)
            reward += pnl_reward
            info["components"]["pnl_direct"] = pnl_reward
            
            # 🔥 WIN/LOSS BALANCEADO
            if pnl > 0:
                win_bonus = self.weights["win_bonus"]
                reward += win_bonus
                info["components"]["win_bonus"] = win_bonus
                
                # ⚡ BÔNUS POR VELOCIDADE (característica essencial day trading)
                speed_reward = self._calculate_speed_rewards(duration, pnl)
                reward += speed_reward
                info["components"]["speed_rewards"] = speed_reward
                
                # 🎯 BÔNUS QUALIDADE SCALPING
                scalping_reward = self._calculate_scalping_quality_rewards(last_trade)
                reward += scalping_reward
                info["components"]["scalping_quality"] = scalping_reward
                
            else:
                loss_penalty = self.weights["loss_penalty"]
                reward += loss_penalty
                info["components"]["loss_penalty"] = loss_penalty
                
                # Penalidade adicional por hold longo com loss
                if duration > self.long_hold_penalty:
                    long_hold_penalty = self.weights["long_hold_penalty"]
                    reward += long_hold_penalty
                    info["components"]["long_hold_penalty"] = long_hold_penalty
            
            # 🛡️ GESTÃO DE RISCO INTRADAY
            risk_reward = self._calculate_daytrading_risk_management(last_trade)
            reward += risk_reward
            info["components"]["risk_management"] = risk_reward
            
            # 📈 ANÁLISE TÉCNICA INTRADAY
            technical_reward = self._calculate_intraday_technical_analysis(env, last_trade)
            reward += technical_reward
            info["components"]["technical_analysis"] = technical_reward
            
            # 📊 ATUALIZAR MÉTRICAS DE SCALPING
            self._update_scalping_metrics(last_trade)
        
        # 🎯 ATIVIDADE CALIBRADA PARA DAY TRADING (com cache otimizado)
        trades_today = current_trades_count  # Já calculado, evitar nova chamada
        activity_reward = self._calculate_daytrading_activity_reward(trades_today)
        reward += activity_reward
        info["components"]["activity_reward"] = activity_reward
        
        # 📊 CONSISTÊNCIA DE SESSÃO (apenas quando há trades)
        if current_trades_count > 0:
            session_reward = self._calculate_session_consistency_reward(env)
            reward += session_reward
            info["components"]["session_consistency"] = session_reward
        
        # 🚫 PENALIDADES ESPECÍFICAS DAY TRADING (apenas quando relevante)
        if trades_today > 10:  # Só calcular quando há activity significativa
            penalties = self._calculate_daytrading_penalties(env, trades_today)
            reward += penalties
            info["components"]["daytrading_penalties"] = penalties
        
        # 🚀 OPUS FASE 1: SCALING ADAPTATIVO (substituir clipping fixo)
        reward = self._adaptive_reward_scaling(reward)
        
        # 🧠 OPUS FASE 2: Aplicar sistemas avançados
        if self.phase2_enabled:
            reward, phase2_info = self._apply_phase2_enhancements(reward, env, old_state, info)
            info.update(phase2_info)
        
        # 📊 INFORMAÇÕES DETALHADAS DAY TRADING
        info.update({
            "trades_today": trades_today,
            "target_trades": self.target_trades_per_day,
            "activity_zone": self._get_daytrading_activity_zone(trades_today),
            "total_reward": reward,
            "step_count": self.step_count,
            "scalping_metrics": {
                "daily_scalps": self.daily_scalps,
                "consecutive_quick_trades": self.consecutive_quick_trades,
                "avg_trade_duration": self.scalping_metrics.avg_trade_duration,
                "quick_trades_ratio": self.scalping_metrics.quick_trades_ratio
            },
            "system_info": {
                "system_type": "daytrading_specialized",
                "focus": "scalping_and_speed",
                "timeframe": "5min_intraday",
                "sl_tp_ranges": f"{self.sl_range_min}-{self.sl_range_max} SL, {self.tp_range_min}-{self.tp_range_max} TP"
            }
        })
        
        # 🔧 CLIPPING FINAL PARA EVITAR REWARDS EXTREMOS
        reward = np.clip(reward, -30.0, 35.0)
        
        return reward, info, done
    
    def _calculate_speed_rewards(self, duration: int, pnl: float) -> float:
        """⚡ Recompensas por velocidade de execução (essencial day trading)"""
        speed_reward = 0.0
        
        # Scalp ultra-rápido (<20 steps = 100min)
        if duration <= self.quick_scalp_max and pnl > 2.0:
            speed_reward += self.weights["quick_scalp_bonus"]
            self.consecutive_quick_trades += 1
            
            # Bônus extra por scalp perfeito (rápido + lucrativo)
            if pnl > 8.0:
                speed_reward += self.weights["perfect_scalp_bonus"]
        
        # Execução rápida geral (até 60 steps = 5h)
        elif duration <= self.optimal_trade_max and pnl > 0:
            speed_reward += self.weights["speed_execution_bonus"]
        
        # Bônus por sequência de trades rápidos
        if self.consecutive_quick_trades >= 3:
            speed_reward += self.weights["precision_entry_streak"]
        
        return speed_reward
    
    def _calculate_scalping_quality_rewards(self, trade: Dict) -> float:
        """🎯 Qualidade específica de scalping"""
        scalping_reward = 0.0
        
        pnl = trade.get('pnl_usd', 0.0)
        duration = trade.get('duration_steps', 0)
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', entry_price)
        
        # Precisão de entrada (movimento favorável imediato)
        price_movement = abs(exit_price - entry_price)
        if pnl > 0 and price_movement > entry_price * 0.001:  # 0.1% movement
            scalping_reward += self.weights["rapid_entry_bonus"]
            self.scalping_metrics.precision_entries += 1
        
        # Take profit rápido e eficiente
        if duration <= 15 and pnl > 5.0:  # 75min com lucro >$5
            scalping_reward += self.weights["rapid_profit_taking"]
            self.scalping_metrics.rapid_exits += 1
        
        # Captura de momentum intraday
        if duration <= 30 and pnl > 3.0:  # 2.5h com lucro >$3
            scalping_reward += self.weights["momentum_capture"]
        
        # Contabilizar scalp
        if duration <= self.quick_scalp_max:
            self.daily_scalps += 1
        
        return scalping_reward
    
    def _calculate_daytrading_risk_management(self, trade: Dict) -> float:
        """🛡️ Gestão de risco específica para day trading"""
        risk_reward = 0.0
        
        sl_points = abs(trade.get('sl_points', 0))
        tp_points = abs(trade.get('tp_points', 0))
        
        if sl_points == 0 or tp_points == 0:
            return 0.0
        
        # Ratio ótimo para day trading (mais flexível: 1.2-2.0)
        risk_reward_ratio = tp_points / sl_points
        if self.optimal_risk_reward_min <= risk_reward_ratio <= self.optimal_risk_reward_max:
            risk_reward += self.weights["optimal_ratio_bonus"]
        
        # Gestão ativa (ajustes de SL/TP durante o trade)
        if trade.get('sl_adjusted', False) or trade.get('tp_adjusted', False):
            risk_reward += self.weights["risk_management_bonus"]
        
        return risk_reward
    
    def _calculate_intraday_technical_analysis(self, env, trade: Dict) -> float:
        """📈 Análise técnica focada em intraday"""
        technical_reward = 0.0
        
        try:
            entry_price = trade.get('entry_price', 0)
            if not entry_price or not hasattr(env, 'df'):
                return 0.0
            
            current_step = getattr(env, 'current_step', 0)
            if current_step < 20:
                return 0.0
            
            # Análise dos últimos 20 períodos (100min de histórico)
            recent_data = env.df.iloc[max(0, current_step-20):current_step]
            
            if len(recent_data) < 10:
                return 0.0
            
            highs = recent_data['high_5m'].values if 'high_5m' in recent_data.columns else []
            lows = recent_data['low_5m'].values if 'low_5m' in recent_data.columns else []
            closes = recent_data['close_5m'].values if 'close_5m' in recent_data.columns else []
            
            if len(highs) < 10:
                return 0.0
            
            # 🎯 SUPORTE/RESISTÊNCIA INTRADAY (últimos 100min)
            recent_highs = sorted(highs[-10:])
            recent_lows = sorted(lows[-10:])
            
            # Verificar se entrada foi próxima a S/R
            for level in recent_lows + recent_highs:
                if abs(entry_price - level) < entry_price * 0.002:  # 0.2% tolerância
                    technical_reward += self.weights["support_resistance_entry"]
                    break
            
            # 🎯 BREAKOUT INTRADAY
            if len(closes) >= 5:
                recent_high = max(highs[-5:])
                recent_low = min(lows[-5:])
                
                trade_type = trade.get('type', 'long')
                if trade_type == 'long' and entry_price > recent_high * 1.0005:  # Breakout acima
                    technical_reward += self.weights["breakout_timing"]
                elif trade_type == 'short' and entry_price < recent_low * 0.9995:  # Breakout abaixo
                    technical_reward += self.weights["breakout_timing"]
            
            # 🎯 CONFIRMAÇÃO DE VOLUME (se disponível)
            if 'volume_5m' in recent_data.columns:
                volumes = recent_data['volume_5m'].values
                if len(volumes) >= 5:
                    avg_volume = np.mean(volumes[-5:])
                    current_volume = volumes[-1]
                    
                    if current_volume > avg_volume * 1.3:  # Volume 30% acima da média
                        technical_reward += self.weights["volume_confirmation"]
            
            # 🎯 ALINHAMENTO TENDÊNCIA INTRADAY (últimos 10 períodos)
            if len(closes) >= 10:
                trend_short = np.mean(closes[-5:])
                trend_long = np.mean(closes[-10:])
                
                trade_type = trade.get('type', 'long')
                trend_up = trend_short > trend_long
                trend_down = trend_short < trend_long
                
                if (trade_type == 'long' and trend_up) or (trade_type == 'short' and trend_down):
                    technical_reward += self.weights["trend_alignment_intraday"]
        
        except Exception as e:
            # 🚀 OPUS FASE 1: Log estruturado + fallback inteligente
            import logging
            logging.warning(f"Technical analysis failed: {e}")
            # Fallback baseado em PnL simples
            technical_reward = max(0, pnl * 0.1) if pnl > 0 else 0
        
        return technical_reward
    
    def _calculate_daytrading_activity_reward(self, trades_today: int) -> float:
        """🎯 Atividade calibrada para day trading (20-50 trades/dia)"""
        activity_reward = 0.0
        
        # Zona alvo expandida para day trading
        if self.target_zone_min <= trades_today <= self.target_zone_max:
            # Bônus graduado dentro da zona
            if 30 <= trades_today <= 40:  # Zona ótima
                activity_reward += 2.0
            else:  # Zona boa
                activity_reward += 1.0
        
        # Bônus por múltiplos scalps
        if self.daily_scalps >= 5:
            activity_reward += self.weights["multiple_scalps_bonus"]
        
        return activity_reward
    
    def _calculate_session_consistency_reward(self, env) -> float:
        """📊 Consistência dentro da sessão de day trading"""
        session_reward = 0.0
        
        try:
            if not hasattr(env, 'trades') or len(env.trades) < 3:
                return 0.0
            
            # Analisar últimos 5 trades para consistência
            recent_trades = env.trades[-5:]
            wins = sum(1 for t in recent_trades if t.get('pnl_usd', 0) > 0)
            
            # Win rate alto na sessão
            if len(recent_trades) >= 3:
                win_rate = wins / len(recent_trades)
                if win_rate >= 0.6:  # 60%+ win rate
                    session_reward += self.weights["session_consistency"]
                
                # Bônus por padrão de scalping bem-sucedido
                quick_trades = sum(1 for t in recent_trades 
                                 if t.get('duration_steps', 999) <= self.quick_scalp_max)
                if quick_trades >= 3 and win_rate >= 0.6:
                    session_reward += self.weights["pattern_recognition"]
        
        except Exception as e:
            # 🚀 OPUS FASE 1: Log e fallback inteligente
            import logging
            logging.warning(f"Session consistency calculation failed: {e}")
            session_reward = 0.0
        
        return session_reward
    
    def _calculate_daytrading_penalties(self, env, trades_today: int) -> float:
        """🚫 Penalidades específicas para day trading"""
        penalties = 0.0
        
        # Overtrading severo (>60 trades/dia)
        if trades_today > 60:
            penalties += self.weights["overtrading_penalty"]
        
        # Penalidade por posições overnight (se detectável)
        if hasattr(env, 'positions'):
            for pos in env.positions:
                duration = self.step_count - pos.get('entry_step', self.step_count)
                if duration > 200:  # >16h (posição overnight)
                    penalties += self.weights["overnight_penalty"]
        
        # Fadiga de sessão (muitos trades consecutivos sem pausa)
        if trades_today > 40 and self.step_count > 0:
            recent_trade_density = trades_today / max(self.step_count / 50, 1)  # Trades por 50 steps
            if recent_trade_density > 3:  # >3 trades por período de 250min
                penalties += self.weights["session_fatigue_penalty"]
        
        return penalties
    
    def _update_scalping_metrics(self, trade: Dict):
        """📊 Atualizar métricas de scalping"""
        duration = trade.get('duration_steps', 0)
        
        # Atualizar duração média
        if hasattr(self, '_total_duration'):
            self._total_duration += duration
            self._total_trades += 1
        else:
            self._total_duration = duration
            self._total_trades = 1
        
        self.scalping_metrics.avg_trade_duration = self._total_duration / self._total_trades
        
        # Ratio de trades rápidos
        if duration <= self.quick_scalp_max:
            self.scalping_metrics.quick_trades_ratio = (
                self.daily_scalps / max(self._total_trades, 1)
            )
    
    def _get_daytrading_activity_zone(self, trades_today: int) -> str:
        """Determina zona de atividade para day trading"""
        if trades_today < self.target_zone_min:
            return "UNDERTRADING"
        elif self.target_zone_min <= trades_today <= self.target_zone_max:
            if 30 <= trades_today <= 40:
                return "OPTIMAL_ZONE"
            else:
                return "TARGET_ZONE"
        elif trades_today > 60:
            return "SEVERE_OVERTRADING"
        else:
            return "MODERATE_OVERTRADING"
    
    def _update_dynamic_weights(self):
        """🚀 OPUS FASE 1: Atualizar pesos dinamicamente baseado em volatilidade e win rate"""
        try:
            # Calcular multiplicador de volatilidade (1.0 - 2.0)
            volatility_multiplier = min(2.0, 1.0 + self.current_volatility * 0.5)
            
            # Calcular momentum de win rate (0.0 - 0.5)
            win_rate_bonus = min(0.5, self.win_rate_momentum)
            
            # Aplicar scaling dinâmico
            self.weights["pnl_direct"] = self.base_weights["pnl_direct"] * volatility_multiplier
            self.weights["win_bonus"] = self.base_weights["win_bonus"] * (1.0 + win_rate_bonus)
            self.weights["perfect_scalp_bonus"] = self.base_weights["perfect_scalp_bonus"] * volatility_multiplier
            
        except Exception as e:
            # 🚀 OPUS FASE 1: Validação robusta - fallback para base weights
            import logging
            logging.warning(f"Dynamic weight update failed: {e}")
            self.weights = self.base_weights.copy()
    
    def _adaptive_reward_scaling(self, raw_reward: float) -> float:
        """🚀 OPUS FASE 1: Scaling adaptativo baseado no progresso de treinamento - AJUSTADO"""
        try:
            base_scale = 20.0  # Reduzido de 30.0 para 20.0
            
            # Growth factor mais conservador (1.0 → 1.5)
            growth_factor = min(1.5, 1.0 + self.episode_count / 15000)  # Mais lento
            
            # Scaling baseado na volatilidade mais conservador
            volatility_scale = 1.0 + (self.current_volatility - 1.0) * 0.2  # Reduzido de 0.3 para 0.2
            
            # Scaling final adaptativo
            max_reward = base_scale * growth_factor * volatility_scale
            min_reward = -base_scale * 0.8  # Penalties menos severas
            
            return np.clip(raw_reward, min_reward, max_reward)
            
        except Exception as e:
            # Fallback para clipping conservador
            import logging
            logging.warning(f"Adaptive scaling failed: {e}")
            return np.clip(raw_reward, -20.0, 25.0)  # Mais conservador
    
    def _validate_trade_data(self, trade: Dict) -> bool:
        """🚀 OPUS FASE 1: Validação robusta de dados de trade"""
        try:
            # Campos obrigatórios
            required_fields = ['pnl_usd', 'duration_steps', 'entry_price']
            
            for field in required_fields:
                if field not in trade or trade[field] is None:
                    return False
                    
                # Validação de tipos e ranges
                if field == 'pnl_usd' and not isinstance(trade[field], (int, float)):
                    return False
                if field == 'duration_steps' and (not isinstance(trade[field], int) or trade[field] < 0):
                    return False
                if field == 'entry_price' and (not isinstance(trade[field], (int, float)) or trade[field] <= 0):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _calculate_market_volatility(self, env) -> float:
        """🚀 OPUS FASE 1: Calcular volatilidade atual do mercado"""
        try:
            if not hasattr(env, 'df') or len(env.df) < 20:
                return 1.0
                
            current_step = getattr(env, 'current_step', 0)
            if current_step < 20:
                return 1.0
                
            # Últimos 20 períodos para volatilidade
            recent_data = env.df.iloc[max(0, current_step-20):current_step]
            
            if 'close_5m' in recent_data.columns and len(recent_data) >= 10:
                closes = recent_data['close_5m'].values
                returns = np.diff(closes) / closes[:-1]
                volatility = np.std(returns) * 100  # Converter para %
                
                # Normalizar volatility (0.5 - 2.0)
                normalized_vol = max(0.5, min(2.0, volatility / 0.5))
                
                # Atualizar histórico
                self.volatility_history.append(normalized_vol)
                if len(self.volatility_history) > 100:
                    self.volatility_history.pop(0)
                
                return normalized_vol
            
            return 1.0
            
        except Exception:
            return 1.0
    
    def _apply_phase2_enhancements(self, base_reward: float, env, old_state: Dict, base_info: Dict) -> Tuple[float, Dict[str, Any]]:
        """🧠 OPUS FASE 2: Aplicar progressive shaping + pattern detection + anti-gaming"""
        enhanced_reward = base_reward
        phase2_info = {
            "phase2_enabled": True,
            "progressive_shaping": {},
            "pattern_detection": {},
            "anti_gaming": {}
        }
        
        try:
            # 1. PROGRESSIVE REWARD SHAPING
            context = self._build_context_for_phase2(env, old_state, base_info)
            shaped_reward, shaping_info = self.progressive_shaper.shape_reward(enhanced_reward, context)
            enhanced_reward = shaped_reward
            phase2_info["progressive_shaping"] = shaping_info
            
            # 2. PATTERN DETECTION BONUS
            if hasattr(env, 'trades') and env.trades:
                # Update pattern detector with latest trade
                last_trade = env.trades[-1] if env.trades else {}
                market_data = self._extract_market_data(env)
                self.pattern_detector.add_trade_data(last_trade, market_data)
                
                # Detect patterns and calculate bonus
                patterns = self.pattern_detector.detect_patterns(context)
                pattern_bonus, pattern_info = self.pattern_detector.calculate_pattern_bonus(patterns)
                enhanced_reward += pattern_bonus
                phase2_info["pattern_detection"] = pattern_info
            
            # 3. ANTI-GAMING DETECTION AND PENALTY
            if hasattr(env, 'trades') and len(env.trades) > 0:
                # Update anti-gaming with latest data
                last_trade = env.trades[-1] if env.trades else {}
                reward_data = base_info
                action_data = old_state.get('last_action', {})
                self.anti_gaming.add_trade_data(last_trade, reward_data, action_data)
                
                # Detect gaming and apply penalty
                gaming_detections = self.anti_gaming.detect_gaming()
                gaming_penalty, gaming_info = self.anti_gaming.calculate_penalty(gaming_detections)
                enhanced_reward -= gaming_penalty
                phase2_info["anti_gaming"] = gaming_info
            
        except Exception as e:
            # Fallback gracefully
            import logging
            logging.warning(f"Phase 2 enhancement failed: {e}")
            phase2_info["error"] = str(e)
            enhanced_reward = base_reward
        
        return enhanced_reward, phase2_info
    
    def _build_context_for_phase2(self, env, old_state: Dict, base_info: Dict) -> Dict[str, Any]:
        """Construir contexto para sistemas Fase 2"""
        context = {
            'current_step': getattr(env, 'current_step', 0),
            'total_trades': len(getattr(env, 'trades', [])),
            'current_hour': 12,  # Default, can be extracted from timestamps
        }
        
        # Recent trade data
        if hasattr(env, 'trades') and env.trades:
            recent_trades = env.trades[-10:] if len(env.trades) >= 10 else env.trades
            
            context.update({
                'recent_trade_durations': [t.get('duration_steps', 0) for t in recent_trades],
                'recent_win_sequence': [1 if t.get('pnl_usd', 0) > 0 else 0 for t in recent_trades],
                'recent_performance_scores': [min(1.0, max(0.0, t.get('pnl_usd', 0) / 10)) for t in recent_trades],
                'avg_profit_per_trade': np.mean([t.get('pnl_usd', 0) for t in recent_trades if t.get('pnl_usd', 0) > 0]) if recent_trades else 0.0,
                'avg_loss_per_trade': np.mean([t.get('pnl_usd', 0) for t in recent_trades if t.get('pnl_usd', 0) < 0]) if recent_trades else 0.0,
                'recent_max_drawdown': 5.0,  # Simplified, can be calculated properly
            })
            
            # SL/TP variety scores (simplified)
            sl_values = [t.get('sl_points', 0) for t in recent_trades if t.get('sl_points', 0) > 0]
            tp_values = [t.get('tp_points', 0) for t in recent_trades if t.get('tp_points', 0) > 0]
            
            context.update({
                'sl_variety_score': min(1.0, np.std(sl_values) / (np.mean(sl_values) + 1e-8)) if sl_values else 0.0,
                'tp_variety_score': min(1.0, np.std(tp_values) / (np.mean(tp_values) + 1e-8)) if tp_values else 0.0,
                'session_variety_score': 0.5,  # Simplified
                'current_risk_level': 1.0,  # Simplified
                'sharpe_ratio': 1.0,  # Simplified
                'profit_per_minute': 0.1,  # Simplified
            })
        
        return context
    
    def _extract_market_data(self, env) -> Dict[str, Any]:
        """Extrair dados de mercado para pattern detection"""
        market_data = {}
        
        try:
            if hasattr(env, 'df') and len(env.df) > 0:
                current_step = getattr(env, 'current_step', 0)
                if current_step < len(env.df):
                    current_row = env.df.iloc[current_step]
                    
                    market_data.update({
                        'price': current_row.get('close_5m', 0.0),
                        'volume': current_row.get('volume_5m', 0.0),
                        'high': current_row.get('high_5m', 0.0),
                        'low': current_row.get('low_5m', 0.0),
                    })
        except Exception:
            # Fallback data
            market_data = {'price': 1.0, 'volume': 1000.0}
        
        return market_data
    
    def _get_trades_today(self, env) -> int:
        """Conta trades realizados hoje"""
        try:
            if hasattr(env, 'trades') and env.trades:
                return len(env.trades)
            return 0
        except Exception:
            return 0
    
    def calculate_reward(self, action, portfolio_value, trades, current_step, old_state):
        """Método de compatibilidade para sistemas legados"""
        class MockEnv:
            def __init__(self, trades, current_step, portfolio_value):
                self.trades = trades
                self.current_step = current_step
                self.portfolio_value = portfolio_value
                self.positions = []
                self.current_drawdown = 0.0
        
        env = MockEnv(trades, current_step, portfolio_value)
        reward, info, done = self.calculate_reward_and_info(env, action, old_state)
        return reward

def create_daytrading_reward_system(initial_balance: float = 1000.0):
    """Factory function para criar sistema de rewards específico para day trading"""
    return DayTradingRewardCalculator(initial_balance)

# Configuração específica para day trading
DAYTRADING_REWARD_CONFIG = {
    "initial_balance": 1000.0,
    "system_type": "daytrading_specialized_opus_v2",
    "target_trades_per_day": 35,
    "sl_range": "2-8 points",
    "tp_range": "3-15 points", 
    "optimal_duration": "5-60 steps (25min-5h)",
    "description": "Sistema de recompensas OPUS FASE 2 COMPLETO - Curriculum learning + Pattern detection + Anti-gaming",
    "version": "2.0_opus_phase2_complete",
    "phase1_improvements": [
        "Adaptive reward scaling based on episode progress",
        "Dynamic weight adjustment based on market volatility", 
        "Robust validation with intelligent fallbacks",
        "Enhanced error handling with structured logging"
    ],
    "phase2_improvements": [
        "Progressive reward shaping with curriculum learning",
        "Simple ML pattern detection (momentum-based)",
        "Anti-gaming system with statistical detection",
        "Context-aware bonus/penalty system",
        "Graceful fallback for maximum reliability"
    ],
    "training_phases": {
        "exploration": "0-100k steps: Focus on discovery + variety",
        "refinement": "100k-500k steps: Focus on consistency + stability", 
        "mastery": "500k+ steps: Focus on performance + efficiency"
    },
    "pattern_detection": [
        "Momentum alignment with entry timing",
        "Volume confirmation signals",
        "Win streak momentum patterns",
        "Time-based performance patterns",
        "Mean reversion detection"
    ],
    "anti_gaming_protection": [
        "Reward farming detection (small profit repetition)",
        "Artificial pattern detection (unnatural uniformity)",
        "Risk-free arbitrage detection (exploit prevention)",
        "Correlation exploit detection (statistical anomalies)"
    ]
}