#!/usr/bin/env python3
"""
🎯 UNIFIED REWARD COM COMPONENTES ESPECIALIZADOS
Sistema de reward único com feedback específico sobre timing vs gestão
"""

import numpy as np
from typing import Dict, Any, Tuple
from collections import deque
import pandas as pd


class UnifiedRewardWithComponents:
    """
    Reward único alimenta o backbone, mas com componentes específicos
    que dão feedback diferenciado sobre timing vs gestão
    """

    def __init__(self, 
                 base_weight: float = 0.8,
                 timing_weight: float = 0.1, 
                 management_weight: float = 0.1,
                 verbose: bool = False):
        
        # PESOS CONSERVADORES - começar pequeno
        self.base_weight = base_weight           # Reward tradicional (dominante)
        self.timing_weight = timing_weight       # Componente de timing  
        self.management_weight = management_weight  # Componente de gestão

        self.verbose = verbose

        # HISTÓRICO para análise
        self.component_history = {
            'base': [], 'timing': [], 'management': [], 'total': []
        }
        
        # Configurações de timing
        self.volatility_threshold = 0.005  # Threshold para mercado ativo
        self.momentum_confluence_threshold_high = 0.6  # Momentum forte
        self.momentum_confluence_threshold_low = 0.3   # Momentum fraco
        
        # Configurações de management
        self.min_duration_for_penalty = 30  # Menos de 150min (30 steps * 5m)
        self.min_activity_threshold = 0.1   # Atividade mínima SL/TP
        self.max_activity_threshold = 1.0   # Atividade máxima SL/TP

    def calculate_unified_reward(self, base_reward: float, action: np.ndarray, info: Dict[str, Any], env) -> Tuple[float, Dict[str, float]]:
        """
        Calcula reward unificado com componentes especializados
        
        Args:
            base_reward: Reward tradicional do sistema existente
            action: Ação do modelo (11 dimensões)
            info: Informações do step
            env: Environment de trading
            
        Returns:
            Tuple[final_reward, components_dict]
        """
        # COMPONENTE 1: TIMING QUALITY
        timing_component = self.calculate_timing_component(action, info, env)
        
        # COMPONENTE 2: MANAGEMENT QUALITY
        management_component = self.calculate_management_component(action, info, env)
        
        # REWARD FINAL UNIFICADO
        final_reward = (
            self.base_weight * base_reward +
            self.timing_weight * timing_component +
            self.management_weight * management_component
        )
        
        # Registrar no histórico
        self.component_history['base'].append(base_reward)
        self.component_history['timing'].append(timing_component)
        self.component_history['management'].append(management_component)  
        self.component_history['total'].append(final_reward)
        
        # Manter histórico limitado (últimos 10000 steps)
        if len(self.component_history['base']) > 10000:
            for key in self.component_history:
                self.component_history[key] = self.component_history[key][-5000:]
        
        components_dict = {
            'base': base_reward,
            'timing': timing_component,
            'management': management_component,
            'final': final_reward
        }
        
        # Log se verboso e há atividade
        if self.verbose and (timing_component != 0 or management_component != 0):
            print(f"📊 Unified Reward: Base={base_reward:.4f}, Timing={timing_component:.4f}, Mgmt={management_component:.4f}, Final={final_reward:.4f}")
        
        return final_reward, components_dict

    def calculate_timing_component(self, action: np.ndarray, info: Dict[str, Any], env) -> float:
        """
        Componente que avalia qualidade do TIMING da entrada
        Não substitui profit/loss, apenas adiciona feedback sobre momento
        """
        timing_bonus = 0.0

        if info.get('new_position_opened', False):
            entry_quality = action[1]  # Entry quality do modelo
            current_step = env.current_step

            # MÉTODO 1: Volatilidade recente (proxy para movimento futuro)
            if current_step >= 20:
                recent_volatility = self._calculate_recent_volatility(env, current_step, window=20)

                if recent_volatility > self.volatility_threshold:
                    # Mercado ativo = bom timing
                    timing_bonus = +0.2 * entry_quality
                else:
                    # Mercado parado = timing questionável
                    timing_bonus = -0.1 * (1.0 - entry_quality)

            # MÉTODO 2: Momentum confluence (múltiplos timeframes)
            momentum_score = self._evaluate_momentum_confluence(env, current_step)
            if momentum_score > self.momentum_confluence_threshold_high:  # Momentum forte
                timing_bonus += 0.1 * entry_quality
            elif momentum_score < self.momentum_confluence_threshold_low:  # Momentum fraco
                timing_bonus -= 0.1 * (1.0 - entry_quality)

        return timing_bonus

    def calculate_management_component(self, action: np.ndarray, info: Dict[str, Any], env) -> float:
        """
        Componente que avalia qualidade da GESTÃO de posições
        Avalia SL/TP placement, duration, trailing usage
        """
        management_bonus = 0.0

        # 1. AVALIAÇÃO DE NOVA POSIÇÃO (SL/TP placement)
        if info.get('new_position_opened', False):
            management_bonus += self._evaluate_sl_tp_placement(action, info, env)

        # 2. AVALIAÇÃO DE FECHAMENTO (efficiency)
        if info.get('position_closed', False):
            management_bonus += self._evaluate_closure_efficiency(action, info, env)

        # 3. AVALIAÇÃO DE GESTÃO ATIVA (trailing, adjustments)
        management_bonus += self._evaluate_active_management(action, info, env)

        return management_bonus

    def _calculate_recent_volatility(self, env, step: int, window: int = 20) -> float:
        """Calcular volatilidade das últimas N barras"""
        if step < window:
            return 0.0

        try:
            recent_prices = env.df['close_5m'].iloc[step-window:step]
            returns = recent_prices.pct_change().dropna()
            return returns.std() if len(returns) > 0 else 0.0
        except (KeyError, IndexError):
            return 0.0

    def _evaluate_momentum_confluence(self, env, step: int) -> float:
        """Avaliar confluência de momentum em diferentes timeframes"""
        if step < 50:
            return 0.5  # Neutro

        try:
            # Momentum 1: Short term (5 bars)
            short_momentum = self._calculate_momentum(env.df['close_5m'], step, 5)

            # Momentum 2: Medium term (20 bars)  
            medium_momentum = self._calculate_momentum(env.df['close_5m'], step, 20)

            # Confluence: quando ambos apontam na mesma direção
            if (short_momentum > 0 and medium_momentum > 0) or (short_momentum < 0 and medium_momentum < 0):
                return 0.8  # Alta confluência
            else:
                return 0.2  # Baixa confluência
        except (KeyError, IndexError):
            return 0.5  # Neutro em caso de erro

    def _calculate_momentum(self, prices: pd.Series, step: int, window: int) -> float:
        """Calcular momentum para uma janela específica"""
        if step < window:
            return 0.0
        
        try:
            start_price = prices.iloc[step - window]
            end_price = prices.iloc[step - 1]
            return (end_price - start_price) / start_price if start_price != 0 else 0.0
        except (IndexError, ZeroDivisionError):
            return 0.0

    def _evaluate_sl_tp_placement(self, action: np.ndarray, info: Dict[str, Any], env) -> float:
        """Avaliar qualidade inicial de SL/TP"""
        if 'new_position' not in info:
            return 0.0

        position = info['new_position']
        entry_price = position.get('entry_price', 0)
        sl_price = position.get('sl_price', 0)  
        tp_price = position.get('tp_price', 0)

        if entry_price == 0 or sl_price == 0 or tp_price == 0:
            return 0.0

        # Calcular distâncias
        sl_distance = abs(sl_price - entry_price)
        tp_distance = abs(tp_price - entry_price)
        risk_reward_ratio = tp_distance / sl_distance if sl_distance > 0 else 1.0

        # REWARD baseado em risk/reward ratio
        if 1.5 <= risk_reward_ratio <= 3.0:  # Ratio saudável
            return +0.1
        elif risk_reward_ratio < 1.0:  # Ratio ruim
            return -0.1
        else:
            return 0.0  # Ratio questionável mas aceitável

    def _evaluate_closure_efficiency(self, action: np.ndarray, info: Dict[str, Any], env) -> float:
        """Avaliar eficiência do fechamento"""
        if 'closed_trade' not in info:
            return 0.0

        trade = info['closed_trade']
        duration = trade.get('duration', 0)
        exit_reason = trade.get('exit_reason', '')
        pnl = trade.get('pnl_usd', 0)

        bonus = 0.0

        # 1. Penalizar trades muito curtos (overtrading)
        if duration < self.min_duration_for_penalty:  # Menos de 150min
            bonus -= 0.15 * (1.0 - duration / self.min_duration_for_penalty)  # Penalidade progressiva

        # 2. Reward por uso eficiente de trailing
        if trade.get('trailing_activated', False) and pnl > 0:
            bonus += 0.1

        # 3. Reward por saídas inteligentes (não apenas SL hit)
        if exit_reason == 'take_profit':
            bonus += 0.05
        elif exit_reason == 'trailing_stop':
            bonus += 0.08
        elif exit_reason == 'time_exit' and pnl > 0:  # Exit inteligente por tempo
            bonus += 0.03

        return bonus

    def _evaluate_active_management(self, action: np.ndarray, info: Dict[str, Any], env) -> float:
        """Avaliar gestão ativa durante a posição"""
        bonus = 0.0

        # Verificar ajustes de SL/TP nas actions atuais
        if len(action) >= 11:  # V7 action space
            sl_adjusts = action[5:8]   # SL adjusts
            tp_adjusts = action[8:11]  # TP adjusts

            # Reward por uso moderado de adjustments (não zero, não excessivo)
            sl_activity = np.mean(np.abs(sl_adjusts))
            tp_activity = np.mean(np.abs(tp_adjusts))

            if self.min_activity_threshold <= sl_activity <= self.max_activity_threshold:  # Atividade moderada
                bonus += 0.02
            if self.min_activity_threshold <= tp_activity <= self.max_activity_threshold:
                bonus += 0.02

        return bonus

    def get_component_stats(self) -> Dict[str, float]:
        """Obter estatísticas dos componentes"""
        if not self.component_history['base']:
            return {}
        
        stats = {}
        for component in ['base', 'timing', 'management', 'total']:
            values = self.component_history[component][-1000:]  # Últimos 1000 steps
            if values:
                stats[f"{component}_mean"] = np.mean(values)
                stats[f"{component}_std"] = np.std(values)
                stats[f"{component}_min"] = np.min(values)
                stats[f"{component}_max"] = np.max(values)
        
        return stats

    def reset_history(self):
        """Reset do histórico de componentes"""
        for key in self.component_history:
            self.component_history[key] = []


class ComponentRewardMonitor:
    """Monitor para analisar eficácia dos componentes"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.components_history = {
            'base': deque(maxlen=window_size),
            'timing': deque(maxlen=window_size), 
            'management': deque(maxlen=window_size),
            'total': deque(maxlen=window_size)
        }

    def log_step(self, base: float, timing: float, management: float, total: float):
        """Log de um step dos componentes"""
        self.components_history['base'].append(base)
        self.components_history['timing'].append(timing)
        self.components_history['management'].append(management)
        self.components_history['total'].append(total)

    def analyze_components(self):
        """Análise dos componentes e seus impactos"""
        if len(self.components_history['total']) < 100:
            return

        base_mean = np.mean(self.components_history['base'])
        timing_mean = np.mean(self.components_history['timing'])
        mgmt_mean = np.mean(self.components_history['management'])

        # Correlações
        base_arr = np.array(self.components_history['base'])
        timing_arr = np.array(self.components_history['timing'])
        mgmt_arr = np.array(self.components_history['management'])

        timing_base_corr = np.corrcoef(timing_arr, base_arr)[0, 1] if len(timing_arr) > 1 else 0.0
        mgmt_base_corr = np.corrcoef(mgmt_arr, base_arr)[0, 1] if len(mgmt_arr) > 1 else 0.0

        print(f"\n📊 REWARD COMPONENTS ANALYSIS:")
        print(f"  Base Reward: {base_mean:.4f}")
        print(f"  Timing Component: {timing_mean:.4f}")
        print(f"  Management Component: {mgmt_mean:.4f}")
        print(f"  Timing-Base Correlation: {timing_base_corr:.3f}")
        print(f"  Management-Base Correlation: {mgmt_base_corr:.3f}")

        # Alertas
        if abs(timing_mean) > 0.05:
            print(f"  ⚠️ Timing component very active: {timing_mean:.4f}")
        if abs(mgmt_mean) > 0.05:
            print(f"  ⚠️ Management component very active: {mgmt_mean:.4f}")

    def get_summary(self) -> Dict[str, float]:
        """Obter resumo das estatísticas"""
        if len(self.components_history['total']) < 10:
            return {}

        return {
            'base_mean': np.mean(self.components_history['base']),
            'timing_mean': np.mean(self.components_history['timing']),
            'management_mean': np.mean(self.components_history['management']),
            'total_mean': np.mean(self.components_history['total']),
            'base_std': np.std(self.components_history['base']),
            'timing_std': np.std(self.components_history['timing']),
            'management_std': np.std(self.components_history['management']),
            'total_std': np.std(self.components_history['total']),
        }