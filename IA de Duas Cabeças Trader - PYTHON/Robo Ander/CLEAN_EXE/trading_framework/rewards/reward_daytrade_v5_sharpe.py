"""
🎯 SHARPE-OPTIMIZED REWARD SYSTEM V5
Sistema de reward que otimiza DIRETAMENTE o Sharpe Ratio
Baseado no V4 INNO com foco em risk-adjusted returns

🎯 FILOSOFIA V5 SHARPE:
- SHARPE RATIO = 60% do reward (dominante) 
- Risk management = 25% do reward
- Volatility control = 15% do reward
- Direct Sharpe ratio optimization (não proxy metrics)
- Risk aversion exponencial para drawdowns
- Volatility targeting dinâmico
- Position sizing baseado em risco atual
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging

class SharpeOptimizedRewardV5:
    """
    Sistema de reward focado DIRETAMENTE no Sharpe Ratio
    Otimiza risk-adjusted returns em vez de PnL absoluto
    Implementa controle de volatilidade e risk management avançado
    """
    
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        self.logger = logging.getLogger(__name__)
        
        # 🎯 V5 SHARPE CONFIGURATIONS (OTIMIZADAS PARA SHARPE RATIO)
        self.target_volatility = 0.15   # Target 15% volatility anual
        self.risk_free_rate = 0.02      # 2% risk-free rate anual
        self.max_drawdown_penalty = 0.10 # Penalty exponencial >10% DD
        self.max_reward = 3.0           # Cap aumentado para Sharpe rewards
        
        # 🎯 V5 SHARPE PARAMETERS 
        self.sharpe_weight = 0.60       # 60% Sharpe ratio direto
        self.risk_weight = 0.25         # 25% Risk management
        self.volatility_weight = 0.15   # 15% Volatility control
        self.min_returns_for_sharpe = 20 # Mínimo 20 returns para calcular Sharpe
        
        # 🎯 V5 SHARPE TRACKING
        self.last_portfolio_value = initial_balance
        self.episode_returns = []       # Lista de returns para Sharpe calculation
        self.portfolio_history = []     # Portfolio values históricos
        self.volatility_history = []    # Volatilidade histórica (rolling)
        self.current_drawdown = 0.0     # Drawdown atual
        
        # 🎯 V5 SHARPE CACHE
        self.step_counter = 0
        self.cached_sharpe_reward = 0.0
        self.cached_risk_reward = 0.0
        self.cached_volatility_reward = 0.0
        self.cached_rolling_sharpe = 0.0
        
        # Tracking para compatibilidade com V4 base code
        self.position_performance = {}  # position_id -> performance_tracking 
        self.recent_pnl_trend = []      # Últimos 10 PnLs para momentum
        
        # Métricas para debugging
        self.total_rewards_given = 0
        self.positive_rewards = 0
        self.negative_rewards = 0
        
    def calculate_reward_and_info(self, env, action: np.ndarray, old_state: Dict) -> Tuple[float, Dict, bool]:
        """
        Calcula reward V5 SHARPE:
        60% Sharpe Ratio + 25% Risk Management + 15% Volatility Control
        
        Returns:
            reward: Valor otimizado para Sharpe Ratio
            info: Informações de debug
            done: Se deve terminar episódio (nunca pelo reward system)
        """
        
        self.step_counter += 1
        
        # 🎯 1. UPDATE PORTFOLIO TRACKING (necessário para todas as métricas)
        self._update_portfolio_tracking(env)
        
        # 🎯 2. SHARPE RATIO REWARD (60% do reward - DOMINANTE)
        if self.step_counter % 3 == 0:  # Cache a cada 3 steps para performance
            self.cached_sharpe_reward, sharpe_info = self._calculate_sharpe_reward(env)
        else:
            sharpe_info = {'cached_sharpe': True}
        sharpe_reward = self.cached_sharpe_reward
        
        # 🎯 3. RISK MANAGEMENT REWARD (25% do reward) - CACHED A CADA 5 STEPS
        if self.step_counter % 5 == 0:
            self.cached_risk_reward, risk_info = self._calculate_risk_management_reward(env)
        else:
            risk_info = {'cached_risk': True}
        risk_reward = self.cached_risk_reward
        
        # 🎯 4. VOLATILITY CONTROL REWARD (15% do reward) - CACHED A CADA 7 STEPS
        if self.step_counter % 7 == 0:
            self.cached_volatility_reward, vol_info = self._calculate_volatility_control_reward(env)
        else:
            vol_info = {'cached_volatility': True}
        volatility_reward = self.cached_volatility_reward
        
        # 🎯 5. REWARD FINAL V5 SHARPE = 60% Sharpe + 25% Risk + 15% Volatility
        sharpe_component = sharpe_reward * self.sharpe_weight
        risk_component = risk_reward * self.risk_weight
        volatility_component = volatility_reward * self.volatility_weight
        total_reward = sharpe_component + risk_component + volatility_component
        
        # 🎯 6. EARLY TERMINATION DESABILITADO - Permitir episódios completos
        done = False  # 🚀 NUNCA terminar episódio por reward system V5
        
        # 🎯 7. NORMALIZAÇÃO para PPO (clipping mais amplo para Sharpe rewards)
        total_reward = np.clip(total_reward, -self.max_reward, self.max_reward)
        
        # Safeguard numérico robusto
        if not np.isfinite(total_reward) or abs(total_reward) > 100:
            total_reward = 0.0
        
        # 🎯 8. LOGGING - REDUCED FREQUENCY  
        if self.step_counter % 10 == 0:
            self._update_stats(total_reward)
        
        # 🎯 9. INFO para debugging V5 SHARPE
        info = {
            'sharpe_reward': sharpe_reward,
            'risk_reward': risk_reward,
            'volatility_reward': volatility_reward,
            'total_reward': total_reward,
            'sharpe_component': sharpe_component,
            'risk_component': risk_component,
            'volatility_component': volatility_component,
            'rolling_sharpe': self.cached_rolling_sharpe,
            'current_drawdown': self.current_drawdown,
            'portfolio_volatility': self._get_current_volatility(),
            'v5_sharpe_mode': True,
            **sharpe_info,
            **risk_info,
            **vol_info
        }
        
        return total_reward, info, done
    
    # 🎯 ============ V5 SHARPE-SPECIFIC METHODS ============
    
    def _update_portfolio_tracking(self, env):
        """🎯 V5: Atualiza tracking necessário para Sharpe calculation"""
        try:
            # Portfolio atual
            current_portfolio = getattr(env, 'portfolio_value', self.initial_balance)
            
            # Calcular return se temos valor anterior
            if len(self.portfolio_history) > 0:
                last_value = self.portfolio_history[-1]
                if last_value > 0:
                    daily_return = (current_portfolio - last_value) / last_value
                    self.episode_returns.append(daily_return)
                    
                    # Manter apenas últimos 100 returns para performance
                    if len(self.episode_returns) > 100:
                        self.episode_returns = self.episode_returns[-100:]
            
            # Update histórico
            self.portfolio_history.append(current_portfolio)
            if len(self.portfolio_history) > 100:  # Manter apenas últimos 100
                self.portfolio_history = self.portfolio_history[-100:]
            
            # Calcular drawdown atual
            if len(self.portfolio_history) >= 2:
                peak = max(self.portfolio_history)
                if peak > 0:
                    self.current_drawdown = (peak - current_portfolio) / peak
                else:
                    self.current_drawdown = 0.0
            
        except Exception as e:
            self.logger.warning(f"V5 Portfolio tracking error: {e}")
    
    def _calculate_sharpe_reward(self, env) -> Tuple[float, Dict]:
        """🎯 V5: Calcula reward baseado diretamente no Sharpe Ratio"""
        try:
            if len(self.episode_returns) < self.min_returns_for_sharpe:
                # Não temos dados suficientes - usar reward básico de PnL
                current_portfolio = getattr(env, 'portfolio_value', self.initial_balance)
                basic_return = (current_portfolio - self.initial_balance) / self.initial_balance
                reward = np.tanh(basic_return * 10) * 0.5  # Reward conservativo inicial
                
                info = {
                    'rolling_sharpe': 0.0,
                    'returns_count': len(self.episode_returns),
                    'using_basic_pnl': True
                }
                return reward, info
            
            # Calcular Sharpe ratio com últimos N returns
            recent_returns = np.array(self.episode_returns[-50:])  # Últimos 50 returns
            
            # Mean e std dos returns
            returns_mean = np.mean(recent_returns)
            returns_std = np.std(recent_returns) + 1e-8  # Evitar divisão por zero
            
            # Ajustar para risk-free rate (diário)
            daily_risk_free = self.risk_free_rate / 252
            excess_return = returns_mean - daily_risk_free
            
            # Sharpe ratio anualizado
            rolling_sharpe = (excess_return / returns_std) * np.sqrt(252)
            self.cached_rolling_sharpe = rolling_sharpe
            
            # Reward baseado no Sharpe (tanh scaling para [-3, +3])
            sharpe_reward = np.tanh(rolling_sharpe / 2.0) * 3.0
            
            info = {
                'rolling_sharpe': rolling_sharpe,
                'returns_mean': returns_mean,
                'returns_std': returns_std,
                'excess_return': excess_return,
                'returns_count': len(recent_returns)
            }
            
            return sharpe_reward, info
            
        except Exception as e:
            self.logger.warning(f"V5 Sharpe calculation error: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_risk_management_reward(self, env) -> Tuple[float, Dict]:
        """🎯 V5: Reward baseado em controle de risco"""
        try:
            risk_reward = 0.0
            
            # 1. Drawdown penalty (exponencial para DD > 10%)
            if self.current_drawdown > self.max_drawdown_penalty:
                dd_penalty = -np.exp((self.current_drawdown - self.max_drawdown_penalty) * 20) * 0.5
                risk_reward += dd_penalty
            else:
                # Small bonus para manter drawdown baixo
                dd_bonus = (self.max_drawdown_penalty - self.current_drawdown) * 2.0
                risk_reward += dd_bonus
            
            # 2. Position sizing reward (verificar se não está over-leveraged)
            total_exposure = abs(getattr(env, 'total_exposure', 0.0))
            portfolio_value = getattr(env, 'portfolio_value', self.initial_balance)
            
            if portfolio_value > 0:
                leverage_ratio = total_exposure / portfolio_value
                if leverage_ratio > 1.5:  # Over-leveraged
                    leverage_penalty = -(leverage_ratio - 1.5) * 2.0
                    risk_reward += leverage_penalty
                elif leverage_ratio < 0.1:  # Under-leveraged  
                    under_leverage_penalty = -(0.1 - leverage_ratio) * 1.0
                    risk_reward += under_leverage_penalty
            
            info = {
                'drawdown_component': dd_penalty if 'dd_penalty' in locals() else (dd_bonus if 'dd_bonus' in locals() else 0),
                'leverage_component': leverage_penalty if 'leverage_penalty' in locals() else 0,
                'current_drawdown': self.current_drawdown,
                'leverage_ratio': leverage_ratio if 'leverage_ratio' in locals() else 0
            }
            
            return risk_reward, info
            
        except Exception as e:
            self.logger.warning(f"V5 Risk management error: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_volatility_control_reward(self, env) -> Tuple[float, Dict]:
        """🎯 V5: Reward baseado em controle de volatilidade"""
        try:
            if len(self.episode_returns) < 10:
                return 0.0, {'insufficient_data': True}
            
            # Calcular volatilidade atual (rolling)
            recent_returns = np.array(self.episode_returns[-30:])  # Últimos 30 returns
            current_vol = np.std(recent_returns) * np.sqrt(252)  # Anualizada
            
            # Reward baseado na proximidade do target volatility
            vol_diff = abs(current_vol - self.target_volatility)
            
            if vol_diff < 0.05:  # Dentro de 5% do target
                vol_reward = 1.0 - (vol_diff / 0.05) * 0.5  # [0.5, 1.0]
            else:  # Fora do target - penalty
                vol_penalty = -min(vol_diff - 0.05, 0.2) * 2.0  # Max penalty -0.4
                vol_reward = vol_penalty
            
            info = {
                'current_volatility': current_vol,
                'target_volatility': self.target_volatility,
                'vol_diff': vol_diff,
                'vol_reward': vol_reward
            }
            
            return vol_reward, info
            
        except Exception as e:
            self.logger.warning(f"V5 Volatility control error: {e}")
            return 0.0, {'error': str(e)}
    
    def _get_current_volatility(self) -> float:
        """🎯 V5: Helper para obter volatilidade atual"""
        if len(self.episode_returns) < 5:
            return 0.0
        recent_returns = np.array(self.episode_returns[-20:])
        return np.std(recent_returns) * np.sqrt(252)
    
    # 🎯 ============ LEGACY V4 METHODS (manter por compatibilidade) ============
    
    def _calculate_pure_pnl_reward_v4(self, env) -> Tuple[float, Dict]:
        """
        V4 INNO: PnL reward otimizado com unrealized PnL valorizado
        """
        try:
            # PnL realizado dos trades fechados
            realized_pnl = getattr(env, 'total_realized_pnl', 0.0)
            
            # PnL não realizado das posições abertas (VALORIZADO V4)
            unrealized_pnl = getattr(env, 'total_unrealized_pnl', 0.0)
            
            # 🚀 V4 INNO: Valorizar unrealized PnL (0.5 -> 0.8)
            total_pnl = realized_pnl + (unrealized_pnl * 0.8)
            pnl_percent = total_pnl / self.initial_balance
            
            # 🚀 V4 INNO: REWARD BASE amplificado e balanceado (5.0 -> 12.0)
            pnl_percent_clipped = np.clip(pnl_percent, -0.15, 0.15)  # Max ±15%
            base_reward = pnl_percent_clipped * 12.0  # 2.4x mais forte, balanceado
            
            # PAIN MULTIPLICATION SUAVE (sem discontinuidades bruscas)
            if pnl_percent_clipped < -0.03:
                # Pain multiplier suave usando tanh para continuidade
                pain_factor = 1.0 + (self.pain_multiplier - 1.0) * np.tanh(abs(pnl_percent_clipped) * 20)
                base_reward *= pain_factor
                
            # BONUS SUAVE para lucros consistentes
            elif pnl_percent_clipped > 0.02:
                bonus_factor = 1.0 + 0.2 * np.tanh(pnl_percent_clipped * 50)  # Bonus aumentado
                base_reward *= bonus_factor
            
            info = {
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'total_pnl': total_pnl,
                'pnl_percent': pnl_percent * 100,
                'pain_applied': pnl_percent < -0.05,
                'v4_unrealized_factor': 0.8,
                'v4_pnl_multiplier': 12.0
            }
            
            return base_reward, info
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo PnL reward V4: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_intelligent_shaping(self, env, action: np.ndarray) -> Tuple[float, Dict]:
        """
        🎯 V4 INNO: SHAPING INTELIGENTE - Resolve sparsity + aumenta atividade
        """
        try:
            shaping_reward = 0.0
            info = {}
            
            # 1. PORTFOLIO PROGRESS SHAPING (mais importante - AMPLIFICADO)
            current_portfolio = getattr(env, 'portfolio_value', self.initial_balance)
            progress = (current_portfolio - self.last_portfolio_value) / self.initial_balance
            
            if abs(progress) > 0.001:  # Só se houve mudança significativa
                progress_reward = progress * self.progress_scaling * 20.0  # 2x amplificado
                shaping_reward += progress_reward
                info['progress_reward'] = progress_reward
                info['portfolio_delta'] = progress * 100
            else:
                info['progress_reward'] = 0.0
                info['portfolio_delta'] = 0.0
            
            # 2. POSITION MOMENTUM SHAPING (peso aumentado)
            momentum_reward = self._calculate_momentum_shaping_v4(env)
            shaping_reward += momentum_reward
            info['momentum_reward'] = momentum_reward
            
            # 3. 🚀 V4 INNO: POSITION PERFORMANCE REWARD (substitui age penalty)
            performance_reward = self._calculate_position_performance_reward(env)
            shaping_reward += performance_reward
            info['performance_reward'] = performance_reward
            
            # 4. ACTION DECISIVENESS AMPLIFICADO (50x mais forte)
            decisiveness_reward = self._calculate_action_decisiveness_v4(action, env)
            shaping_reward += decisiveness_reward
            info['decisiveness_reward'] = decisiveness_reward
            
            return shaping_reward, info
            
        except Exception as e:
            self.logger.error(f"Erro no shaping inteligente V4: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_position_performance_reward(self, env) -> float:
        """
        🚀 V4 INNO: Reward baseado em PERFORMANCE, não em idade
        """
        try:
            positions = getattr(env, 'positions', [])
            if not positions:
                self.position_performance.clear()
                return 0.0
            
            total_reward = 0.0
            
            for pos in positions:
                pos_id = pos.get('id', str(hash(str(pos))))
                unrealized_pnl = pos.get('unrealized_pnl', 0.0)
                pnl_percent = unrealized_pnl / self.initial_balance
                
                # Reward por posições lucrativas
                if pnl_percent > 0.01:  # >1% profit
                    performance_bonus = self.position_performance_weight * pnl_percent * 50.0
                    total_reward += performance_bonus
                elif pnl_percent < -0.02:  # <-2% loss
                    performance_penalty = -self.position_performance_weight * abs(pnl_percent) * 25.0
                    total_reward += performance_penalty
                
                # Track performance
                self.position_performance[pos_id] = pnl_percent
            
            return total_reward
            
        except:
            return 0.0
    
    def _calculate_momentum_shaping_v4(self, env) -> float:
        """V4 INNO: Momentum reward amplificado"""
        try:
            # Obter PnL do último trade fechado
            trades = getattr(env, 'trades', [])
            if not trades:
                return 0.0
            
            # Pegar último PnL
            last_pnl = trades[-1].get('pnl_usd', 0.0)
            
            # Adicionar ao trend tracking
            self.recent_pnl_trend.append(last_pnl)
            if len(self.recent_pnl_trend) > 10:
                self.recent_pnl_trend.pop(0)
            
            # Calcular momentum (tendência recente)
            if len(self.recent_pnl_trend) >= 3:
                recent_avg = np.mean(self.recent_pnl_trend[-3:])
                momentum = recent_avg / self.initial_balance
                
                # 🚀 V4 INNO: Momentum reward amplificado
                if momentum > 0.01:  # >1% momentum
                    return self.momentum_weight * momentum * 25.0  # 2.5x amplificado
                elif momentum < -0.01:  # <-1% momentum negativo
                    return self.momentum_weight * momentum * 10.0  # 2x amplificado
            
            return 0.0
            
        except:
            return 0.0
    
    def _calculate_action_decisiveness_v4(self, action: np.ndarray, env) -> float:
        """🚀 V4 INNO: Action decisiveness AMPLIFICADO (50x)"""
        try:
            if len(action) == 0:
                return -0.02  # Penalty aumentada por não-ação
            
            # Calcular "força" da decisão
            action_magnitude = np.sum(np.abs(action))
            
            # 🚀 V4 INNO: Action bonus AMPLIFICADO
            if action_magnitude > 0.1:
                # Bonus proporcional ao PnL atual
                current_portfolio = getattr(env, 'portfolio_value', self.initial_balance)
                pnl_percent = (current_portfolio - self.initial_balance) / self.initial_balance
                pnl_magnitude = abs(pnl_percent)
                
                # Amplificado: 0.001 -> 0.05 (50x)
                return self.action_decisiveness_weight * (1.0 + pnl_magnitude * 10.0)
            else:
                # Penalty amplificada por indecisão: 0.0005 -> 0.02 (40x)
                return -self.action_decisiveness_weight * 0.4
                
        except:
            return 0.0
    
    def _calculate_activity_bonus(self, env) -> Tuple[float, Dict]:
        """
        🚀 V4 INNO: Bonus por atividade de trading saudável
        """
        try:
            trades = getattr(env, 'trades', [])
            trades_count = len(trades)
            
            # Estimar trades por episódio (assumindo 500 steps = 1 episódio)
            current_step = getattr(env, 'current_step', self.step_counter)
            if current_step > 0:
                estimated_episodes = max(1, current_step / 500.0)
                trades_per_episode = trades_count / estimated_episodes
            else:
                trades_per_episode = 0
            
            # 🚀 V4 INNO: Target 3-12 trades por episódio
            if 3 <= trades_per_episode <= 12:
                bonus = 0.08  # Bonus significativo para range ideal
                status = 'optimal'
            elif trades_per_episode < 3:
                # Penalty por poucos trades (aumenta atividade)
                shortage = 3 - trades_per_episode
                bonus = -0.04 * shortage  # Penalty progressiva
                status = 'too_low'
            else:
                # Penalty leve por overtrading
                excess = trades_per_episode - 12
                bonus = -0.02 * excess
                status = 'too_high'
            
            # Bonus adicional por consistência
            if len(trades) >= 2:
                # Calcular volatilidade dos PnLs
                pnls = [trade.get('pnl_usd', 0) for trade in trades[-5:]]  # Últimos 5 trades
                if len(pnls) >= 3:
                    pnl_std = np.std(pnls)
                    avg_pnl = np.mean(pnls)
                    if avg_pnl > 0 and pnl_std < abs(avg_pnl) * 2:  # Baixa volatilidade com lucro médio
                        bonus += 0.02  # Bonus por consistência
                        status += '_consistent'
            
            info = {
                'trades_count': trades_count,
                'trades_per_episode': trades_per_episode,
                'activity_bonus': bonus,
                'activity_status': status,
                'target_range': '3-12 trades/episode'
            }
            
            return bonus, info
            
        except Exception as e:
            self.logger.error(f"Erro no activity bonus: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_portfolio_drawdown(self, env) -> float:
        """
        Calcula drawdown do portfolio (mantido do V3)
        """
        try:
            current_balance = getattr(env, 'portfolio_value', self.initial_balance)
            peak_balance = getattr(env, 'peak_portfolio_value', self.initial_balance)
            
            if peak_balance <= 0:
                return 0.0
                
            drawdown = (peak_balance - current_balance) / peak_balance
            return max(0.0, drawdown)
            
        except:
            return 0.0
    
    def _update_tracking_v4(self, env):
        """V4 INNO: Update tracking variables otimizado"""
        try:
            # Update last portfolio value
            self.last_portfolio_value = getattr(env, 'portfolio_value', self.initial_balance)
            
            # Limpar posições antigas do tracking
            current_positions = getattr(env, 'positions', [])
            current_pos_ids = set(pos.get('id', str(hash(str(pos)))) for pos in current_positions)
            
            # Remover posições que não existem mais
            for pos_id in list(self.position_performance.keys()):
                if pos_id not in current_pos_ids:
                    del self.position_performance[pos_id]
                    
        except:
            pass
    
    def _update_stats(self, reward: float):
        """Atualizar estatísticas para debugging"""
        self.total_rewards_given += 1
        
        if reward > 0:
            self.positive_rewards += 1
        elif reward < 0:
            self.negative_rewards += 1
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do reward system V4"""
        total = max(1, self.total_rewards_given)
        return {
            'total_rewards': self.total_rewards_given,
            'positive_ratio': self.positive_rewards / total,
            'negative_ratio': self.negative_rewards / total,
            'zero_ratio': (total - self.positive_rewards - self.negative_rewards) / total,
            'version': 'V4_INNO'
        }
    
    def reset(self):
        """Reset para novo episódio"""
        # Reset tracking para novo episódio
        self.last_portfolio_value = self.initial_balance
        self.position_performance.clear()
        self.recent_pnl_trend.clear()
        
        # Manter apenas stats cumulativas


def test_reward_balance():
    """🧪 TESTE DE BALANCE - V3 vs V4 COMPARAÇÃO"""
    
    class MockEnv:
        def __init__(self, realized_pnl, unrealized_pnl, portfolio_value, peak_value, trades=None, positions=None, current_step=0):
            self.total_realized_pnl = realized_pnl
            self.total_unrealized_pnl = unrealized_pnl
            self.portfolio_value = portfolio_value
            self.peak_portfolio_value = peak_value
            self.trades = trades or []
            self.positions = positions or []
            self.current_step = current_step
    
    # Importar V3 para comparação
    import sys
    sys.path.append('D:/Projeto/trading_framework/rewards/')
    from reward_daytrade_v3_brutal import BrutalMoneyReward
    
    reward_v3 = BrutalMoneyReward(initial_balance=10000)
    reward_v4 = InnovativeMoneyReward(initial_balance=10000)
    
    # Cenários críticos para testar balance
    test_scenarios = [
        ("Neutro - sem PnL", MockEnv(0, 0, 10000, 10000, [], [], 100)),
        ("Pequeno lucro +1%", MockEnv(100, 0, 10100, 10100, [{'pnl_usd': 100}], [], 200)),
        ("Lucro médio +3%", MockEnv(300, 0, 10300, 10300, [{'pnl_usd': 300}], [], 300)),
        ("Grande lucro +8%", MockEnv(800, 0, 10800, 10800, [{'pnl_usd': 800}], [], 400)),
        ("Pequena perda -2%", MockEnv(-200, 0, 9800, 10000, [{'pnl_usd': -200}], [], 500)),
        ("Perda média -5%", MockEnv(-500, 0, 9500, 10000, [{'pnl_usd': -500}], [], 600)),
        ("Grande perda -10%", MockEnv(-1000, 0, 9000, 10000, [{'pnl_usd': -1000}], [], 700)),
        ("Posição aberta lucrativa +2%", MockEnv(0, 200, 10200, 10200, [], [{'id': 'pos1', 'unrealized_pnl': 200}], 800)),
        ("Posição aberta perdedora -3%", MockEnv(0, -300, 9700, 10000, [], [{'id': 'pos2', 'unrealized_pnl': -300}], 900)),
    ]
    
    print("🧪 TESTE DE BALANCE V3 BRUTAL vs V4 INNO")
    print("=" * 80)
    print(f"{'CENÁRIO':<25} {'V3 REWARD':<12} {'V4 REWARD':<12} {'DIFERENÇA':<12} {'RATIO':<8}")
    print("-" * 80)
    
    action_decisive = np.array([0.8, 0.6, 0.4, 0.2])  # Ação decisiva padrão
    
    for scenario, env in test_scenarios:
        # Testar V3
        reward_v3_val, info_v3, _ = reward_v3.calculate_reward_and_info(env, action_decisive, {})
        
        # Testar V4
        reward_v4_val, info_v4, _ = reward_v4.calculate_reward_and_info(env, action_decisive, {})
        
        # Calcular diferença
        diff = reward_v4_val - reward_v3_val
        ratio = reward_v4_val / reward_v3_val if reward_v3_val != 0 else float('inf') if reward_v4_val != 0 else 1.0
        
        print(f"{scenario:<25} {reward_v3_val:+8.4f}    {reward_v4_val:+8.4f}    {diff:+8.4f}    {ratio:6.2f}x")
    
    print("\n" + "=" * 80)
    print("📊 ANÁLISE DETALHADA - CASOS ESPECÍFICOS")
    print("=" * 80)
    
    # Teste detalhado em casos críticos
    critical_cases = [
        ("Lucro pequeno +1%", MockEnv(100, 0, 10100, 10100, [{'pnl_usd': 100}], [], 100)),
        ("Unrealized +2%", MockEnv(0, 200, 10200, 10200, [], [{'id': 'pos1', 'unrealized_pnl': 200}], 200)),
        ("Perda -5%", MockEnv(-500, 0, 9500, 10000, [{'pnl_usd': -500}], [], 300)),
    ]
    
    for case_name, env in critical_cases:
        print(f"\n🔍 {case_name}")
        print("-" * 40)
        
        # V3 Detalhado
        reward_v3_val, info_v3, _ = reward_v3.calculate_reward_and_info(env, action_decisive, {})
        print(f"V3 Brutal:")
        print(f"  Total: {reward_v3_val:+.4f}")
        print(f"  PnL: {info_v3.get('pnl_reward', 0):+.4f}")
        print(f"  Shaping: {info_v3.get('proportional_shaping', 0):+.4f}")
        print(f"  PnL %: {info_v3.get('pnl_percent', 0):+.1f}%")
        
        # V4 Detalhado
        reward_v4_val, info_v4, _ = reward_v4.calculate_reward_and_info(env, action_decisive, {})
        print(f"V4 Inno:")
        print(f"  Total: {reward_v4_val:+.4f}")
        print(f"  PnL Component (80%): {info_v4.get('pure_pnl_component', 0):+.4f}")
        print(f"  Shaping Component (20%): {info_v4.get('shaping_component', 0):+.4f}")
        print(f"  Raw PnL Reward: {info_v4.get('pnl_reward', 0):+.4f}")
        print(f"  PnL %: {info_v4.get('pnl_percent', 0):+.1f}%")
        print(f"  Unrealized Factor: {info_v4.get('v4_unrealized_factor', 'N/A')}")
        print(f"  PnL Multiplier: {info_v4.get('v4_pnl_multiplier', 'N/A')}")
        
        # Comparação
        improvement = ((reward_v4_val - reward_v3_val) / abs(reward_v3_val) * 100) if reward_v3_val != 0 else 0
        print(f"📈 Melhoria V4: {improvement:+.1f}%")


def test_innovative_reward_v4():
    """Teste básico do sistema V4 INNO"""
    
    class MockEnv:
        def __init__(self, realized_pnl, unrealized_pnl, portfolio_value, peak_value, trades=None, positions=None, current_step=0):
            self.total_realized_pnl = realized_pnl
            self.total_unrealized_pnl = unrealized_pnl
            self.portfolio_value = portfolio_value
            self.peak_portfolio_value = peak_value
            self.trades = trades or []
            self.positions = positions or []
            self.current_step = current_step
    
    reward_system = InnovativeMoneyReward(initial_balance=10000)
    
    # Simular sequência de trading V4
    scenarios = [
        ("Início - sem trades", MockEnv(0, 0, 10000, 10000, [], [], 0)),
        ("Poucos trades (2)", MockEnv(200, 0, 10200, 10200, [{'pnl_usd': 100}, {'pnl_usd': 100}], [], 250)),  # 2 trades em 250 steps = 4 trades/episódio
        ("Atividade ótima (6 trades)", MockEnv(500, 0, 10500, 10500, [{'pnl_usd': 80} for _ in range(6)], [], 500)),  # 6 trades em 500 steps = 6 trades/episódio
        ("Overtrading (20 trades)", MockEnv(400, 0, 10400, 10400, [{'pnl_usd': 20} for _ in range(20)], [], 500)),  # 20 trades em 500 steps = 20 trades/episódio
        ("Trades consistentes", MockEnv(400, 0, 10400, 10400, [{'pnl_usd': 80}, {'pnl_usd': 85}, {'pnl_usd': 75}, {'pnl_usd': 90}, {'pnl_usd': 70}], [], 500)),  # Trades consistentes
    ]
    
    print("🚀 TESTE INNOVATIVE MONEY REWARD V4")
    print("=" * 70)
    
    for i, (scenario, env) in enumerate(scenarios):
        # Simular diferentes ações
        if i % 3 == 0:
            action = np.array([0.9, 0.7, 0.5, 0.3])  # Ação muito decisiva
        elif i % 3 == 1:
            action = np.array([0.3, 0.2, 0.1, 0.1])  # Ação moderada
        else:
            action = np.array([0.01, 0.005, 0.0, 0.0])  # Ação indecisa
        
        reward, info, done = reward_system.calculate_reward_and_info(env, action, {})
        
        print(f"\n{i+1}. {scenario}")
        print(f"  🎯 Total Reward: {reward:+.4f}")
        print(f"    ├─ PnL Component (70%): {info.get('pure_pnl_component', 0):+.4f}")
        print(f"    ├─ Shaping Component (20%): {info.get('shaping_component', 0):+.4f}")
        print(f"    └─ Activity Component (10%): {info.get('activity_component', 0):+.4f}")
        print(f"  📈 PnL Details:")
        print(f"    ├─ PnL %: {info.get('pnl_percent', 0):+.1f}%")
        print(f"    ├─ Unrealized Factor: {info.get('v4_unrealized_factor', 0)}")
        print(f"    └─ PnL Multiplier: {info.get('v4_pnl_multiplier', 0)}")
        print(f"  🎭 Shaping Details:")
        print(f"    ├─ Progress: {info.get('progress_reward', 0):+.4f}")
        print(f"    ├─ Momentum: {info.get('momentum_reward', 0):+.4f}")
        print(f"    ├─ Performance: {info.get('performance_reward', 0):+.4f}")
        print(f"    └─ Decisiveness: {info.get('decisiveness_reward', 0):+.4f}")
        print(f"  📊 Activity Details:")
        print(f"    ├─ Trades/Episode: {info.get('trades_per_episode', 0):.1f}")
        print(f"    ├─ Status: {info.get('activity_status', 'N/A')}")
        print(f"    └─ Target: {info.get('target_range', 'N/A')}")
        print(f"  ⚠️ Risk: Drawdown {info.get('portfolio_drawdown', 0):.1f}% | Done: {done}")


# Factory function para compatibilidade
def create_innovative_daytrade_reward_system(initial_balance: float = 1000.0):
    """Factory function para o sistema V4 INNO"""
    return InnovativeMoneyReward(initial_balance)

if __name__ == "__main__":
    print("Escolha o teste:")
    print("1. Teste básico V4")
    print("2. Teste de balance V3 vs V4")
    
    choice = input("Digite 1 ou 2: ")
    
    if choice == "2":
        test_reward_balance()
    else:
        test_innovative_reward_v4()